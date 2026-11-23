"""Stage 6: compose the statements and names texts for every file that has declarations.

Reads   data/interim/files.parquet, data/interim/decls.parquet,
        the records' declaration shards (for `uses`), the source tree
Writes  data/full/statements.parquet   one row per file, in embedding order

The recipe is the pilot's "statements" variant (stage 2): for theorems and lemmas
"kind shortName signature" from the records, for definition-like declarations the whole
declaration from the source, capped. No path, no imports, no `open` lines, no proofs.

One addition. 3.9% of theorems are stated as `theorem foo : Foo`, where `Foo` is a
proposition defined elsewhere, so the statement carries no mathematics and such files
cluster by form. When `Foo` is defined in another file, its definition is appended under
the theorem, once per file. The records list what each declaration uses by full name,
which resolves `Foo` without guessing at namespaces. One level only: a definition that
itself names further propositions is not expanded again.

The names text is the pilot's "names" variant, unchanged: the path components, then the
full name of every declaration. It carries the directory tree and no mathematics, and
exists so a layout can blend some of it in.

Row order is the embedding order: a stratified subset of DENSE_N files first (the pilot's
sqrt allocation over second-level directories), then every other file by path. Stage 7
can therefore stop after the subset, and the layout can be checked at four times the
pilot's density before the rest is embedded.
"""

from __future__ import annotations

import glob
import gzip
import hashlib
import json

import numpy as np
import pandas as pd

from riemann_map import (
    FULL,
    INTERIM,
    PILOT,
    RECORDS_DIR,
    SOURCE_DIR,
    allocate,
    parse_opens,
    write_parquet_safely,
)

DENSE_N = 40_000
SEED = 0
DEF_CAP = 2_000  # characters kept from one definition-like declaration
PROOF_KINDS = {"theorem", "lemma"}
UNIVERSES = r"\.\{[^}]*\}"  # `.{u}` and `.{0, 0}` after a name
BARE_SIG = r"^:\s*([^\s():]+)$"  # a signature that is one name and nothing else


def main() -> None:
    files = pd.read_parquet(
        INTERIM / "files.parquet", columns=["module", "path", "sha256", "dir2", "n_decls"]
    )
    eligible = files[files["n_decls"] > 0].sort_values("path").reset_index(drop=True)
    print(
        f"files: {len(files):,} in → {len(eligible):,} eligible "
        f"({len(files) - len(eligible):,} dropped: no declarations to embed)"
    )

    decls = pd.read_parquet(
        INTERIM / "decls.parquet",
        columns=["name", "kind", "module", "span_start", "span_end", "sig"],
    )
    decls = decls.sort_values(["module", "span_start"], kind="stable").reset_index(drop=True)
    decls["sig"] = decls["sig"].fillna("")
    is_proof = decls["kind"].isin(PROOF_KINDS)

    # --- theorems stated as a bare named proposition -------------------------------
    sig_plain = decls["sig"].str.replace(UNIVERSES, "", regex=True)
    bare = sig_plain.str.extract(BARE_SIG)[0].str.removeprefix("_root_.").where(is_proof)
    n_bare = int(bare.notna().sum())
    print(f"theorems stated as one bare name: {n_bare:,} of {int(is_proof.sum()):,}")

    # (module, span_start) identifies a declaration; names are not unique.
    wanted = dict(zip(zip(decls["module"][bare.notna()], decls["span_start"][bare.notna()]), bare.dropna()))
    resolved: dict[tuple[str, int], str] = {}
    n_none = n_many = 0
    for shard in sorted(glob.glob(str(RECORDS_DIR / "index" / "decls" / "*.jsonl.gz"))):
        with gzip.open(shard, "rt") as f:
            for line in f:
                r = json.loads(line)
                key = (r["module"], r["span"][0])
                x = wanted.get(key)
                if x is None:
                    continue
                cands = {u for u in r["uses"] + r["weak"] if u == x or u.endswith("." + x)}
                if len(cands) == 1:
                    resolved[key] = cands.pop()
                elif cands:
                    n_many += 1
                else:
                    n_none += 1
    print(
        f"  the records' uses name the proposition for {len(resolved):,}; "
        f"{n_none:,} name none (local notation, `letI`, universe forms), {n_many:,} name several"
    )

    # The proposition's own declaration: it must be the only declaration of that name.
    name_counts = decls["name"].value_counts()
    unique_names = set(name_counts[name_counts == 1].index)
    prop_rows = decls[decls["name"].isin(set(resolved.values()) & unique_names) & ~is_proof]
    prop_at = {
        r.name: (r.module, r.span_start, r.span_end)
        for r in prop_rows[["name", "module", "span_start", "span_end"]].itertuples(index=False)
    }
    expand: list[str | None] = [None] * len(decls)
    pos = {key: i for i, key in enumerate(zip(decls["module"], decls["span_start"]))}
    n_same_file = n_no_decl = 0
    for key, prop in resolved.items():
        where = prop_at.get(prop)
        if where is None:
            n_no_decl += 1
        elif where[0] == key[0]:
            n_same_file += 1  # the definition is already part of this file's text
        else:
            expand[pos[key]] = prop
    needed = {p for p in expand if p is not None}
    print(
        f"  defined in another file → expanded: {sum(p is not None for p in expand):,} theorems, "
        f"{len(needed):,} distinct propositions; defined in the same file: {n_same_file:,}; "
        f"no single definition found: {n_no_decl:,}"
    )

    path_of = dict(zip(files["module"], files["path"]))
    by_module: dict[str, list[str]] = {}
    for p in needed:
        by_module.setdefault(prop_at[p][0], []).append(p)
    prop_text: dict[str, str] = {}
    for module, props in by_module.items():
        raw = (SOURCE_DIR / path_of[module]).read_bytes()
        for p in props:
            _, s0, s1 = prop_at[p]
            prop_text[p] = raw[s0:s1].decode("utf-8", errors="replace")[:DEF_CAP].strip()

    # --- compose ------------------------------------------------------------------------
    codes = pd.factorize(decls["module"])[0]
    first = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    last = np.r_[first[1:], len(decls)]
    bounds = dict(zip(decls["module"].to_numpy()[first], zip(first.tolist(), last.tolist())))
    kind = decls["kind"].tolist()
    name = decls["name"].tolist()
    sig = decls["sig"].tolist()
    s0s = decls["span_start"].tolist()
    s1s = decls["span_end"].tolist()

    texts, name_texts, firsts, opens, n_expanded, only_bare = [], [], [], [], [], []
    sha_bad = capped = 0
    for row in eligible.itertuples(index=False):
        raw = (SOURCE_DIR / row.path).read_bytes()
        sha_bad += hashlib.sha256(raw).hexdigest() != row.sha256
        lo, hi = bounds[row.module]
        stmts: list[str] = []
        seen: set[str] = set()
        all_bare = True
        for i in range(lo, hi):
            if kind[i] in PROOF_KINDS:
                s = f"{kind[i]} {name[i].rsplit('.', 1)[-1]} {sig[i]}".strip()
                prop = expand[i]
                if prop is not None and prop not in seen:
                    seen.add(prop)
                    s += "\n" + prop_text[prop]
                all_bare &= (row.module, s0s[i]) in wanted
                stmts.append(s)
            else:
                span = raw[s0s[i] : s1s[i]].decode("utf-8", errors="replace")
                capped += len(span) > DEF_CAP
                stmts.append(span[:DEF_CAP].strip())
                all_bare = False
        texts.append("\n\n".join(stmts))
        path_words = " / ".join(row.path.removesuffix(".lean").split("/"))
        name_texts.append(path_words + "\n" + "\n".join(name[lo:hi]))
        firsts.append(stmts[0][:600])
        opens.append(parse_opens(raw.decode("utf-8")))
        n_expanded.append(len(seen))
        only_bare.append(all_bare)
    assert sha_bad == 0, f"{sha_bad} source files differ from the files the records describe"
    print(f"sha256 matches the records: {len(eligible):,} of {len(eligible):,}")

    out = eligible[["module", "path", "dir2"]].copy()
    out["text_statements"] = texts
    out["text_names"] = name_texts
    out["first_statement"] = firsts
    out["opens"] = opens
    out["n_props_expanded"] = np.array(n_expanded, dtype="int32")
    out["only_bare_theorems"] = only_bare
    n_chars = out["text_statements"].str.len()
    assert (n_chars > 0).all(), f"{int((n_chars == 0).sum())} files composed to an empty text"
    print(f"definition-like declarations cut at {DEF_CAP} chars: {capped:,}")
    print(
        f"statements chars: total {n_chars.sum():,}, median {n_chars.median():,.0f}, "
        f"p90 {n_chars.quantile(0.9):,.0f}, max {n_chars.max():,}"
    )
    ob = out["only_bare_theorems"]
    print(
        f"files holding only bare-name theorems: {int(ob.sum()):,}; "
        f"of those, now carrying at least one definition: {int((ob & (out['n_props_expanded'] > 0)).sum()):,}"
    )
    print(f"files whose text differs from the pilot recipe: {int((out['n_props_expanded'] > 0).sum()):,}")
    print(f"names chars: total {out['text_names'].str.len().sum():,}")

    # The pilot's texts must come back unchanged wherever nothing was expanded.
    pilot_path = PILOT / "sample.parquet"
    if pilot_path.exists():
        pilot = pd.read_parquet(pilot_path, columns=["module", "text_statements", "text_names"])
        both = pilot.merge(out, on="module", suffixes=("_pilot", ""), validate="one_to_one")
        assert (both["text_names_pilot"] == both["text_names"]).all(), "a names text differs from the pilot"
        same = both["text_statements_pilot"] == both["text_statements"]
        untouched = both["n_props_expanded"] == 0
        assert (same | ~untouched).all(), "a file with no expansion differs from the pilot text"
        print(
            f"pilot check: {int((same & untouched).sum()):,} of {int(untouched.sum()):,} unexpanded "
            f"pilot files are identical; {int((~untouched).sum()):,} pilot files gained a definition"
        )

    # --- embedding order: a dense stratified subset first -------------------------------
    counts = out["dir2"].value_counts().sort_index()
    alloc = allocate(counts, DENSE_N)
    assert alloc.sum() == DENSE_N and (alloc <= counts).all()
    rng = np.random.default_rng(SEED)
    dense = np.zeros(len(out), dtype=bool)
    for dir2, idx in out.groupby("dir2", sort=True).indices.items():
        k = int(alloc[dir2])
        if k:
            dense[np.sort(rng.choice(idx, size=k, replace=False))] = True
    assert dense.sum() == DENSE_N
    out["dense_subset"] = dense
    out = out.sort_values(["dense_subset", "path"], ascending=[False, True]).reset_index(drop=True)
    assert out["dense_subset"].iloc[:DENSE_N].all() and not out["dense_subset"].iloc[DENSE_N:].any()
    assert out["module"].is_unique
    top = pd.DataFrame({"files": counts, "dense": alloc}).sort_values("files", ascending=False)
    top["rate"] = (top["dense"] / top["files"]).round(3)
    print(
        f"dense subset: {DENSE_N:,} files, {int((alloc == counts).sum())} of {len(counts)} parts taken whole"
    )
    print(top.head(6).to_string())

    write_parquet_safely(out, FULL / "statements.parquet")
    print(
        f"\nwrote {FULL / 'statements.parquet'} ({len(out):,} rows, first {DENSE_N:,} are the dense subset)"
    )


if __name__ == "__main__":
    main()
