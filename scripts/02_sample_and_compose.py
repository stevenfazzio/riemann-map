"""Stage 2 (pilot): draw a stratified sample of files and compose the text variants.

Reads   data/interim/files.parquet, data/interim/decls.parquet, the source tree
Writes  data/pilot/sample.parquet

Sampling: strata are the 331 second-level directories ("dir2"). Each gets an allocation
proportional to sqrt(file count), capped at its size, so AbelianVarietyTheory (57% of
all files) does not swamp the rest while every part still appears.

Text variants, one string per file:
  names       the path components and the declaration names. No mathematical content;
              this is the "directory and naming" baseline.
  statements  what the file says, without proofs and without the path: for theorems and
              lemmas "kind shortName signature" from the records, for definitions,
              structures, instances and the like the whole declaration from the source
              (capped), because a definition's content is its body, not its type.
  file        the whole source file: imports, opens, statements and proofs.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from riemann_map import INTERIM, PILOT, SOURCE_DIR, allocate, parse_opens, write_parquet_safely

N_SAMPLE = 10_000
SEED = 0
DEF_CAP = 2_000  # characters kept from one definition-like declaration
PROOF_KINDS = {"theorem", "lemma"}


def main() -> None:
    files = pd.read_parquet(INTERIM / "files.parquet")
    eligible = files[files["n_decls"] > 0].sort_values("path").reset_index(drop=True)
    print(
        f"files: {len(files):,} in → {len(eligible):,} eligible "
        f"({len(files) - len(eligible):,} dropped: no declarations to embed)"
    )

    counts = eligible["dir2"].value_counts().sort_index()
    alloc = allocate(counts, N_SAMPLE)
    assert alloc.sum() == N_SAMPLE and (alloc <= counts).all()
    rng = np.random.default_rng(SEED)
    picked = []
    for dir2, grp in eligible.groupby("dir2", sort=True):
        k = int(alloc[dir2])
        if k:
            picked.append(grp.iloc[np.sort(rng.choice(len(grp), size=k, replace=False))])
    sample = pd.concat(picked).sort_values("path").reset_index(drop=True)
    assert len(sample) == N_SAMPLE
    top = pd.DataFrame({"files": counts, "sampled": alloc}).sort_values("files", ascending=False)
    top["rate"] = (top["sampled"] / top["files"]).round(3)
    print(
        f"sample: {len(sample):,} files from {int((alloc > 0).sum())} of {len(counts)} parts; "
        f"{int((alloc == counts).sum())} parts taken whole"
    )
    print(top.head(12).to_string())

    decls = pd.read_parquet(
        INTERIM / "decls.parquet",
        columns=["name", "kind", "module", "line", "span_start", "span_end", "sig"],
    )
    decls = decls[decls["module"].isin(set(sample["module"]))].sort_values(["module", "span_start"])
    by_module = {m: g for m, g in decls.groupby("module", sort=False)}
    print(f"declarations in sampled files: {len(decls):,}")

    out = {
        k: []
        for k in [
            "text_names",
            "text_statements",
            "text_file",
            "opens",
            "decl_names",
            "first_statement",
            "sha_ok",
        ]
    }
    sig_found = sig_missing = header_ok = n_decl_seen = capped = 0
    for row in sample.itertuples(index=False):
        raw = (SOURCE_DIR / row.path).read_bytes()
        out["sha_ok"].append(hashlib.sha256(raw).hexdigest() == row.sha256)
        source = raw.decode("utf-8")
        g = by_module[row.module]
        stmts = []
        for d in g.itertuples(index=False):
            n_decl_seen += 1
            # Spans are byte offsets into the file.
            span = raw[d.span_start : d.span_end].decode("utf-8", errors="replace")
            short = d.name.rsplit(".", 1)[-1]
            header_ok += (d.kind in span[:400]) and (short in span[:2000])
            if d.kind in PROOF_KINDS:
                # The records' sig is the signature with whitespace normalised, so it is
                # only sometimes a verbatim substring of the source. Compose every theorem
                # the same way from the record instead of slicing the source.
                i = span.find(d.sig) if d.sig else -1
                sig_found += i >= 0
                sig_missing += i < 0
                stmts.append(f"{d.kind} {short} {d.sig}".strip())
            else:
                if len(span) > DEF_CAP:
                    capped += 1
                stmts.append(span[:DEF_CAP].strip())
        names = g["name"].tolist()
        path_words = " / ".join(row.path.removesuffix(".lean").split("/"))
        out["text_names"].append(path_words + "\n" + "\n".join(names))
        out["text_statements"].append("\n\n".join(stmts))
        out["text_file"].append(source)
        out["opens"].append(parse_opens(source))
        out["decl_names"].append(names)
        out["first_statement"].append(stmts[0][:600])
    for k, v in out.items():
        sample[k] = v

    print(f"\nsha256 matches the records: {sample['sha_ok'].sum():,} of {len(sample):,}")
    assert sample["sha_ok"].all(), "the cloned source differs from the files the records describe"
    print(f"declaration spans that show their keyword and name: {header_ok:,} of {n_decl_seen:,}")
    n_proof = sig_found + sig_missing
    print(
        f"theorem signatures that are a verbatim substring of the source: {sig_found:,} of {n_proof:,} "
        f"(informational; all are composed as kind + short name + sig)"
    )
    print(f"definition-like declarations cut at {DEF_CAP} chars: {capped:,}")
    for c in ["text_names", "text_statements", "text_file"]:
        n = sample[c].str.len()
        print(
            f"{c:16s} chars: median {n.median():,.0f}  p90 {n.quantile(0.9):,.0f}  "
            f"max {n.max():,}  empty {int((n == 0).sum())}"
        )
    opens = pd.Series([o for xs in sample["opens"] for o in xs])
    print(
        f"\nfiles with at least one `open`: {(sample['opens'].map(len) > 0).sum():,}; "
        f"distinct opened namespaces: {opens.nunique():,}"
    )
    print("most opened namespaces (files):")
    print(opens.value_counts().head(70).to_string())

    write_parquet_safely(sample, PILOT / "sample.parquet")
    print(f"\nwrote {PILOT / 'sample.parquet'} ({len(sample):,} rows)")


if __name__ == "__main__":
    main()
