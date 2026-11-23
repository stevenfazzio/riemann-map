"""Stage 1: turn the CC0 records into two tables.

Reads   data/raw/records-r2/x/records/{index,spine}
Writes  data/interim/decls.parquet   one row per declaration (name, kind, module, sig, ...)
        data/interim/files.parquet   one row per source file, with path-derived and
                                     dependency-derived fields but no text

The records come from a third-party source scan (tamnd/riemann, `rh index` / `rh spine`),
not from a Lean build: `uses` edges are syntactic. Field meanings observed here:
  modules.depth  longest import chain from Main, in edges (Main = 0, max 202)
  spine.depth    distance from riemannHypothesis in the uses-graph
  spine.layer    the named branch of the proof a declaration sits under; absent for
                 declarations the records do not assign to a single branch
"""

from __future__ import annotations

import glob
import gzip
import json

import pandas as pd

from riemann_map import INTERIM, RECORDS_DIR, write_parquet_safely


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        for line in f:
            yield json.loads(line)


def main() -> None:
    summary = json.loads((RECORDS_DIR / "index" / "summary.json").read_text())

    # --- modules -----------------------------------------------------------------
    rows = []
    for d in read_jsonl_gz(RECORDS_DIR / "index" / "modules.jsonl.gz"):
        imports = d["imports"]
        rows.append(
            {
                "module": d["module"],
                "path": d["path"],
                "lines": d["lines"],
                "bytes": d["bytes"],
                "sha256": d["sha256"],
                "n_decls_records": d["decls"],
                "import_depth": d["depth"],
                "imports": imports,
            }
        )
    files = pd.DataFrame(rows)
    assert len(files) == summary["modules"], (len(files), summary["modules"])
    assert files["module"].is_unique and files["path"].is_unique
    print(f"modules: {len(files):,} rows (summary says {summary['modules']:,})")

    internal = set(files["module"])
    files["n_imports_internal"] = files["imports"].map(lambda xs: sum(x in internal for x in xs))
    files["n_imports_external"] = files["imports"].map(lambda xs: sum(x not in internal for x in xs))
    fan_in = pd.Series([x for xs in files["imports"] for x in xs if x in internal]).value_counts()
    files["import_fan_in"] = files["module"].map(fan_in).fillna(0).astype("int32")

    parts = files["path"].str.split("/")
    files["library"] = parts.str[0].str.replace(r"\.lean$", "", regex=True)
    # "part" is the second path component; top-level files and one-level libraries get "(root)".
    files["part"] = parts.map(lambda p: p[1] if len(p) > 2 else "(root)")
    files["dir2"] = parts.map(lambda p: "/".join(p[:2]) if len(p) > 2 else p[0])
    files["dir3"] = parts.map(lambda p: "/".join(p[:3]) if len(p) > 3 else "/".join(p[:-1]) or p[0])
    files["parent_dir"] = parts.map(lambda p: "/".join(p[:-1]) or "(root)")
    files["path_depth"] = parts.map(len)

    # --- declarations ------------------------------------------------------------
    shards = sorted(glob.glob(str(RECORDS_DIR / "index" / "decls" / "*.jsonl.gz")))
    drows = []
    for shard in shards:
        for d in read_jsonl_gz(shard):
            drows.append(
                (
                    d["name"],
                    d["kind"],
                    d["module"],
                    d["line"],
                    d["span"][0],
                    d["span"][1],
                    d["sig"],
                    len(d["uses"]),
                    len(d["weak"]),
                )
            )
    decls = pd.DataFrame(
        drows,
        columns=["name", "kind", "module", "line", "span_start", "span_end", "sig", "n_uses", "n_weak"],
    )
    del drows
    print(f"decls: {len(decls):,} rows from {len(shards)} shards (summary says {summary['decls']:,})")
    assert len(decls) == summary["decls"], (len(decls), summary["decls"])
    orphan = ~decls["module"].isin(internal)
    assert not orphan.any(), f"{orphan.sum()} declarations name a module that is not in modules"
    print("decl kinds:", decls["kind"].value_counts().to_dict())
    print(f"sig chars: total {decls['sig'].str.len().sum():,}, median {decls['sig'].str.len().median():.0f}")

    # --- spine (declarations the proof uses) ---------------------------------------
    srows = [
        (d["name"], d["module"], d["depth"], d["fanin"], d["size"], d.get("layer"))
        for d in read_jsonl_gz(RECORDS_DIR / "spine" / "spine.jsonl.gz")
    ]
    spine = pd.DataFrame(
        srows, columns=["name", "module", "use_depth", "use_fan_in", "closure_size", "branch"]
    )
    print(f"spine: {len(spine):,} declarations used by riemannHypothesis")
    print(
        "spine branch (None = not assigned to one branch):",
        spine["branch"].value_counts(dropna=False).to_dict(),
    )

    # Names are not unique (summary: duplicate_names), so join spine to decls on (module, name).
    dup = decls.duplicated(["module", "name"], keep=False).sum()
    print(f"decls sharing (module, name) with another decl: {dup:,}")
    spine_key = spine.drop_duplicates(["module", "name"])
    print(f"spine: {len(spine) - len(spine_key):,} duplicate (module, name) rows collapsed for the join")
    decls = decls.merge(spine_key, on=["module", "name"], how="left", validate="many_to_one")
    decls["used_by_proof"] = decls["use_depth"].notna()
    print(f"decls marked used_by_proof: {decls['used_by_proof'].sum():,} of {len(decls):,}")
    assert len(decls) == summary["decls"], "the spine join changed the number of declarations"

    # --- per-file aggregates -------------------------------------------------------
    g = decls.groupby("module", sort=False)
    agg = g.agg(
        n_decls=("name", "size"),
        n_theorems=("kind", lambda s: int(s.isin(["theorem", "lemma"]).sum())),
        n_defs=(
            "kind",
            lambda s: int(s.isin(["def", "abbrev", "structure", "inductive", "class", "instance"]).sum()),
        ),
        n_used_by_proof=("used_by_proof", "sum"),
        min_use_depth=("use_depth", "min"),
        max_closure_size=("closure_size", "max"),
        sig_chars=("sig", lambda s: int(s.str.len().sum())),
    )

    def top_branch(s: pd.Series):
        s = s.dropna()
        return s.value_counts().index[0] if len(s) else None

    agg["branch"] = g["branch"].agg(top_branch)
    n_before = len(files)
    files = files.merge(agg, left_on="module", right_index=True, how="left", validate="one_to_one")
    assert len(files) == n_before
    no_decls = files["n_decls"].isna()
    print(f"files with no declarations in the records: {no_decls.sum():,} (kept, n_decls = 0)")
    for c in ["n_decls", "n_theorems", "n_defs", "n_used_by_proof", "sig_chars"]:
        files[c] = files[c].fillna(0).astype("int32")
    mismatch = (files["n_decls"] != files["n_decls_records"]).sum()
    print(f"files whose declaration count differs from modules.decls: {mismatch:,}")

    print("\nfiles per library:")
    print(files["library"].value_counts().to_string())
    print(
        f"\nparts (dir2): {files['dir2'].nunique():,}; dir3: {files['dir3'].nunique():,}; "
        f"parent dirs: {files['parent_dir'].nunique():,}"
    )
    print("decls per file: ", files["n_decls"].describe(percentiles=[0.5, 0.9, 0.99]).round(1).to_dict())

    write_parquet_safely(decls, INTERIM / "decls.parquet")
    write_parquet_safely(files, INTERIM / "files.parquet")
    print(
        f"\nwrote {INTERIM / 'decls.parquet'} ({len(decls):,} rows), "
        f"{INTERIM / 'files.parquet'} ({len(files):,} rows)"
    )


if __name__ == "__main__":
    main()
