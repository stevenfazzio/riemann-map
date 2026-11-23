"""Shared paths, safe-write helpers and sampling for the riemann-map pipeline."""

from __future__ import annotations

import os
import re
import stat
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
# Byte-for-byte copy of the 2026-11-20 release (github.com/tamnd/riemann-lean, lean/src).
SOURCE_DIR = RAW / "riemann-lean" / "lean" / "src"
# CC0 records from github.com/tamnd/riemann, release records-9110120-r2.
RECORDS_DIR = RAW / "records-r2" / "x" / "records"
INTERIM = DATA / "interim"
PILOT = DATA / "pilot"
FULL = DATA / "full"
OUTPUT = ROOT / "output"

SOURCE_URL = "https://github.com/tamnd/riemann-lean/blob/main/lean/src/"

# Rows per resumable piece of the full embedding run (stage 7).
EMBED_CHUNK = 1000


def _match_mode(tmp_path: str, output_path: Path) -> None:
    # mkstemp creates 0600 and os.replace keeps the source mode, so set it explicitly.
    if output_path.exists():
        os.chmod(tmp_path, stat.S_IMODE(output_path.stat().st_mode))
    else:
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp_path, 0o666 & ~umask)


def write_parquet_safely(df, output_path) -> None:
    """Write df so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".parquet.tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp_path, compression="zstd")
        _match_mode(tmp_path, output_path)
        meta = pq.read_metadata(tmp_path)
        assert meta.num_rows == len(df), f"row count {meta.num_rows} != {len(df)}"
        on_disk = set(meta.schema.to_arrow_schema().names)
        assert on_disk >= set(df.columns), f"columns missing on disk: {set(df.columns) - on_disk}"
        os.replace(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def write_npy_safely(arr: np.ndarray, output_path) -> None:
    """Write an array so output_path only ever holds a complete, verified file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=output_path.parent, suffix=".npy.tmp")
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        _match_mode(tmp_path, output_path)
        back = np.load(tmp_path, mmap_mode="r")
        assert back.shape == arr.shape, f"shape {back.shape} != {arr.shape}"
        os.replace(tmp_path, output_path)
    except BaseException:
        os.unlink(tmp_path)
        raise


def write_text_safely(text: str, output_path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_name(output_path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, output_path)


# --- the mapped sets -------------------------------------------------------------------
# "pilot": the 10,000-file sample of stages 2-5.
# "dense": the first 40,000 rows of stage 6's table: "statements" from the local model
#          (stage 7) and "q8_inst" from Qwen3 (stage 8), read from finished files or pieces.
# "full":  every file with declarations, Qwen3 only.
RUN_DIRS = {"pilot": PILOT, "dense": FULL / "dense", "full": FULL / "map"}
# Pilot variants. Local model (gte-modernbert-base): names, statements, file, and
# "expanded", the full run's statements recipe. Hosted Qwen3-Embedding-8B (stage 8):
# q8_statements, q8_inst (the same text behind a subject instruction), q8_names.
# A blend mixes a names embedding into a statements embedding; the weight is the names
# share of the cosine similarity: cos = w * cos_names + (1 - w) * cos_statements.
BLENDS = {
    "blend30": ("names", "expanded", 0.30),
    "q8_blend15": ("q8_names", "q8_statements", 0.15),
    "q8_blend30": ("q8_names", "q8_statements", 0.30),
    "q8_blend50": ("q8_names", "q8_statements", 0.50),
    "q8i_blend15": ("q8_names", "q8_inst", 0.15),
    "q8i_blend30": ("q8_names", "q8_inst", 0.30),
}
RUN_VARIANTS = {
    "pilot": ["names", "statements", "file", "expanded", "q8_names", "q8_statements", "q8_inst", *BLENDS],
    "dense": ["statements", "q8_inst"],
    "full": ["q8_inst", "q8i_blend15", "q8i_blend30"],
}


def load_sample(run: str) -> pd.DataFrame:
    """One row per mapped file, with the columns the pilot's sample.parquet carries."""
    if run == "pilot":
        return pd.read_parquet(PILOT / "sample.parquet")
    st = pd.read_parquet(
        FULL / "statements.parquet",
        columns=["module", "path", "opens", "first_statement", "dense_subset"],
    )
    n = len(st)
    if run == "dense":
        n = int(st["dense_subset"].sum())
        assert st["dense_subset"].iloc[:n].all(), "the dense subset must be the first rows"
    st = st.iloc[:n].drop(columns="dense_subset")
    files = pd.read_parquet(INTERIM / "files.parquet").drop(columns=["path"])
    out = st.merge(files, on="module", how="left", validate="one_to_one")
    decls = pd.read_parquet(INTERIM / "decls.parquet", columns=["module", "name", "span_start"])
    decls = decls[decls["module"].isin(set(out["module"]))].sort_values(["module", "span_start"])
    out["decl_names"] = out["module"].map(decls.groupby("module", sort=False)["name"].agg(list))
    assert len(out) == n and out["decl_names"].notna().all()
    return out


def load_embeddings(run: str, variant: str, n: int) -> np.ndarray | None:
    """Embeddings for the first n rows of a run, or None if they do not exist yet."""
    if variant in BLENDS:
        first, second, w = BLENDS[variant]
        a, b = load_embeddings(run, first, n), load_embeddings(run, second, n)
        if a is None or b is None:
            return None
        # Both parts are unit vectors, so the stacked vector is too and its cosine is
        # the weighted sum of the two cosines.
        # float32 weights: a NumPy float64 scalar would promote 175,013 x 2,048 values to float64.
        return np.hstack([np.float32(np.sqrt(w)) * a, np.float32(np.sqrt(1 - w)) * b])
    if run == "pilot":
        path = PILOT / f"emb_{variant}.npy"
        return np.load(path) if path.exists() else None
    final = FULL / f"emb_{variant}.npy"
    if final.exists():
        return np.ascontiguousarray(np.load(final, mmap_mode="r")[:n])
    pieces = [FULL / f"emb_{variant}.chunks" / f"{s:06d}.npy" for s in range(0, n, EMBED_CHUNK)]
    if not all(p.exists() for p in pieces):
        return None
    return np.concatenate([np.load(p) for p in pieces])[:n]


# --- stratified sampling ---------------------------------------------------------------
def allocate(counts: pd.Series, total: int) -> pd.Series:
    """alloc_i = min(n_i, c * sqrt(n_i)), with c chosen so the allocations sum to total."""
    root = np.sqrt(counts.to_numpy(dtype=float))
    n = counts.to_numpy(dtype=float)
    lo, hi = 0.0, float(n.max())
    for _ in range(100):
        mid = (lo + hi) / 2
        if np.minimum(n, mid * root).sum() < total:
            lo = mid
        else:
            hi = mid
    raw = np.minimum(n, hi * root)
    base = np.floor(raw).astype(int)
    short = total - base.sum()
    # Hand the remainder to the largest fractional parts that still have room.
    order = np.argsort(-(raw - base))
    for i in order:
        if short <= 0:
            break
        if base[i] < n[i]:
            base[i] += 1
            short -= 1
    return pd.Series(base, index=counts.index)


# --- `open` lines ----------------------------------------------------------------------
OPEN_RE = re.compile(r"^\s*open\s+(?:scoped\s+)?(.*)$")
IDENT_RE = re.compile(r"[A-Za-z_À-￿][\w.'À-￿]*")
OPEN_STOP = {"in", "hiding", "renaming", "scoped"}


def parse_opens(source: str) -> list[str]:
    """The namespaces a Lean file opens, sorted and de-duplicated."""
    names: list[str] = []
    for line in source.splitlines():
        m = OPEN_RE.match(line)
        if not m:
            continue
        rest = m.group(1).split("(")[0]
        for tok in IDENT_RE.findall(rest):
            if tok in OPEN_STOP:
                break
            names.append(tok)
    return sorted(set(names))


# --- which opened namespaces are Mathlib's ------------------------------------------
# A namespace root counts as the project's own when the project declares at least this
# many names under it. Checked by eye on the pilot sample: the rule misfiles exactly one
# frequently opened Mathlib namespace, in which the project declares 43 names.
INTERNAL_ROOT_MIN_DECLS = 20
MATHLIB_ROOTS_FORCED = {"AlgebraicGeometry"}


def project_namespace_roots(decl_names) -> set[str]:
    counts = decl_names.str.split(".", n=1).str[0].value_counts()
    return set(counts[counts >= INTERNAL_ROOT_MIN_DECLS].index) - MATHLIB_ROOTS_FORCED
