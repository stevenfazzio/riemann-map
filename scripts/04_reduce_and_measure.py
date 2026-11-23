"""Stage 4: reduce each variant to 2-d and measure what its neighbourhoods agree on.

Reads   the run's sample and embeddings (riemann_map.load_sample / load_embeddings)
Writes  {run dir}/coords_{variant}.npy   UMAP layout, row order = sample order
        {run dir}/metrics.json           the numbers printed below

    uv run python -I scripts/04_reduce_and_measure.py               # the 10k pilot
    uv run python -I scripts/04_reduce_and_measure.py --run dense   # first 40k of the full run
    uv run python -I scripts/04_reduce_and_measure.py --run full    # every file; about an hour

The question: does a layout built from the text only reproduce the directory tree, or
does it also group files by mathematics the tree does not show?

For each variant, in the embedding space (cosine) and in the 2-d layout, over each
file's K nearest neighbours:
  directory agreement   share of neighbours in the same library / dir2 / dir3 / parent
                        directory, against the share expected from a random neighbour
  Mathlib-open overlap  mean Jaccard overlap between the Mathlib namespaces two files
                        `open`, against random pairs. Neither `open` lines nor paths are
                        in the "statements" text, so for that variant this is an
                        independent check that neighbours share subject matter.
  cross-directory       the same overlap restricted to neighbour pairs from different
                        dir2 parts, against random pairs from different parts

And for the layout alone, how much shape it has:
  kept neighbours       share of a file's 2-d neighbours that are also among its
                        neighbours in the embedding the layout was made from
  largest cluster       share of all files in the biggest density cluster (HDBSCAN's
                        default selection); near 100% means one continuous cloud
  outside pockets       share of files in no dense pocket (HDBSCAN leaf selection)
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import umap
from sklearn.cluster import HDBSCAN
from sklearn.neighbors import NearestNeighbors

from riemann_map import (
    INTERIM,
    INTERNAL_ROOT_MIN_DECLS,
    RUN_DIRS,
    RUN_VARIANTS,
    load_embeddings,
    load_sample,
    project_namespace_roots,
    write_npy_safely,
    write_text_safely,
)

K = 15
SEED = 42
LEVELS = ["library", "dir2", "dir3", "parent_dir"]
# Density-cluster sizes for a 10,000-point layout, scaled with the point count.
PILOT_N, MIN_CLUSTER, MIN_SAMPLES = 10_000, 25, 10


def knn(x: np.ndarray, metric: str) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=K + 1, metric=metric).fit(x)
    idx = nn.kneighbors(x, return_distance=False)
    # Drop self. Exact duplicates can put another row first, so remove by identity.
    rows = np.arange(len(x))[:, None]
    keep = idx != rows
    out = np.empty((len(x), K), dtype=np.int64)
    for i in range(len(x)):
        out[i] = idx[i][keep[i]][:K]
    return out


def jaccard_pairs(sets: list[frozenset], a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.empty(len(a))
    for n, (i, j) in enumerate(zip(a, b)):
        u = len(sets[i] | sets[j])
        out[n] = len(sets[i] & sets[j]) / u if u else np.nan
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUN_DIRS), default="pilot")
    run = ap.parse_args().run
    run_dir, variants = RUN_DIRS[run], RUN_VARIANTS[run]
    sample = load_sample(run)[["path", "opens"] + LEVELS]
    n = len(sample)

    internal_roots = project_namespace_roots(
        pd.read_parquet(INTERIM / "decls.parquet", columns=["name"])["name"]
    )
    all_opens = pd.Series([o for xs in sample["opens"] for o in xs])
    is_internal = all_opens.str.split(".", n=1).str[0].isin(internal_roots)
    print(
        f"opened namespaces: {all_opens.nunique():,} distinct; "
        f"{all_opens[~is_internal].nunique():,} treated as Mathlib "
        f"(root has < {INTERNAL_ROOT_MIN_DECLS} project declarations)"
    )
    print("most opened Mathlib namespaces:", all_opens[~is_internal].value_counts().head(25).index.tolist())
    mathlib_sets = [
        frozenset(o for o in xs if o.split(".", 1)[0] not in internal_roots) for xs in sample["opens"]
    ]
    has_tags = np.array([len(s) > 0 for s in mathlib_sets])
    print(f"files that open at least one Mathlib namespace: {has_tags.sum():,} of {n:,}")

    codes = {lv: sample[lv].astype("category").cat.codes.to_numpy() for lv in LEVELS}
    chance = {lv: float((np.bincount(codes[lv]) / n) @ (np.bincount(codes[lv]) / n)) for lv in LEVELS}

    rng = np.random.default_rng(SEED)
    ra, rb = rng.integers(0, n, 200_000), rng.integers(0, n, 200_000)
    ok = (ra != rb) & has_tags[ra] & has_tags[rb]
    base_all = float(np.nanmean(jaccard_pairs(mathlib_sets, ra[ok], rb[ok])))
    cross = ok & (codes["dir2"][ra] != codes["dir2"][rb])
    base_cross = float(np.nanmean(jaccard_pairs(mathlib_sets, ra[cross], rb[cross])))

    metrics = {
        "k": K,
        "n": n,
        "chance": chance,
        "open_overlap_random": base_all,
        "open_overlap_random_cross_dir2": base_cross,
        "variants": {},
    }
    neighbours = {}
    for v in variants:
        emb = load_embeddings(run, v, n)
        if emb is None:
            print(f"{v}: no embeddings yet, skipping")
            continue
        assert emb.shape[0] == n, (emb.shape, n)
        coords_path = run_dir / f"coords_{v}.npy"
        if coords_path.exists():
            coords = np.load(coords_path)
        else:
            # One reduction, used for both clustering and plotting; min_dist stays low so
            # regions come out dense. A fixed random_state makes UMAP single-threaded.
            coords = umap.UMAP(
                n_neighbors=K,
                min_dist=0.05,
                metric="cosine",
                random_state=SEED,
                # The 0.5.x pipeline, stated so a umap upgrade cannot move the layout.
                compatibility_layout=True,
            ).fit_transform(emb)
            write_npy_safely(coords.astype(np.float32), coords_path)
        res = {}
        for space, x, metric in [("embedding", emb, "cosine"), ("layout", coords, "euclidean")]:
            nb = knn(x, metric)
            neighbours[(v, space)] = nb
            r = {lv: float((codes[lv][nb] == codes[lv][:, None]).mean()) for lv in LEVELS}
            src = np.repeat(np.arange(n), K)
            dst = nb.ravel()
            both = has_tags[src] & has_tags[dst]
            r["open_overlap"] = float(np.nanmean(jaccard_pairs(mathlib_sets, src[both], dst[both])))
            xdir = both & (codes["dir2"][src] != codes["dir2"][dst])
            r["open_overlap_cross_dir2"] = float(
                np.nanmean(jaccard_pairs(mathlib_sets, src[xdir], dst[xdir]))
            )
            r["share_neighbours_cross_dir2"] = float((codes["dir2"][src] != codes["dir2"][dst]).mean())
            res[space] = r
        emb_nb, lay_nb = neighbours[(v, "embedding")], neighbours[(v, "layout")]
        shape = {
            "kept_neighbours": float(np.mean([len(set(a) & set(b)) / K for a, b in zip(lay_nb, emb_nb)]))
        }
        sizes = {
            "min_cluster_size": round(MIN_CLUSTER * n / PILOT_N),
            "min_samples": round(MIN_SAMPLES * n / PILOT_N),
        }
        eom = HDBSCAN(copy=True, **sizes).fit_predict(coords)
        leaf = HDBSCAN(copy=True, cluster_selection_method="leaf", **sizes).fit_predict(coords)
        shape["largest_cluster"] = float(np.bincount(eom[eom >= 0]).max() / n) if (eom >= 0).any() else 0.0
        shape["outside_pockets"] = float(np.mean(leaf < 0))
        shape["pockets"] = int(leaf.max() + 1)
        res["shape"] = shape
        metrics["variants"][v] = res
        print(f"{v}: measured", flush=True)

    done = [v for v in variants if v in metrics["variants"]]
    overlap = {}
    for space in ["embedding", "layout"]:
        for i, a in enumerate(done):
            for b in done[i + 1 :]:
                na, nb_ = neighbours[(a, space)], neighbours[(b, space)]
                shared = np.mean([len(set(na[r]) & set(nb_[r])) / K for r in range(n)])
                overlap[f"{space}:{a}~{b}"] = float(shared)
    metrics["neighbour_set_overlap"] = overlap

    print(f"\nshare of the {K} nearest neighbours that match, n = {n:,}")
    head = (
        f"{'variant':14s} {'space':10s}"
        + "".join(f"{lv:>12s}" for lv in LEVELS)
        + f"{'open ovl':>10s}{'x-dir ovl':>11s}{'x-dir nbrs':>12s}"
    )
    print(head)
    print(
        f"{'(random)':14s} {'':10s}"
        + "".join(f"{chance[lv]:12.3f}" for lv in LEVELS)
        + f"{base_all:10.3f}{base_cross:11.3f}{'':>12s}"
    )
    for v in done:
        for space in ["embedding", "layout"]:
            r = metrics["variants"][v][space]
            print(
                f"{v:14s} {space:10s}"
                + "".join(f"{r[lv]:12.3f}" for lv in LEVELS)
                + f"{r['open_overlap']:10.3f}{r['open_overlap_cross_dir2']:11.3f}"
                + f"{r['share_neighbours_cross_dir2']:12.3f}"
            )
    print(
        f"\n{'layout shape':14s}{'kept nbrs':>11s}{'largest cluster':>17s}"
        f"{'outside pockets':>17s}{'pockets':>9s}"
    )
    for v in done:
        sh = metrics["variants"][v]["shape"]
        print(
            f"{v:14s}{sh['kept_neighbours']:11.3f}{sh['largest_cluster']:17.1%}"
            f"{sh['outside_pockets']:17.1%}{sh['pockets']:9d}"
        )
    print("\nshare of neighbour sets two variants have in common (embedding space):")
    for key, val in overlap.items():
        if key.startswith("embedding:"):
            print(f"  {key:40s} {val:.3f}")

    write_text_safely(json.dumps(metrics, indent=2), run_dir / "metrics.json")
    print(f"\nwrote {run_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
