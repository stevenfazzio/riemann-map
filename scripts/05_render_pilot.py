"""Stage 5: render one diagnostic map per text variant of a run.

Reads   the run's sample (riemann_map.load_sample), {run dir}/coords_{variant}.npy
Writes  output/{run}/map_{variant}.html

    uv run python -I scripts/05_render_pilot.py                # the 10k pilot, every variant
    uv run python -I scripts/05_render_pilot.py --run dense    # first 40k of the full run
    uv run python -I scripts/05_render_pilot.py --run full     # every file, Qwen3 and two blends

These are diagnostic maps, not the final design. There is no Toponymy naming yet, so the
region labels are computed, not written: each label is a density cluster found in the
2-d layout, named after the directory most of its files come from, with that share.
A label reading "Proofs/GlobalInterpolation 92%" is a region the directory tree already
explains; a low percentage marks a region that mixes directories.

The maps carry Lean statements in the hovercard, so they are for local use only.
"""

from __future__ import annotations

import argparse
import colorsys
import html
from urllib.parse import quote

import datamapplot
import glasbey
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex, to_rgb
from sklearn.cluster import HDBSCAN

from riemann_map import (
    INTERIM,
    OUTPUT,
    RUN_DIRS,
    RUN_VARIANTS,
    SOURCE_URL,
    load_sample,
    project_namespace_roots,
)

VARIANTS = {
    "names": "path and declaration names only",
    "statements": "statements only (no proofs, no path)",
    "file": "whole source file",
    "expanded": "statements, named propositions expanded",
    "q8_names": "path and declaration names (Qwen3-Embedding-8B)",
    "q8_statements": "statements (Qwen3-Embedding-8B)",
    "q8_inst": "statements behind a subject instruction (Qwen3-Embedding-8B)",
    "blend30": "30% names + 70% statements",
    "q8_blend15": "15% names + 85% statements (Qwen3-Embedding-8B)",
    "q8_blend30": "30% names + 70% statements (Qwen3-Embedding-8B)",
    "q8_blend50": "50% names + 50% statements (Qwen3-Embedding-8B)",
    "q8i_blend15": "15% names + 85% instructed statements (Qwen3-Embedding-8B)",
    "q8i_blend30": "30% names + 70% instructed statements (Qwen3-Embedding-8B)",
}
PILOT_N = 10_000  # the cluster sizes below were set by eye at this many points
# How HDBSCAN picks clusters from its tree. The default ("eom") puts 99% of the dense
# run's points in one cluster, because that layout is one continuous mass with a few
# islands; "leaf" returns the dense pockets inside it and leaves the rest unlabelled.
# The pilot keeps the default so its reported cluster counts stay reproducible.
SELECTION = {"pilot": "eom", "dense": "leaf", "full": "leaf"}
TOP_PARTS = 14

NEUTRAL_COLORS = {
    "Other": "#8c8c8c",
    "None opened": "#e2e2e2",
    "Other Mathlib": "#b5b5b5",
    "Used, no single branch": "#a3a3a3",
    "Not used by the proof": "#dcdcdc",
}

# Hand grouping of the Mathlib namespaces these files `open`, by eye from the most
# frequent ones in the sample. A pilot convenience, not a taxonomy. A file gets the first
# area in this order that it opens anything from, so the order runs from the most
# distinctive subjects to the ambient ones: nearly everything here is about number fields
# and matrices, so those only label a file that opens nothing more specific.
MATHLIB_AREAS = {
    "Algebraic geometry": ["AlgebraicGeometry", "Scheme"],
    "Category theory": [
        "CategoryTheory",
        "Limits",
        "MonoidalCategory",
        "CartesianMonoidalCategory",
        "MonObj",
        "Opposite",
        "CochainComplex",
        "HomologicalComplex",
        "ZeroObject",
    ],
    "Polynomials": ["MvPolynomial", "Polynomial", "PowerSeries"],
    "Analysis and measure": [
        "MeasureTheory",
        "ENNReal",
        "NNReal",
        "Filter",
        "Topology",
        "ContDiff",
        "Complex",
        "ComplexConjugate",
        "ComplexOrder",
        "InnerProductSpace",
        "Real",
        "TopologicalSpace",
        "Asymptotics",
        "Metric",
        "Manifold",
        "intervalIntegral",
    ],
    "Linear algebra": ["Matrix", "TensorProduct", "Module", "LinearMap", "Submodule", "Representation"],
    "Number fields": [
        "NumberField",
        "IsDedekindDomain",
        "ValuativeRel",
        "IsLocalRing",
        "Valuation",
        "HeightOneSpectrum",
        "DirichletCharacter",
        "ArithmeticFunction",
        "Nat",
        "Int",
    ],
    "Sets and logic": ["Classical", "Finset", "BigOperators", "Set", "Pointwise", "Function", "Fintype"],
}
AREA_OF = {ns: area for area, nss in MATHLIB_AREAS.items() for ns in nss}
AREA_ORDER = list(MATHLIB_AREAS)


def categorical_palette(n_colors: int, avoid: list[str]) -> list[str]:
    seed = avoid or ["#8c8c8c"]
    palette = glasbey.extend_palette(
        seed,
        palette_size=len(seed) + n_colors,
        colorblind_safe=True,
        cvd_severity=50.0,
        lightness_bounds=(35, 75),
        chroma_bounds=(20, 90),
        hue_bounds=(0, 360),
    )
    return palette[len(seed) :]


def color_mapping(values: pd.Series) -> dict[str, str]:
    order = values.value_counts().index.tolist()
    named = [c for c in order if c not in NEUTRAL_COLORS]
    neutrals = {c: NEUTRAL_COLORS[c] for c in order if c in NEUTRAL_COLORS}
    mapping = dict(zip(named, categorical_palette(len(named), list(neutrals.values()))))
    mapping.update(neutrals)
    assert len(set(mapping.values())) == len(mapping), f"two categories share a colour: {mapping}"
    return mapping


def relative_luminance(rgb) -> float:
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(a, b) -> float:
    la, lb = sorted([relative_luminance(a), relative_luminance(b)], reverse=True)
    return (la + 0.05) / (lb + 0.05)


def pill_colors(hex_color: str) -> tuple[str, str]:
    h, _, s = colorsys.rgb_to_hls(*to_rgb(hex_color))
    bg = colorsys.hls_to_rgb(h, 0.92, min(s, 0.65))
    lightness = 0.35
    fg = colorsys.hls_to_rgb(h, lightness, s)
    while contrast(fg, bg) < 4.5:
        lightness -= 0.03
        fg = colorsys.hls_to_rgb(h, lightness, s)
    return to_hex(bg), to_hex(fg)


def mathlib_area(opens, internal_roots: set[str]) -> str:
    ext = [o for o in opens if o.split(".", 1)[0] not in internal_roots]
    if not ext:
        return "None opened"
    areas = {AREA_OF[r] for r in (o.split(".", 1)[0] for o in ext) if r in AREA_OF}
    return min(areas, key=AREA_ORDER.index) if areas else "Other Mathlib"


def cluster_labels(
    coords: np.ndarray, keys: pd.Series, min_cluster_size: int, min_samples: int, method: str
) -> tuple[np.ndarray, pd.DataFrame]:
    """Density-cluster the layout and name each cluster after its dominant directory."""
    # min_samples below min_cluster_size: sklearn defaults them equal, which leaves a large
    # share of a 10k-point layout unlabelled.
    raw = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=method,
        copy=True,
    ).fit_predict(coords)
    names = np.full(len(coords), "Unlabelled", dtype=object)
    used: dict[str, int] = {}
    rows = []
    for c in sorted(set(raw) - {-1}):
        m = raw == c
        vc = keys[m].value_counts(normalize=True)
        top, share = vc.index[0], float(vc.iloc[0])
        label = f"{'/'.join(top.split('/')[-2:])} {share:.0%}"
        used[label] = used.get(label, 0) + 1
        if used[label] > 1:  # same name twice would merge two regions under one label
            label = f"{label} ·{used[label]}"
        names[m] = label
        rows.append({"label": label, "size": int(m.sum()), "dominant": top, "share": share})
    return names, pd.DataFrame(rows)


DOT = (
    "display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; "
    "box-shadow: inset 0 0 0 1px rgba(0, 0, 0, 0.15); background: {color};"
)
LABEL = "color: #6b7280; font-size: 11px; padding-top: 1px;"
HOVER_TEMPLATE = f"""
<div style="max-width: 360px; white-space: normal; font-weight: 400; color: #1f2328;">
  <span style="display: inline-block; padding: 1px 8px; margin-bottom: 5px; border-radius: 999px;
    font-size: 11px; font-weight: 600; background: {{pill_bg}}; color: {{pill_fg}};">{{part}}</span>
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25; overflow-wrap: anywhere;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 78px minmax(0, 1fr); gap: 3px 10px; margin-top: 6px;
    font-size: 12px; line-height: 1.35;">
    <div style="{LABEL}">Mathlib area</div>
    <div><span style="{DOT.format(color="{area_dot}")}"></span>{{area}}</div>
    <div style="{LABEL}">Proof branch</div>
    <div style="overflow-wrap: anywhere;">
      <span style="{DOT.format(color="{branch_dot}")}"></span>{{branch}}</div>
    <div style="{LABEL}">Import depth</div><div>{{import_depth}}</div>
    <div style="{LABEL}">Declarations</div><div>{{decl_line}}</div>
  </div>
  <div style="font-family: ui-monospace, Menlo, monospace; font-size: 11px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4; overflow-wrap: anywhere;">{{statement}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px; overflow-wrap: anywhere;">{{path}}</div>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*")
    ap.add_argument("--run", choices=list(RUN_DIRS), default="pilot")
    ap.add_argument("--selection", choices=["eom", "leaf"], help="override the run's cluster selection")
    args = ap.parse_args()
    run = args.run
    variants = args.variants or RUN_VARIANTS[run]
    sample = load_sample(run)
    n = len(sample)
    # Keep the clusters the same share of the map as the point count grows.
    scale = n / PILOT_N
    parts = sample["path"].str.split("/")
    sample["dir4"] = parts.map(lambda p: "/".join(p[:4]) if len(p) > 4 else "/".join(p[:-1]) or p[0])

    internal_roots = project_namespace_roots(
        pd.read_parquet(INTERIM / "decls.parquet", columns=["name"])["name"]
    )

    # --- fields ----------------------------------------------------------------------
    top_parts = sample["part"].value_counts().head(TOP_PARTS).index
    part = sample["part"].where(sample["part"].isin(top_parts), "Other")
    area = sample["opens"].map(lambda xs: mathlib_area(list(xs), internal_roots))
    branch = sample["branch"].map(lambda b: b.rsplit(".", 1)[-1] if isinstance(b, str) else None)
    branch = branch.where(
        branch.notna(),
        np.where(sample["n_used_by_proof"] > 0, "Used, no single branch", "Not used by the proof"),
    )
    library = sample["library"]
    print("Mathlib area:", area.value_counts().to_dict())
    print("Proof branch:", branch.value_counts().to_dict())

    maps = {
        "part": color_mapping(part),
        "area": color_mapping(area),
        "branch": color_mapping(branch),
        "library": color_mapping(library),
    }
    pills = {k: pill_colors(v) for k, v in maps["part"].items()}

    names_short = sample["decl_names"].map(lambda xs: ", ".join(x.rsplit(".", 1)[-1] for x in list(xs)[:4]))
    more = (sample["n_decls"] - 4).clip(lower=0)
    decl_line = [
        f"{k}: {s}" + (f" +{m}" if m else "") for k, s, m in zip(sample["n_decls"], names_short, more)
    ]
    statement = sample["first_statement"].map(lambda s: s[:320] + ("…" if len(s) > 320 else ""))

    def esc(s: pd.Series) -> pd.Series:
        return s.map(lambda x: html.escape(str(x)))

    extra = pd.DataFrame(
        {
            "title": esc(parts.str[-1].str.removesuffix(".lean")),
            "part": esc(sample["part"]),
            "pill_bg": [pills[p][0] for p in part],
            "pill_fg": [pills[p][1] for p in part],
            "area": esc(area),
            "area_dot": [maps["area"][a] for a in area],
            "branch": esc(branch),
            "branch_dot": [maps["branch"][b] for b in branch],
            "import_depth": sample["import_depth"].astype(str),
            "decl_line": esc(pd.Series(decl_line)),
            "statement": esc(statement),
            "path": esc(sample["path"]),
            "url": [SOURCE_URL + quote(p) for p in sample["path"]],
            "search_text": esc(
                sample["path"]
                + " "
                + sample["decl_names"].map(lambda xs: " ".join(xs)).str[:3000]
                + " "
                + sample["opens"].map(lambda xs: " ".join(xs))
            ),
        }
    )

    colormap_rawdata = [
        part.to_numpy(),
        area.to_numpy(),
        branch.to_numpy(),
        library.to_numpy(),
        sample["import_depth"].to_numpy(),
        np.log10(sample["n_decls"].to_numpy()),
    ]
    colormap_metadata = [
        {
            "field": "part",
            "description": "Part (directory)",
            "kind": "categorical",
            "color_mapping": maps["part"],
        },
        {
            "field": "area",
            "description": "Mathlib area opened",
            "kind": "categorical",
            "color_mapping": maps["area"],
        },
        {
            "field": "branch",
            "description": "Proof branch",
            "kind": "categorical",
            "color_mapping": maps["branch"],
        },
        {
            "field": "library",
            "description": "Library",
            "kind": "categorical",
            "color_mapping": maps["library"],
        },
        {
            "field": "import_depth",
            "description": "Import depth from Main",
            "kind": "continuous",
            "cmap": "viridis",
        },
        {
            "field": "log_decls",
            "description": "Declarations in file (log10)",
            "kind": "continuous",
            "cmap": "magma",
        },
    ]

    out_dir = OUTPUT / run
    out_dir.mkdir(parents=True, exist_ok=True)
    for v in variants:
        coords_path = RUN_DIRS[run] / f"coords_{v}.npy"
        if not coords_path.exists():
            print(f"{v}: no layout yet, skipping")
            continue
        coords = np.load(coords_path)
        assert len(coords) == n
        # The full run's statements are the pilot's "expanded" recipe.
        described = VARIANTS["expanded" if (run, v) == ("dense", "statements") else v]
        ms = round(10 * scale)
        how = args.selection or SELECTION[run]
        fine, fine_tbl = cluster_labels(coords, sample["dir4"], round(25 * scale), ms, how)
        coarse, coarse_tbl = cluster_labels(coords, sample["dir2"], round(110 * scale), ms, how)
        layers = [fine, coarse]  # DataMapPlot wants finest first
        assert pd.Series(layers[0]).nunique() >= pd.Series(layers[-1]).nunique(), (
            "label layers must be finest-first"
        )
        for name, tbl, lab in [("fine", fine_tbl, fine), ("coarse", coarse_tbl, coarse)]:
            w = tbl["size"] / tbl["size"].sum()
            print(
                f"{v}: {name} clusters {len(tbl)}, unlabelled {np.mean(lab == 'Unlabelled'):.1%}, "
                f"dominant-directory share: size-weighted mean {(tbl['share'] * w).sum():.2f}, "
                f"clusters under 50%: {(tbl['share'] < 0.5).sum()}"
            )

        plot = datamapplot.create_interactive_plot(
            coords,
            *layers,
            hover_text=extra["title"].to_numpy(),
            title=f"RH proof map · {run} · {v}",
            sub_title=f"{n:,} of 175,187 files · layout from {described}",
            font_family="IBM Plex Sans",
            extra_point_data=extra,
            hover_text_html_template=HOVER_TEMPLATE,
            on_click="window.open(`{url}`)",
            enable_search=True,
            search_field="search_text",
            colormap_rawdata=colormap_rawdata,
            colormap_metadata=colormap_metadata,
            cvd_safer=True,
            initial_zoom_fraction=0.9,  # a few far-out islands otherwise squeeze the cloud
            custom_css=".deck-tooltip { max-width: min(420px, 92vw) !important; }",
            custom_js="datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.04, smooth: true}}});",
        )
        out = out_dir / f"map_{v}.html"
        plot.save(str(out))
        print(f"{v}: wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
