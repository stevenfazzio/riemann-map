"""Stage 10: render the finished map, private build.

Reads   data/full/map/coords_q8_inst.npy              the statements-only layout
        data/full/naming/cluster_labels_b120.npy      the regions (stage 9)
        data/full/naming/full/regions.parquet         their names (stage 9)
        the full run's sample (riemann_map.load_sample)
Writes  output/full/rh_map.html

    uv run python -I scripts/10_render_map.py

This is stage 5's diagnostic map with the 412 region names from stage 9, at four levels, in
place of its computed directory labels. The hovercard, the search text and the palettes are
stage 5's. The colour overlays differ: Mathlib area comes first because it follows this
layout best (54% of a file's 15 nearest neighbours share its area, 19% by chance), Library
is gone (81% of files are in one library) and steps from the main theorem is new.

The page carries Lean statements in the hovercard and declaration names in the search
text, and the source has no licence, so it is for local use only.
"""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import re
from pathlib import Path
from urllib.parse import quote

import datamapplot
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import to_hex

from riemann_map import (
    FULL,
    INTERIM,
    OUTPUT,
    RUN_DIRS,
    SOURCE_URL,
    load_sample,
    project_namespace_roots,
    write_text_safely,
)

VARIANT = "q8_inst"  # the layout Steven chose on 2026-11-22: statements only
BASE = 120  # stage 9's smallest region at the finest layer, in files
NAMING = FULL / "naming"
# Hand changes to stage 9's names: (layer, cluster index) -> (the name replaced, the new name).
# Stage 9 gave this top-level region the name of its largest sub-region as well, so one
# label appeared twice on the map. The wording is mine; Steven was shown it on 2026-11-23.
RENAMES = {(3, 4): ("Hermitian Lattices over CM Fields", "CM Field Arithmetic")}
# Steps from riemannHypothesis in the uses-graph, for the nearest declaration in the file.
# Binned, not continuous: DataMapPlot draws a missing number as a transparent point, and
# 18% of files hold nothing the proof uses.
STEP_BINS = [(0, 15), (16, 25), (26, 35), (36, 50), (51, 75), (76, None)]
NOT_USED = "Not used by the proof"
PILLS = ["Source released 20 Nov 2026", "Names by Claude Opus 5.5", "Private build"]

CUSTOM_CSS = """
.deck-tooltip { max-width: min(420px, 92vw) !important; }
#main-title { line-height: 1.1 !important; letter-spacing: -0.01em; text-wrap: balance; }
#main-title + br + span { display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty; }
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: #636c76; background: rgba(99, 108, 118, 0.08);
  border: 1px solid rgba(99, 108, 118, 0.15); border-radius: 12px;
}
"""
# DataMapPlot hard-codes a scroll-zoom speed of 0.01, which Steven finds too slow.
CUSTOM_JS = "datamap.deckgl.setProps({controller: {scrollZoom: {speed: 0.04, smooth: true}}});"


def load_stage(name: str):
    """Import another stage's script, whose file name cannot be imported directly."""
    spec = importlib.util.spec_from_file_location(
        name.lstrip("0123456789_"), Path(__file__).with_name(f"{name}.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hover_template(label: str, dot: str) -> str:
    """Stage 5's card with one more row, for the steps overlay."""
    return f"""
<div style="max-width: 360px; white-space: normal; font-weight: 400; color: #1f2328;">
  <span style="display: inline-block; padding: 1px 8px; margin-bottom: 5px; border-radius: 999px;
    font-size: 11px; font-weight: 600; background: {{pill_bg}}; color: {{pill_fg}};">{{part}}</span>
  <div style="font-weight: 600; font-size: 15px; line-height: 1.25; overflow-wrap: anywhere;">{{title}}</div>
  <div style="display: grid; grid-template-columns: 82px minmax(0, 1fr); gap: 3px 10px; margin-top: 6px;
    font-size: 12px; line-height: 1.35;">
    <div style="{label}">Mathlib area</div>
    <div><span style="{dot.format(color="{area_dot}")}"></span>{{area}}</div>
    <div style="{label}">Proof branch</div>
    <div style="overflow-wrap: anywhere;">
      <span style="{dot.format(color="{branch_dot}")}"></span>{{branch}}</div>
    <div style="{label}">Import depth</div><div>{{import_depth}}</div>
    <div style="{label}">From theorem</div>
    <div><span style="{dot.format(color="{steps_dot}")}"></span>{{steps}}</div>
    <div style="{label}">Declarations</div><div>{{decl_line}}</div>
  </div>
  <div style="font-family: ui-monospace, Menlo, monospace; font-size: 11px; margin-top: 8px; padding-top: 7px;
    border-top: 1px solid #e5e7eb; line-height: 1.4; overflow-wrap: anywhere;">{{statement}}</div>
  <div style="color: #6b7280; font-size: 11px; margin-top: 7px; overflow-wrap: anywhere;">{{path}}</div>
</div>
"""


def region_layers(labels: np.ndarray, regions: pd.DataFrame) -> list[np.ndarray]:
    """One name per file per layer, finest layer first, "Unlabelled" outside every region."""
    names = regions.set_index(["layer", "cluster_idx"])["name"].copy()
    for key, (old, new) in RENAMES.items():
        assert names.loc[key] == old, f"region {key} is named {names.loc[key]!r}, not {old!r}"
        names.loc[key] = new
    assert (names.str.strip() != "").all(), "a region has no name"
    layers = []
    for layer, row in enumerate(labels):
        of_layer = names.loc[layer]
        # DataMapPlot treats two regions with one name in a layer as a single region.
        assert of_layer.is_unique, f"layer {layer} uses a name twice"
        vec = np.full(len(row), "Unlabelled", dtype=object)
        inside = row >= 0
        vec[inside] = of_layer.loc[row[inside]].to_numpy()  # a KeyError here is a region without a row
        layers.append(vec)
    counts = [len(names.loc[layer]) for layer in range(len(labels))]
    assert counts[0] >= counts[-1], "label layers must be finest-first for DataMapPlot"
    return layers


def main() -> None:
    s5 = load_stage("05_render_pilot")
    sample = load_sample("full")
    n = len(sample)
    coords = np.load(RUN_DIRS["full"] / f"coords_{VARIANT}.npy")
    labels = np.load(NAMING / f"cluster_labels_b{BASE}.npy")
    regions = pd.read_parquet(NAMING / "full" / "regions.parquet")
    assert len(coords) == n == labels.shape[1], (coords.shape, n, labels.shape)
    made_from = json.loads((NAMING / f"cluster_labels_b{BASE}.json").read_text())["coords_sha256"]
    assert made_from == hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest(), (
        "the saved regions were made from a different layout"
    )
    layers = region_layers(labels, regions)
    in_none = float((labels < 0).all(axis=0).mean())
    print(
        f"{n:,} files; regions per level, finest first: {[int(row.max()) + 1 for row in labels]}; "
        f"{in_none:.0%} of files are in no region at any level"
    )

    # --- fields --------------------------------------------------------------------------
    internal_roots = project_namespace_roots(
        pd.read_parquet(INTERIM / "decls.parquet", columns=["name"])["name"]
    )
    parts = sample["path"].str.split("/")
    top_parts = sample["part"].value_counts().head(s5.TOP_PARTS).index
    part = sample["part"].where(sample["part"].isin(top_parts), "Other")
    area = sample["opens"].map(lambda xs: s5.mathlib_area(list(xs), internal_roots))
    branch = sample["branch"].map(lambda b: b.rsplit(".", 1)[-1] if isinstance(b, str) else None)
    branch = branch.where(
        branch.notna(), np.where(sample["n_used_by_proof"] > 0, "Used, no single branch", NOT_USED)
    )

    depth = sample["min_use_depth"]
    assert (depth.isna() == (sample["n_used_by_proof"] == 0)).all(), "a used file has no distance"
    bin_names = [f"{lo}–{hi} steps" if hi is not None else f"{lo} or more steps" for lo, hi in STEP_BINS]
    which = np.searchsorted([lo for lo, _ in STEP_BINS], depth.fillna(0).to_numpy(), side="right") - 1
    steps_bin = pd.Series(np.where(depth.isna(), NOT_USED, np.array(bin_names, dtype=object)[which]))
    steps_text = [
        NOT_USED if np.isnan(d) else f"{int(d)} step{'' if d == 1 else 's'} in the uses-graph" for d in depth
    ]
    print("Mathlib area:", area.value_counts().to_dict())
    print("Proof branch:", branch.value_counts().to_dict())
    print("Steps from the theorem:", steps_bin.value_counts().reindex([*bin_names, NOT_USED]).to_dict())

    maps = {
        "part": s5.color_mapping(part),
        "area": s5.color_mapping(area),
        "branch": s5.color_mapping(branch),
    }
    # The same ramp as import depth, so the two depth overlays can be compared; dict order
    # is the legend's order.
    maps["steps"] = {
        name: to_hex(colormaps["viridis"](x))
        for name, x in zip(bin_names, np.linspace(0, 0.95, len(bin_names)))
    }
    maps["steps"][NOT_USED] = s5.NEUTRAL_COLORS[NOT_USED]
    assert len(set(maps["steps"].values())) == len(maps["steps"]), "two step bins share a colour"
    pills = {k: s5.pill_colors(v) for k, v in maps["part"].items()}

    names_short = sample["decl_names"].map(lambda xs: ", ".join(x.rsplit(".", 1)[-1] for x in list(xs)[:4]))
    more = (sample["n_decls"] - 4).clip(lower=0)
    decl_line = [
        f"{k}: {s}" + (f" +{m}" if m else "") for k, s, m in zip(sample["n_decls"], names_short, more)
    ]
    statement = sample["first_statement"].map(lambda s: s[:320] + ("…" if len(s) > 320 else ""))

    def esc(s: pd.Series) -> pd.Series:
        # DataMapPlot puts these values into the card as raw HTML.
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
            "steps": esc(pd.Series(steps_text)),
            "steps_dot": [maps["steps"][b] for b in steps_bin],
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

    # DataMapPlot always lists its own "Clusters" colouring first; these follow in this order.
    def categorical(field: str, description: str) -> dict:
        return {
            "field": field,
            "description": description,
            "kind": "categorical",
            "color_mapping": maps[field],
        }

    def continuous(field: str, description: str, cmap: str) -> dict:
        return {"field": field, "description": description, "kind": "continuous", "cmap": cmap}

    overlays = [
        (area.to_numpy(), categorical("area", "Mathlib area opened")),
        (branch.to_numpy(), categorical("branch", "Proof branch")),
        (part.to_numpy(), categorical("part", "Part (directory)")),
        (sample["import_depth"].to_numpy(), continuous("import_depth", "Import depth from Main", "viridis")),
        (steps_bin.to_numpy(), categorical("steps", "Steps from the main theorem")),
        (
            np.log10(sample["n_decls"].to_numpy()),
            continuous("log_decls", "Declarations in file (log10)", "magma"),
        ),
    ]

    plot = datamapplot.create_interactive_plot(
        coords,
        *layers,
        hover_text=extra["title"].to_numpy(),
        title="RH proof map",
        # Short enough for one line: at 1280 px the panel stops growing and a longer line wraps.
        sub_title=f"{n:,} Lean files, placed by statement text · {len(regions)} named regions",
        title_font_size=24,
        sub_title_font_size=13,
        font_family="IBM Plex Sans",
        extra_point_data=extra,
        hover_text_html_template=hover_template(s5.LABEL, s5.DOT),
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=[values for values, _ in overlays],
        colormap_metadata=[meta for _, meta in overlays],
        cvd_safer=True,
        initial_zoom_fraction=0.9,  # a few far-out islands otherwise squeeze the cloud
        custom_css=CUSTOM_CSS,
        custom_js=CUSTOM_JS,
    )

    # The title panel holds only spans, so its first closing </div> is its own.
    pills_html = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in PILLS)
    text, found = re.subn(
        r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
        lambda m: m.group(1) + f"<div>{pills_html}</div>" + m.group(2),
        str(plot),
        count=1,
        flags=re.DOTALL,
    )
    assert found == 1, "title panel not found"
    out = OUTPUT / "full" / "rh_map.html"
    write_text_safely(text, out)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
