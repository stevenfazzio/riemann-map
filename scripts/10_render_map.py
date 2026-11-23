"""Stage 10: render the finished map, as a private file or as the public site.

Reads   data/full/map/coords_q8_inst.npy              the statements-only layout
        data/full/naming/cluster_labels_b120.npy      the regions (stage 9)
        data/full/naming/full/regions.parquet         their names (stage 9)
        the full run's sample (riemann_map.load_sample)
Writes  output/full/rh_map.html                       private: one file, data inside
        docs/index.html and docs/rh_*.zip             --public: the folder GitHub Pages serves

    uv run python -I scripts/10_render_map.py
    uv run python -I scripts/10_render_map.py --public
    uv run python -I scripts/10_render_map.py --public --light   # the first published look
    uv run python -I scripts/11_social_preview.py     # the public page's link-card image

This is stage 5's diagnostic map with the 412 region names from stage 9, at four levels, in
place of its computed directory labels. The hovercard, the search text and the palettes are
stage 5's. The colour overlays differ: Mathlib area comes first because it follows this
layout best (54% of a file's 15 nearest neighbours share its area, 19% by chance), Library
is gone (81% of files are in one library) and steps from the main theorem is new.

The page is dark unless --light is given (see THEMES). Stage 5's colours were chosen for a
white page, so the dark theme replaces the ones that depend on the background and keeps
every named category's colour.

Both builds carry the author's text: a 320-character excerpt of each file's first statement
in the hovercard, and declaration names in the search text. The release states no licence
or reuse terms. Steven's position (2026-11-23) is to publish the map and honour any request
to remove it, so the public build keeps the excerpts and adds what a public page needs:
credit to the author, an "unofficial" mark, a way to ask for removal, link-preview tags, and
its data in separate files so the page itself is small. This script publishes nothing.
"""

from __future__ import annotations

import argparse
import colorsys
import hashlib
import html
import json
import re
from string import Template
from urllib.parse import quote

import datamapplot
import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap, to_hex, to_rgb

from riemann_map import (
    FULL,
    INTERIM,
    OUTPUT,
    ROOT,
    RUN_DIRS,
    SOURCE_URL,
    load_sample,
    load_stage,
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
# Steven's other maps are served by GitHub Pages from the repository's docs/ folder at
# stevenfazzio.com/<repository>/. These two addresses assume the same for this one; neither
# exists until he publishes.
DOCS = ROOT / "docs"
PUBLIC_HTML = DOCS / "index.html"
SOCIAL_PREVIEW = DOCS / "social-preview.png"
DATA_PREFIX = "rh"  # the data files sit beside index.html as rh_*.zip
PUBLIC_URL = "https://stevenfazzio.com/riemann-map/"
REPO_URL = "https://github.com/stevenfazzio/riemann-map"
AUTHOR_NOTE_URL = "https://alpo.ge/remark.pdf"
MIRROR_URL = "https://github.com/tamnd/riemann-lean"
PRIVATE_HTML = OUTPUT / "full" / "rh_map.html"

TITLE = {"private": "RH proof map", "public": "Riemann Hypothesis Proof Map"}
PILLS = {
    "private": ["Source released 20 Nov 2026", "Names by Claude Opus 5.5", "Private build"],
    "public": ["Source released 20 Nov 2026", "Names by Claude Opus 5.5", "Unofficial"],
}
# Under the pills of the public page. The subtitle credits the author; this says what the
# map is, what it quotes, and how to have something removed.
ABOUT = f"""
<details class="about"><summary>About this map</summary>
<p>Each point is one source file of the Lean 4 proof of the Riemann hypothesis that
<a href="{AUTHOR_NOTE_URL}" target="_blank" rel="noopener">Levent Alpöge released</a> on
20 November 2026. A file sits near others that state similar mathematics: the layout comes
from the text of each file's theorem statements and definitions. Region names were written by
Claude Opus 5.5 from sample files and have not been checked by a mathematician.</p>
<p>This map is unofficial and has no connection to the author. Hovercards quote short excerpts
of the source, which is the author's work, and a click opens the file in a
<a href="{MIRROR_URL}" target="_blank" rel="noopener">public copy</a>. To have anything
corrected or removed, <a href="{REPO_URL}/issues" target="_blank" rel="noopener">open an
issue</a>. <a href="{REPO_URL}" target="_blank" rel="noopener">Code and method</a>.</p>
</details>
"""

# Dark is the default: Steven compared the two on 2026-11-23 and found dark much better for
# this map. "light" is the first published look, and is stage 5's colours unchanged.
#   darkmode    DataMapPlot's own switch: page, panels, tooltip and region-label colours
#   colors      what CUSTOM_CSS and the hovercard are filled in with
#   neutrals    greys for the catch-all categories, in place of stage 5's. A catch-all
#               recedes by being pale on white and by being dark on black. Each is distinct,
#               because the legend selects points by colour.
#   ramp_start  the share of a continuous ramp to skip: the dark end of viridis and magma
#               (which starts at black) cannot be seen on a black page
THEMES = {
    "dark": {
        "darkmode": True,
        "colors": {
            "panel_muted": "#a4adb7",
            "panel_tint": "rgba(164, 173, 183, 0.12)",
            "panel_line": "rgba(164, 173, 183, 0.25)",
            "about_text": "#b6bdc5",
            "link": "#7ab0ff",
            "card_text": "#e6e8eb",
            "card_muted": "#9aa3ad",
            "card_rule": "#30363d",
            "dot_ring": "rgba(255, 255, 255, 0.28)",
        },
        "neutrals": {
            "Other": "#7d7d7d",
            "None opened": "#3a3a3a",
            "Other Mathlib": "#5a5a5a",
            "Used, no single branch": "#6b6b6b",
            "Not used by the proof": "#454545",
        },
        "ramp_start": {"viridis": 0.15, "magma": 0.30},
    },
    "light": {
        "darkmode": False,
        "colors": {
            "panel_muted": "#636c76",
            "panel_tint": "rgba(99, 108, 118, 0.08)",
            "panel_line": "rgba(99, 108, 118, 0.15)",
            "about_text": "#4b5563",
            "link": "#2563eb",
            "card_text": "#1f2328",
            "card_muted": "#6b7280",
            "card_rule": "#e5e7eb",
            "dot_ring": "rgba(0, 0, 0, 0.15)",
        },
        "neutrals": {},
        "ramp_start": {"viridis": 0.0, "magma": 0.0},
    },
}

# A string.Template: $name is one of a theme's colors.
CUSTOM_CSS = """
.deck-tooltip { max-width: min(420px, 92vw) !important; }
#main-title { line-height: 1.1 !important; letter-spacing: -0.01em; text-wrap: balance; }
#main-title + br + span { display: inline-block; margin-top: 4px; line-height: 1.3; text-wrap: pretty; }
.title-pill {
  display: inline-block; margin: 8px 6px 0 0; padding: 3px 10px;
  font-size: 11px; font-weight: 500; letter-spacing: 0.04em; text-transform: uppercase;
  color: $panel_muted; background: $panel_tint;
  border: 1px solid $panel_line; border-radius: 12px;
}
.about { margin-top: 8px; max-width: 520px; font-size: 12px; line-height: 1.45; color: $about_text; }
.about summary {
  cursor: pointer; font-size: 11px; font-weight: 500; letter-spacing: 0.04em;
  text-transform: uppercase; color: $panel_muted;
}
.about p { margin: 6px 0 0; }
.about a { color: $link; }
"""
# DataMapPlot hard-codes a scroll-zoom speed of 0.01, which Steven finds too slow. Stage 11
# reads this speed to keep its framing.
ZOOM_SPEED = 0.04
CUSTOM_JS = f"datamap.deckgl.setProps({{controller: {{scrollZoom: {{speed: {ZOOM_SPEED}, smooth: true}}}}}});"


def ramp(name: str, start: float) -> str:
    """The name of a matplotlib colormap: `name` without the first `start` share of it."""
    if start == 0:
        return name
    cut = ListedColormap(colormaps[name](np.linspace(start, 1.0, 256)), name=f"{name}_from_{start:.2f}")
    colormaps.register(cut, force=True)  # DataMapPlot looks colormaps up by name
    return cut.name


def dark_pill_colors(hex_color: str, contrast) -> tuple[str, str]:
    """Stage 5's pill for a dark card: a deep shade of the hue behind, a pale tint in front."""
    h, _, s = colorsys.rgb_to_hls(*to_rgb(hex_color))
    bg = colorsys.hls_to_rgb(h, 0.20, min(s, 0.55))
    lightness = 0.78
    fg = colorsys.hls_to_rgb(h, lightness, s)
    while contrast(fg, bg) < 4.5:
        lightness += 0.03
        fg = colorsys.hls_to_rgb(h, lightness, s)
    return to_hex(bg), to_hex(fg)


def hover_template(colors: dict) -> str:
    """Stage 5's card with one more row, for the steps overlay, in a theme's colours."""
    text, muted, rule = colors["card_text"], colors["card_muted"], colors["card_rule"]
    label = f"color: {muted}; font-size: 11px; padding-top: 1px;"
    dot = (
        "display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; "
        f"box-shadow: inset 0 0 0 1px {colors['dot_ring']}; background: {{color}};"
    )
    return f"""
<div style="max-width: 360px; white-space: normal; font-weight: 400; color: {text};">
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
    border-top: 1px solid {rule}; line-height: 1.4; overflow-wrap: anywhere;">{{statement}}</div>
  <div style="color: {muted}; font-size: 11px; margin-top: 7px; overflow-wrap: anywhere;">{{path}}</div>
</div>
"""


def open_graph_tags(n: int) -> str:
    og = {
        "og:title": TITLE["public"],
        "og:description": (
            f"{n:,} source files of the Lean proof of the Riemann hypothesis, laid out by what "
            "they state and named at four zoom levels. Pan, zoom, hover and search."
        ),
        "og:type": "website",
        "og:url": PUBLIC_URL,
        # Always declared: stage 11 screenshots the rendered page, so the image comes second.
        "og:image": PUBLIC_URL + SOCIAL_PREVIEW.name,
    }
    tags = [f'<meta property="{k}" content="{html.escape(v, quote=True)}">' for k, v in og.items()]
    tags.append(f'<meta name="description" content="{html.escape(og["og:description"], quote=True)}">')
    tags.append('<meta name="twitter:card" content="summary_large_image">')
    return "\n".join(tags)


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true", help="build docs/ for the web, not the private file")
    ap.add_argument("--light", action="store_true", help="a white page, not the default dark one")
    args = ap.parse_args()
    build = "public" if args.public else "private"
    theme = THEMES["light" if args.light else "dark"]
    s5 = load_stage("05_render_pilot")
    assert set(theme["neutrals"]) <= set(s5.NEUTRAL_COLORS), "a themed grey is not a stage 5 catch-all"
    neutrals = {**s5.NEUTRAL_COLORS, **theme["neutrals"]}
    ramps = {name: ramp(name, start) for name, start in theme["ramp_start"].items()}
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

    def color_mapping(values: pd.Series) -> dict[str, str]:
        # Stage 5 solves the palette against its own greys in either theme, so a named
        # category has one colour on the dark map and the light one. Only the greys change.
        mapping = {k: neutrals.get(k, v) for k, v in s5.color_mapping(values).items()}
        assert len(set(mapping.values())) == len(mapping), f"two categories share a colour: {mapping}"
        return mapping

    maps = {
        "part": color_mapping(part),
        "area": color_mapping(area),
        "branch": color_mapping(branch),
    }
    # The same ramp as import depth, so the two depth overlays can be compared; dict order
    # is the legend's order.
    maps["steps"] = {
        name: to_hex(colormaps[ramps["viridis"]](x))
        for name, x in zip(bin_names, np.linspace(0, 0.95, len(bin_names)))
    }
    maps["steps"][NOT_USED] = neutrals[NOT_USED]
    assert len(set(maps["steps"].values())) == len(maps["steps"]), "two step bins share a colour"
    if theme["darkmode"]:
        pills = {k: dark_pill_colors(v, s5.contrast) for k, v in maps["part"].items()}
    else:
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
            # 3,000 characters of names keep 96% of the 1.06M declarations findable by
            # name; 300 would keep 53%.
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
        (
            sample["import_depth"].to_numpy(),
            continuous("import_depth", "Import depth from Main", ramps["viridis"]),
        ),
        (steps_bin.to_numpy(), categorical("steps", "Steps from the main theorem")),
        (
            np.log10(sample["n_decls"].to_numpy()),
            continuous("log_decls", "Declarations in file (log10)", ramps["magma"]),
        ),
    ]

    # Each subtitle is short enough for one line: at 1280 px the panel stops growing and a
    # longer line wraps. The public one names the author and says that clicking works.
    sub_title = {
        "private": f"{n:,} Lean files, placed by statement text · {len(regions)} named regions",
        "public": f"{n:,} files of Levent Alpöge's Lean proof. Click one to open it.",
    }
    where = {}
    if args.public:
        # Data in files beside the page: one 50 MB HTML file is too heavy for the web. The
        # page refers to them by file name, so they must stay next to index.html.
        DOCS.mkdir(parents=True, exist_ok=True)
        for old in DOCS.glob(f"{DATA_PREFIX}_*"):
            old.unlink()  # a build that names its files differently must not leave these behind
        where = {"inline_data": False, "offline_data_path": str(DOCS / DATA_PREFIX)}
    plot = datamapplot.create_interactive_plot(
        coords,
        *layers,
        hover_text=extra["title"].to_numpy(),
        title=TITLE[build],
        sub_title=sub_title[build],
        title_font_size=24,
        sub_title_font_size=13,
        font_family="IBM Plex Sans",
        extra_point_data=extra,
        hover_text_html_template=hover_template(theme["colors"]),
        on_click="window.open(`{url}`)",
        enable_search=True,
        search_field="search_text",
        colormap_rawdata=[values for values, _ in overlays],
        colormap_metadata=[meta for _, meta in overlays],
        cvd_safer=True,
        initial_zoom_fraction=0.9,  # a few far-out islands otherwise squeeze the cloud
        darkmode=theme["darkmode"],
        custom_css=Template(CUSTOM_CSS).substitute(theme["colors"]),
        custom_js=CUSTOM_JS,
        **where,
    )

    # The title panel holds only spans, so its first closing </div> is its own.
    pills_html = "".join(f'<span class="title-pill">{html.escape(p)}</span>' for p in PILLS[build])
    in_panel = f"<div>{pills_html}</div>" + (ABOUT if args.public else "")
    text, found = re.subn(
        r'(<div\s+id="title-container"[^>]*>.*?)(</div>)',
        lambda m: m.group(1) + in_panel + m.group(2),
        str(plot),
        count=1,
        flags=re.DOTALL,
    )
    assert found == 1, "title panel not found"
    if not args.public:
        write_text_safely(text, PRIVATE_HTML)
        print(f"wrote {PRIVATE_HTML} ({PRIVATE_HTML.stat().st_size / 1e6:.1f} MB)")
        return

    # After the charset declaration, which browsers honour only in the first 1024 bytes.
    anchor = re.search(r"<meta[^>]*charset[^>]*>", text) or re.search("<head>", text)
    assert anchor is not None, "no <head> in the rendered page"
    text = text[: anchor.end()] + "\n" + open_graph_tags(n) + text[anchor.end() :]
    write_text_safely(text, PUBLIC_HTML)
    files = sorted(p for p in DOCS.iterdir() if p.is_file())
    for p in files:
        print(f"  {p.stat().st_size / 1e6:7.2f} MB  {p.name}")
    print(f"wrote {DOCS}: {sum(p.stat().st_size for p in files) / 1e6:.1f} MB in {len(files)} files")
    print(f"next: uv run python -I scripts/11_social_preview.py to refresh {SOCIAL_PREVIEW.name}")


if __name__ == "__main__":
    main()
