# A Map of the Lean Proof of the Riemann Hypothesis

An interactive map of the Lean 4 proof of the Riemann hypothesis that
[Levent Alpöge released](https://alpo.ge/remark.pdf) on 20 November 2026. Each point
is one of the 175,013 source files that declare something (of 175,187 in the release),
placed by the text of its theorem statements and definitions and named at four zoom
levels.

**Live map:** https://stevenfazzio.com/riemann-map/

This is an unofficial project and is not affiliated with the author of the proof. The
published page quotes the Lean source in two places: a hovercard shows up to 320
characters of each file's first statement, and the search box matches file paths and
declaration names. Clicking a point opens the file in a
[public copy](https://github.com/tamnd/riemann-lean) of the release. The release states
no licence, so that text remains its author's work. To have anything corrected or
removed, [open an issue](https://github.com/stevenfazzio/riemann-map/issues).

## What the map shows

Files that sit near each other state similar mathematics. The layout uses only what a
file says: no paths, imports or proofs go into the text that is embedded, so
directories play no part in where a file lands.

The 412 region names were written by Claude Opus 5.5, each from eight files near the
middle of a region, the region's most distinctive identifiers and the names of the
regions inside it. Nobody who knows the proof has reviewed them. Some are named after the proof's own identifiers, and about a
fifth of the files sit in no named region at any level.

The colour menu offers six overlays: the Mathlib area a file opens, the branch of the
proof it sits under, its directory, its import depth, its distance from the main
theorem, and how many declarations it holds. The dependency fields come from a source
scan (the CC0 records below), not from a Lean build, so they are approximate.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Tables | `01_build_tables.py` | One row per file and per declaration, from the [CC0 records](https://github.com/tamnd/riemann) of the release |
| Text | `06_compose_statements.py` | Each file's statements text: theorem signatures and whole definitions, without proofs or paths |
| Embed | `08_embed_api.py` | Qwen3-Embedding-8B through OpenRouter, behind an instruction asking for the area of mathematics |
| Layout | `04_reduce_and_measure.py` | UMAP to 2-d, and measures of what neighbouring files have in common |
| Names | `09_name_regions.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) finds regions on the 2-d layout and names them with Claude |
| Render | `10_render_map.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds a private single-file map, or `docs/` with `--public` |
| Card | `11_social_preview.py` | Screenshots the public page in Chrome for the link-preview image |

Regions are found on the same 2-d layout that is plotted, so every named region
matches something visible on the map.

Stages 2, 3, 5 and 7 are the earlier work that chose this recipe: a 10,000-file pilot
with three candidate texts and a local embedding model, diagnostic maps, and a local
embedding run of every file that the hosted model replaced. The finished map does not
need their output, though stage 10 borrows helpers from stage 5.

## Reproducing

Requires [uv](https://docs.astral.sh/uv/), `OPENROUTER_API_KEY` and
`ANTHROPIC_API_KEY`. The social card needs a local Chrome. Two downloads go under
`data/raw/`:

- a clone of [tamnd/riemann-lean](https://github.com/tamnd/riemann-lean) at
  `data/raw/riemann-lean/`, so the sources are in `data/raw/riemann-lean/lean/src/`
- `records.tar` from the `records-9110120-r2` release of
  [tamnd/riemann](https://github.com/tamnd/riemann), extracted so that
  `data/raw/records-r2/x/records/index/` exists

```bash
uv run python -I scripts/01_build_tables.py
uv run python -I scripts/06_compose_statements.py
uv run python -I scripts/08_embed_api.py --run full --variants q8_inst    # about $1.30, and hours: the provider's quota sets the pace
uv run python -I scripts/04_reduce_and_measure.py --run full
uv run python -I scripts/09_name_regions.py --tag full --max-dollars 15   # about $14 on Claude Opus 5.5
uv run python -I scripts/10_render_map.py --public
uv run python -I scripts/11_social_preview.py
```

Intermediate data lives in `data/` and is not committed. Embeddings and model answers
are cached on disk, so re-running a stage does not repeat paid calls, and both paid
stages stop at a dollar limit.

A fresh run will not give exactly this map. The region finder is not deterministic,
the names are model output, and stage 10 carries one hand-written name that is tied to
this run's regions.

## License

The code is released under the [MIT License](LICENSE). The Lean source, including the
excerpts in `docs/`, is the work of Levent Alpöge and is not covered by that licence.
