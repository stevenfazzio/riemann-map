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
directories play no part in where a file lands. What counts as similar is decided by a
text embedding model; [How files are placed](#how-files-are-placed) says which one and
how much it matters.

The 412 region names were written by Claude Opus 5.5, each from eight files near the
middle of a region, the region's most distinctive identifiers and the names of the
regions inside it. Nobody who knows the proof has reviewed them. Some are named after the proof's own identifiers, and about a
fifth of the files sit in no named region at any level.

The colour menu offers seven overlays: the Mathlib area a file opens, the branch of the
proof it sits under, its directory, its import depth, its distance from the main
theorem, how many declarations it holds, and how isolated it is among files that state
similar things. The dependency fields come from a source scan (the CC0 records below),
not from a Lean build, so they are approximate.
[Colour overlays](#colour-overlays) says how each one is worked out.

## How files are placed

Where a file lands is decided by a text embedding model. The map is that model's
reading of the statements, and another model would draw a different map.

1. **Text.** Each file is reduced to its statements: the signature of every theorem and
   lemma, and every definition as written, up to 2,000 characters each. Proofs, paths,
   imports and `open` lines are left out. A theorem stated as a bare named proposition
   (`theorem foo : Foo`) also carries the definition of `Foo`.
2. **Embedding.** [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B),
   a general-purpose text embedding model, embeds each text behind this instruction:

   > Given the theorem statements and definitions of a Lean 4 source file, identify the
   > area of mathematics it belongs to

   The first 1,024 of the model's 4,096 dimensions are kept. Texts over 40,000
   characters are cut to that length, which affects 90 files. One provider (Nebius,
   through OpenRouter) embedded every file: in a 200-file test a second provider's
   vectors changed a tenth of each file's nearest neighbours.
3. **Layout.** UMAP reduces the vectors to two dimensions (15 neighbours, cosine
   distance, `min_dist` 0.05, a fixed seed).

The model was chosen on the 10,000-file pilot, by what a file has in common with its 15
nearest neighbours on the map. The measure is the overlap between the Mathlib namespaces
two files `open`, from 0 to 1; random pairs score 0.09. `open` lines are not in the
embedded text, so this checks the layout against something the model was not given.

| Embedding of the same text | Neighbours' overlap | Embedding neighbours kept on the map |
|---|---|---|
| gte-modernbert-base, run locally | 0.26 | 13% |
| Qwen3-Embedding-8B | 0.31 | 20% |
| Qwen3-Embedding-8B behind the instruction | 0.33 | 29% |

The last column is the share of a file's 15 nearest neighbours on the map that are also
among its 15 nearest in the embedding.

On the finished map of 175,013 files the overlap is 0.43 (0.54 in the embedding before
UMAP, 0.14 for random pairs), and 13% of embedding neighbours are kept. Two dimensions
hold the broad regions and lose most of the fine structure, so the map is better read
by region than by which file is nearest to which.

## Colour overlays

The default colouring is by named region. The colour menu offers seven more. None of
them is given to the embedding model; the last is worked out from what the model returns.

| Overlay | What colours a file |
|---|---|
| Mathlib area opened | The most specific of seven hand-grouped Mathlib areas among the namespaces the file opens |
| Proof branch | Which of eight large sub-proofs under the main theorem the file's declarations belong to |
| Part (directory) | The second component of the file's path. The 14 largest are named and the rest are "Other" |
| Import depth from Main | The longest chain of imports from `Main` down to the file (`Main` is 0, the deepest file 202) |
| Steps from the main theorem | The distance from `riemannHypothesis` in the uses-graph, for the nearest declaration in the file, in six bins. Files the proof does not use are grey |
| Declarations in file | How many declarations the file holds, on a log scale |
| Isolation | How far the file is from the 15 files most like it, in seven percentile bands. See [Isolation](#isolation) |

Proof branch, import depth and steps come from the CC0 records. Those are a scan of the
source text and not a Lean build, so what uses what is approximate. Category colours are
a colour-blind-safe Glasbey palette, handed out in order of how many files each category
holds, with greys for the catch-all categories.

### Mathlib area opened

A Lean file's `open` lines name the namespaces it works in. The project's own
namespaces are set aside, and the root of each remaining one is looked up in a grouping
of 53 Mathlib namespaces into seven areas (`MATHLIB_AREAS` in `05_render_pilot.py`). The
grouping was made by hand from the most frequently opened namespaces. It is not a
taxonomy.

70% of files open namespaces from more than one area. Such a file takes the first of its
areas in the order below, however many namespaces it opens from each. The order runs
from the most distinctive subjects to the most ambient, so a common area colours a file
only when the file opens nothing more specific.

| Area, in order | Files coloured | Files that open it |
|---|---|---|
| Algebraic geometry | 40,356 | 40,356 |
| Category theory | 2,124 | 42,399 |
| Polynomials | 22,526 | 25,872 |
| Analysis and measure | 52,533 | 69,529 |
| Linear algebra | 20,953 | 77,966 |
| Number fields | 16,798 | 98,535 |
| Sets and logic | 3,354 | 38,196 |

"None opened" (14,475 files) opens no Mathlib namespace: 6,987 of them open nothing and
the rest open only the project's own. "Other Mathlib" (1,894 files) opens only
namespaces outside the grouping.

### Proof branch

The records trace everything `riemannHypothesis` uses, directly or not: 757,483 of the
release's 1,061,531 declarations. They put eight declarations on a trunk that leads down
from the theorem, and name eight large declarations that hang off it. A declaration
belongs to a branch when the proof reaches it only through that branch's head. 300,318
declarations do. The others are not confined to any one of the eight and have no branch.

A file takes the most common branch among its declarations that have one. Of the 64,687
files with a branch, 64,443 hold declarations of one branch only. The legend names a
branch by the last part of its head's Lean name (`vwin_of_row` is one), and the four
files that hold the trunk are `trunk`.

"Used, no single branch" (79,679 files) means the proof uses the file but none of its
declarations sits under a single branch. "Not used by the proof" (30,647 files) means
the scan found nothing in the file that the theorem depends on.

### Isolation

A file's isolation is the mean cosine distance from its embedding to those of its 15
nearest files, measured before UMAP. The bands are percentiles, because the values bunch
up: the median is 0.062 and the most isolated 1% run from 0.17 to 0.32. UMAP spreads
files out roughly evenly, so a tight group of near-identical statements and a loose
scatter of unrelated ones look much the same on the map. This overlay shows which is
which. It follows the layout closely: the Spearman correlation between a file's band and
the median band of its 15 nearest files on the map is 0.55, against 0.37 for import depth
and 0.40 for steps from the theorem.

It describes the embedding, not the proof. Of the 1,751 most isolated files, 64% state a
single theorem whose type chains names defined elsewhere, such as
`theorem slice_of : Dict → FinTr → Slice`. Only 2% of all files look like that. The
statements text adds a definition only to a theorem stated as one bare name, so these
files reach the model as a few abbreviations and nothing more. UMAP puts files it cannot
place well near the middle: half of the most isolated 1% sit among the 7% of files
closest to the centre of the map.

## How it's built

| Stage | Script | What it does |
|---|---|---|
| Tables | `01_build_tables.py` | One row per file and per declaration, from the [CC0 records](https://github.com/tamnd/riemann) of the release |
| Text | `06_compose_statements.py` | Each file's statements text: theorem signatures and whole definitions, without proofs or paths |
| Embed | `08_embed_api.py` | Qwen3-Embedding-8B through OpenRouter, behind an instruction asking for the area of mathematics |
| Layout | `04_reduce_and_measure.py` | UMAP to 2-d, and measures of what neighbouring files have in common |
| Names | `09_name_regions.py` | [Toponymy](https://github.com/TutteInstitute/toponymy) finds regions on the 2-d layout and names them with Claude |
| Render | `10_render_map.py` | [DataMapPlot](https://github.com/TutteInstitute/datamapplot) builds a private single-file map, or `docs/` with `--public`; the page is dark unless `--light` is given. The first run also finds each file's 15 nearest files in the embedding for the isolation overlay, which takes about 5 minutes and is kept in `data/` |
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
