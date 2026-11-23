"""Stage 9: name the regions of the full map with Toponymy and Claude.

    uv run python -I scripts/09_name_regions.py --coarsest 2 --tag pilot --dry-run --count-tokens
    uv run python -I scripts/09_name_regions.py --coarsest 2 --tag pilot
    uv run python -I scripts/09_name_regions.py --tag full --max-dollars 15

Regions are found by density clustering in the plotted layout (statements only, q8_inst),
so a named region is something a viewer can see. The cluster labels are saved the first
time and read back afterwards: Toponymy 0.5.1 builds its spanning tree with
reproducible=False, so a second fit need not give the same regions, and a pilot must name
regions the full run will also have.

What Claude is shown for a region: 8 files nearest the region's centre (their statements
text, cut to EXEMPLAR_CHARS; Toponymy does not shorten them), 16 key phrases, and for a
layer above the finest one the names of the regions inside it. Toponymy then re-prompts
for groups of names it finds too alike, with every group member's files in one prompt.

--coarsest N names only the N coarsest layers. They get the name lengths they would get
in a run of all layers, but the finest of them is then named from files and key phrases
alone, without the sub-region names a full run would give it.

Key phrases and region names are embedded with the same model as the files (Qwen3 through
OpenRouter, Nebius, stage 8's request code) but without the subject instruction and into
their own cache. Read in the installed source: Toponymy compares these vectors only with
each other (key phrase with key phrase, name with name), never with a file's vector, and
the instruction describes a Lean source file, which neither is.

Every Claude answer is kept on disk under a hash of the model and the prompt, so a re-run
pays only for prompts it has not sent before. --dry-run sends nothing to Claude: it
builds every prompt, answers with placeholder names and reports prompt sizes
(--count-tokens asks Anthropic's free token counter for exact input sizes, which does send
the prompts there). A dry run's re-prompting is no guide to a real run's, in either
direction: over all four layers placeholder names had 120 regions re-prompted and the real
names 154 (41% more input on top of the naming prompts), while in the two-layer pilot the
placeholders had 19 and the real names 5.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import threading
import time
import warnings

import anthropic
import numpy as np
import pandas as pd
import scipy.sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from toponymy.cluster_layer import ClusterLayerText
from toponymy.clustering import Clusterer, ToponymyClusterer, build_cluster_tree, centroids_from_labels
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.llm_wrappers import AsyncLLMWrapper, FailFastLLMError
from toponymy.toponymy import Toponymy

from riemann_map import (
    FULL,
    RUN_DIRS,
    load_stage,
    write_npy_safely,
    write_parquet_safely,
    write_text_safely,
)

VARIANT = "q8_inst"  # the layout Steven chose on 2026-11-22: statements only
NAMING = FULL / "naming"
EXEMPLAR_CHARS = 1500  # 31% of files are longer; the mean after cutting is about 900
DIMS = 1024  # as stage 8 keeps for the files
OBJECTS = "Lean 4 source files, each shown as its theorem statements and definitions with proofs removed"
CORPUS = "a Lean 4 formalisation of a proof of the Riemann hypothesis, built on Mathlib"
# With four layers this asks for 4-8, 3-6, 2-5 and 1-4 words, finest first. Steven's usual
# 0.2-0.8 would ask the finest layer for 6-12, and the pilot's names already ran a word or
# two over what was asked. The two coarsest layers get the same request under either range.
DETAIL = (0.4, 0.8)
# Dollars per million input and output tokens, from Anthropic's model table as of 2026-11-03.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5-5": (2.0, 10.0),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-fable-5-2": (8.0, 40.0),
}
# Room for the model's thinking as well as the one-line answer. Toponymy asks for 128.
MAX_TOKENS = 8000
# Lean syntax, so that key phrases are identifiers and not "theorem foo". A phrase is
# dropped when its first or last word is a stop word.
LEAN_WORDS = frozenset(
    "theorem lemma def abbrev instance structure class inductive noncomputable private protected "
    "variable variables universe namespace section end open scoped import where fun by let have show "
    "from match with then else if do at in forall exists type prop sort true false set_option deriving "
    "extends macro syntax notation local attribute mutual opaque axiom example unsafe partial nonrec "
    "omit include export hiding renaming using sorry".split()
)


def cut(text: str) -> str:
    if len(text) <= EXEMPLAR_CHARS:
        return text
    head = text[:EXEMPLAR_CHARS]
    newline = head.rfind("\n")
    if newline >= EXEMPLAR_CHARS // 2:
        head = head[:newline]
    return head + "\n…"


def sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode())
        h.update(b"\0")
    return h.hexdigest()


# --- regions ---------------------------------------------------------------------------
def cluster_labels(coords: np.ndarray, emb: np.ndarray, base: int) -> np.ndarray:
    """Cluster labels per layer, finest first, shape (layers, files); -1 is unclustered."""
    path = NAMING / f"cluster_labels_b{base}.npy"
    meta_path = path.with_suffix(".json")
    coords_hash = hashlib.sha256(np.ascontiguousarray(coords).tobytes()).hexdigest()
    if path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["coords_sha256"] != coords_hash:
            raise SystemExit(f"{path} was made from a different layout; move it aside to cluster again")
        return np.load(path)
    clusterer = ToponymyClusterer(min_clusters=6, min_samples=5, base_min_cluster_size=base)
    clusterer.fit(clusterable_vectors=coords, embedding_vectors=emb)
    labels = np.vstack([layer.cluster_labels for layer in clusterer.cluster_layers_]).astype(np.int32)
    meta = {
        "coords_sha256": coords_hash,
        "base_min_cluster_size": base,
        "min_samples": 5,
        "regions_per_layer": [int(row.max()) + 1 for row in labels],
        "share_unclustered_per_layer": [round(float((row < 0).mean()), 4) for row in labels],
    }
    write_text_safely(json.dumps(meta, indent=2), meta_path)
    write_npy_safely(labels, path)  # written last: its presence marks the labels as usable
    return labels


class SavedLayers(Clusterer):
    """Layers built from saved labels. Toponymy reuses a clusterer that already has them."""

    def __init__(self, labels: list[np.ndarray], emb: np.ndarray):
        super().__init__()
        labels = [np.asarray(row, dtype=np.int64) for row in labels]
        self.cluster_layers_ = [
            ClusterLayerText(row, centroids_from_labels(row, emb), layer_id=i) for i, row in enumerate(labels)
        ]
        self.cluster_tree_ = build_cluster_tree(labels)

    def fit(self, *args, **kwargs):
        return self

    def fit_predict(self, *args, **kwargs):
        return self.cluster_layers_, self.cluster_tree_


# --- key phrases -------------------------------------------------------------------------
class CachedKeyphrases(KeyphraseBuilder):
    """Counting phrases over every file takes minutes, so the result is kept on disk."""

    def fit(self, objects):
        settings = repr((self.ngram_range, self.token_pattern, self.max_features, self.min_occurrences))
        stem = NAMING / f"keyphrases_{sha(*objects, settings, *sorted(self.stop_words))[:16]}"
        matrix_path, list_path = stem.with_suffix(".npz"), stem.with_suffix(".json")
        if list_path.exists():
            self.object_x_keyphrase_matrix_ = scipy.sparse.load_npz(matrix_path)
            self.keyphrase_list_ = json.loads(list_path.read_text())
            print(f"key phrases: read {len(self.keyphrase_list_):,} from {list_path.name}", flush=True)
        else:
            t0 = time.perf_counter()
            super().fit(objects)
            tmp = stem.with_suffix(".tmp.npz")
            scipy.sparse.save_npz(tmp, self.object_x_keyphrase_matrix_.tocsr())
            os.replace(tmp, matrix_path)
            write_text_safely(json.dumps(self.keyphrase_list_), list_path)  # last: marks it done
            print(
                f"key phrases: counted {len(self.keyphrase_list_):,} over {len(objects):,} files "
                f"in {(time.perf_counter() - t0) / 60:.1f} min",
                flush=True,
            )
        assert self.object_x_keyphrase_matrix_.shape == (len(objects), len(self.keyphrase_list_))
        self.keyphrase_vectors_ = None  # Toponymy asks the embedder for them
        return self


class ShortTextEmbedder:
    """Qwen3 vectors for key phrases and region names, kept on disk in append-only parts."""

    def __init__(self, api, key: str, workers: int):
        self.api, self.key, self.workers = api, key, workers
        self.dir = NAMING / "short_text_emb"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.vectors: dict[str, np.ndarray] = {}
        self.parts = 0
        self.tokens = 0
        for path in sorted(self.dir.glob("part_*.npz")):
            with np.load(path) as part:
                self.vectors.update(zip(part["texts"].tolist(), part["vecs"]))
            self.parts += 1

    def encode(self, texts, show_progress_bar=None, **kwargs) -> np.ndarray:
        # An empty name (a failed naming call) cannot be embedded; it still needs a vector.
        texts = [str(t) if str(t).strip() else "unnamed" for t in texts]
        missing = sorted(set(texts) - self.vectors.keys())
        for start in range(0, len(missing), 4096):
            chunk = missing[start : start + 4096]
            vecs, tokens, cut_short, providers = self.api.embed_texts(self.key, chunk, self.workers)
            assert cut_short == 0 and vecs.shape[1] >= DIMS, (cut_short, vecs.shape)
            vecs = vecs[:, :DIMS]
            vecs = (vecs / np.linalg.norm(vecs, axis=1, keepdims=True)).astype(np.float32)
            path = self.dir / f"part_{self.parts:05d}.npz"
            tmp = self.dir / f"tmp_{self.parts:05d}.npz"  # not part_*: a crash must not leave one
            np.savez(tmp, texts=np.array(chunk), vecs=vecs)
            os.replace(tmp, path)
            self.parts += 1
            self.vectors.update(zip(chunk, vecs))
            self.tokens += tokens
            if len(missing) > 100:
                print(
                    f"short texts: embedded {min(start + 4096, len(missing)):,} of {len(missing):,} "
                    f"({self.tokens:,} tokens so far, provider {sorted(providers)})  "
                    f"[{time.strftime('%H:%M:%S')}]",
                    flush=True,
                )
        return np.vstack([self.vectors[t] for t in texts])


# --- Claude ------------------------------------------------------------------------------
class OverBudget(Exception):
    """The dollar limit for this run is reached; no further request is sent."""


class ClaudeNamer(AsyncLLMWrapper):
    """One request per prompt through the Anthropic SDK, answers kept on disk.

    Toponymy's own Anthropic wrapper always sends a temperature, which Opus 5.5 rejects, and
    asks for 128 output tokens, which leaves no room for thinking the model cannot turn off.
    Toponymy runs each layer and each re-prompting pass under its own asyncio.run(), so
    nothing here may be bound to an event loop: the blocking client runs in worker threads.
    """

    FAIL_FAST_EXCEPTIONS = (
        anthropic.AuthenticationError,
        anthropic.PermissionDeniedError,
        anthropic.NotFoundError,
        anthropic.BadRequestError,
        OverBudget,
    )

    def __init__(self, model, effort, max_dollars, concurrency, dry_run, count_tokens, log_path):
        self.model, self.effort, self.max_dollars = model, effort, max_dollars
        self.dry_run, self.count_tokens, self.log_path = dry_run, count_tokens, log_path
        self.cache = NAMING / "llm_cache"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.client = anthropic.Anthropic(max_retries=5) if (not dry_run or count_tokens) else None
        self.slots = threading.BoundedSemaphore(concurrency)
        self.lock = threading.Lock()
        self.calls: list[dict] = []
        self.spent = 0.0  # dollars billed by this run; cached answers cost nothing

    def dollars(self, usage: dict) -> float:
        price_in, price_out = PRICES[self.model]
        return (usage["input_tokens"] * price_in + usage["output_tokens"] * price_out) / 1e6

    def record(self, **call) -> None:
        with self.lock:
            self.calls.append(call)
            with open(self.log_path, "a") as f:
                f.write(json.dumps(call) + "\n")

    def ask(self, system: str, user: str) -> str:
        key = sha(self.model, self.effort or "", system, user)
        # A re-prompt lists its regions as '"1. Old name":'; a naming prompt has none.
        regions = len(re.findall(r'^"\d+\. ', user, flags=re.M))
        call = {"key": key, "kind": "rename" if regions else "name", "regions": regions or 1}
        call["chars"] = len(system) + len(user)
        request = {"model": self.model, "messages": [{"role": "user", "content": user}]}
        if system:
            request["system"] = system
        cached = self.cache / f"{key}.json"
        if cached.exists():
            saved = json.loads(cached.read_text())
            self.record(**call, cached=True, **saved["usage"])
            return saved["text"]
        if self.dry_run:
            if self.count_tokens:
                with self.slots:
                    counted = self.client.messages.count_tokens(**request)
                call["input_tokens"] = counted.input_tokens
            self.record(**call, dry_run=True, system=system, user=user)
            return json.dumps({"topic_name": f"Placeholder {key[:6]}", "topic_specificity": 0.5})
        with self.slots:
            with self.lock:
                if self.spent >= self.max_dollars:
                    raise OverBudget(f"${self.spent:.2f} billed, limit ${self.max_dollars:.2f}")
            request["max_tokens"] = MAX_TOKENS
            if self.effort:
                request["output_config"] = {"effort": self.effort}
            t0 = time.monotonic()
            message = self.client.messages.create(**request)
        usage = {"input_tokens": message.usage.input_tokens, "output_tokens": message.usage.output_tokens}
        details = getattr(message.usage, "output_tokens_details", None)
        if details is not None:
            usage["output_tokens_details"] = details.to_dict()
        with self.lock:
            self.spent += self.dollars(usage)
        self.record(
            **call,
            cached=False,
            stop_reason=message.stop_reason,
            seconds=round(time.monotonic() - t0, 1),
            **usage,
        )
        if message.stop_reason != "end_turn":
            # A refusal or a cut-off answer is billed but not kept; Toponymy tries again.
            raise RuntimeError(f"the model stopped with {message.stop_reason}")
        text = "".join(block.text for block in message.content if block.type == "text")
        saved = {"model": self.model, "effort": self.effort, "system": system, "user": user, "text": text}
        write_text_safely(json.dumps({**saved, "usage": usage}, ensure_ascii=False, indent=1), cached)
        return text

    async def _call_single_llm_with_system(self, system_prompt, user_prompt, temperature, max_tokens) -> str:
        return await asyncio.to_thread(self.ask, system_prompt, user_prompt)

    async def _call_single_llm(self, prompt, temperature, max_tokens) -> str:
        return await asyncio.to_thread(self.ask, "", prompt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=int, default=120, help="smallest region at the finest layer, in files")
    ap.add_argument("--coarsest", type=int, help="name only this many of the coarsest layers")
    ap.add_argument("--model", default="claude-opus-5-5", choices=list(PRICES))
    ap.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"], help="default: the model's"
    )
    ap.add_argument("--max-dollars", type=float, default=2.0, help="stop once Claude has billed this much")
    ap.add_argument("--concurrency", type=int, default=4, help="Claude requests at once")
    ap.add_argument("--tag", required=True, help="folder name for this run's outputs")
    ap.add_argument("--dry-run", action="store_true", help="build the prompts, send nothing to Claude")
    ap.add_argument("--count-tokens", action="store_true", help="with --dry-run: exact input sizes")
    args = ap.parse_args()
    for name in ["OPENROUTER_API_KEY"] + (
        [] if args.dry_run and not args.count_tokens else ["ANTHROPIC_API_KEY"]
    ):
        if not os.environ.get(name):
            raise SystemExit(f"{name} is not set")

    out = NAMING / args.tag
    out.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(FULL / "statements.parquet", columns=["module", "text_statements"])
    modules = table["module"].to_numpy()
    texts = [cut(t) for t in table["text_statements"]]
    del table
    coords = np.load(RUN_DIRS["full"] / f"coords_{VARIANT}.npy")
    emb = np.load(FULL / f"emb_{VARIANT}.npy")
    assert len(texts) == len(coords) == len(emb), (len(texts), coords.shape, emb.shape)

    labels = cluster_labels(coords, emb, args.base)
    n_layers = len(labels)
    kept = list(range(n_layers))[-args.coarsest :] if args.coarsest else list(range(n_layers))
    levels = np.linspace(*DETAIL, n_layers)
    counts = [int(row.max()) + 1 for row in labels]
    print(
        f"{len(texts):,} files; regions per layer, finest first: {counts}; naming layers {kept} "
        f"({sum(counts[i] for i in kept)} regions) with {args.model}"
        + (" — dry run, nothing is sent to Claude" if args.dry_run else ""),
        flush=True,
    )

    api = load_stage("08_embed_api")
    api.BATCH_TEXTS = 128  # short texts; a request's time barely depends on how many it holds
    embedder = ShortTextEmbedder(api, os.environ["OPENROUTER_API_KEY"], workers=6)
    log_path = out / f"calls_{time.strftime('%Y%m%d_%H%M%S')}{'_dry' if args.dry_run else ''}.jsonl"
    namer = ClaudeNamer(
        args.model, args.effort, args.max_dollars, args.concurrency, args.dry_run, args.count_tokens, log_path
    )
    topo = Toponymy(
        llm_wrapper=namer,
        text_embedding_model=embedder,
        clusterer=SavedLayers([labels[i] for i in kept], emb),
        keyphrase_builder=CachedKeyphrases(stop_words=ENGLISH_STOP_WORDS | LEAN_WORDS),
        object_description=OBJECTS,
        corpus_description=CORPUS,
        # linspace over the kept layers gives each the level it has among all layers.
        lowest_detail_level=float(levels[kept[0]]),
        highest_detail_level=float(levels[kept[-1]]),
        exemplar_delimiters=["```lean\n", "\n```"],
        verbose=False,
    )
    t0 = time.perf_counter()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            # The two vector slots are in the opposite order from the clusterer's fit.
            topo.fit(texts, embedding_vectors=emb, clusterable_vectors=coords)
    except FailFastLLMError as e:
        # Over budget, or a request Claude will never accept. Answers so far stay in the cache.
        raise SystemExit(
            f"stopped: {e}\n${namer.spent:.2f} billed in this run over {len(namer.calls)} calls; "
            f"the answers received are kept in {namer.cache} and a re-run does not pay for them again"
        ) from e
    problems = [str(w.message) for w in caught if "ailed" in str(w.message) or "ignore" in str(w.message)]

    calls = namer.calls
    by_kind = {
        kind: {
            "calls": sum(c["kind"] == kind for c in calls),
            "regions": sum(c["regions"] for c in calls if c["kind"] == kind),
            "chars": sum(c["chars"] for c in calls if c["kind"] == kind),
            "input_tokens": sum(c.get("input_tokens", 0) for c in calls if c["kind"] == kind),
            "output_tokens": sum(c.get("output_tokens", 0) for c in calls if c["kind"] == kind),
        }
        for kind in ("name", "rename")
    }
    summary = {
        "model": args.model,
        "effort": args.effort,
        "layers_named": kept,
        "regions_named": sum(counts[i] for i in kept),
        "detail_levels": [float(levels[i]) for i in kept],
        "calls": by_kind,
        "calls_answered_from_cache": sum(bool(c.get("cached")) for c in calls),
        "dollars_billed_this_run": round(namer.spent, 4),
        "short_text_tokens_embedded_this_run": embedder.tokens,
        "minutes": round((time.perf_counter() - t0) / 60, 1),
        "problems": problems,
    }
    print(json.dumps(summary, indent=2), flush=True)
    if args.dry_run:
        print(f"dry run: the prompts are in {log_path}")
        return

    parent = {child: p for p, children in topo.cluster_tree_.items() for child in children}
    rows = []
    for i, layer in enumerate(topo.cluster_layers_):
        sizes = np.bincount(layer.cluster_labels[layer.cluster_labels >= 0], minlength=len(layer.topic_names))
        for c, name in enumerate(layer.topic_names):
            up = parent.get((i, c))
            rows.append(
                {
                    "layer": kept[i],
                    "cluster_idx": c,
                    "name": name,
                    "n_files": int(sizes[c]),
                    # A parent one past the last layer is Toponymy's root for unadopted regions.
                    "parent_layer": kept[up[0]] if up and up[0] < len(kept) else -1,
                    "parent_idx": int(up[1]) if up and up[0] < len(kept) else -1,
                    "keyphrases": list(layer.keyphrases[c]),
                    "exemplar_modules": [str(modules[j]) for j in layer.exemplar_indices[c]],
                }
            )
    regions = pd.DataFrame(rows)
    write_parquet_safely(regions, out / "regions.parquet")
    write_text_safely(json.dumps(summary, indent=2), out / "summary.json")
    empty = int((regions["name"].str.strip() == "").sum())
    print(f"wrote {out / 'regions.parquet'}: {len(regions)} regions, {empty} without a name")


if __name__ == "__main__":
    main()
