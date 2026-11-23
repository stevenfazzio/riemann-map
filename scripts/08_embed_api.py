"""Stage 8: embed with a hosted model, Qwen3-Embedding-8B, through OpenRouter.

    --run pilot   the 10,000-file sample (stages 2-5), three variants, 4,096 dimensions
    --run full    every file, in stage 6's row order, two variants, first 1,024 dimensions

Reads   data/full/statements.parquet (and data/pilot/sample.parquet for the pilot's rows)
Writes  {run dir}/emb_{variant}.npy   float32, L2-normalised, row order = the run's rows
        {run dir}/api_meta.json       model, provider, tokens billed, texts cut

Variants
  q8_statements  the statements recipe (stage 6), as is                      pilot only
  q8_inst        the same text behind an instruction asking for the subject
  q8_names       path and declaration names

What leaves the machine: the texts, to OpenRouter and the one provider pinned below.
The request carries the API key and nothing else that identifies the sender. The key is
read from OPENROUTER_API_KEY and never printed.

One provider is pinned with fallbacks off, Nebius unless --provider names another. Vectors
from two providers do not share a space: on 200 files DeepInfra's differed from Nebius's
by cosine 0.997 where two Nebius runs agree to 0.99995, and a tenth of each file's 15
neighbours changed. So one variant is embedded on one provider from start to finish, and
the manifest refuses pieces made on another. Two variants may use different providers,
because their vectors are never compared with each other.

The full run keeps the first 1,024 of the model's 4,096 dimensions and renormalises.
The model is trained for that; on the pilot it cost nothing measurable (neighbours'
Mathlib overlap 0.376 against 0.377) and it cuts the stored matrix from 2.9 GB to 0.7.

The provider enforces a token quota (HTTP 429, "exceeded your tokens quota"). A refused
request waits and is sent again, so the run settles at whatever rate the quota allows; on
2026-11-22 on Nebius that was 0.15-0.27M tokens a minute. See post().

Resumable: each variant is embedded in pieces under emb_{variant}.chunks/, with a
manifest that pins model, provider, dimensions and a hash of the texts. --max-dollars
stops the run if the tokens billed in this invocation pass the budget.

    uv run python -I scripts/08_embed_api.py --probe 20                          # writes nothing
    uv run python -I scripts/08_embed_api.py                                     # the pilot
    uv run python -I scripts/08_embed_api.py --run full --variants q8_inst --limit 40000
    uv run python -I scripts/08_embed_api.py --run full                          # everything
    uv run python -I scripts/08_embed_api.py --run full --variants q8_names --provider deepinfra
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

from riemann_map import EMBED_CHUNK, FULL, PILOT, write_npy_safely, write_text_safely

API_URL = "https://openrouter.ai/api/v1/embeddings"
MODEL = "qwen/qwen3-embedding-8b"
# main() puts --provider into "order"; the other two settings are fixed.
PROVIDER = {"order": ["nebius"], "allow_fallbacks": False, "data_collection": "deny"}
PRICE_PER_M = 0.01  # dollars per million tokens, from OpenRouter's model list on 2026-11-22
# The model card's format for an instructed input; documents normally carry none.
INSTRUCTION = (
    "Given the theorem statements and definitions of a Lean 4 source file, "
    "identify the area of mathematics it belongs to"
)
RUNS = {
    "pilot": {"dir": PILOT, "variants": ["q8_statements", "q8_inst", "q8_names"], "piece": 250, "dims": 4096},
    "full": {"dir": FULL, "variants": ["q8_inst", "q8_names"], "piece": EMBED_CHUNK, "dims": 1024},
}
# Characters per billed token, measured on the pilot; used only to state a cost up front.
CHARS_PER_TOKEN = {"q8_statements": 2.25, "q8_inst": 2.3, "q8_names": 3.4}
MAX_CHARS = 40_000  # one text; about 20k tokens of Lean, inside the provider's 40,960
BATCH_TEXTS = 32  # texts per request; --batch-texts sets it
BATCH_CHARS = 60_000  # characters per request
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 522, 524, 529}
TIMEOUT_STATUS = {408, 504, 524}  # the provider did not answer OpenRouter in time (60 s there)
# Seconds one request keeps being sent again before the run gives up; --patience sets it.
# Refused and timed-out requests are not billed, so waiting out an outage costs only time.
PATIENCE = 30 * 60
QUOTA_WAITS = [6, 12, 24, 30]  # seconds before a refused request is sent again
# A request of several texts that times out this often is split. Not sooner: on the evening
# of 2026-11-22 about half of all requests to Nebius hung, and in five timed requests a
# 28k-token one was answered in 6 s between two that hung, so size was not the cause.
SPLIT_AFTER_TIMEOUTS = 4
PIECE_ATTEMPTS = 5  # tries at one piece, two minutes apart, before the run stops
_local = threading.local()
_stats_lock = threading.Lock()
# Requests refused for quota, answered, timed out, failed otherwise; tokens billed.
_stats = {"hits": 0, "ok": 0, "slow": 0, "other": 0, "tokens": 0}


class TooLong(Exception):
    """The provider rejected the request as malformed or too long (HTTP 400/413)."""


class TooSlow(Exception):
    """A request of several texts keeps timing out; smaller requests may get through."""


def session() -> requests.Session:
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
    return _local.s


def count(what: str, n: int = 1) -> None:
    with _stats_lock:
        _stats[what] += n


def heartbeat(every: float = 600) -> None:
    """Print the request counts now and then, so a provider outage does not look like a hang."""
    while True:
        time.sleep(every)
        with _stats_lock:
            s = dict(_stats)
        print(
            f"  [{time.strftime('%H:%M:%S')}] requests so far: {s['ok']} answered, {s['hits']} refused, "
            f"{s['slow']} timed out, {s['other']} failed otherwise; "
            f"${s['tokens'] / 1e6 * PRICE_PER_M:.2f} billed",
            flush=True,
        )


def post(key: str, texts: list[str]) -> dict:
    """One request, sent again until it is answered or PATIENCE runs out.

    Measured on 2026-11-22 through OpenRouter on Nebius: about half of all requests are
    refused for quota (HTTP 429, no Retry-After) and the sustained pace is 0.15-0.27M
    tokens a minute, whether refused requests retry on their own, as here, or all workers
    hold off together. OpenRouter gives the provider 60 s; slower answers come back as 504.
    """
    payload = {"model": MODEL, "input": texts, "provider": PROVIDER, "encoding_format": "float"}
    last = "no attempt made"
    refusals = timeouts = failures = 0
    started = time.monotonic()
    while True:
        code, timed_out = None, False
        try:
            r = session().post(API_URL, headers={"Authorization": f"Bearer {key}"}, json=payload, timeout=180)
        except requests.Timeout:
            last, timed_out = "no answer in 180 s", True
        except requests.ConnectionError as e:
            last = type(e).__name__
        else:
            code, body = r.status_code, None
            try:
                body = r.json()
            except ValueError:
                last = f"HTTP {code}, body is not JSON"
            if isinstance(body, dict) and "data" in body and code == 200:
                count("ok")
                return body
            if isinstance(body, dict) and "error" in body:
                # OpenRouter can answer 200 with an error object from the provider.
                err = body["error"]
                code = int(err.get("code", code)) if str(err.get("code", "")).isdigit() else code
                last = f"HTTP {code}: {str(err.get('message', ''))[:300]}"
            elif body is not None:
                last = f"HTTP {code}: unexpected body {str(body)[:200]}"
            if code in (400, 413):
                raise TooLong(last)
            if body is not None and code not in RETRY_STATUS:
                # Bad key, no credit, unknown model or provider: retrying would hide it.
                raise RuntimeError(last)
            timed_out = code in TIMEOUT_STATUS
        if time.monotonic() - started > PATIENCE:
            raise RuntimeError(f"still failing after {PATIENCE / 60:.0f} minutes: {last}")
        if code == 429:
            count("hits")
            wait = QUOTA_WAITS[min(refusals, len(QUOTA_WAITS) - 1)]
            refusals += 1
        elif timed_out:
            count("slow")
            timeouts += 1
            if len(texts) > 1 and timeouts >= SPLIT_AFTER_TIMEOUTS:
                raise TooSlow(last)
            wait = min(10 * timeouts, 60)
        else:
            count("other")
            failures += 1
            wait = min(2**failures * 3, 60)
        time.sleep(wait * random.uniform(0.75, 1.25))


def embed_batch(key: str, texts: list[str]) -> tuple[np.ndarray, int, int, set[str]]:
    """Vectors for texts, tokens billed, texts cut shorter than given, providers seen."""
    try:
        body = post(key, texts)
    except TooSlow:
        mid = len(texts) // 2
        a, b = embed_batch(key, texts[:mid]), embed_batch(key, texts[mid:])
        return np.vstack([a[0], b[0]]), a[1] + b[1], a[2] + b[2], a[3] | b[3]
    except TooLong as e:
        if len(texts) > 1:
            mid = len(texts) // 2
            a, b = embed_batch(key, texts[:mid]), embed_batch(key, texts[mid:])
            return np.vstack([a[0], b[0]]), a[1] + b[1], a[2] + b[2], a[3] | b[3]
        if len(texts[0]) < 2_000:
            raise RuntimeError(f"a {len(texts[0])}-character text was rejected: {e}") from e
        vec, tok, _, prov = embed_batch(key, [texts[0][: len(texts[0]) // 2]])
        return vec, tok, 1, prov
    data = sorted(body["data"], key=lambda d: d["index"])
    assert len(data) == len(texts), (len(data), len(texts))
    vecs = np.array([d["embedding"] for d in data], dtype=np.float32)
    usage = body.get("usage") or {}
    tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
    count("tokens", tokens)
    provider = {str(body["provider"])} if body.get("provider") else set()
    return vecs, tokens, 0, provider


def batches(texts: list[str]) -> list[tuple[int, int]]:
    """Consecutive (start, end) ranges holding at most BATCH_TEXTS texts and BATCH_CHARS chars."""
    out, start, chars = [], 0, 0
    for i, t in enumerate(texts):
        if i > start and (i - start >= BATCH_TEXTS or chars + len(t) > BATCH_CHARS):
            out.append((start, i))
            start, chars = i, 0
        chars += len(t)
    if start < len(texts):
        out.append((start, len(texts)))
    return out


def embed_texts(key: str, texts: list[str], workers: int) -> tuple[np.ndarray, int, int, set[str]]:
    ranges = batches(texts)
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda r: embed_batch(key, texts[r[0] : r[1]]), ranges))
    vecs = np.vstack([r[0] for r in results])
    assert vecs.shape[0] == len(texts) and np.isfinite(vecs).all()
    providers = set().union(*(r[3] for r in results))
    return vecs, sum(r[1] for r in results), sum(r[2] for r in results), providers


def variant_texts(run: str, variant: str) -> list[str]:
    column = "text_names" if variant == "q8_names" else "text_statements"
    table = pd.read_parquet(FULL / "statements.parquet", columns=["module", column])
    if run == "pilot":
        sample = pd.read_parquet(PILOT / "sample.parquet", columns=["module"])
        table = sample.merge(table, on="module", how="left", validate="one_to_one")
        assert table[column].notna().all(), "a pilot file is missing from stage 6"
    texts = [t[:MAX_CHARS] for t in table[column]]
    if variant == "q8_inst":
        texts = [f"Instruct: {INSTRUCTION}\nQuery:{t}" for t in texts]
    return texts


def main() -> None:
    global PATIENCE, BATCH_TEXTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=list(RUNS), default="pilot")
    ap.add_argument("--variants", nargs="+")
    ap.add_argument("--limit", type=int, help="embed only the first N rows; finalises nothing")
    ap.add_argument("--probe", type=int, default=0, help="embed N random texts per variant, write nothing")
    ap.add_argument("--workers", type=int, default=6, help="concurrent requests")
    ap.add_argument("--max-dollars", type=float, default=2.0, help="stop once this much is billed")
    ap.add_argument("--provider", default="nebius", help="the one OpenRouter provider to use, by its slug")
    ap.add_argument("--patience", type=float, default=30, help="minutes one request is retried for")
    ap.add_argument("--batch-texts", type=int, default=BATCH_TEXTS, help="most texts in one request")
    args = ap.parse_args()
    PATIENCE = args.patience * 60
    BATCH_TEXTS = args.batch_texts
    PROVIDER["order"] = [args.provider]
    cfg = RUNS[args.run]
    variants = args.variants or cfg["variants"]
    assert set(variants) <= set(cfg["variants"]), f"{args.run} has variants {cfg['variants']}"
    out_dir, piece, dims = cfg["dir"], cfg["piece"], cfg["dims"]
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    meta_path = out_dir / "api_meta.json"
    billed = 0  # tokens, this invocation
    if not args.probe:
        threading.Thread(target=heartbeat, daemon=True).start()
    for variant in variants:
        texts = variant_texts(args.run, variant)
        n = len(texts)
        chars = sum(map(len, texts))
        n_capped = sum(len(t) >= MAX_CHARS for t in texts)

        if args.probe:
            idx = np.random.default_rng(0).choice(n, size=args.probe, replace=False)
            sub = [texts[i] for i in idx]
            t0 = time.perf_counter()
            vecs, tokens, cut, providers = embed_texts(key, sub, args.workers)
            dt = time.perf_counter() - t0
            norms = np.linalg.norm(vecs, axis=1)
            ratio = sum(map(len, sub)) / max(tokens, 1)
            est = chars / ratio
            print(
                f"{variant}: {len(sub)} texts in {dt:.1f}s, dim {vecs.shape[1]}, norms {norms.min():.3f}–"
                f"{norms.max():.3f}, provider {sorted(providers) or 'not reported'}, "
                f"{tokens:,} tokens billed, {ratio:.2f} chars/token → all {n:,} texts ≈ "
                f"{est / 1e6:.1f}M tokens ≈ ${est / 1e6 * PRICE_PER_M:.2f}; "
                f"{n_capped} texts capped at {MAX_CHARS:,} chars"
            )
            continue

        final = out_dir / f"emb_{variant}.npy"
        if final.exists():
            print(f"{variant}: {final.name} already exists, skipping")
            continue
        h = hashlib.sha256()
        for t in texts:
            h.update(t.encode())
            h.update(b"\0")
        manifest = {
            "model": MODEL,
            "provider": PROVIDER,
            "max_chars": MAX_CHARS,
            "piece": piece,
            "dims": dims,
            "n_rows": n,
            "texts_sha256": h.hexdigest(),
        }
        chunk_dir = out_dir / f"emb_{variant}.chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = chunk_dir / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text())
            differs = sorted(
                k for k in manifest.keys() | existing.keys() if manifest.get(k) != existing.get(k)
            )
            if differs:
                raise SystemExit(
                    f"the pieces in {chunk_dir} were made with a different {', '.join(differs)}; "
                    "pass the same settings, or move the folder aside to start again"
                )
        else:
            write_text_safely(json.dumps(manifest, indent=2), manifest_path)

        starts = list(range(0, n, piece))
        limit = n if args.limit is None else min(args.limit, n)
        todo = [s for s in starts if s < limit and not (chunk_dir / f"{s:06d}.npy").exists()]
        todo_chars = sum(len(t) for s in todo for t in texts[s : s + piece])
        est = todo_chars / CHARS_PER_TOKEN[variant]
        print(
            f"{variant}: {n:,} texts, {chars / 1e6:.1f}M chars, {n_capped} capped at {MAX_CHARS:,} chars; "
            f"{len(todo)} of {len(starts)} pieces to embed now ≈ {est / 1e6:.1f}M tokens "
            f"≈ ${est / 1e6 * PRICE_PER_M:.2f}",
            flush=True,
        )
        t0 = time.perf_counter()
        done_texts = done_tokens = 0
        for k, s in enumerate(todo, start=1):
            for attempt in range(1, PIECE_ATTEMPTS + 1):
                try:
                    vecs, tokens, cut, providers = embed_texts(key, texts[s : s + piece], args.workers)
                    break
                except RuntimeError as e:
                    if attempt == PIECE_ATTEMPTS:
                        raise
                    print(
                        f"{variant}: the piece at row {s:,} failed ({e}); trying again in 2 minutes",
                        flush=True,
                    )
                    time.sleep(120)
            assert vecs.shape[1] >= dims, f"the model returned {vecs.shape[1]} dimensions, wanted {dims}"
            vecs = vecs[:, :dims]
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            assert (norms > 0).all(), f"zero vector in piece {s}"
            usage = {"tokens": tokens, "cut_by_provider_limit": cut, "providers": sorted(providers)}
            write_text_safely(json.dumps(usage), chunk_dir / f"{s:06d}.json")
            # The vector file marks the piece as done, so it is written last.
            write_npy_safely((vecs / norms).astype(np.float32), chunk_dir / f"{s:06d}.npy")
            billed = _stats["tokens"]  # every answered request, failed pieces included
            done_tokens += tokens
            done_texts += len(vecs)
            elapsed = time.perf_counter() - t0
            rate = done_texts / elapsed
            left = sum(min(piece, n - x) for x in todo[k:])
            refused = _stats["hits"] / max(_stats["hits"] + _stats["ok"], 1)
            print(
                f"{variant}: piece {k}/{len(todo)}  {rate:5.1f} texts/s  "
                f"{done_tokens / 1e6 / (elapsed / 60):.2f}M tokens/min  "
                f"~{left / rate / 60:5.1f} min left  ${billed / 1e6 * PRICE_PER_M:.2f} billed  "
                f"{_stats['hits']} quota refusals ({refused:.0%} of requests), "
                f"{_stats['slow']} timeouts, {_stats['other']} other failures  "
                f"[{time.strftime('%H:%M:%S')}]",
                flush=True,
            )
            if billed / 1e6 * PRICE_PER_M > args.max_dollars:
                raise SystemExit(
                    f"stopped: ${billed / 1e6 * PRICE_PER_M:.2f} billed in this run, over the "
                    f"${args.max_dollars:.2f} budget. The pieces done are kept; re-run to continue."
                )

        missing = [s for s in starts if not (chunk_dir / f"{s:06d}.npy").exists()]
        if missing:
            print(f"{variant}: {len(starts) - len(missing)} of {len(starts)} pieces exist; not finalised")
            continue
        emb = np.concatenate([np.load(chunk_dir / f"{s:06d}.npy") for s in starts])
        usages = [json.loads((chunk_dir / f"{s:06d}.json").read_text()) for s in starts]
        assert emb.shape == (n, dims) and np.isfinite(emb).all(), emb.shape
        assert np.allclose(np.linalg.norm(emb, axis=1), 1.0, atol=1e-3)
        write_npy_safely(emb, final)
        tokens = sum(u["tokens"] for u in usages)
        # Read here, not at the start: another variant's run may have finished meanwhile.
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
        meta[variant] = {
            "manifest": manifest,
            "instruction": INSTRUCTION if variant == "q8_inst" else None,
            "dim": int(emb.shape[1]),
            "tokens_billed": tokens,
            "dollars_at_list_price": round(tokens / 1e6 * PRICE_PER_M, 4),
            "texts_capped_at_max_chars": n_capped,
            "texts_cut_by_provider_limit": sum(u["cut_by_provider_limit"] for u in usages),
            "providers_reported": sorted({p for u in usages for p in u["providers"]}),
        }
        write_text_safely(json.dumps(meta, indent=2), meta_path)
        shutil.rmtree(chunk_dir)  # only after the final file is written and verified
        print(
            f"{variant}: wrote {final} {emb.shape}; {tokens / 1e6:.2f}M tokens billed "
            f"≈ ${tokens / 1e6 * PRICE_PER_M:.2f}"
        )


if __name__ == "__main__":
    main()
