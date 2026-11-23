"""Stage 7: embed the statements text of every file.

Reads   data/full/statements.parquet     row order = embedding order (stage 6)
Writes  data/full/emb_statements.npy     float32, L2-normalised, row order = statements.parquet
        data/full/tokens_statements.npy  untruncated token count per file
        data/full/embed_meta.json        model, token statistics, truncation

Resumable. The rows are cut into EMBED_CHUNK-row pieces saved under
data/full/emb_statements.chunks/; a re-run skips the pieces that exist. A manifest in
that directory pins the model, the token limit and a hash of the ordered texts, and a
re-run refuses to continue if any of them changed, so pieces from two recipes can never
be mixed. Nothing is deleted on a mismatch: the pieces cost hours.

    uv run python -I scripts/07_embed_full.py --limit 40000   # the dense subset only
    uv run python -I scripts/07_embed_full.py                 # every file, then finalise

Measured on the pilot (M3, MPS, fp32, batch 16): about 6 files a second, so roughly
8 hours for all 175,013 files. Same model and token limit as the pilot.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from riemann_map import EMBED_CHUNK, FULL, write_npy_safely, write_text_safely

MODEL = "Alibaba-NLP/gte-modernbert-base"
MAX_TOKENS = 2048
BATCH = 16


def token_counts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    enc = model.tokenizer(texts, add_special_tokens=True, truncation=False)
    return np.array([len(x) for x in enc["input_ids"]], dtype=np.int32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="embed only the first N rows")
    args = ap.parse_args()

    final = FULL / "emb_statements.npy"
    chunk_dir = FULL / "emb_statements.chunks"
    table = pd.read_parquet(FULL / "statements.parquet", columns=["module", "text_statements"])
    texts = table["text_statements"].tolist()
    n = len(texts)
    limit = n if args.limit is None else min(args.limit, n)

    h = hashlib.sha256()
    for module, text in zip(table["module"], texts):
        h.update(module.encode())
        h.update(b"\0")
        h.update(text.encode())
        h.update(b"\0")
    manifest = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "chunk": EMBED_CHUNK,
        "n_rows": n,
        "texts_sha256": h.hexdigest(),
    }

    meta_path = FULL / "embed_meta.json"
    if final.exists():
        done = json.loads(meta_path.read_text())["manifest"]
        if done != manifest:
            raise SystemExit(
                f"{final} was built from different texts or settings. Move it aside to re-embed."
            )
        print(f"{final.name} already holds all {n:,} rows; nothing to do")
        return

    chunk_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = chunk_dir / "manifest.json"
    if manifest_path.exists():
        found = json.loads(manifest_path.read_text())
        if found != manifest:
            diff = {k: (found.get(k), v) for k, v in manifest.items() if found.get(k) != v}
            raise SystemExit(
                f"the pieces in {chunk_dir} were made with different inputs: {diff}. "
                "Move that directory aside to start again; nothing was deleted."
            )
    else:
        write_text_safely(json.dumps(manifest, indent=2), manifest_path)

    starts = list(range(0, n, EMBED_CHUNK))
    in_scope = [s for s in starts if s < limit]
    todo = [s for s in in_scope if not (chunk_dir / f"{s:06d}.npy").exists()]
    print(
        f"{n:,} rows in {len(starts)} pieces; scope is the first {limit:,} rows ({len(in_scope)} pieces): "
        f"{len(in_scope) - len(todo)} already done, {len(todo)} to go",
        flush=True,
    )

    if todo:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(MODEL, device=device, local_files_only=True)
        model.max_seq_length = MAX_TOKENS
        dim = model.get_embedding_dimension()
        print(f"model {MODEL} on {device}, dim {dim}, max_seq_length {MAX_TOKENS}", flush=True)
        t0 = time.perf_counter()
        done_rows = 0
        for k, s in enumerate(todo, start=1):
            piece = texts[s : s + EMBED_CHUNK]
            n_tok = token_counts(model, piece)
            emb = model.encode(piece, batch_size=BATCH, normalize_embeddings=True, show_progress_bar=False)
            emb = emb.astype(np.float32)
            assert emb.shape == (len(piece), dim), emb.shape
            assert np.isfinite(emb).all(), f"non-finite embedding in piece {s}"
            # The embedding file is what marks a piece as done, so it is written last.
            write_npy_safely(n_tok, chunk_dir / f"{s:06d}.tok.npy")
            write_npy_safely(emb, chunk_dir / f"{s:06d}.npy")
            if device == "mps":
                # A precaution for a run of several hours, not a fix for an observed leak.
                torch.mps.empty_cache()
            done_rows += len(piece)
            rate = done_rows / (time.perf_counter() - t0)
            left = sum(min(EMBED_CHUNK, n - x) for x in todo[k:])
            print(
                f"piece {k}/{len(todo)} (rows {s:,}–{s + len(piece) - 1:,})  {rate:5.2f} files/s  "
                f"~{left / rate / 3600:4.2f} h left in scope  [{time.strftime('%H:%M:%S')}]",
                flush=True,
            )

    missing = [s for s in starts if not (chunk_dir / f"{s:06d}.npy").exists()]
    if missing:
        print(
            f"{len(starts) - len(missing)} of {len(starts)} pieces exist; "
            "run without --limit to embed the rest and finalise"
        )
        return

    emb = np.concatenate([np.load(chunk_dir / f"{s:06d}.npy") for s in starts])
    n_tok = np.concatenate([np.load(chunk_dir / f"{s:06d}.tok.npy") for s in starts])
    assert emb.shape[0] == n and n_tok.shape == (n,), (emb.shape, n_tok.shape, n)
    assert np.isfinite(emb).all(), "non-finite values in the embeddings"
    norms = np.linalg.norm(emb, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (norms.min(), norms.max())
    write_npy_safely(emb, final)
    write_npy_safely(n_tok, FULL / "tokens_statements.npy")
    meta = {
        "manifest": manifest,
        "tokens_median": float(np.median(n_tok)),
        "tokens_p90": float(np.quantile(n_tok, 0.9)),
        "tokens_total": int(n_tok.sum()),
        "share_truncated": float((n_tok > MAX_TOKENS).mean()),
    }
    write_text_safely(json.dumps(meta, indent=2), meta_path)
    shutil.rmtree(chunk_dir)  # only after the final files are written and verified
    print(
        f"wrote {final} {emb.shape}; tokens median {meta['tokens_median']:,.0f}, "
        f"p90 {meta['tokens_p90']:,.0f}, {meta['share_truncated']:.1%} of files truncated at {MAX_TOKENS}"
    )


if __name__ == "__main__":
    main()
