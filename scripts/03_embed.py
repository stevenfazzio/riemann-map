"""Stage 3 (pilot): embed each text variant with one local model.

Reads   data/pilot/sample.parquet
Writes  data/pilot/emb_{variant}.npy   float32, L2-normalised, row order = sample order
        data/pilot/embed_meta.json     model, token counts, truncation, throughput

Resumable: each variant is embedded in chunks saved under emb_{variant}.chunks/, and a
re-run skips chunks that already exist. A finished variant leaves no chunks behind.

A fourth variant, "expanded", is the full run's statements recipe (stage 6: bare-name
theorems carry the definition of the proposition they name) on the same sample. It is
not embedded by default. Only the files whose text changed are embedded; the rest reuse
their rows of emb_statements.npy, so the two variants differ only where the text does.

    uv run python -I scripts/03_embed.py --time-only   # throughput on 48 texts per variant
    uv run python -I scripts/03_embed.py
    uv run python -I scripts/03_embed.py --variants expanded   # after stage 6
"""

from __future__ import annotations

import argparse
import json
import shutil
import time

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from riemann_map import FULL, PILOT, write_npy_safely, write_text_safely

MODEL = "Alibaba-NLP/gte-modernbert-base"  # 8192-token context, no remote code, no prompt
MAX_TOKENS = 2048
VARIANTS = {"names": 32, "statements": 16, "file": 8}  # variant -> batch size
EXPANDED = "expanded"
CHUNK = 500


def token_counts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    tok = model.tokenizer
    out = []
    for i in range(0, len(texts), 256):
        enc = tok(texts[i : i + 256], add_special_tokens=True, truncation=False)
        out.extend(len(x) for x in enc["input_ids"])
    return np.array(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=[*VARIANTS, EXPANDED])
    ap.add_argument(
        "--time-only", action="store_true", help="time 48 random texts per variant, write nothing"
    )
    args = ap.parse_args()

    text_cols = {f"text_{'statements' if v == EXPANDED else v}" for v in args.variants}
    sample = pd.read_parquet(PILOT / "sample.parquet", columns=["module", *sorted(text_cols)])
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer(MODEL, device=device, local_files_only=True)
    model.max_seq_length = MAX_TOKENS
    print(f"model {MODEL} on {device}, dim {model.get_embedding_dimension()}, max_seq_length {MAX_TOKENS}")

    meta_path = PILOT / "embed_meta.json"
    meta = (
        json.loads(meta_path.read_text())
        if meta_path.exists()
        else {"model": MODEL, "max_tokens": MAX_TOKENS, "variants": {}}
    )

    for variant in args.variants:
        base = None
        if variant == EXPANDED:
            full = pd.read_parquet(FULL / "statements.parquet", columns=["module", "text_statements"])
            new_text = sample[["module"]].merge(full, on="module", how="left", validate="one_to_one")
            assert new_text["text_statements"].notna().all(), "a pilot file is missing from stage 6"
            changed = (new_text["text_statements"] != sample["text_statements"]).to_numpy()
            rows = np.flatnonzero(changed)
            texts = new_text["text_statements"].to_numpy()[rows].tolist()
            base = np.load(PILOT / "emb_statements.npy")
            assert len(base) == len(sample)
            print(f"{variant}: {len(rows):,} of {len(sample):,} texts changed; the rest reuse emb_statements")
            bs = VARIANTS["statements"]
        else:
            texts = sample[f"text_{variant}"].tolist()
            bs = VARIANTS[variant]
        if args.time_only:
            idx = np.random.default_rng(0).choice(len(texts), size=48, replace=False)
            sub = [texts[i] for i in idx]
            t0 = time.perf_counter()
            model.encode(sub, batch_size=bs, normalize_embeddings=True, show_progress_bar=False)
            dt = time.perf_counter() - t0
            print(
                f"{variant:11s} 48 texts in {dt:5.1f}s → {48 / dt:5.1f} texts/s → "
                f"{len(texts) / (48 / dt) / 60:5.1f} min for {len(texts):,}"
            )
            continue

        final = PILOT / f"emb_{variant}.npy"
        if final.exists():
            print(f"{variant}: {final.name} already exists, skipping")
            continue
        n_tok = token_counts(model, texts)
        truncated = float((n_tok > MAX_TOKENS).mean())
        print(
            f"{variant}: tokens median {np.median(n_tok):,.0f}, p90 {np.quantile(n_tok, 0.9):,.0f}, "
            f"max {n_tok.max():,}; {truncated:.1%} of texts exceed {MAX_TOKENS} and are truncated"
        )

        chunk_dir = PILOT / f"emb_{variant}.chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        starts = list(range(0, len(texts), CHUNK))
        todo = [s for s in starts if not (chunk_dir / f"{s:06d}.npy").exists()]
        print(f"{variant}: {len(starts) - len(todo)} chunks already done, {len(todo)} to go")
        t0 = time.perf_counter()
        done_texts = 0
        for n, s in enumerate(todo, start=1):
            emb = model.encode(
                texts[s : s + CHUNK], batch_size=bs, normalize_embeddings=True, show_progress_bar=False
            )
            write_npy_safely(emb.astype(np.float32), chunk_dir / f"{s:06d}.npy")
            done_texts += len(emb)
            rate = done_texts / (time.perf_counter() - t0)
            left = sum(min(CHUNK, len(texts) - x) for x in todo[n:])
            print(
                f"{variant}: chunk {n}/{len(todo)}  {rate:5.1f} texts/s  ~{left / rate / 60:4.1f} min left",
                flush=True,
            )
        elapsed = time.perf_counter() - t0

        emb = np.concatenate([np.load(chunk_dir / f"{s:06d}.npy") for s in starts])
        assert emb.shape == (len(texts), model.get_embedding_dimension()), emb.shape
        if base is not None:
            base[rows] = emb
            emb = base
        assert np.isfinite(emb).all(), "non-finite values in the embeddings"
        norms = np.linalg.norm(emb, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3), (norms.min(), norms.max())
        write_npy_safely(emb, final)
        shutil.rmtree(chunk_dir)
        meta["variants"][variant] = {
            "tokens_median": float(np.median(n_tok)),
            "tokens_p90": float(np.quantile(n_tok, 0.9)),
            "tokens_total": int(n_tok.sum()),
            "texts_embedded": len(texts),  # fewer than the sample when rows are reused
            "share_truncated": truncated,
            "texts_per_second": done_texts / elapsed if done_texts else None,
        }
        write_text_safely(json.dumps(meta, indent=2), meta_path)
        print(f"{variant}: wrote {final.name} {emb.shape}")


if __name__ == "__main__":
    main()
