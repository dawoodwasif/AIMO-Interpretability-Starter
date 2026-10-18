"""Cache internal features (ALL layers) for every unique problem text of a model.

Uses solution/aimo_core/internals.py -> exactly what the bundle computes.
Output: data/cache/features/<model_slug>.npz with
    keys            (n,) text hashes
    <feature_set>   (n, d) for internal:last:L*, internal:mean:L*, internal:logits

Usage (GPU box, model in HF cache or downloadable):
    python research/extract_features.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from common import CACHE, load_labeled, slug, text_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--all-texts", action="store_true",
                    help="extract for every labeled problem text, not only this model's rows")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    import os
    os.environ.setdefault("HF_HUB_OFFLINE", "0")
    from aimo_core import config, internals

    rows = load_labeled()
    if not args.all_texts:
        rows = [r for r in rows if r["model_id"] == args.model] or rows
    texts = {text_hash(r["original_problem"]): r["original_problem"] for r in rows}
    keys = sorted(texts)
    print(f"{len(keys)} unique problem texts for {args.model}")

    # local_files_only=True inside load_model mirrors Codabench; pre-download first:
    #   huggingface-cli download <model>
    tok, model = internals.load_model(args.model)
    t0 = time.monotonic()
    feats = internals.extract(tok, model, [texts[k] for k in keys], layers=None, batch_size=args.batch)
    dt = time.monotonic() - t0
    print(f"extracted {len(feats)} feature sets in {dt:.1f}s ({dt / max(1, len(keys)):.3f}s/problem)")
    print(f"-> update PER_ITEM_COST_S[{args.model!r}] in solution/aimo_core/config.py (GPU differs from Codabench's RTX PRO 6000)")
    out = CACHE / "features" / f"{slug(args.model)}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(keys), **{k: v.astype(np.float16) for k, v in feats.items()},
                        _meta_system_prompt=np.array(config.SYSTEM_PROMPT))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
