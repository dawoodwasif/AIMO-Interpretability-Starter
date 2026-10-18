"""Cache internal features for every unique problem text (labeled rows + pool).

Uses solution/aimo_core/internals.py -> exactly what the bundle computes
(same prompt, same perturbations, same statistics).
Output: data/cache/features/<model_slug>[_<effort>].npz with
    keys            (n,) text hashes (common.text_hash)
    <feature_set>   (n, d) fp16 for internal:{last,mean}:L*, internal:logits,
                    internal:nll, internal:shift
    _meta_*         prompt / layers / timing
Layers: ~12 evenly spaced (incl. the shift depths) to keep files < 100 MB.

Usage (GPU box; model must be in the HF cache: huggingface-cli download <id>):
    python research/extract_features.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import CACHE, POOL, load_labeled, read_jsonl, slug, text_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None, help="gpt-oss: reasoning effort passed to the chat template")
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--no-pool", action="store_true", help="only labeled texts")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from aimo_core import config, internals

    texts = {text_hash(r["original_problem"]): r["original_problem"] for r in load_labeled()}
    if not args.no_pool and (POOL / "problems.jsonl").exists():
        texts.update({text_hash(p["problem"]): p["problem"] for p in read_jsonl(POOL / "problems.jsonl")})
    keys = sorted(texts)
    print(f"{len(keys)} unique problem texts for {args.model}")

    t0 = time.monotonic()
    tok, model = internals.load_model(args.model)
    load_s = time.monotonic() - t0
    L = internals.n_layers(model)
    layers = sorted({int(round(x)) for x in np.linspace(1, L, args.n_layers)} | set(internals.shift_layers(model)))
    t0 = time.monotonic()
    feats = internals.extract(tok, model, [texts[k] for k in keys], layers=layers, batch_size=args.batch, shift=True,
                              effort=args.effort)
    dt = time.monotonic() - t0
    per = dt / max(1, len(keys))
    print(f"load {load_s:.1f}s; extracted {len(feats)} feature sets in {dt:.1f}s ({per:.3f}s/problem incl. "
          f"{len(internals.SHIFT_KINDS)} perturbed copies) -> PER_ITEM_COST_S hint (GPU differs from Codabench)")
    out = CACHE / "features" / f"{slug(args.model)}{'_' + args.effort if args.effort else ''}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, keys=np.array(keys), **{k: v.astype(np.float16) for k, v in feats.items()},
                        _meta_system_prompt=np.array(config.SYSTEM_PROMPT), _meta_layers=np.array(layers),
                        _meta_n_layers=np.array(L), _meta_s_per_problem=np.array(per), _meta_load_s=np.array(load_s))
    print(f"saved {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
