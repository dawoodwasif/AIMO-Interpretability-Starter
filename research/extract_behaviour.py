"""Cache test-time self-consistency features (H9) with the SHIPPED code (GPU).

Runs aimo_core.behaviour.run - greedy decoding of each problem plus its three
rule-based perturbed copies, forced answers on truncation - for every labeled
problem of a model, at a fixed token cap that mimics the eval budget.
Output: data/cache/features/<model_slug>[_<effort>]_behav.npz
    keys (n,) text hashes, behav (n, 4) [agree, parsed, len, trunc], _meta_max_tokens

train.py picks this up as feature set "behav" (and "behav+internal:nll").

Usage (GPU box):
    python research/extract_behaviour.py --model Qwen/Qwen3.5-4B --max-tokens 4096 --chunk-problems 16
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np

from common import CACHE, load_labeled, slug, text_hash


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--chunk-problems", type=int, default=16, help="problems per behaviour.run call")
    args = ap.parse_args()

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from aimo_core import behaviour, config, internals

    config.DECODE_TOK_S["_default"] = config.DECODE_TOK_S[args.model] = 1e9  # cap by tokens, not time
    rows = [r for r in load_labeled() if r["model_id"] == args.model
            and (r.get("reasoning_effort") or "default") == (args.effort or "default")]
    texts = {text_hash(r["original_problem"]): r["original_problem"] for r in rows}
    keys = sorted(texts)
    out = CACHE / "features" / f"{slug(args.model)}{'_' + args.effort if args.effort else ''}_behav.npz"
    done = {}
    if out.exists():
        z = np.load(out)
        done = {str(k): v for k, v in zip(z["keys"], z["behav"])}
    todo = [k for k in keys if k not in done]
    print(f"{len(keys)} labeled problems for {args.model}, {len(todo)} to do")
    tok, model = internals.load_model(args.model)
    t0 = time.monotonic()
    for s in range(0, len(todo), args.chunk_problems):
        ks = todo[s:s + args.chunk_problems]
        f = behaviour.run(tok, model, [texts[k] for k in ks], time.monotonic() + 1e6, min_tokens=1,
                          max_tokens=args.max_tokens, model_id=args.model, effort=args.effort)
        done.update(dict(zip(ks, f)))
        out.parent.mkdir(parents=True, exist_ok=True)
        kk = sorted(done)
        np.savez_compressed(out, keys=np.array(kk), behav=np.stack([done[k] for k in kk]).astype(np.float32),
                            _meta_max_tokens=np.array(args.max_tokens))
        el = time.monotonic() - t0
        print(f"  {s + len(ks)}/{len(todo)} problems, {el:.0f}s ({el / (s + len(ks)):.1f}s/problem)", flush=True)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
