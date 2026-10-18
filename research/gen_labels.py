"""Self-generated robustness labels for one competition model (GPU job, vLLM).

Stage A: sample the model --n-base times on each pool problem   -> base accuracy
Stage B: for problems with base >= --min-base, sample --n-var times on every
         variant (data/pool/variants.jsonl) of every family    -> family accuracy
Label   : one row per (model, problem, effort) - exactly what the harness scores:
          decay_f = base - acc_f; robust if max_f decay <= --robust-max,
          spurious if >= --spurious-min, else ambiguous (dropped; official labels
          keep a wide margin); conjunction over families.

Raw per-sample results are checkpointed after every chunk to
data/selfgen/<model>[_<effort>].jsonl, so a crashed job resumes (--resume) and labels
can be recomputed with another rule without regenerating (--relabel-only).

Usage (GPU box):
    python research/gen_labels.py --model Qwen/Qwen3.5-4B --limit 10            # smoke test, ~5-10 min
    python research/gen_labels.py --model Qwen/Qwen3.5-4B --resume              # full pool
    python research/gen_labels.py --model openai/gpt-oss-120b --effort low --resume
    python research/gen_labels.py --model ... --relabel-only --robust-max 0.1 --spurious-min 0.3
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

from common import DATA, LABELED, POOL, read_jsonl, slug, text_hash, write_jsonl
from generate import extract_answer, generate, is_correct

SOLVE_PROMPT = ("Solve the following math problem. Reason step by step and put the final answer in "
                "\\boxed{{}}.\n\n{p}")


def _raw_path(args):
    return DATA / "selfgen" / f"{slug(args.model)}{'_' + args.effort if args.effort else ''}.jsonl"


def sample(args, items: list[dict], kind: str, n: int, done: set) -> list[dict]:
    items = [it for it in items if (kind, it["id"]) not in done]
    rows = []
    for c in range(0, len(items), args.chunk):
        chunk = items[c:c + args.chunk]
        t0 = time.time()
        prompts = [[{"role": "user", "content": SOLVE_PROMPT.format(p=it["problem"])}] for it in chunk]
        outs = generate(args.model, prompts, n=n, max_tokens=args.max_tokens, thinking=True,
                        reasoning_effort=args.effort, detail=True)
        new = []
        for it, resp in zip(chunk, outs):
            new.append({"kind": kind, "id": it["id"], "parent_id": it.get("parent_id", it["id"]),
                        "type": it.get("type", "original"), "answer": it.get("answer"),
                        "preds": [extract_answer(d["text"]) for d in resp],
                        "correct": [is_correct(d["text"], it.get("answer")) for d in resp],
                        "n_tokens": [d["n_tokens"] for d in resp], "truncated": [d["truncated"] for d in resp],
                        "forced": [d["forced"] for d in resp]})
        with open(_raw_path(args), "a", encoding="utf-8") as f:
            for r in new:
                f.write(json.dumps(r) + "\n")
        rows += new
        toks = sum(sum(r["n_tokens"]) for r in new)
        print(f"  {kind} {c + len(chunk)}/{len(items)}: {toks / max(1e-9, time.time() - t0):.0f} tok/s, "
              f"trunc={sum(sum(r['truncated']) for r in new) / max(1, sum(len(r['truncated']) for r in new)):.2f}",
              flush=True)
    return rows


def label(args, raw: list[dict], problems: dict) -> list[dict]:
    base = {}
    for r in raw:
        if r["kind"] == "original":
            base.setdefault(r["id"], []).extend(r["correct"])
    fam = defaultdict(lambda: defaultdict(list))
    for r in raw:
        if r["kind"] == "variant":
            fam[r["parent_id"]][r["type"]].extend(r["correct"])
    out = []
    for pid, cs in base.items():
        b = sum(cs) / len(cs)
        accs = {t: sum(v) / len(v) for t, v in fam.get(pid, {}).items() if v}
        lab, max_drop = None, None
        if b >= args.min_base and len(accs) >= args.min_families:
            max_drop = max(b - a for a in accs.values())
            lab = True if max_drop <= args.robust_max else False if max_drop >= args.spurious_min else None
        p = problems.get(pid, {})
        out.append({
            "case_id": f"self:{slug(args.model)}:{args.effort or 'default'}:{pid}",
            "problem_id": text_hash(p.get("problem", pid)), "model_id": args.model,
            "reasoning_effort": args.effort or "default", "original_problem": p.get("problem", ""),
            "label": lab, "source": "self", "origin": p.get("origin"), "leak": False,
            "base_accuracy": b, "family_acc": accs, "max_drop": max_drop, "n_base": len(cs),
            "rule": {"min_base": args.min_base, "robust_max": args.robust_max, "spurious_min": args.spurious_min,
                     "max_tokens": args.max_tokens},
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--effort", default=None, help="gpt-oss reasoning effort: low|medium|high")
    ap.add_argument("--problems", default=str(POOL / "problems.jsonl"))
    ap.add_argument("--variants", default=str(POOL / "variants.jsonl"))
    ap.add_argument("--n-base", type=int, default=8)
    ap.add_argument("--n-var", type=int, default=4, help="samples per variant (2 variants/family -> 8/family)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--chunk", type=int, default=64, help="prompts per vLLM call (checkpoint granularity)")
    ap.add_argument("--limit", type=int, default=None, help="max original problems (smoke test)")
    ap.add_argument("--min-base", type=float, default=0.6)
    ap.add_argument("--min-families", type=int, default=3)
    ap.add_argument("--robust-max", type=float, default=0.125)
    ap.add_argument("--spurious-min", type=float, default=0.3)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--relabel-only", action="store_true")
    args = ap.parse_args()

    path = _raw_path(args)
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = read_jsonl(args.problems)[: args.limit]
    pmap = {p["id"]: p for p in probs}
    if not args.relabel_only:
        if path.exists() and not args.resume:
            raise SystemExit(f"{path} exists: pass --resume to continue or --relabel-only")
        raw = read_jsonl(path) if path.exists() else []
        done = {(r["kind"], r["id"]) for r in raw}
        t0 = time.time()
        raw += sample(args, probs, "original", args.n_base, done)
        base = defaultdict(list)
        for r in raw:
            if r["kind"] == "original":
                base[r["id"]] += r["correct"]
        solved = {k for k, v in base.items() if sum(v) / len(v) >= args.min_base and k in pmap}
        vars_ = [v for v in read_jsonl(args.variants) if v["parent_id"] in solved]
        print(f"stage A done in {time.time() - t0:.0f}s: {len(solved)}/{len(probs)} solved (base>={args.min_base}); "
              f"stage B: {len(vars_)} variants x {args.n_var}", flush=True)
        raw += sample(args, vars_, "variant", args.n_var, done)
        print(f"total {time.time() - t0:.0f}s", flush=True)
    raw = read_jsonl(path)
    rows = label(args, raw, pmap)
    out = LABELED / f"self_{path.stem}.jsonl"
    write_jsonl(out, rows)
    lab = [r for r in rows if r["label"] is not None]
    print(f"{len(rows)} rows, {len(lab)} labeled ({sum(r['label'] for r in lab)} robust, "
          f"{sum(not r['label'] for r in lab)} spurious) -> {out}")


if __name__ == "__main__":
    main()
