"""Self-generated robustness labels for one cached competition model (GPU job).

  1. sample the model n times on each original problem  -> base_accuracy
  2. sample it on each variant (by perturbation family) -> permuted accuracy
  3. label (problem, model, perm types) robust / spurious using a margin rule
Raw generations are saved first, so labels can be recomputed with a different
rule WITHOUT regenerating (--relabel-only).

Label rule (mirrors what is known about the official labels; see CLAUDE.md):
  * keep only originals the model solves: base_accuracy >= --min-base
  * decay_t = max(0, base - acc_t) per family t
  * family robust   if decay_t <= --robust-max
    family spurious if decay_t >= --spurious-min
    otherwise ambiguous -> dropped (official sets keep a wide margin)
  * a multi-family row is spurious if ANY family is spurious (conjunction),
    robust if ALL are robust, else dropped.

Usage (GPU box):
    python research/gen_labels.py --model deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --n 8 --max-tokens 8192
    python research/gen_labels.py --model ... --relabel-only --robust-max 0.1 --spurious-min 0.4
"""
from __future__ import annotations

import argparse
import itertools
from collections import defaultdict

from common import DATA, LABELED, POOL, read_jsonl, slug, write_jsonl
from generate import generate, is_correct

SOLVE_PROMPT = ("Solve the following math problem. Reason step by step and put the final answer in "
                "\\boxed{{}}.\n\n{p}")


def sample(args, items: list[dict], tag: str) -> list[dict]:
    prompts = [[{"role": "user", "content": SOLVE_PROMPT.format(p=it["problem"])}] for it in items]
    outs = generate(args.model, prompts, n=args.n, max_tokens=args.max_tokens,
                    thinking=True, reasoning_effort=args.reasoning_effort)
    rows = []
    for it, resp in zip(items, outs):
        corr = [is_correct(r, it.get("answer")) for r in resp]
        rows.append({**it, "kind": tag, "n": len(corr), "acc": sum(corr) / max(1, len(corr)),
                     "responses_tail": [r[-400:] for r in resp]})
    return rows


def label(args, raw: list[dict]) -> list[dict]:
    base = {r["id"]: r for r in raw if r["kind"] == "original"}
    fam = defaultdict(lambda: defaultdict(list))  # parent -> type -> [acc...]
    for r in raw:
        if r["kind"] == "variant" and r["parent_id"] in base:
            fam[r["parent_id"]][r["type"]].append(r["acc"])

    out = []
    for pid, types in fam.items():
        b = base[pid]
        if b["acc"] < args.min_base:
            continue
        verdict = {}
        for t, accs in types.items():
            decay = max(0.0, b["acc"] - sum(accs) / len(accs))
            verdict[t] = (True if decay <= args.robust_max else False if decay >= args.spurious_min else None, decay)
        combos = [[t] for t in verdict]
        if len(verdict) > 1:
            combos.append(sorted(verdict))
        for combo in combos:
            vs = [verdict[t][0] for t in combo]
            lab = False if any(v is False for v in vs) else True if all(v is True for v in vs) else None
            if lab is None:
                continue
            out.append({
                "case_id": f"self:{slug(args.model)}:{pid}:{'+'.join(combo)}",
                "problem_id": pid, "model_id": args.model, "original_problem": b["problem"],
                "permutation_type": combo, "label": lab, "source": "self",
                "base_accuracy": b["acc"], "absolute_accuracy_decay": max(verdict[t][1] for t in combo),
            })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--problems", default=str(POOL / "problems.jsonl"))
    ap.add_argument("--variants", default=str(POOL / "variants.jsonl"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--reasoning-effort", default=None, help="gpt-oss: low|medium|high")
    ap.add_argument("--limit", type=int, default=None, help="max original problems (for smoke tests)")
    ap.add_argument("--min-base", type=float, default=0.6)
    ap.add_argument("--robust-max", type=float, default=0.1)
    ap.add_argument("--spurious-min", type=float, default=0.4)
    ap.add_argument("--relabel-only", action="store_true")
    args = ap.parse_args()

    raw_path = DATA / "selfgen" / f"{slug(args.model)}{'_' + args.reasoning_effort if args.reasoning_effort else ''}.jsonl"
    if not args.relabel_only:
        probs = read_jsonl(args.problems)[: args.limit]
        raw = sample(args, probs, "original")
        write_jsonl(raw_path, raw)  # checkpoint
        solved = {r["id"] for r in raw if r["acc"] >= args.min_base}
        vars_ = [v for v in read_jsonl(args.variants) if v["parent_id"] in solved]
        print(f"{len(solved)}/{len(probs)} originals solved; sampling {len(vars_)} variants")
        raw += sample(args, vars_, "variant")
        write_jsonl(raw_path, raw)
    raw = read_jsonl(raw_path)
    rows = label(args, raw)
    write_jsonl(LABELED / f"self_{raw_path.stem}.jsonl", rows)
    pos = sum(r["label"] for r in rows)
    print(f"{len(rows)} labeled rows ({pos} robust, {len(rows) - pos} spurious) -> data/labeled/self_{raw_path.stem}.jsonl")
    for t, g in itertools.groupby(sorted(rows, key=lambda r: "+".join(r["permutation_type"])),
                                  key=lambda r: "+".join(r["permutation_type"])):
        g = list(g)
        print(f"  {t:60s} n={len(g):4d} robust={sum(r['label'] for r in g) / len(g):.2f}")


if __name__ == "__main__":
    main()
