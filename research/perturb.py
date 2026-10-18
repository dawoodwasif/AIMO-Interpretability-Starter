"""Generate answer-preserving variants of pool problems (the 7 official families).

Input : data/pool/problems.jsonl   rows {id, problem, answer}
Output: data/pool/variants.jsonl   rows {id, parent_id, type, problem, answer}

Rule-based (no model needed): rename, typos, distract.
LLM-based  (needs a generator): rephrase, domain, expert_perturbations.
expert_no_solution is NOT generated here: it needs real math expertise. Mine
examples from data/raw/aimo-interp-challenge-sample-full.jsonl
(`permutations_causing_decay` holds actual variant questions) instead.

Usage:
    python research/perturb.py --k 5 --types rename typos distract
    python research/perturb.py --k 5 --types rephrase domain --llm Qwen/Qwen3.5-4B
"""
from __future__ import annotations

import argparse
import random
import re

from common import POOL, read_jsonl, write_jsonl
from aimo_core.perturb import RULES, distract, rename, typos  # noqa: F401  (shared with the bundle)

LLM_PROMPTS = {
    "rephrase": "Rewrite the following math problem using different wording and sentence structure. "
                "Keep every mathematical fact, number and the question EXACTLY equivalent so the answer is unchanged. "
                "Output only the rewritten problem.\n\nProblem:\n{p}",
    "domain": "Rewrite the following math problem as a story in a different real-world setting "
              "(e.g. logistics, biology, sports), keeping the underlying mathematics, all numbers and the answer "
              "exactly the same. Output only the new problem.\n\nProblem:\n{p}",
    "expert_perturbations": "You are an olympiad problem setter. Make a subtle, answer-preserving edit to this "
                            "problem that a pattern-matching solver might misread (e.g. restate a convention, "
                            "make an implicit assumption explicit, reorder conditions). The answer must not change. "
                            "Output only the edited problem.\n\nProblem:\n{p}",
}


def clean_llm(txt: str) -> str:
    """Strip reasoning: <think> blocks (Qwen/DeepSeek) or gpt-oss harmony analysis channel."""
    txt = re.sub(r"(?s)<think>.*?</think>", "", txt)
    for marker in ("assistantfinal", "<|channel|>final<|message|>"):
        if marker in txt:
            txt = txt.rsplit(marker, 1)[1]
    txt = re.sub(r"<\|[^|]+\|>", "", txt).strip()
    txt = re.sub(r"(?is)^(\*\*)?(rewritten|new|edited)? ?problem:?(\*\*)?\s*", "", txt)
    return txt.strip()


def plausible(orig: str, var: str) -> bool:
    """Cheap answer-preservation filter: every integer of the original must survive,
    length within a sane ratio, and no leaked answer/solution text."""
    nums = set(re.findall(r"\d+", orig))
    if not nums.issubset(set(re.findall(r"\d+", var))):
        return False
    r = len(var) / max(1, len(orig))
    return 0.5 <= r <= 3.0 and "\\boxed" not in var and "answer is" not in var.lower()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(POOL / "problems.jsonl"))
    ap.add_argument("--out", default=str(POOL / "variants.jsonl"))
    ap.add_argument("--types", nargs="+", default=["rename", "typos", "distract"])
    ap.add_argument("--k", type=int, default=2, help="variants per problem per type")
    ap.add_argument("--llm", default="openai/gpt-oss-120b",
                    help="writer for LLM families (organizers used gpt-oss-120b at low effort)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--verify", action="store_true",
                    help="keep an LLM variant only if the writer solves it correctly in >=1 of 2 samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N problems only (smoke test)")
    args = ap.parse_args()

    probs = read_jsonl(args.problems)[: args.limit]
    out = read_jsonl(args.out) if args.append else []
    have = {v["id"] for v in out}
    for p in probs:
        for t in args.types:
            if t not in RULES:
                continue
            for k in range(args.k):
                rng = random.Random(f"{args.seed}:{p['id']}:{t}:{k}")
                v = RULES[t](p["problem"], rng)
                if v != p["problem"] and f"{p['id']}:{t}:{k}" not in have:
                    out.append({"id": f"{p['id']}:{t}:{k}", "parent_id": p["id"], "type": t,
                                "problem": v, "answer": p["answer"], "writer": "rule"})

    llm_types = [t for t in args.types if t in LLM_PROMPTS]
    if llm_types:
        from generate import generate, is_correct  # noqa: F401

        done_pt = {(v["parent_id"], v["type"]) for v in out}
        reqs = [(p, t) for p in probs for t in llm_types if (p["id"], t) not in done_pt]
        print(f"LLM variants: {len(reqs)} (problem, family) requests")
        prompts = [[{"role": "user", "content": LLM_PROMPTS[t].format(p=p["problem"])}] for p, t in reqs]
        gens = generate(args.llm, prompts, n=args.k + 1, temperature=0.9, max_tokens=4096, thinking=False,
                        reasoning_effort=args.effort, force_answer=False)
        cand = []
        for (p, t), texts in zip(reqs, gens):
            kept = [c for c in (clean_llm(x) for x in texts) if len(c) > 20 and plausible(p["problem"], c)]
            for k, txt in enumerate(kept[: args.k]):
                cand.append({"id": f"{p['id']}:{t}:{k}", "parent_id": p["id"], "type": t, "problem": txt,
                             "answer": p["answer"], "writer": f"{args.llm}:{args.effort}"})
        if args.verify and cand:
            sol = generate(args.llm, [[{"role": "user", "content": "Solve the following math problem. Put the final "
                                        "answer in \\boxed{}.\n\n" + c["problem"]}] for c in cand],
                           n=2, max_tokens=16384, reasoning_effort="medium")
            ok = [any(is_correct(x, c["answer"]) for x in s_) for c, s_ in zip(cand, sol)]
            print(f"verification kept {sum(ok)}/{len(cand)} LLM variants")
            cand = [c for c, o in zip(cand, ok) if o]
        out += [c for c in cand if c["id"] not in have]
    write_jsonl(args.out, out)
    from collections import Counter

    print(f"wrote {len(out)} variants -> {args.out}: {dict(Counter(v['type'] for v in out))}")


if __name__ == "__main__":
    main()
