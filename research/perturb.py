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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default=str(POOL / "problems.jsonl"))
    ap.add_argument("--out", default=str(POOL / "variants.jsonl"))
    ap.add_argument("--types", nargs="+", default=["rename", "typos", "distract"])
    ap.add_argument("--k", type=int, default=5, help="variants per problem per type")
    ap.add_argument("--llm", default=None, help="HF model id used to write LLM-based variants")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    probs = read_jsonl(args.problems)
    out = read_jsonl(args.out) if args.append else []
    for p in probs:
        for t in args.types:
            if t not in RULES:
                continue
            for k in range(args.k):
                rng = random.Random(f"{args.seed}:{p['id']}:{t}:{k}")
                out.append({"id": f"{p['id']}:{t}:{k}", "parent_id": p["id"], "type": t,
                            "problem": RULES[t](p["problem"], rng), "answer": p["answer"]})

    llm_types = [t for t in args.types if t in LLM_PROMPTS]
    if llm_types:
        if not args.llm:
            raise SystemExit(f"--llm is required for {llm_types}")
        from generate import generate

        reqs = [(p, t) for p in probs for t in llm_types]
        prompts = [[{"role": "user", "content": LLM_PROMPTS[t].format(p=p["problem"])}] for p, t in reqs]
        gens = generate(args.llm, prompts, n=args.k, temperature=0.9, max_tokens=1024, thinking=False)
        for (p, t), texts in zip(reqs, gens):
            for k, txt in enumerate(texts):
                txt = re.sub(r"(?s)<think>.*?</think>", "", txt).strip()
                if len(txt) > 20:
                    out.append({"id": f"{p['id']}:{t}:{k}", "parent_id": p["id"], "type": t,
                                "problem": txt, "answer": p["answer"]})
    write_jsonl(args.out, out)
    print(f"wrote {len(out)} variants -> {args.out}")


if __name__ == "__main__":
    main()
