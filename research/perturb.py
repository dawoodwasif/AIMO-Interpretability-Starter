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

NAMES = ["Alice", "Bob", "Carol", "Dave", "Eve", "Frank", "Grace", "Heidi", "Ivan",
         "Judy", "Mallory", "Oscar", "Peggy", "Trent", "Victor", "Walter", "Carla", "Daniel"]
NEW_NAMES = ["Priya", "Tomasz", "Aiko", "Kwame", "Sofia", "Mateo", "Leila", "Ravi",
             "Ingrid", "Chen", "Amara", "Diego", "Yuki", "Omar", "Nadia", "Lars"]
DISTRACTORS = [
    "Note that the problem was first posed at a regional contest in {y}.",
    "A student initially guessed the answer was {r}, but this guess was not checked.",
    "Assume all computations are done carefully; a calculator shows {r}.{d} for an unrelated quantity.",
    "(For context, {r} other problems in the same set were about geometry.)",
    "The number {r} also appears on the cover of the booklet but plays no role here.",
]
_MATH = re.compile(r"(\$\$.*?\$\$|\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])", re.S)
_VAR = re.compile(r"(?<![\\a-zA-Z])([a-zA-Z])(?![a-zA-Z])")


def _split_math(text: str):
    parts, last = [], 0
    for m in _MATH.finditer(text):
        parts.append((False, text[last:m.start()]))
        parts.append((True, m.group(0)))
        last = m.end()
    parts.append((False, text[last:]))
    return parts


def rename(text: str, rng: random.Random) -> str:
    used = set(_VAR.findall("".join(s for is_m, s in _split_math(text) if is_m)))
    lower = [c for c in "abcdefghijklmnopqrstuvwxyz" if c not in used and c not in "ei"]
    upper = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c not in used]
    rng.shuffle(lower)
    rng.shuffle(upper)
    mapping = {}
    for v in sorted(used):
        pool = upper if v.isupper() else lower
        if pool:
            mapping[v] = pool.pop()
    new_names = NEW_NAMES[:]
    rng.shuffle(new_names)
    name_map = {n: new_names[i % len(new_names)] for i, n in enumerate(NAMES)}
    out = []
    for is_m, s in _split_math(text):
        if is_m:
            s = _VAR.sub(lambda m: mapping.get(m.group(1), m.group(1)), s)
            # also rename plain-text single-letter references like "point A"
        else:
            for a, b in name_map.items():
                s = re.sub(rf"\b{a}\b", b, s)
            s = re.sub(r"(?<![A-Za-z])([A-Z])(?![A-Za-z])", lambda m: mapping.get(m.group(1), m.group(1)), s)
        out.append(s)
    return "".join(out)


def typos(text: str, rng: random.Random, rate: float = 0.08) -> str:
    out = []
    for is_m, s in _split_math(text):
        if is_m:
            out.append(s)
            continue
        words = s.split(" ")
        for i, w in enumerate(words):
            if len(w) > 3 and w.isalpha() and rng.random() < rate:
                j = rng.randrange(1, len(w) - 2)
                op = rng.choice(["swap", "drop", "double"])
                if op == "swap":
                    w = w[:j] + w[j + 1] + w[j] + w[j + 2:]
                elif op == "drop":
                    w = w[:j] + w[j + 1:]
                else:
                    w = w[:j] + w[j] + w[j:]
                words[i] = w
        out.append(" ".join(words))
    return "".join(out)


def distract(text: str, rng: random.Random) -> str:
    sent = rng.choice(DISTRACTORS).format(y=rng.randint(1985, 2023), r=rng.randint(2, 999), d=rng.randint(1, 99))
    sents = re.split(r"(?<=[.?!])\s+", text.strip())
    pos = rng.randint(0, max(0, len(sents) - 1))  # never after the final question
    sents.insert(pos, sent)
    return " ".join(sents)


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
RULES = {"rename": rename, "typos": typos, "distract": distract}


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
