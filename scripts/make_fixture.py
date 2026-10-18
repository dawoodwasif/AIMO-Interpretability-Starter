"""Write a small synthetic fixture (data/fixtures/cases.jsonl) for offline tests.

Labels are synthetic and meaningless - the fixture only checks the contract,
timing and determinism. Never report accuracy on it.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
from aimo_core.config import KNOWN_MODELS, PERTURBATION_TYPES  # noqa: E402

TEMPLATES = [
    "Find the remainder when $3^{{{a}}}$ is divided by ${b}$.",
    "Triangle $ABC$ has $AB={a}$ and $BC={b}$ with a right angle at $B$. Find $AC^2$.",
    "How many positive integers less than ${a}$ are divisible by ${b}$?",
    "Alice writes the numbers $1$ to ${a}$ on a board. Bob erases those divisible by ${b}$. How many remain?",
    "Let $f(x)=x^2+{a}x+{b}$. Find the sum of the squares of the roots of $f$.",
]


def main() -> None:
    rng = random.Random(0)
    out = ROOT / "data" / "fixtures" / "cases.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for mi, mid in enumerate(KNOWN_MODELS):
        for k in range(6):
            t = TEMPLATES[(k + mi) % len(TEMPLATES)].format(a=rng.randint(10, 999), b=rng.randint(3, 97))
            types = list(PERTURBATION_TYPES) if k % 3 == 0 else [rng.choice(PERTURBATION_TYPES)]
            rows.append({"case_id": f"fx-{mi}-{k}", "problem_id": f"p{(k + mi) % 5}", "model_id": mid,
                         "original_problem": t, "permutation_type": types, "label": rng.random() < 0.5,
                         "source": "fixture"})
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} fixture cases -> {out}")


if __name__ == "__main__":
    main()
