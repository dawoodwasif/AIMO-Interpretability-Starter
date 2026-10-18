"""Deterministic rule-based perturbations (rename / typos / distract).

Shipped in the bundle (used for test-time representation-shift features) and
imported by research/perturb.py, so training and serving use identical code.
Pure stdlib. Seeds are strings -> random.Random is reproducible across runs.
"""
from __future__ import annotations

import random
import re

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


# Letters never renamed: indistinguishable from English words in prose.
_KEEP = set("aAI")
# Inside math: a backslash command, a \text{...}-like group, or a single letter.
_MATH_TOK = re.compile(r"(\\(?:text|mathrm|operatorname|textbf|mbox)\{[^}]*\}|\\[A-Za-z]+|[A-Za-z])")
# Prose: an isolated letter or an all-caps point label like BC / ABC.
_PROSE_TOK = re.compile(r"(?<![A-Za-z\\'])([A-Za-z]|[A-Z]{2,4})(?![A-Za-z'])")


def _math_letters(seg: str) -> set[str]:
    return {t for t in _MATH_TOK.findall(seg) if len(t) == 1}


def rename(text: str, rng: random.Random) -> str:
    """Rename variables/point labels consistently in math AND prose, and person names."""
    parts = _split_math(text)
    used = set().union(*[_math_letters(s) for is_m, s in parts if is_m] or [set()]) - _KEEP
    taken = used | _KEEP | set(re.findall(r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])", text))
    lower = [c for c in "bcdfghjkmnpqrstuvwxyz" if c not in taken]
    upper = [c for c in "BCDEFGHJKLMNPQRSTUVWXYZ" if c not in taken]
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

    def _prose(m):
        tok = m.group(1)
        if all(c in mapping for c in tok):
            return "".join(mapping[c] for c in tok)
        return tok

    out = []
    for is_m, seg in parts:
        if is_m:
            seg = _MATH_TOK.sub(lambda m: mapping.get(m.group(1), m.group(1)), seg)
        else:
            for a, b in name_map.items():
                seg = re.sub(rf"\b{a}\b", b, seg)
            seg = _PROSE_TOK.sub(_prose, seg)
        out.append(seg)
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


RULES = {"rename": rename, "typos": typos, "distract": distract}


def variants(text: str, kinds=("rename", "typos", "distract"), k: int = 1, seed: str = "aimo") -> list[tuple[str, str]]:
    """[(kind, perturbed_text)] for each kind, k copies each, seeded by the text."""
    out = []
    for kind in kinds:
        for j in range(k):
            rng = random.Random(f"{seed}:{kind}:{j}:{text}")
            out.append((kind, RULES[kind](text, rng)))
    return out
