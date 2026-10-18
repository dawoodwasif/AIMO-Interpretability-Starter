"""Cheap, deterministic, CPU-only features from the problem text.

No model needed, so these always run (they are the fallback when internals
are skipped for time). Pure Python + numpy: identical in training and bundle.
"""
from __future__ import annotations

import math
import re
import zlib

import numpy as np

from .config import PERTURBATION_TYPES

HASH_DIM = 512
_TOKEN = re.compile(r"[a-z]+|\d+|\\[a-z]+|[^\sa-z\d]")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
KEYWORDS = [
    "remainder", "modulo", "prime", "divisible", "digit", "triangle", "circle",
    "angle", "area", "probability", "expected", "sequence", "polynomial", "root",
    "integer", "real", "maximum", "greatest", "minimum", "smallest", "how many",
    "sum", "product", "function", "grid", "coloring", "game", "fibonacci",
    "gcd", "lcm", "coprime", "m+n", "permutation", "subset",
]


def perm_types(raw) -> list[str]:
    """Normalise permutation_type (list, JSON string, or plain string)."""
    if raw is None:
        return []
    if isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            import json

            try:
                raw = json.loads(s)
            except ValueError:
                raw = [s]
        else:
            raw = [s]
    return [str(t).strip().lower() for t in raw]


def perm_features(types: list[str]) -> np.ndarray:
    uniq = set(types)
    v = [1.0 if t in uniq else 0.0 for t in PERTURBATION_TYPES]
    v.append(math.log1p(len(uniq)))
    v.append(1.0 if len(uniq) >= 5 else 0.0)  # "all perturbations" style case
    return np.asarray(v, dtype=np.float32)


def _hashed_ngrams(text: str, dim: int = HASH_DIM) -> np.ndarray:
    toks = _TOKEN.findall(text.lower())
    vec = np.zeros(dim, dtype=np.float32)
    grams = toks + [a + " " + b for a, b in zip(toks, toks[1:])]
    for g in grams:
        h = zlib.crc32(g.encode("utf-8"))
        vec[h % dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
    vec = np.sign(vec) * np.log1p(np.abs(vec))
    n = float(np.linalg.norm(vec))
    return vec / n if n > 0 else vec


def _stats(text: str) -> np.ndarray:
    low = text.lower()
    nums = _NUMBER.findall(text)
    max_digits = max((len(n.split(".")[0]) for n in nums), default=0)
    math_chars = sum(text.count(c) for c in "$\\^_{}=")
    feats = [
        math.log1p(len(text)),
        math.log1p(len(low.split())),
        math.log1p(len(nums)),
        math.log1p(max_digits),
        math_chars / max(1, len(text)),
        math.log1p(low.count(".") + low.count("?")),
    ]
    feats += [1.0 if k in low else 0.0 for k in KEYWORDS]
    return np.asarray(feats, dtype=np.float32)


def featurize(text: str, types) -> np.ndarray:
    """Feature vector for feature_set == 'text'."""
    t = perm_types(types)
    return np.concatenate([perm_features(t), _stats(text or ""), _hashed_ngrams(text or "")])


def featurize_meta(text: str, types) -> np.ndarray:
    """Small feature set (no hashing) for very low-data heads: 'meta'."""
    t = perm_types(types)
    return np.concatenate([perm_features(t), _stats(text or "")])


FEATURE_FUNCS = {"text": featurize, "meta": featurize_meta}
