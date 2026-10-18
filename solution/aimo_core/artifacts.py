"""Loads artifacts/manifest.json and the per-model heads it references.

manifest.json schema
--------------------
{
  "version": 1,
  "default_prior": 0.5,                 # P(robust) when nothing else is known
  "threshold": 0.5,                     # global decision threshold
  "models": {
    "<hf model id or '*'>": {
      "prior": 0.45,                    # P(robust) for this model
      "perm_prior": {"typos": 0.7},     # optional P(robust | single perm type)
      "threshold": 0.5,                 # optional per-model threshold
      "heads": [ {"file": "deepseek_text.npz", "weight": 1.0} ]
    }
  }
}
'*' is the shared/fallback entry used for unknown models.
"""
from __future__ import annotations

import json
import os
import sys
from functools import lru_cache

from . import config
from .linear import LinearHead


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


def _norm(mid: str) -> str:
    mid = (mid or "").strip().lower()
    for suffix in (":low", ":medium", ":high"):
        if mid.endswith(suffix):
            mid = mid[: -len(suffix)]
    return mid


@lru_cache(maxsize=1)
def manifest() -> dict:
    path = os.path.join(config.ARTIFACT_DIR, "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        _log(f"no usable manifest at {path}: {exc!r}")
        return {"version": 1, "default_prior": 0.5, "threshold": 0.5, "models": {}}


def model_entry(model_id: str) -> dict:
    models = manifest().get("models", {})
    if model_id in models:
        return models[model_id]
    want = _norm(model_id)
    for key, val in models.items():
        if _norm(key) == want:
            return val
    base = want.split("/")[-1]
    for key, val in models.items():
        if key != "*" and _norm(key).split("/")[-1] == base:
            return val
    return models.get("*", {})


@lru_cache(maxsize=64)
def load_head(filename: str) -> LinearHead | None:
    try:
        return LinearHead.load(os.path.join(config.ARTIFACT_DIR, filename))
    except Exception as exc:  # noqa: BLE001
        _log(f"failed to load head {filename}: {exc!r}")
        return None
