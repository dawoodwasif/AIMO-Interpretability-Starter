"""Shared helpers for the offline research pipeline (NOT shipped in the bundle)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLUTION = ROOT / "solution"
ARTIFACTS = SOLUTION / "artifacts"
DATA = ROOT / "data"
RAW = DATA / "raw"
LABELED = DATA / "labeled"      # unified labeled rows: *.jsonl
POOL = DATA / "pool"            # problems with answers + generated variants
CACHE = DATA / "cache"          # feature caches (.npz), gitignored
RESULTS = ROOT / "results"      # one JSON per experiment

if str(SOLUTION) not in sys.path:
    sys.path.insert(0, str(SOLUTION))

# Unified labeled-row schema (one JSON object per line):
#   case_id, problem_id, model_id, original_problem, permutation_type (list[str]),
#   label (bool: robust), source ("official" | "self" | ...),
#   base_accuracy, absolute_accuracy_decay (optional floats)


def read_jsonl(path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def slug(model_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", model_id).strip("_").lower()


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_labeled(paths=None, sources=None) -> list[dict]:
    paths = paths or sorted(LABELED.glob("*.jsonl"))
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            if r.get("label") is None:
                continue
            if sources and r.get("source") not in sources:
                continue
            rows.append(r)
    return rows


def env_info() -> dict:
    info = {"python": sys.version.split()[0]}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        info["torch"] = None
    info["hf_home"] = os.environ.get("HF_HOME")
    return info
