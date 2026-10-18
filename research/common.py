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

# Unified labeled-row schema (one JSON object per line), ONE row per eval case
# (model_id, problem, reasoning_effort) - the harness never sees perturbation types:
#   case_id, problem_id (= text_hash(problem)), model_id, reasoning_effort,
#   original_problem, label (bool robust | None = ambiguous, dropped),
#   source ("official" | "official_math" | "byeolki" | "self"), origin (problem source),
#   leak (bool: organizer leakage advice), base_accuracy, max_drop, family_acc (optional)


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
    """Stable id of a problem text (whitespace-normalised; same as import_external.pid)."""
    return hashlib.sha1(re.sub(r"\s+", " ", text.strip()).encode("utf-8")).hexdigest()[:12]


def load_labeled(paths=None, sources=None, include_unlabeled: bool = False) -> list[dict]:
    """Labeled rows; include_unlabeled=True also keeps label=None rows that carry a
    base_accuracy (used to train solve-rate probes, H8)."""
    paths = paths or sorted(LABELED.glob("*.jsonl"))
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            if r.get("label") is None and not (include_unlabeled and r.get("base_accuracy") is not None):
                continue
            if sources and r.get("source") not in sources:
                continue
            r.setdefault("reasoning_effort", "default")
            r["problem_id"] = text_hash(r["original_problem"])
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
