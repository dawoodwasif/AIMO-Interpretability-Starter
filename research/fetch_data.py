"""Download the public aimo-interp datasets and normalise them to data/labeled/.

Usage:
    python research/fetch_data.py                 # official sets
    python research/fetch_data.py --pool          # + public problem pool with answers

Every dataset's raw rows are saved to data/raw/<name>.jsonl so you can inspect
schemas (do it! schemas are not documented and may change).
Rows that carry `model_is_robust` become labeled rows (source="official").
"""
from __future__ import annotations

import argparse
import json

from common import LABELED, POOL, RAW, text_hash, write_jsonl
from aimo_core.config import MODEL_ALIASES

OFFICIAL = [
    "aimo-interp/train-main-v2",            # competition models (byeolki: 34 small-track rows)
    "aimo-interp/val-sample",               # private: needs HF_TOKEN with access
    "aimo-interp/aimo-interp-challenge-sample-full",
    "aimo-interp/augmented-sample-math",
    "aimo-interp/augmented-sample-math-agg",
    "aimo-interp/augmented-sample-math-agg-filtered",
    "aimo-interp/augmented-sample-math-full-filtered",
    "aimo-interp/problems-public",  # was gated (401) in Oct 2026; may open later
]

# Candidate public problem sources with integer answers. VERIFY ids/columns.
POOL_SOURCES = [
    "AI-MO/aimo-validation-aime",
    "AI-MO/aimo-validation-amc",
    "MathArena/aime_2025",
    "MathArena/hmmt_feb_2025",
]


def _load(name: str) -> list[dict]:
    from datasets import load_dataset

    import os

    ds = load_dataset(name, token=os.environ.get("HF_TOKEN") or None)
    rows = []
    for split, d in ds.items():
        for r in d:
            r = dict(r)
            r["_split"] = split
            rows.append(r)
    return rows


def _parse_types(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = [v]
    return sorted(set(map(str, v or [])))


LABEL_FIELDS = ("model_is_robust", "is_robust")
PROBLEM_FIELDS = ("original_problem", "problem")


def normalise(name: str, rows: list[dict]) -> list[dict]:
    """One labeled row per (model, problem, effort); per-family rows collapsed by
    conjunction (robust only if all robust), like the official importer requires."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        lab = next((r[k] for k in LABEL_FIELDS if k in r and r[k] is not None), None)
        text = next((r[k] for k in PROBLEM_FIELDS if isinstance(r.get(k), str)), None)
        if lab is None or not text:
            continue
        mid = MODEL_ALIASES.get(str(r.get("model_id", "")), str(r.get("model_id", "")))
        eff = str(r.get("reasoning_effort") or "default")
        k = (mid, " ".join(text.split()), eff)
        g = groups.setdefault(k, {"label": True, "families": set(), "base": [], "decay": []})
        g["label"] = g["label"] and bool(lab)
        g["families"] |= set(_parse_types(r.get("permutation_type")))
        if r.get("base_accuracy") is not None:
            g["base"].append(float(r["base_accuracy"]))
        if r.get("absolute_accuracy_decay") is not None:
            g["decay"].append(float(r["absolute_accuracy_decay"]))
    out = []
    for (mid, text, eff), g in groups.items():
        out.append({
            "case_id": f"{name.split('/')[-1]}:{text_hash(mid + eff + text)}",
            "problem_id": text_hash(text), "model_id": mid, "reasoning_effort": eff,
            "original_problem": text, "label": g["label"], "source": "official", "dataset": name,
            "families": sorted(g["families"]),
            "base_accuracy": min(g["base"]) if g["base"] else None,
            "max_drop": max(g["decay"]) if g["decay"] else None,
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", action="store_true", help="also fetch problem pool sources")
    args = ap.parse_args()

    for name in OFFICIAL:
        try:
            rows = _load(name)
        except Exception as exc:  # noqa: BLE001
            print(f"SKIP {name}: {exc!r}")
            continue
        fn = name.split("/")[-1]
        write_jsonl(RAW / f"{fn}.jsonl", rows)
        lab = normalise(name, rows)
        if lab:
            write_jsonl(LABELED / f"official_{fn}.jsonl", lab)
        cols = sorted(rows[0].keys()) if rows else []
        print(f"{name}: {len(rows)} rows, {len(lab)} labeled; columns={cols}")

    if args.pool:
        for name in POOL_SOURCES:
            try:
                rows = _load(name)
            except Exception as exc:  # noqa: BLE001
                print(f"SKIP pool {name}: {exc!r}")
                continue
            write_jsonl(POOL / "raw" / f"{name.replace('/', '__')}.jsonl", rows)
            print(f"pool {name}: {len(rows)} rows; columns={sorted(rows[0].keys()) if rows else []}")
        print("Next: write data/pool/problems.jsonl rows {id, problem, answer} (see research/README.md)")


if __name__ == "__main__":
    main()
