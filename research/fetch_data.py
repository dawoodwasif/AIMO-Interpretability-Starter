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

OFFICIAL = [
    "aimo-interp/val-sample",
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

    ds = load_dataset(name)
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


def normalise(name: str, rows: list[dict]) -> list[dict]:
    out = []
    for i, r in enumerate(rows):
        if "model_is_robust" not in r or "original_problem" not in r:
            continue
        out.append({
            "case_id": f"{name.split('/')[-1]}:{i}",
            "problem_id": str(r.get("problem_id") or text_hash(r["original_problem"])),
            "model_id": str(r.get("model_id", "")),
            "original_problem": r["original_problem"],
            "permutation_type": _parse_types(r.get("permutation_type")),
            "label": bool(r["model_is_robust"]),
            "source": "official",
            "dataset": name,
            "base_accuracy": r.get("base_accuracy"),
            "absolute_accuracy_decay": r.get("absolute_accuracy_decay"),
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
