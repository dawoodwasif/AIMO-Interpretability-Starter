"""Score a bundle locally the way Codabench does (separate process, hard timeout).

Usage:
    python scripts/run_local.py solution --cases data/fixtures/cases.jsonl
    python scripts/run_local.py dist/latest.zip --cases data/labeled/official_val-sample.jsonl
    python scripts/run_local.py solution --cases data/val-sample/input/cases.jsonl \
        --labels data/val-sample/reference/labels.jsonl          # official getting-started layout

Cases need model_id, original_problem, permutation_type. Labels come from
--labels (joined on case_id, else by line order) or from a `label` /
`model_is_robust` field in the cases themselves.
Prints JSON: accuracy, balanced_accuracy, coverage, invalid_predictions,
runtime_s, per-model accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_KEYS = ("label", "model_is_robust", "is_robust", "robust")


def _label(d):
    for k in LABEL_KEYS:
        if k in d and d[k] is not None:
            return bool(d[k])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="solution dir or bundle .zip")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--no-internals", action="store_true", help="set AIMO_DISABLE_INTERNALS=1")
    ap.add_argument("--out", default=None, help="write metrics JSON here too")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="aimo_eval_")
    bundle = Path(args.bundle).resolve()
    if bundle.suffix == ".zip":
        with zipfile.ZipFile(bundle) as z:
            z.extractall(tmp + "/bundle")
        bundle = Path(tmp + "/bundle")
    if not (bundle / "solution.py").exists():
        raise SystemExit(f"solution.py not found at bundle root: {bundle}")

    cases = [json.loads(l) for l in open(args.cases, encoding="utf-8") if l.strip()]
    labels = [_label(c) for c in cases]
    if args.labels:
        lab = [json.loads(l) for l in open(args.labels, encoding="utf-8") if l.strip()]
        if all("case_id" in c for c in cases) and all("case_id" in l for l in lab):
            m = {l["case_id"]: _label(l) for l in lab}
            labels = [m.get(c["case_id"]) for c in cases]
        else:
            labels = [_label(l) for l in lab]

    preds_path = tmp + "/predictions.jsonl"
    env = dict(os.environ, AIMO_BUDGET_S=str(args.time_limit))
    if args.no_internals:
        env["AIMO_DISABLE_INTERNALS"] = "1"
    t0 = time.monotonic()
    timed_out = False
    try:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "_ingest_runner.py"), str(bundle),
                        str(Path(args.cases).resolve()), preds_path], env=env, timeout=args.time_limit, check=False)
    except subprocess.TimeoutExpired:
        timed_out = True
    runtime = time.monotonic() - t0

    preds = [json.loads(l) for l in open(preds_path)] if os.path.exists(preds_path) else []
    pmap = {p["idx"]: p for p in preds}
    correct = valid = tp = tn = pos = neg = 0
    per_model = defaultdict(lambda: [0, 0])
    scored = 0
    for i, (c, y) in enumerate(zip(cases, labels)):
        if y is None:
            continue
        scored += 1
        p = pmap.get(i, {})
        ok = p.get("valid", False)
        valid += ok
        hit = ok and p["is_robust"] == y
        correct += hit
        per_model[c["model_id"]][0] += hit
        per_model[c["model_id"]][1] += 1
        if y:
            pos += 1
            tp += hit
        else:
            neg += 1
            tn += hit
    res = {
        "accuracy": round(correct / scored, 4) if scored else None,
        "balanced_accuracy": round(((tp / pos if pos else 0) + (tn / neg if neg else 0)) / 2, 4) if scored else None,
        "coverage": round(valid / scored, 4) if scored else None,
        "invalid_predictions": scored - valid,
        "n": scored,
        "label_balance": round(pos / scored, 3) if scored else None,
        "runtime_s": round(runtime, 1),
        "timed_out": timed_out,
        "per_model": {m: round(a / b, 3) for m, (a, b) in per_model.items()},
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    if timed_out or (scored and valid < scored):
        sys.exit(2)


if __name__ == "__main__":
    main()
