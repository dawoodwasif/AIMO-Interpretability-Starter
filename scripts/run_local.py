"""Score a bundle locally with the OFFICIAL Codabench ingestion + scoring code.

Runs third_party/getting-started/components/{ingestion,scoring}_program verbatim
in a subprocess with a hard timeout, so the call signature (keyword
reasoning_effort/problems, string problems, batching by (model, effort)), the
small.txt track guard and the metric are exactly Codabench's.

Usage:
    python scripts/run_local.py solution --cases data/fixtures/cases.jsonl
    python scripts/run_local.py dist/main-v1.zip --cases data/labeled/official_train-main-v2.jsonl
    python scripts/run_local.py dist/small-v1.zip --small --cases ...
    python scripts/run_local.py solution --input-dir data/val-sample/input \
        --reference-dir data/val-sample/reference            # official importer layout

--cases accepts our labeled-row schema (model_id, original_problem|problem,
label|model_is_robust|is_robust, optional reasoning_effort, case_id|id) and is
converted to the official input/reference layout. Rows for the same
(model, problem, effort) are collapsed with the conjunction rule (robust only if
every row is robust), like the official importer requires.
Prints JSON: the official scores plus balanced accuracy, per-model accuracy,
label balance and runtime.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = ROOT / "third_party" / "getting-started" / "components"
LABEL_KEYS = ("label", "model_is_robust", "is_robust", "robust")


def _label(d):
    for k in LABEL_KEYS:
        if k in d and d[k] is not None:
            return bool(d[k])
    return None


def to_official(cases_path: Path, out_dir: Path, models=None) -> int:
    """Write <out>/input/cases.jsonl and <out>/reference/labels.jsonl."""
    rows = [json.loads(l) for l in open(cases_path, encoding="utf-8") if l.strip()]
    groups: dict[tuple, dict] = {}
    for r in rows:
        mid = r["model_id"]
        if models and mid not in models:
            continue
        prob = r.get("problem") if isinstance(r.get("problem"), str) else r.get("original_problem")
        eff = r.get("reasoning_effort") or "default"
        key = (mid, prob, eff)
        y = _label(r)
        g = groups.setdefault(key, {"label": True, "has": False})
        if y is not None:
            g["label"] = g["label"] and y
            g["has"] = True
    (out_dir / "input").mkdir(parents=True, exist_ok=True)
    (out_dir / "reference").mkdir(parents=True, exist_ok=True)
    with open(out_dir / "input" / "cases.jsonl", "w") as fc, open(out_dir / "reference" / "labels.jsonl", "w") as fl:
        for (mid, prob, eff), g in groups.items():
            if not g["has"]:
                continue
            cid = "case-" + hashlib.sha256(json.dumps([mid, prob, eff]).encode()).hexdigest()[:24]
            c = {"id": cid, "model_id": mid, "problem": prob}
            if eff != "default":
                c["reasoning_effort"] = eff
            fc.write(json.dumps(c) + "\n")
            fl.write(json.dumps({"id": cid, "is_robust": g["label"]}) + "\n")
    return sum(g["has"] for g in groups.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle", help="solution dir or bundle .zip")
    ap.add_argument("--cases", default=None, help="our labeled-row jsonl (converted to the official layout)")
    ap.add_argument("--input-dir", default=None, help="official layout: dir with cases.jsonl")
    ap.add_argument("--reference-dir", default=None, help="official layout: dir with labels.jsonl")
    ap.add_argument("--small", action="store_true", help="small-model track (bundle must contain small.txt)")
    ap.add_argument("--models", nargs="*", default=None, help="only these model ids (with --cases)")
    ap.add_argument("--time-limit", type=float, default=3600)
    ap.add_argument("--no-internals", action="store_true", help="set AIMO_DISABLE_INTERNALS=1")
    ap.add_argument("--out", default=None, help="write metrics JSON here too")
    args = ap.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="aimo_eval_"))
    try:
        bundle = Path(args.bundle).resolve()
        if bundle.suffix == ".zip":
            with zipfile.ZipFile(bundle) as z:
                z.extractall(tmp / "bundle")
            bundle = tmp / "bundle"
        elif args.small and not any(p.name.casefold() == "small.txt" for p in bundle.iterdir()):
            # a source dir is not a track bundle; stage a copy with the marker
            shutil.copytree(bundle, tmp / "bundle", ignore=shutil.ignore_patterns("__pycache__"))
            (tmp / "bundle" / "small.txt").write_text("small\n")
            bundle = tmp / "bundle"
        if not (bundle / "solution.py").exists():
            raise SystemExit(f"solution.py not found at bundle root: {bundle}")

        if args.cases:
            models = args.models
            if args.small and models is None:
                sys.path.insert(0, str(ROOT / "solution"))
                from aimo_core.config import SMALL_MODELS  # noqa: E402
                models = list(SMALL_MODELS)
            n =to_official(Path(args.cases), tmp / "data", models)
            if n == 0:
                raise SystemExit("no labeled cases after conversion")
            input_dir, ref_dir = tmp / "data" / "input", tmp / "data" / "reference"
        else:
            input_dir, ref_dir = Path(args.input_dir).resolve(), Path(args.reference_dir).resolve()

        track = "small" if args.small else "main"
        res_dir, score_in, score_out = tmp / "input" / "res", tmp / "input", tmp / "output"
        shutil.copytree(ref_dir, score_in / "ref")
        env = dict(os.environ, AIMO_BUDGET_S=str(args.time_limit), PYTHONDONTWRITEBYTECODE="1")
        if args.no_internals:
            env["AIMO_DISABLE_INTERNALS"] = "1"
        t0 = time.monotonic()
        timed_out = False
        try:
            r = subprocess.run([sys.executable, str(COMPONENTS / "ingestion_program" / "ingestion.py"),
                                str(input_dir), str(res_dir), str(bundle), "--track", track],
                               env=env, timeout=args.time_limit, capture_output=True, text=True)
            sys.stderr.write(r.stderr[-4000:])
            if r.returncode != 0:
                raise SystemExit(f"ingestion failed (exit {r.returncode})")
        except subprocess.TimeoutExpired:
            timed_out = True
        runtime = time.monotonic() - t0
        if timed_out:
            res = {"accuracy": 0.0, "coverage": 0.0, "timed_out": True, "runtime_s": round(runtime, 1)}
        else:
            subprocess.run([sys.executable, str(COMPONENTS / "scoring_program" / "scoring.py"),
                            str(score_in), str(score_out), "--track", track], check=True)
            res = json.loads((score_out / "scores.json").read_text())
            cases = {c["id"]: c for c in map(json.loads, open(input_dir / "cases.jsonl")) if c}
            labels = {l["id"]: l["is_robust"] for l in map(json.loads, open(ref_dir / "labels.jsonl")) if l}
            preds = {p["id"]: p for p in map(json.loads, open(res_dir / "predictions.jsonl")) if p}
            per_model = defaultdict(lambda: [0, 0])
            tp = tn = pos = neg = 0
            for cid, y in labels.items():
                p = preds.get(cid, {})
                hit = bool(p.get("valid")) and p.get("is_robust") == y
                m = cases[cid]["model_id"] + ("@" + cases[cid]["reasoning_effort"] if "reasoning_effort" in cases[cid] else "")
                per_model[m][0] += hit
                per_model[m][1] += 1
                pos, tp = (pos + 1, tp + hit) if y else (pos, tp)
                neg, tn = (neg, tn) if y else (neg + 1, tn + hit)
            res.update({
                "balanced_accuracy": round(((tp / pos if pos else 0) + (tn / neg if neg else 0)) / 2, 4),
                "n": len(labels), "label_balance": round(pos / max(1, len(labels)), 3),
                "predicted_robust": sum(bool(p.get("is_robust")) for p in preds.values()),
                "runtime_s": round(runtime, 1), "timed_out": False, "track": track,
                "per_model": {m: round(a / b, 3) for m, (a, b) in sorted(per_model.items())},
            })
        print(json.dumps(res, indent=1))
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(res, indent=1))
        if res.get("timed_out") or res.get("coverage", 0) < 1.0:
            sys.exit(2)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
