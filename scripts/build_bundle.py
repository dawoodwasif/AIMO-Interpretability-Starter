"""Build a Codabench submission ZIP from solution/ and verify it.

Checks (any failure -> non-zero exit, no zip kept):
  1. solution.py at archive root, no __pycache__/.pyc
  2. every import is stdlib or in the competition runtime allowlist
  3. manifest.json parses and every referenced head file exists & loads
  4. smoke run on the fixture in a separate process with internals disabled:
     coverage == 1.0, no invalid predictions, deterministic across 2 runs

Usage:
    python scripts/build_bundle.py --name main-v3
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "solution"
DIST = ROOT / "dist"
# Dockerfile.competition: torch 2.12.1, accelerate, huggingface_hub, joblib,
# pandas, safetensors, scikit-learn, tokenizers, transformers (+ their deps).
ALLOWED = {"torch", "transformers", "accelerate", "huggingface_hub", "joblib", "pandas",
           "safetensors", "sklearn", "tokenizers", "numpy", "aimo_core", "solution"}
MAX_MB = 500  # no documented limit; keep bundles small anyway


def check_imports(files):
    bad = []
    std = set(sys.stdlib_module_names)
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            mods = []
            if isinstance(node, ast.Import):
                mods = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                mods = [node.module]
            for m in mods:
                top = m.split(".")[0]
                if top not in std and top not in ALLOWED:
                    bad.append(f"{f.relative_to(SRC)}: import {m}")
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None)
    ap.add_argument("--cases", default=str(ROOT / "data" / "fixtures" / "cases.jsonl"))
    args = ap.parse_args()

    files = [p for p in SRC.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    py = [p for p in files if p.suffix == ".py"]
    errors = check_imports(py)

    try:
        manifest = json.loads((SRC / "artifacts" / "manifest.json").read_text())
        sys.path.insert(0, str(SRC))
        from aimo_core.linear import LinearHead

        for mid, e in manifest.get("models", {}).items():
            for h in e.get("heads", []):
                LinearHead.load(str(SRC / "artifacts" / h["file"]))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manifest/heads: {exc!r}")
    if errors:
        print("BUNDLE CHECK FAILED:\n  " + "\n  ".join(errors))
        sys.exit(1)

    DIST.mkdir(exist_ok=True)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except OSError:
        sha = ""
    name = args.name or f"bundle-{time.strftime('%m%d-%H%M')}{'-' + sha if sha else ''}"
    out = DIST / f"{name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(SRC).as_posix())
    with zipfile.ZipFile(out) as z:
        assert "solution.py" in z.namelist(), "solution.py must be at archive root"
    mb = out.stat().st_size / 1e6
    if mb > MAX_MB:
        out.unlink()
        raise SystemExit(f"bundle too large: {mb:.1f} MB")

    results = []
    for k in range(2):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_local.py"), str(out), "--cases", args.cases,
                            "--no-internals", "--time-limit", "600"], capture_output=True, text=True)
        try:
            results.append(json.loads(r.stdout))
        except ValueError:
            print(r.stdout, r.stderr)
            out.unlink()
            raise SystemExit("smoke run produced no metrics")
    m = results[0]
    if m["coverage"] != 1.0 or m["invalid_predictions"] or m["timed_out"]:
        out.unlink()
        raise SystemExit(f"smoke run failed: {m}")
    if results[0]["accuracy"] != results[1]["accuracy"]:
        out.unlink()
        raise SystemExit("non-deterministic predictions across runs")

    digest = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
    meta = {"zip": out.name, "size_mb": round(mb, 2), "sha256_16": digest, "git": sha,
            "manifest_exp": manifest.get("exp"), "smoke_fixture": m, "built": time.strftime("%Y-%m-%d %H:%M")}
    (DIST / f"{name}.json").write_text(json.dumps(meta, indent=1))
    latest = DIST / "latest.zip"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.write_bytes(out.read_bytes())
    print(json.dumps(meta, indent=1))
    print(f"OK -> {out}  (upload this file to Codabench)")


if __name__ == "__main__":
    main()
