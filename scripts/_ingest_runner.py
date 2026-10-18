"""Child process that mimics Codabench ingestion: import solution.py from a
bundle directory and call are_robust once per model batch.

argv: <bundle_dir> <cases.jsonl> <predictions_out.jsonl>
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]


def _types(v):
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = [v]
    return list(v or [])


def main() -> None:
    bundle, cases_path, out_path = sys.argv[1:4]
    cases = [json.loads(l) for l in open(cases_path, encoding="utf-8") if l.strip()]
    os.chdir(bundle)
    sys.path.insert(0, bundle)
    t0 = time.monotonic()
    import solution  # noqa: E402

    order, groups = [], {}
    for i, c in enumerate(cases):
        mid = c["model_id"]
        if mid not in groups:
            order.append(mid)
            groups[mid] = []
        groups[mid].append(i)

    preds = {}
    for mid in order:
        idx = groups[mid]
        probs = [Problem(cases[i]["original_problem"], _types(cases[i].get("permutation_type"))) for i in idx]
        t = time.monotonic()
        try:
            out = solution.are_robust(mid, probs)
        except Exception as exc:  # noqa: BLE001
            print(f"[runner] are_robust raised for {mid}: {exc!r}", file=sys.stderr)
            out = None
        ok = isinstance(out, list) and len(out) == len(idx)
        for k, i in enumerate(idx):
            v = out[k] if ok else None
            preds[i] = {"idx": i, "is_robust": v if type(v) is bool else None, "valid": type(v) is bool}
        print(f"[runner] {mid}: {len(idx)} cases in {time.monotonic() - t:.1f}s", file=sys.stderr)

    with open(out_path, "w", encoding="utf-8") as f:
        for i in range(len(cases)):
            f.write(json.dumps(preds.get(i, {"idx": i, "is_robust": None, "valid": False})) + "\n")
    print(f"[runner] total {time.monotonic() - t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
