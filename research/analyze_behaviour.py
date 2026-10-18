"""H6: do a few samples on the ORIGINAL problem predict robustness? (split-sample, CPU)

Uses the raw byeolki generations (data/external/byeolki_genstats.jsonl). For each
(model, problem) with 8 original samples and 3 rule-based variant families:
  * features from samples A (first --k):  acc_A, answer agreement (gold-free),
    mean reasoning length, truncation rate
  * label from samples B (the rest) + all variant samples, with our rule
    (base_B >= min_base; robust if max family drop <= robust_max, spurious if >= spurious_min)
so features and label never share a sample. Reports AUC per feature overall and
within-problem pairwise accuracy on mixed problems (the balanced-test analogue).
This decides whether test-time sampling is worth its runtime.

Usage: python research/analyze_behaviour.py [--k 4] [--allow-leak]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from common import DATA, RESULTS, read_jsonl
from import_external import LEAK_SOURCES


def feats(o, k):
    c = o["correct"][:k]
    a = [x for x in o["answers"][:k] if x is not None]
    agree = Counter(a).most_common(1)[0][1] / k if a else 0.0
    return {"acc_A": float(np.mean(c)), "agree_A": agree, "len_A": float(np.mean(o["lengths"][:k])),
            "trunc_A": float(np.mean(o["truncated"][:k])), "parsed_A": len(a) / k}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--min-base", type=float, default=0.5)
    ap.add_argument("--robust-max", type=float, default=0.125)
    ap.add_argument("--spurious-min", type=float, default=0.25)
    ap.add_argument("--allow-leak", action="store_true")
    args = ap.parse_args()

    rows = []
    leak_ids = {r["problem_id"] for r in read_jsonl(DATA / "labeled" / "ext_byeolki.jsonl") if r.get("leak")}
    for g in read_jsonl(DATA / "external" / "byeolki_genstats.jsonl"):
        if "original" not in g or not all(f in g for f in ("rename", "typos", "distract")):
            continue
        if not args.allow_leak and (g["problem_id"] in leak_ids or g.get("origin") in LEAK_SOURCES):
            continue
        o = g["original"]
        B = o["correct"][args.k:]
        base = float(np.mean(B))
        if base < args.min_base:
            continue
        drop = max(base - float(np.mean(g[f]["correct"])) for f in ("rename", "typos", "distract"))
        lab = True if drop <= args.robust_max else False if drop >= args.spurious_min else None
        if lab is None:
            continue
        rows.append({"model": g["model_id"], "pid": g["problem_id"], "label": lab, "base_B": base,
                     **feats(o, args.k)})
    y = np.array([r["label"] for r in rows], bool)
    print(f"{len(rows)} labelled rows ({y.mean():.2f} robust), models {Counter(r['model'] for r in rows)}")
    names = ["acc_A", "agree_A", "len_A", "trunc_A", "parsed_A"]
    res = {"n": len(rows), "robust_rate": round(float(y.mean()), 3), "auc": {}, "mixed_pairwise": {}}
    for n in names:
        s = np.array([r[n] for r in rows])
        res["auc"][n] = round(float(roc_auc_score(y, s)), 3) if 0 < y.mean() < 1 else None
    by = defaultdict(list)
    for r in rows:
        by[r["pid"]].append(r)
    mixed = [v for v in by.values() if len({r["label"] for r in v}) == 2]
    res["mixed_problems"] = len(mixed)
    for n in names:
        hit = tot = 0.0
        for v in mixed:
            for a in v:
                for b in v:
                    if a["label"] and not b["label"]:
                        tot += 1
                        hit += 1.0 if a[n] > b[n] else 0.5 if a[n] == b[n] else 0.0
        res["mixed_pairwise"][n] = round(hit / tot, 3) if tot else None
        res.setdefault("mixed_pairs", int(tot))
    print(json.dumps(res, indent=1))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "EXP-002.json").write_text(json.dumps({"args": vars(args), **res}, indent=1))


if __name__ == "__main__":
    main()
