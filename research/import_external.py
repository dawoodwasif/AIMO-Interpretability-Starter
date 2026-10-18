"""Import public labeled data + a problem pool from cloned GitHub repos (no HF needed).

Sources (clone them first, see third_party/README.md):
  * third_party/baselines/data/math-robust-agg.csv   (organizers, Apache-2.0)
      DeepSeek-R1-0528-Qwen3-8B (alias qwen3-8b:low) on 137 MATH problems
      -> data/labeled/official_math-agg.jsonl  (one row per (model, problem), conjunction)
      -> data/raw/official_variants.jsonl     (real LLM-written variants that caused decay)
  * third_party/byeolki/data/generations/*.jsonl  (byeolki/aimo-interp, MIT)
      raw sampled correctness for Qwen3.5-4B / Skywork / DeepSeek on 150 problems
      (8 samples on the original, 6 on one rename/typos/distract variant each;
      8k reasoning budget with forced answers). Relabelled with OUR rule:
      -> data/labeled/ext_byeolki.jsonl  + data/external/byeolki_genstats.jsonl
  * third_party/byeolki/data/problems.jsonl + qwen25math eval sets -> data/pool/problems.jsonl
      minus organizer leakage advice (AIME 2024-2026, AIMO refs) and minus problems
      that already have labels (kept for validation only).

Usage:
    python research/import_external.py [--min-base 0.6 --robust-max 0.10 --spurious-min 0.25]
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict

import pandas as pd

from common import DATA, LABELED, POOL, RAW, ROOT, write_jsonl
from aimo_core.config import MODEL_ALIASES

TP = ROOT / "third_party"
LEAK_SOURCES = {"aime_2024", "aime_2025", "aime_2026", "aimo2", "aimo3", "jmo", "problems-public"}


def norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def pid(text: str) -> str:
    import hashlib

    return hashlib.sha1(norm(text).encode("utf-8")).hexdigest()[:12]


def key(t: str) -> str:
    """Loose key for duplicate / leakage matching across sources."""
    return re.sub(r"[^a-z0-9]", "", norm(t).lower())[:160]


def import_official_math() -> list[dict]:
    p = TP / "baselines" / "data" / "math-robust-agg.csv"
    if not p.exists():
        print(f"skip: {p} missing")
        return []
    d = pd.read_csv(p)
    groups = defaultdict(list)
    for _, r in d.iterrows():
        groups[(r.model_id, norm(r.original_problem))].append(r)
    rows = []
    for (mid, text), rs in groups.items():
        rows.append({
            "case_id": f"official_math:{pid(text)}", "problem_id": pid(text),
            "model_id": MODEL_ALIASES.get(mid, mid), "reasoning_effort": "default",
            "original_problem": text, "label": bool(all(bool(r.model_is_robust) for r in rs)),
            "source": "official_math", "origin": "hendrycks-math",
            "base_accuracy": float(min(r.base_accuracy for r in rs)),
            "max_drop": float(max(r.absolute_accuracy_decay for r in rs)),
            "n_detrimental": int(max(r.n_detrimental_permutations for r in rs)),
        })
    # real variants (questions that caused decay) for perturbation analysis
    f = pd.read_csv(TP / "baselines" / "data" / "math-robust-final.csv")
    var = []
    for _, r in f.iterrows():
        for x in json.loads(r.permutations_causing_decay):
            var.append({"parent_id": pid(r.original_problem), "type": x["permutation_type"],
                        "source": x.get("permutation_source"), "variant_accuracy": x["variant_accuracy"],
                        "n": x["n_predictions"], "original": norm(r.original_problem), "variant": x["question"]})
    write_jsonl(RAW / "official_variants.jsonl", var)
    print(f"official_math: {len(rows)} rows ({sum(r['label'] for r in rows)} robust); {len(var)} decay variants")
    return rows


def import_byeolki(args) -> list[dict]:
    gdir = TP / "byeolki" / "data" / "generations"
    probs = {json.loads(l)["problem_id"]: json.loads(l) for l in open(TP / "byeolki" / "data" / "problems.jsonl")}
    model_of = {"Qwen_Qwen3.5-4B": "Qwen/Qwen3.5-4B", "Skywork_Skywork-OR1-Math-7B": "Skywork/Skywork-OR1-Math-7B",
                "deepseek-ai_DeepSeek-R1-0528-Qwen3-8B": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
    rows, stats = [], []
    for stem, mid in model_of.items():
        p = gdir / f"{stem}.jsonl"
        if not p.exists():
            continue
        by = defaultdict(lambda: defaultdict(list))
        raw = defaultdict(dict)
        for line in open(p):
            r = json.loads(line)
            by[r["problem_id"]][r["family"]].extend(r["correct"])
            raw[r["problem_id"]][r["family"]] = {k: r[k] for k in ("answers", "correct", "lengths", "truncated")}
        for q, fams in by.items():
            text = norm(probs[q]["problem"])
            orig = fams.get("original", [])
            base = sum(orig) / max(1, len(orig))
            fam_acc = {f: sum(v) / len(v) for f, v in fams.items() if f != "original" and v}
            drops = {f: base - a for f, a in fam_acc.items()}
            max_drop = max(drops.values()) if drops else None
            if base < args.min_base or len(fam_acc) < 3:
                label = None
            elif max_drop <= args.robust_max:
                label = True
            elif max_drop >= args.spurious_min:
                label = False
            else:
                label = None
            src = probs[q].get("source", "")
            row = {"case_id": f"byeolki:{stem}:{pid(text)}", "problem_id": pid(text), "model_id": mid,
                   "reasoning_effort": "default", "original_problem": text, "label": label,
                   "source": "byeolki", "origin": src, "answer": probs[q].get("answer"),
                   "base_accuracy": base, "family_acc": fam_acc, "max_drop": max_drop,
                   "n_base": len(orig)}
            rows.append(row)
            stats.append({"model_id": mid, "problem_id": pid(text), "origin": src, **{f"{f}": v for f, v in raw[q].items()}})
    write_jsonl(DATA / "external" / "byeolki_genstats.jsonl", stats)
    lab = [r for r in rows if r["label"] is not None]
    print(f"byeolki: {len(rows)} (model,problem) rows, {len(lab)} labelled "
          f"({sum(r['label'] for r in lab)} robust) with min_base={args.min_base} "
          f"robust<={args.robust_max} spurious>={args.spurious_min}")
    return rows


def leak_keys() -> set[str]:
    ks = set()
    p = TP / "qwen25math" / "evaluation" / "data" / "aime24" / "test.jsonl"
    if p.exists():
        ks |= {key(json.loads(l)["problem"]) for l in open(p)}
    else:
        print("WARNING: aime24 texts missing; AIME 2024 rows of aimo-validation-aime cannot be excluded")
    return ks


def build_pool(labeled_keys: set[str], official_keys: set[str], args) -> list[dict]:
    leak = leak_keys()
    cands = []
    for l in open(TP / "byeolki" / "data" / "problems.jsonl"):
        r = json.loads(l)
        cands.append({"problem": norm(r["problem"]), "answer": r["answer"], "origin": r["source"]})
    amc = TP / "qwen25math" / "evaluation" / "data" / "amc23" / "test.jsonl"
    if amc.exists():
        for l in open(amc):
            r = json.loads(l)
            cands.append({"problem": norm(r["question"]), "answer": r["answer"], "origin": "amc23"})
    math = TP / "qwen25math" / "evaluation" / "data" / "math" / "test.jsonl"
    if math.exists() and args.math_l5:
        rng = random.Random(0)
        m = [json.loads(l) for l in open(math)]
        m = [r for r in m if r.get("level") in ("Level 5", 5) and re.fullmatch(r"-?\d+", str(r.get("answer", "")).strip())]
        rng.shuffle(m)
        for r in m[: args.math_l5]:
            cands.append({"problem": norm(r["problem"]), "answer": int(r["answer"]), "origin": "math_l5"})
    out, seen, drop = [], set(), Counter()
    for c in cands:
        k = key(c["problem"])
        if c["origin"] in LEAK_SOURCES or k in leak:
            drop["leakage"] += 1
            continue
        if k in official_keys:
            drop["official sample (validation only)"] += 1
            continue
        if k in seen:
            drop["duplicate"] += 1
            continue
        try:
            ans = int(str(c["answer"]).strip())
        except ValueError:
            drop["non-integer answer"] += 1
            continue
        seen.add(k)
        out.append({"id": pid(c["problem"]), "problem": c["problem"], "answer": ans, "origin": c["origin"],
                    "has_labels": k in labeled_keys})
    print(f"pool: {len(out)} problems ({sum(p['has_labels'] for p in out)} already labelled); dropped {dict(drop)}; "
          f"by origin {dict(Counter(p['origin'] for p in out))}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-base", type=float, default=0.6)
    ap.add_argument("--robust-max", type=float, default=0.10)
    ap.add_argument("--spurious-min", type=float, default=0.25)
    ap.add_argument("--math-l5", type=int, default=80, help="MATH level-5 integer problems added to the pool")
    args = ap.parse_args()

    off = import_official_math()
    write_jsonl(LABELED / "official_math-agg.jsonl", off)
    by = import_byeolki(args)
    leak = leak_keys()
    for r in by:
        r["leak"] = r["origin"] in LEAK_SOURCES or key(r["original_problem"]) in leak
    write_jsonl(LABELED / "ext_byeolki.jsonl", by)
    print(f"byeolki rows flagged leak (AIME 2024-26): {sum(r['leak'] for r in by)}")
    pool = build_pool({key(r["original_problem"]) for r in by}, {key(r["original_problem"]) for r in off}, args)
    write_jsonl(POOL / "problems.jsonl", pool)


if __name__ == "__main__":
    main()
