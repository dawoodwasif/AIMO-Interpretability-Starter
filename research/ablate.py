"""Route comparison + White-Box ablation per model (run when GPU caches exist).

For every target (model[@effort]) with labeled rows, computes OUT-OF-FOLD scores
(GroupKFold by problem, repeated over seeds; everything trainable is fit inside
the training folds only) for these routes:

  prior          constant (per-model base rate)                 -> chance on balanced batches
  behav          test-time self-consistency, no training        (black-box, H9)
  behav_frame    + domain-style frame copy                      (H10)
  solve_probe    ridge(own activations -> base accuracy)        (white-box, H8)
                 layer/pool chosen by inner CV on the training folds
  nll_shift      logistic on internal:nll + internal:shift      (white-box, H3/H4)
  combo          mean rank of solve_probe and behav             (full system)

Metrics per route: within-model AUC and batch-rank accuracy on balanced batches
(can be inflated by problem difficulty), and the TEST-SHAPED metrics pooled over
models on mixed problems: within-problem pairwise accuracy across models and
`test_sim` (per-model batch-rank labelling of mixed rows). Decide on test_sim.
White-Box ablation = combo vs behav (white-box removed) and vs solve_probe
(black-box removed), on test_sim. Results -> results/<exp>.json + a markdown table.

Usage:
    python research/ablate.py --exp EXP-010 [--seeds 3] [--sources self byeolki official_math]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.model_selection import GroupKFold

from common import RESULTS, load_labeled
from train import (auc, balanced_batches, build_X, dedupe, head, load_cache, mixed_problems, solve_head,
                   target_of)

PROBE_SETS_REL = (0.5, 0.75)  # relative depths tried for the solve probe (mean + last pooling)


def pairwise(rows, s, mixed):
    by = defaultdict(list)
    for r, v in zip(rows, s):
        if r["problem_id"] in mixed:
            by[r["problem_id"]].append((r["label"], v))
    hit = tot = 0.0
    for v in by.values():
        for a in v:
            for b in v:
                if a[0] and not b[0]:
                    tot += 1
                    hit += 1.0 if a[1] > b[1] else 0.5 if a[1] == b[1] else 0.0
    return (round(hit / tot, 4), int(tot)) if tot else (None, 0)


def probe_sets(cache):
    names = sorted(cache[1])
    layers = sorted({int(n.rsplit("L", 1)[1]) for n in names if ":L" in n})
    if not layers:
        return []
    L = max(layers)
    pick = sorted({min(layers, key=lambda x: abs(x - d * L)) for d in PROBE_SETS_REL})
    return [f"internal:{p}:L{l}" for l in pick for p in ("mean", "last") if f"internal:{p}:L{l}" in cache[1]]


def fit_probe(fs, tr_rows, extra, cache, alpha=100.0, pca=16):
    X = build_X(fs, tr_rows, cache)
    Xe = build_X(fs, extra, cache) if extra else None
    Xr = X if Xe is None else np.concatenate([X, Xe], 0)
    b = np.array([r["base_accuracy"] for r in tr_rows] + [r["base_accuracy"] for r in (extra if Xe is not None else [])])
    return solve_head("p", fs, Xr, b, pca, alpha, 0.0)


def oof(rows, unl, cache, seed, folds=5):
    y = np.array([r["label"] for r in rows], bool)
    g = np.array([r["problem_id"] for r in rows])
    rng = np.random.RandomState(seed)
    perm = {p: rng.rand() for p in set(g)}
    order = np.argsort([perm[p] for p in g], kind="stable")  # shuffle groups across folds
    rows_o, y_o, g_o = [rows[i] for i in order], y[order], g[order]
    out = {k: np.full(len(rows), np.nan) for k in ("behav", "behav_frame", "solve_probe", "nll_shift")}
    from aimo_core.behaviour import score as bscore

    for name in ("behav", "behav_frame"):
        X = build_X(name, rows_o, cache)
        if X is not None:
            out[name] = bscore(X)
    k = min(folds, len(set(g_o)))
    psets = probe_sets(cache)
    have_base = all(r.get("base_accuracy") is not None for r in rows_o)
    for tr, te in GroupKFold(n_splits=k).split(np.zeros(len(y_o)), y_o, g_o):
        test_p = set(g_o[te])
        tr_rows = [rows_o[i] for i in tr]
        te_rows = [rows_o[i] for i in te]
        extra = [r for r in unl if r["problem_id"] not in test_p]
        if psets and have_base:  # inner selection on training folds only
            best, best_auc = None, -1
            gi = np.array([r["problem_id"] for r in tr_rows])
            yi = np.array([r["label"] for r in tr_rows], bool)
            for fs in psets:
                s_in = np.zeros(len(tr_rows))
                for itr, ite in GroupKFold(n_splits=min(3, len(set(gi)))).split(s_in, yi, gi):
                    ip = set(gi[ite])
                    h = fit_probe(fs, [tr_rows[i] for i in itr], [r for r in extra if r["problem_id"] not in ip], cache)
                    s_in[ite] = h.decision(build_X(fs, [tr_rows[i] for i in ite], cache))
                a = auc(yi, s_in) or 0
                if a > best_auc:
                    best, best_auc = fs, a
            h = fit_probe(best, tr_rows, extra, cache)
            out["solve_probe"][te] = h.decision(build_X(best, te_rows, cache))
        Xn = build_X("internal:nll+internal:shift", tr_rows, cache)
        if Xn is not None and len(set(y_o[tr])) == 2:
            h = head("n", "x", Xn, y_o[tr], 0.03, 16)
            out["nll_shift"][te] = h.proba(build_X("internal:nll+internal:shift", te_rows, cache))
    inv = np.argsort(order)
    res = {k: v[inv] for k, v in out.items() if not np.isnan(v).all()}
    if "solve_probe" in res and "behav" in res:
        from scipy.stats import rankdata

        res["combo"] = (rankdata(res["solve_probe"]) + rankdata(res["behav"])) / (2 * len(rows))
    res["prior"] = np.zeros(len(rows))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--allow-leak", action="store_true")
    args = ap.parse_args()

    rows = [r for r in load_labeled(sources=args.sources) if args.allow_leak or not r.get("leak")]
    rows = dedupe(rows)
    unl = [r for r in load_labeled(sources=args.sources, include_unlabeled=True)
           if r.get("label") is None and (args.allow_leak or not r.get("leak"))]
    mixed = mixed_problems(rows)
    report, lines = {}, ["| target | n | route | AUC (within model) | balanced-batch rank acc |", "|---|---|---|---|---|"]
    pooled = defaultdict(list)  # route -> [(row, score)] across targets (for test-shaped metrics)
    for t in sorted({target_of(r) for r in rows}):
        tr = [r for r in rows if target_of(r) == t]
        cache = load_cache(t)
        if cache is None or len({r["label"] for r in tr}) < 2:
            continue
        tr = [r for r in tr if build_X(sorted(cache[1])[0], [r], cache) is not None]
        tu = [r for r in unl if target_of(r) == t and build_X(sorted(cache[1])[0], [r], cache) is not None]
        y = np.array([r["label"] for r in tr], bool)
        agg = defaultdict(list)
        for seed in range(args.seeds):
            for k, s in oof(tr, tu, cache, seed).items():
                agg[k].append(s)
        report[t] = {}
        for k, ss in agg.items():
            s = np.mean(ss, 0)
            pooled[k] += list(zip(tr, s))
            bb = balanced_batches(y, s, thr=np.median(s) + 1e-12)
            m = {"auc": auc(y, s) if k != "prior" else 0.5, "bb_rank": bb["rank"] if (bb and k != "prior") else 0.5}
            report[t][k] = m
            lines.append(f"| {t} | {len(tr)} | {k} | {m['auc']} | {m['bb_rank']} |")
    # test-shaped metrics on MIXED problems, pooled over models:
    #   pairwise: robust model's score > spurious model's score on the same problem
    #   test_sim: label each model's mixed rows by batch rank (top half robust), accuracy
    lines += ["", "| route | mixed pairwise across models (pairs) | test_sim acc on mixed rows (n) |", "|---|---|---|"]
    report["_pooled"] = {}
    for k, rs in pooled.items():
        rr = [r for r, _ in rs]
        sc = np.array([v for _, v in rs])
        pw, npairs = pairwise(rr, sc, mixed)
        hit = n = 0
        for t in {target_of(r) for r in rr}:
            idx = [i for i, r in enumerate(rr) if target_of(r) == t and r["problem_id"] in mixed]
            if len(idx) < 2:
                continue
            order = sorted(idx, key=lambda i: -sc[i])
            top = set(order[: len(idx) // 2]) if k != "prior" else set()
            hit += sum((i in top) == bool(rr[i]["label"]) for i in idx)
            n += len(idx)
        ts = 0.5 if k == "prior" else (round(hit / n, 4) if n else None)  # prior = chance on a balanced test
        report["_pooled"][k] = {"mixed_pairwise": pw, "pairs": npairs, "test_sim": ts, "n": n}
        lines.append(f"| {k} | {pw} ({npairs}) | {report['_pooled'][k]['test_sim']} ({n}) |")
    P = report["_pooled"]
    if "combo" in P and "behav" in P and P["combo"]["test_sim"] is not None:
        report["white_box_ablation"] = {
            "test_sim_combo_minus_behav": round(P["combo"]["test_sim"] - P["behav"]["test_sim"], 4),
            "test_sim_combo_minus_probe": round(P["combo"]["test_sim"] - P["solve_probe"]["test_sim"], 4)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{args.exp}.json").write_text(json.dumps({"args": vars(args), "report": report}, indent=1))
    (RESULTS / f"{args.exp}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("White-Box ablation:", report.get("white_box_ablation"))


if __name__ == "__main__":
    main()
