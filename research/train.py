"""Train per-model heads with problem-grouped CV, score once on a holdout, export.

Rows = one per eval case (model_id, problem, reasoning_effort), from data/labeled/.
For each target (model[@effort]) and '*' (shared fallback over all rows):
  * candidate feature sets: meta, text (problem-only), and every internal:* set in
    data/cache/features/<model_slug>.npz (+ "internal:nll+internal:shift")
  * selection by GroupKFold CV (groups = problem_id) on the SELECTION split;
    the winner is scored ONCE on held-out problems
  * metrics next to: majority (per target, from the selection split), and on
    MIXED problems (problems with both labels across models; mimics the balanced
    test), plus a balanced-batch simulation (threshold vs batch-rank labelling)
  * solve-rate confound: AUC of base_accuracy alone, corr(head score, base_accuracy)
  * export rule (CLAUDE.md §5): holdout acc >= majority holdout + --min-gain AND
    CV acc > majority CV. Otherwise the target ships its prior only.
Writes solution/artifacts/{manifest.json, *.npz} and results/<exp>.json.

Usage:
    python research/train.py --exp EXP-001 --dry-run
    python research/train.py --exp EXP-004 --sources self byeolki official_math --pca 16
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from common import ARTIFACTS, CACHE, RESULTS, load_labeled, slug, text_hash
from aimo_core.config import KNOWN_MODELS
from aimo_core.linear import LinearHead
from aimo_core.text_features import FEATURE_FUNCS

warnings.filterwarnings("ignore")
CS = [0.003, 0.03, 0.3]
SOURCE_RANK = {"self": 0, "official": 1, "official_math": 2, "byeolki": 3}


def target_of(r) -> str:
    e = r.get("reasoning_effort") or "default"
    return r["model_id"] if e == "default" else f"{r['model_id']}@{e}"


def dedupe(rows):
    """One row per (target, problem); prefer self > official > official_math > byeolki."""
    best = {}
    for r in rows:
        k = (target_of(r), r["problem_id"])
        if k not in best or SOURCE_RANK.get(r["source"], 9) < SOURCE_RANK.get(best[k]["source"], 9):
            best[k] = r
    return list(best.values())


def mixed_problems(rows) -> set:
    labs = defaultdict(set)
    for r in rows:
        labs[r["problem_id"]].add(bool(r["label"]))
    return {p for p, s in labs.items() if len(s) == 2}


def _fit(X, y, C, pca_k):
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-6] = 1.0
    Z = (X - mean) / scale
    proj = None
    if pca_k and Z.shape[1] > pca_k and Z.shape[0] > pca_k:
        p = PCA(n_components=pca_k, random_state=0).fit(Z)
        proj = p.components_.T
        Z = Z @ proj
    if len(set(y)) < 2:
        coef, b = np.zeros(Z.shape[1]), float(np.log((y.mean() + 1e-3) / (1 - y.mean() + 1e-3)))
    else:
        clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Z, y)
        coef, b = clf.coef_[0], float(clf.intercept_[0])
    return mean, scale, proj, coef, b


def head(name, fs, X, y, C, pca_k) -> LinearHead:
    mean, scale, proj, coef, b = _fit(X, y, C, pca_k)
    return LinearHead(name=name, feature_set=fs, mean=mean, scale=scale, coef=coef, intercept=b, proj=proj,
                      meta={"C": C, "pca": pca_k, "n": int(len(y))})


def metrics(y, pred, groups=None, mixed=None):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    out = {"acc": round(float((y == pred).mean()), 4) if len(y) else None, "n": int(len(y))}
    tpr = float(pred[y].mean()) if y.any() else 0.0
    tnr = float((~pred[~y]).mean()) if (~y).any() else 0.0
    out["bal_acc"] = round((tpr + tnr) / 2, 4)
    if groups is not None and mixed:
        m = np.array([g in mixed for g in groups])
        out["mixed_acc"] = round(float((y[m] == pred[m]).mean()), 4) if m.any() else None
        out["mixed_n"] = int(m.sum())
    return out


def balanced_batches(y, score, thr=0.5, size=8, reps=400, seed=0):
    """Simulate per-model-balanced test batches: threshold vs batch-rank accuracy."""
    y = np.asarray(y, bool)
    pos, neg = np.where(y)[0], np.where(~y)[0]
    if len(pos) < 2 or len(neg) < 2:
        return None
    rng = np.random.RandomState(seed)
    h = size // 2
    acc_t, acc_r = [], []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(pos, min(h, len(pos)), replace=len(pos) < h),
                              rng.choice(neg, min(h, len(neg)), replace=len(neg) < h)])
        s, yy = np.asarray(score)[idx], y[idx]
        acc_t.append(float(((s >= thr) == yy).mean()))
        order = np.argsort(-s, kind="stable")
        pr = np.zeros(len(idx), bool)
        pr[order[: len(idx) // 2]] = True
        acc_r.append(float((pr == yy).mean()))
    return {"threshold": round(float(np.mean(acc_t)), 4), "rank": round(float(np.mean(acc_r)), 4)}


def solve_head(name, fs, X, base, pca_k, alpha, shift) -> LinearHead:
    """H8: ridge X -> base accuracy; decision = prediction - shift (monotone in predicted solve rate)."""
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-6] = 1.0
    Z = (X - mean) / scale
    proj = None
    if pca_k and Z.shape[1] > pca_k and Z.shape[0] > pca_k:
        proj = PCA(n_components=pca_k, random_state=0).fit(Z).components_.T
        Z = Z @ proj
    r = Ridge(alpha=alpha).fit(Z, base)
    return LinearHead(name=name, feature_set=fs, mean=mean, scale=scale, coef=r.coef_ * 8.0,
                      intercept=float((r.intercept_ - shift) * 8.0), proj=proj,
                      meta={"kind": "solve_rate_probe", "alpha": alpha, "pca": pca_k, "n": int(len(base))})


def solve_cv(fs, sel, extra, cache, alpha, pca_k, folds):
    """Grouped CV for H8. Regression rows = labeled selection rows + unlabeled rows (any
    base_accuracy) of the same target whose problem is in the training folds."""
    Xl = build_X(fs, sel, cache)
    Xe = build_X(fs, extra, cache) if extra else None
    if Xl is None:
        return None, None
    y = np.array([r["label"] for r in sel], bool)
    g = np.array([r["problem_id"] for r in sel])
    ge = np.array([r["problem_id"] for r in extra]) if extra else np.array([])
    k = min(folds, len(set(g)))
    if k < 2:
        return None, None
    score = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(Xl, y, g):
        test_p = set(g[te])
        X_tr, b_tr = Xl[tr], [sel[i]["base_accuracy"] for i in tr]
        if Xe is not None and len(ge):
            m = np.array([p not in test_p for p in ge])
            X_tr = np.concatenate([X_tr, Xe[m]], 0)
            b_tr = b_tr + [extra[i]["base_accuracy"] for i in np.where(m)[0]]
        h0 = solve_head("cv", fs, X_tr, np.array(b_tr, float), pca_k, alpha, 0.0)
        shift = float(np.median(h0.decision(Xl[tr]) / 8.0))
        score[te] = solve_head("cv", fs, X_tr, np.array(b_tr, float), pca_k, alpha, shift).proba(Xl[te])
    return metrics(y, score >= 0.5), score


def cv(X, y, groups, C, pca_k, folds):
    k = min(folds, len(set(groups)))
    if k < 2:
        return None, None
    score = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        score[te] = head("cv", "x", X[tr], y[tr], C, pca_k).proba(X[te])
    return metrics(y, score >= 0.5), score


def _npz(path, prefix):
    z = np.load(path)
    keys = [str(k) for k in z["keys"]]
    return {k: {n: z[n][i].astype(np.float64) for n in z.files if n.startswith(prefix)} for i, k in enumerate(keys)}


def load_cache(target):
    """(key -> row index, {feature_set: matrix}) merging internals and behaviour caches."""
    mid, _, eff = target.partition("@")
    per_key = {}
    for suffix, prefix in (("", "internal:"), ("_behav", "behav")):
        for name in ([f"{slug(mid)}_{eff}{suffix}"] if eff else []) + [f"{slug(mid)}{suffix}"]:
            p = CACHE / "features" / f"{name}.npz"
            if p.exists():
                for k, d in _npz(p, prefix).items():
                    per_key.setdefault(k, {}).update(d)
                break
    if not per_key:
        return None
    combos = [("internal:nll", "internal:shift"), ("behav", "internal:nll"), ("behav", "internal:shift")]
    names = sorted({n for d in per_key.values() for n in d})
    # keep only keys that have every feature set (rows missing one are skipped by build_X)
    keys = [k for k, d in per_key.items() if all(n in d for n in names)]
    idx = {k: i for i, k in enumerate(keys)}
    feats = {n: np.stack([per_key[k][n] for k in keys]) for n in names}
    for a, b in combos:
        if a in feats and b in feats:
            feats[f"{a}+{b}"] = np.concatenate([feats[a], feats[b]], 1)
    return idx, feats


def build_X(fs, rows, cache):
    if fs in FEATURE_FUNCS:
        return np.stack([FEATURE_FUNCS[fs](r["original_problem"]) for r in rows]).astype(np.float64)
    if cache is None or fs not in cache[1]:
        return None
    ii = [cache[0].get(text_hash(r["original_problem"])) for r in rows]
    if any(i is None for i in ii):
        return None
    return cache[1][fs][ii]


def auc(y, s):
    y = np.asarray(y, bool)
    if y.all() or (~y).all():
        return None
    return round(float(roc_auc_score(y, s)), 4)


def train_target(target, rows, mixed, args, unlabeled=()):
    groups_all = np.array([r["problem_id"] for r in rows])
    uniq = sorted(set(groups_all))
    rng = np.random.RandomState(args.seed)
    rng.shuffle(uniq)
    n_hold = int(round(len(uniq) * args.holdout)) if len(uniq) >= 10 else 0
    hold = set(uniq[:n_hold])
    sel = [r for r in rows if r["problem_id"] not in hold]
    ho = [r for r in rows if r["problem_id"] in hold]
    y_sel = np.array([r["label"] for r in sel], bool)
    g_sel = np.array([r["problem_id"] for r in sel])
    y_ho = np.array([r["label"] for r in ho], bool)
    g_ho = [r["problem_id"] for r in ho]

    prior = float((y_sel.sum() + 1) / (len(y_sel) + 2)) if len(y_sel) else 0.5
    maj = prior >= 0.5
    rep = {"target": target, "n_rows": len(rows), "n_problems": len(uniq), "n_holdout_rows": len(ho),
           "label_balance": round(float(np.mean([r["label"] for r in rows])), 3),
           "mixed_rows": int(sum(r["problem_id"] in mixed for r in rows)),
           "majority_cv": metrics(y_sel, np.full(len(y_sel), maj)),
           "majority_holdout": metrics(y_ho, np.full(len(y_ho), maj), g_ho, mixed),
           "candidates": []}
    # per-model prior baseline (differs from majority only for the shared '*' target)
    per = defaultdict(list)
    for r in sel:
        per[target_of(r)].append(r["label"])
    pm = {k: (sum(v) + 1) / (len(v) + 2) >= 0.5 for k, v in per.items()}
    rep["model_prior_holdout"] = metrics(y_ho, np.array([pm.get(target_of(r), maj) for r in ho], bool), g_ho, mixed)
    base = [r.get("base_accuracy") for r in ho]
    if ho and all(b is not None for b in base):
        rep["base_acc_auc_holdout"] = auc(y_ho, base)  # oracle-ish: needs sampling

    cache = load_cache(target) if target != "*" else None
    fsets = ["meta", "text"] + (sorted(cache[1]) if cache else [])
    best = None
    for fs in fsets:
        X = build_X(fs, sel, cache)
        if X is None:
            continue
        for C in CS:
            for pk in ([0] if X.shape[1] <= 64 else [args.pca]):
                m, _ = cv(X, y_sel, g_sel, C, pk, args.folds)
                if m is None:
                    continue
                cand = {"feature_set": fs, "C": C, "pca": pk, **m}
                rep["candidates"].append(cand)
                if best is None or (m["bal_acc"], m["acc"]) > (best["bal_acc"], best["acc"]):
                    best = cand
    # H8 solve-rate probes on internal feature sets (need base_accuracy on rows)
    extra_sel = [r for r in unlabeled if r["problem_id"] not in hold]
    if cache and args.solve_probe and all(r.get("base_accuracy") is not None for r in sel):
        for fs in sorted(cache[1]):
            for alpha in (10.0, 100.0, 1000.0):
                pk = 0 if cache[1][fs].shape[1] <= 64 else args.pca
                m, _ = solve_cv(fs, sel, extra_sel, cache, alpha, pk, args.folds)
                if m is None:
                    continue
                cand = {"feature_set": fs, "kind": "solve", "C": alpha, "pca": pk, **m}
                rep["candidates"].append(cand)
                if best is None or (m["bal_acc"], m["acc"]) > (best["bal_acc"], best["acc"]):
                    best = cand
    rep["best"] = best
    entry = {"prior": round(prior, 4), "heads": []}

    def fit_best(train_rows, extra_rows):
        fs = best["feature_set"]
        X = build_X(fs, train_rows, cache)
        if best.get("kind") != "solve":
            return head("sel", fs, X, np.array([r["label"] for r in train_rows], bool), best["C"], best["pca"])
        Xr = np.concatenate([X] + ([build_X(fs, extra_rows, cache)] if extra_rows else []), 0)
        b = np.array([r["base_accuracy"] for r in list(train_rows) + list(extra_rows)], float)
        h0 = solve_head("sel", fs, Xr, b, best["pca"], best["C"], 0.0)
        return solve_head(f"{target}:solve:{fs}", fs, Xr, b, best["pca"], best["C"],
                          float(np.median(h0.decision(X) / 8.0)))

    if best and ho:
        h = fit_best(sel, extra_sel)
        s_ho = h.proba(build_X(best["feature_set"], ho, cache))
        rep["holdout"] = metrics(y_ho, s_ho >= 0.5, g_ho, mixed)
        rep["holdout_auc"] = auc(y_ho, s_ho)
        rep["balanced_batches"] = balanced_batches(y_ho, s_ho)
        if all(b is not None for b in base) and len(set(base)) > 1:
            rep["corr_score_base_acc"] = round(float(np.corrcoef(s_ho, base)[0, 1]), 4)
        ref = max(rep["majority_holdout"]["acc"], rep["model_prior_holdout"]["acc"])
        keep = (rep["holdout"]["acc"] >= ref + args.min_gain
                and best["acc"] > rep["majority_cv"]["acc"])
        if target == "*" and not args.shared_heads:
            keep = False
        rep["export"] = bool(keep)
        if keep:
            kind = "solve__" if best.get("kind") == "solve" else ""
            fname = f"{slug(target) if target != '*' else 'shared'}__{kind}{best['feature_set'].replace(':', '_').replace('+', '-')}.npz"
            if not args.dry_run:
                fit_best(rows, list(unlabeled)).save(str(ARTIFACTS / fname))
            entry["heads"].append({"file": fname, "weight": 1.0})
            if args.batch_rank:
                entry["batch_rank"] = True
    entry["prior"] = round(float((sum(r["label"] for r in rows) + 1) / (len(rows) + 2)), 4)
    return entry, rep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id, e.g. EXP-003")
    ap.add_argument("--sources", nargs="*", default=None, help="self official official_math byeolki (default all)")
    ap.add_argument("--models", nargs="*", default=None, help="targets (model or model@effort); default all known")
    ap.add_argument("--allow-leak", action="store_true", help="keep rows flagged by the organizers' leakage advice")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.3)
    ap.add_argument("--pca", type=int, default=16)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--batch-rank", action="store_true", help="exported heads label by batch rank (H5)")
    ap.add_argument("--solve-probe", action="store_true", help="H8: add ridge solve-rate probe candidates")
    ap.add_argument("--shared-heads", action="store_true",
                    help="allow exporting a head for the shared '*' target (off: EXP-001 showed the shared "
                         "problem-only head wins only through a dataset-source confound)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not touch artifacts")
    args = ap.parse_args()

    rows = load_labeled(sources=args.sources)
    if not args.allow_leak:
        rows = [r for r in rows if not r.get("leak")]
    rows = dedupe(rows)
    unl = [r for r in load_labeled(sources=args.sources, include_unlabeled=True)
           if r.get("label") is None and (args.allow_leak or not r.get("leak"))]
    if not rows:
        raise SystemExit("no labeled rows in data/labeled/")
    mixed = mixed_problems(rows)
    print(f"{len(rows)} rows, {len({r['problem_id'] for r in rows})} problems, {len(mixed)} mixed; "
          f"by target {Counter(target_of(r) for r in rows).most_common()}")

    if not args.dry_run:
        for f in ARTIFACTS.glob("*.npz"):
            f.unlink()
    manifest = {"version": 1, "exp": args.exp, "created": time.strftime("%Y-%m-%d %H:%M"),
                "default_prior": 0.5, "threshold": 0.5, "models": {}}
    reports = []
    known = args.models or sorted({target_of(r) for r in rows if r["model_id"] in KNOWN_MODELS})
    for t in known + ["*"]:
        trows = rows if t == "*" else [r for r in rows if target_of(r) == t]
        if not trows:
            continue
        tun = [] if t == "*" else [r for r in unl if target_of(r) == t]
        entry, rep = train_target(t, trows, mixed, args, tun)
        manifest["models"][t] = entry
        reports.append(rep)
        b = rep.get("best") or {}
        print(f"[{t}] rows={rep['n_rows']} mixed_rows={rep['mixed_rows']} bal={rep['label_balance']} "
              f"maj_cv={rep['majority_cv']['acc']} best={b.get('feature_set')}(C={b.get('C')}) cv={b.get('acc')}/{b.get('bal_acc')} "
              f"| holdout {rep.get('holdout')} vs maj {rep['majority_holdout']['acc']} model-prior {rep['model_prior_holdout']['acc']} auc={rep.get('holdout_auc')} "
              f"base_auc={rep.get('base_acc_auc_holdout')} bb={rep.get('balanced_batches')} export={rep.get('export')}")
    manifest["default_prior"] = manifest["models"]["*"]["prior"]
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.exp}.json"
    out.write_text(json.dumps({"args": vars(args), "n_rows": len(rows), "n_mixed": len(mixed), "reports": reports},
                              indent=1, default=str))
    if not args.dry_run:
        (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"artifacts -> {ARTIFACTS}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
