"""Train per-model heads with problem-grouped CV and export bundle artifacts.

For each target model (plus '*' = shared fallback over all rows):
  * candidate feature sets: meta, text, and every internal:* set cached in
    data/cache/features/<model_slug>.npz (optionally PCA-reduced)
  * candidates are scored by GroupKFold CV (groups = problem_id) on a
    SELECTION split; the winner is then scored once on a held-out split of
    problems it never saw (honest estimate; selection over many layers
    inflates CV accuracy)
  * a head is exported only if it beats the prior baseline on selection CV
    by --min-gain; otherwise the model falls back to priors
Writes solution/artifacts/{manifest.json, *.npz} and results/<exp>.json.

Usage:
    python research/train.py --exp EXP-001 --sources official self
    python research/train.py --exp EXP-002 --models deepseek-ai/DeepSeek-R1-0528-Qwen3-8B --pca 32
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from common import ARTIFACTS, CACHE, RESULTS, load_labeled, slug, text_hash
from aimo_core.config import KNOWN_MODELS
from aimo_core.linear import LinearHead
from aimo_core.text_features import FEATURE_FUNCS

warnings.filterwarnings("ignore")
CS = [0.01, 0.1, 1.0]


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
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Z, y)
        coef, b = clf.coef_[0], float(clf.intercept_[0])
    return mean, scale, proj, coef, b


def _head(name, fs, X, y, C, pca_k) -> LinearHead:
    mean, scale, proj, coef, b = _fit(X, y, C, pca_k)
    return LinearHead(name=name, feature_set=fs, mean=mean, scale=scale, coef=coef, intercept=b, proj=proj,
                      meta={"C": C, "pca": pca_k, "n": int(len(y))})


def _metrics(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    acc = float((y == pred).mean())
    tpr = float(pred[y].mean()) if y.any() else 0.0
    tnr = float((~pred[~y]).mean()) if (~y).any() else 0.0
    return {"acc": round(acc, 4), "bal_acc": round((tpr + tnr) / 2, 4), "n": int(len(y))}


def cv(X, y, groups, C, pca_k, folds):
    k = min(folds, len(set(groups)))
    if k < 2:
        return None
    pred = np.zeros(len(y), bool)
    for tr, te in GroupKFold(n_splits=k).split(X, y, groups):
        h = _head("cv", "x", X[tr], y[tr], C, pca_k)
        pred[te] = h.proba(X[te]) >= 0.5
    return _metrics(y, pred)


def build_X(fs, rows, cache):
    if fs in FEATURE_FUNCS:
        return np.stack([FEATURE_FUNCS[fs](r["original_problem"], r["permutation_type"]) for r in rows]).astype(np.float64)
    keys, arr = cache
    idx = {k: i for i, k in enumerate(keys)}
    ii = [idx.get(text_hash(r["original_problem"])) for r in rows]
    if any(i is None for i in ii):
        return None
    return arr[fs][ii].astype(np.float64)


def load_cache(model_id):
    p = CACHE / "features" / f"{slug(model_id)}.npz"
    if not p.exists():
        return None
    z = np.load(p)
    return [str(k) for k in z["keys"]], {k: z[k] for k in z.files if k.startswith("internal:")}


def priors(rows):
    y = np.array([r["label"] for r in rows], float)
    prior = float((y.sum() + 1) / (len(y) + 2)) if len(y) else 0.5
    by = defaultdict(list)
    for r in rows:
        key = r["permutation_type"][0] if len(r["permutation_type"]) == 1 else "__all__"
        by[key].append(r["label"])
    pp = {k: round((sum(v) + prior * 2) / (len(v) + 2), 4) for k, v in by.items()}
    return round(prior, 4), pp


def prior_pred(rows, prior, pp):
    out = []
    for r in rows:
        t = r["permutation_type"]
        p = pp.get(t[0], prior) if len(t) == 1 else pp.get("__all__", prior)
        out.append(p >= 0.5)
    return np.array(out)


def train_target(target, rows, args, layer_stride):
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

    prior, pp = priors(sel)
    base = _metrics(y_sel, prior_pred(sel, prior, pp))
    report = {"target": target, "n_rows": len(rows), "n_problems": len(set(groups_all)),
              "label_balance": round(float(np.mean([r["label"] for r in rows])), 3),
              "prior_baseline_sel": base, "candidates": []}

    cache = load_cache(target) if target != "*" else None
    fsets = ["meta", "text"]
    if cache:
        names = sorted(cache[1], key=lambda s: (s.split(":")[1], int(s.rsplit("L", 1)[1]) if ":L" in s else -1))
        fsets += [n for n in names if ":L" not in n or int(n.rsplit("L", 1)[1]) % layer_stride == 0]

    best = None
    for fs in fsets:
        X = build_X(fs, sel, cache)
        if X is None:
            continue
        for C in CS:
            for pk in ([0] if X.shape[1] <= 64 else [args.pca]):
                m = cv(X, y_sel, g_sel, C, pk, args.folds)
                if m is None:
                    continue
                cand = {"feature_set": fs, "C": C, "pca": pk, **m}
                report["candidates"].append(cand)
                if best is None or (m["acc"], m["bal_acc"]) > (best["acc"], best["bal_acc"]):
                    best = cand

    report["best"] = best
    entry = {"prior": prior, "perm_prior": pp, "heads": []}
    if best and best["acc"] >= base["acc"] + args.min_gain:
        Xs = build_X(best["feature_set"], sel, cache)
        h_sel = _head("sel", best["feature_set"], Xs, y_sel, best["C"], best["pca"])
        if ho:
            y_ho = np.array([r["label"] for r in ho], bool)
            Xh = build_X(best["feature_set"], ho, cache)
            report["holdout"] = {"head": _metrics(y_ho, h_sel.proba(Xh) >= 0.5),
                                 "prior": _metrics(y_ho, prior_pred(ho, prior, pp))}
        # refit on ALL rows for export
        Xa = build_X(best["feature_set"], rows, cache)
        ya = np.array([r["label"] for r in rows], bool)
        fname = f"{slug(target) if target != '*' else 'shared'}__{best['feature_set'].replace(':', '_')}.npz"
        if not args.dry_run:
            _head(f"{target}:{best['feature_set']}", best["feature_set"], Xa, ya, best["C"], best["pca"]).save(
                str(ARTIFACTS / fname))
        entry["heads"].append({"file": fname, "weight": 1.0})
    entry["prior"], entry["perm_prior"] = priors(rows)  # export priors from all rows
    return entry, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiment id, e.g. EXP-003")
    ap.add_argument("--sources", nargs="*", default=None, help="official / self (default: all)")
    ap.add_argument("--models", nargs="*", default=KNOWN_MODELS)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--holdout", type=float, default=0.25)
    ap.add_argument("--pca", type=int, default=32)
    ap.add_argument("--layer-stride", type=int, default=2)
    ap.add_argument("--min-gain", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="report only; do not touch artifacts")
    args = ap.parse_args()

    rows = load_labeled(sources=args.sources)
    if not rows:
        raise SystemExit("no labeled rows in data/labeled/ - run research/fetch_data.py or gen_labels.py")
    print(f"{len(rows)} labeled rows; by model: {Counter(r['model_id'] for r in rows).most_common()}")

    if not args.dry_run:
        for f in ARTIFACTS.glob("*.npz"):
            f.unlink()
    manifest = {"version": 1, "exp": args.exp, "created": time.strftime("%Y-%m-%d %H:%M"),
                "default_prior": 0.5, "threshold": 0.5, "models": {}}
    reports = []
    targets = [m for m in args.models if any(r["model_id"] == m for r in rows)] + ["*"]
    for t in targets:
        trows = rows if t == "*" else [r for r in rows if r["model_id"] == t]
        entry, rep = train_target(t, trows, args, args.layer_stride)
        manifest["models"][t] = entry
        reports.append(rep)
        b = rep.get("best") or {}
        print(f"[{t}] rows={rep['n_rows']} problems={rep['n_problems']} balance={rep['label_balance']} "
              f"prior={rep['prior_baseline_sel']['acc']} best={b.get('feature_set')} cv={b.get('acc')} "
              f"holdout={rep.get('holdout')} exported={bool(entry['heads'])}")
    manifest["default_prior"] = manifest["models"]["*"]["prior"]

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{args.exp}.json"
    out.write_text(json.dumps({"args": vars(args), "reports": reports}, indent=1, default=str))
    if not args.dry_run:
        (ARTIFACTS / "manifest.json").write_text(json.dumps(manifest, indent=1))
        print(f"artifacts -> {ARTIFACTS}")
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
