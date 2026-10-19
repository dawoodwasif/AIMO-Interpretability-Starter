"""Prediction orchestration with hard time budgeting and graceful degradation.

Order of preference per (model, reasoning_effort) batch:
  1. internal heads (needs torch + cached model + enough time)
  2. text/meta heads (CPU, milliseconds)
  3. prior from the manifest (P(robust | model))
Optionally the final scores are turned into labels by batch rank instead of a
fixed threshold (``batch_rank``), which only uses the batch's own inputs.
Never raises; always returns len(problems) Python bools.
"""
from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from . import artifacts, clock, config, diagnostics, internals
from .text_features import FEATURE_FUNCS


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


def problem_text(p) -> str:
    """The harness passes strings; accept dicts/objects too (local tools, legacy)."""
    if isinstance(p, str):
        return p
    if isinstance(p, dict):
        for k in ("problem", "original_problem", "text"):
            if isinstance(p.get(k), str):
                return p[k]
        return ""
    for k in ("problem", "original_problem"):
        v = getattr(p, k, None)
        if isinstance(v, str):
            return v
    return str(p or "")


def to_labels(scores: np.ndarray, entry: dict, thr: float) -> list[bool]:
    """Threshold, or rank within the batch when the manifest asks for it."""
    scores = np.asarray(scores, dtype=np.float64)
    n = len(scores)
    if entry.get("batch_rank") and n >= int(entry.get("rank_min_n", 4)) and np.ptp(scores) > 1e-9:
        k = int(round(float(entry.get("rank_frac", 0.5)) * n))
        order = np.argsort(-scores, kind="stable")  # deterministic tie-break by position
        out = np.zeros(n, bool)
        out[order[:k]] = True
        return [bool(x) for x in out]
    return [bool(x >= thr) for x in scores]


def predict_batch(model_id: str, effort: str, problems) -> list[bool]:
    n = len(problems)
    if n == 0:
        return []
    t_start = time.monotonic()
    call = clock.register_call()
    share = clock.share_for_this_call()
    entry = artifacts.model_entry(model_id, effort)
    thr = float(entry.get("threshold", artifacts.manifest().get("threshold", 0.5)))
    prior_p = float(entry.get("prior", artifacts.manifest().get("default_prior", 0.5)))
    _log(f"call {call}: model={model_id} effort={effort} n={n} track={config.TRACK} "
         f"share={share:.0f}s elapsed={clock.elapsed():.0f}s")

    try:
        diagnostics.run_once(model_id, effort, n)
    except Exception:  # noqa: BLE001
        pass
    try:
        texts = [problem_text(p) for p in problems]
        heads = [(artifacts.load_head(h["file"]), float(h.get("weight", 1.0)))
                 for h in entry.get("heads", [])]
        heads = [(h, w) for h, w in heads if h is not None and w > 0]

        cheap = [(h, w) for h, w in heads if h.feature_set in FEATURE_FUNCS]
        deep = [(h, w) for h, w in heads if h.feature_set.startswith("internal:")]
        scores, weights = [], []

        for h, w in cheap:
            X = np.stack([FEATURE_FUNCS[h.feature_set](t) for t in texts])
            scores.append(h.proba(X))
            weights.append(w)

        behav = entry.get("behav")
        if (deep or behav) and internals.available():
            est = config.cost(config.LOAD_COST_S, model_id) + n * config.cost(config.PER_ITEM_COST_S, model_id)
            if est <= share:
                deadline = t_start + share
                tok = model = None
                try:
                    tok, model = internals.load_model(model_id)
                    if deep:
                        feats = internals.features_for([h.feature_set for h, _ in deep], tok, model, texts,
                                                       deadline=deadline, effort=effort)
                        for h, w in deep:
                            if h.feature_set in feats:
                                scores.append(h.proba(feats[h.feature_set]))
                                weights.append(w)
                    if behav:
                        from . import behaviour

                        bf = behaviour.run(tok, model, texts, deadline, kinds=tuple(behav.get("kinds", behaviour.DEFAULT_KINDS)),
                                           min_tokens=int(behav.get("min_tokens", 512)),
                                           max_tokens=int(behav.get("max_tokens", 16384)), model_id=model_id,
                                           effort=effort)
                        if bf is not None:
                            scores.append(behaviour.score(bf))
                            weights.append(float(behav.get("weight", 1.0)))
                except Exception as exc:  # noqa: BLE001
                    _log(f"internals failed for {model_id}: {exc!r}")
                finally:
                    if model is not None:
                        internals.unload(model)
            else:
                _log(f"skip internals for {model_id}: est {est:.0f}s > share {share:.0f}s")

        if scores:
            w = np.asarray(weights, dtype=np.float64)
            final = (np.stack(scores, 0) * w[:, None]).sum(0) / w.sum()
            preds = to_labels(final, entry, thr)
        else:
            preds = [bool(prior_p >= thr)] * n
    except Exception:  # noqa: BLE001
        _log("predict_batch error, falling back to prior:\n" + traceback.format_exc())
        preds = [bool(prior_p >= thr)] * n

    if len(preds) != n:
        preds = (list(preds) + [True] * n)[:n]
    _log(f"call {call} done in {time.monotonic() - t_start:.1f}s; robust={sum(preds)}/{n}")
    return [bool(x) for x in preds]
