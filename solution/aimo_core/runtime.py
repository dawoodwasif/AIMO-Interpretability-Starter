"""Prediction orchestration with hard time budgeting and graceful degradation.

Order of preference per model batch:
  1. internal heads (needs torch + cached model + enough time)
  2. text/meta heads (CPU, milliseconds)
  3. priors from the manifest (P(robust | model, perm type))
Never raises; always returns len(problems) Python bools.
"""
from __future__ import annotations

import sys
import time
import traceback

import numpy as np

from . import artifacts, clock, config, internals
from .text_features import FEATURE_FUNCS, perm_types


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


def _coerce(p) -> tuple[str, list[str]]:
    if isinstance(p, dict):
        text, types = p.get("original_problem", ""), p.get("permutation_type", [])
    else:
        text = getattr(p, "original_problem", "")
        types = getattr(p, "permutation_type", [])
    return str(text or ""), perm_types(types)


def _prior(entry: dict, types: list[str]) -> float:
    base = float(entry.get("prior", artifacts.manifest().get("default_prior", 0.5)))
    pp = entry.get("perm_prior") or {}
    vals = [float(pp[t]) for t in types if t in pp]
    if len(types) == 1 and vals:
        return vals[0]
    if len(types) > 1 and "__all__" in pp:
        return float(pp["__all__"])
    return base


def predict_batch(model_id: str, problems) -> list[bool]:
    n = len(problems)
    if n == 0:
        return []
    t_start = time.monotonic()
    call = clock.register_call()
    share = clock.share_for_this_call()
    entry = artifacts.model_entry(model_id)
    thr = float(entry.get("threshold", artifacts.manifest().get("threshold", 0.5)))
    _log(f"call {call}: model={model_id} n={n} share={share:.0f}s elapsed={clock.elapsed():.0f}s")

    try:
        items = [_coerce(p) for p in problems]
        prior = np.array([_prior(entry, t) for _, t in items], dtype=np.float64)
        heads = [(artifacts.load_head(h["file"]), float(h.get("weight", 1.0)))
                 for h in entry.get("heads", [])]
        heads = [(h, w) for h, w in heads if h is not None and w > 0]

        cheap = [(h, w) for h, w in heads if h.feature_set in FEATURE_FUNCS]
        deep = [(h, w) for h, w in heads if h.feature_set.startswith("internal:")]
        scores, weights = [], []

        for h, w in cheap:
            X = np.stack([FEATURE_FUNCS[h.feature_set](txt, t) for txt, t in items])
            scores.append(h.proba(X))
            weights.append(w)

        if deep and internals.available():
            est = config.cost(config.LOAD_COST_S, model_id) + n * config.cost(config.PER_ITEM_COST_S, model_id)
            if est <= share:
                deadline = time.monotonic() + share
                tok = model = None
                try:
                    tok, model = internals.load_model(model_id)
                    need = internals.layers_needed([h.feature_set for h, _ in deep])
                    feats = internals.extract(tok, model, [t for t, _ in items], layers=need, deadline=deadline)
                    for h, w in deep:
                        if h.feature_set in feats:
                            scores.append(h.proba(feats[h.feature_set]))
                            weights.append(w)
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
        else:
            final = prior
        preds = [bool(x >= thr) for x in final]
    except Exception:  # noqa: BLE001
        _log("predict_batch error, falling back to prior:\n" + traceback.format_exc())
        try:
            preds = [bool(_prior(entry, perm_types(getattr(p, "permutation_type", []))) >= thr)
                     for p in problems]
        except Exception:  # noqa: BLE001
            preds = [True] * n

    if len(preds) != n:
        preds = (preds + [True] * n)[:n]
    _log(f"call {call} done in {time.monotonic() - t_start:.1f}s; robust={sum(preds)}/{n}")
    return [bool(x) for x in preds]
