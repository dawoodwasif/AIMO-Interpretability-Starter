"""Codabench entry point for the AIMO Interpretability Challenge.

Official contract (aimo-interp/getting-started, components/ingestion_program):
    are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]
  * called once per (model_id, reasoning_effort) batch, with keyword arguments
    ``reasoning_effort=`` and ``problems=`` because our signature names them
  * ``problems`` are plain problem strings (no perturbation type is given)
  * reasoning_effort is "default", "low" or "medium" (gpt-oss)
  * must return a ``list`` of exactly len(problems) real Python ``bool``s
  * True = robust, False = not robust

The legacy positional call ``are_robust(model_id, problems)`` is also handled.
Everything heavy lives in ``aimo_core``; this file must stay tiny and never raise.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def are_robust(model_id: str, reasoning_effort: str = "default", problems: list | None = None,
               **kwargs) -> list[bool]:
    if problems is None and isinstance(reasoning_effort, (list, tuple)):
        # legacy positional call: are_robust(model_id, problems)
        problems, reasoning_effort = reasoning_effort, "default"
    if problems is None:
        problems = kwargs.get("problems") or []
    problems = list(problems)
    try:
        from aimo_core import runtime

        preds = runtime.predict_batch(model_id, str(reasoning_effort or "default"), problems)
        if len(preds) == len(problems):
            return [bool(p) for p in preds]
    except BaseException as exc:  # noqa: BLE001 - never crash the ingestion run
        print(f"[aimo] FATAL in are_robust: {exc!r}", file=sys.stderr)
    return [True for _ in problems]
