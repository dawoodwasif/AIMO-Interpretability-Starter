"""Codabench entry point for the AIMO Interpretability Challenge.

Contract (from aimo-interp/getting-started):
    are_robust(model_id: str, problems: list[Problem]) -> list[bool]
  * exactly one prediction per problem, same order
  * real Python ``bool`` values (not numpy / int)
  * True = robust, False = spurious / not robust

Everything heavy lives in ``aimo_core``; this file must stay tiny and must
never raise. ``aimo_core.runtime`` already falls back to priors on any error,
and the outer try/except below is a last line of defence.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


@dataclass(frozen=True)
class Problem:
    original_problem: str
    permutation_type: list[str]


def are_robust(model_id: str, problems: list[Problem]) -> list[bool]:
    try:
        from aimo_core import runtime

        preds = runtime.predict_batch(model_id, problems)
        if len(preds) == len(problems):
            return [bool(p) for p in preds]
    except BaseException as exc:  # noqa: BLE001 - never crash the ingestion run
        print(f"[aimo] FATAL in are_robust: {exc!r}", file=sys.stderr)
    return [True for _ in problems]
