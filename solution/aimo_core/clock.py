"""Global wall clock for the whole prediction run (all models share 3600 s)."""
from __future__ import annotations

import time

from . import config

_T0 = time.monotonic()
_calls = 0


def elapsed() -> float:
    return time.monotonic() - _T0


def remaining() -> float:
    """Seconds left before the safety margin is hit."""
    return config.TOTAL_BUDGET_S - config.SAFETY_MARGIN_S - elapsed()


def register_call() -> int:
    global _calls
    _calls += 1
    return _calls


def share_for_this_call() -> float:
    """Fair share of the remaining budget for the current model batch."""
    calls_left = max(1, config.EXPECTED_MODEL_CALLS - _calls + 1)
    return max(0.0, remaining() / calls_left)
