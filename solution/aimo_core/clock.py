"""Run-level wall clock that survives one-process-per-case ingestion.

Codabench (Oct 2026) calls are_robust once per CASE, apparently in a fresh
process each time (per-case timings are constant; see docs/COMPETITION.md), so a
module-level clock only sees one case. We persist the run start time and the
number of problems already handled in a small JSON file in the temp dir, keyed
by the parent (ingestion) process id. In a single-process harness this degrades
to the same behaviour as a module clock.

Budget per call = min(n * CASE_TARGET_S[model],
                      n * remaining_run_time / max(1, EXPECTED_CASES - problems_done))
"""
from __future__ import annotations

import json
import os
import tempfile
import time

from . import config

_T0 = time.monotonic()
_calls = 0
_STATE = os.path.join(tempfile.gettempdir(), f"aimo_run_{os.getppid()}.json")
_STALE_S = 6 * 3600  # an older state file belongs to a previous run


def _load() -> dict:
    try:
        with open(_STATE) as f:
            s = json.load(f)
        if time.time() - float(s.get("t0", 0)) < _STALE_S:
            return s
    except (OSError, ValueError, TypeError):
        pass
    return {"t0": time.time(), "done": 0, "calls": 0}


def _save(s: dict) -> None:
    try:
        tmp = _STATE + f".{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(s, f)
        os.replace(tmp, _STATE)
    except OSError:
        pass


def elapsed() -> float:
    """Seconds since the first case of this run (falls back to this process)."""
    s = _load()
    return max(time.monotonic() - _T0, time.time() - float(s["t0"]))


def remaining() -> float:
    """Seconds left in the run before the safety margin is hit."""
    return config.TOTAL_BUDGET_S - config.SAFETY_MARGIN_S - elapsed()


def register_call(n: int = 1) -> int:
    """Count this call; returns the run-level call index."""
    global _calls
    _calls += 1
    s = _load()
    s["calls"] = int(s.get("calls", 0)) + 1
    s.setdefault("done", 0)
    s["_pending"] = n
    _save(s)
    return s["calls"]


def finish_call(n: int) -> None:
    s = _load()
    s["done"] = int(s.get("done", 0)) + n
    _save(s)


def problems_done() -> int:
    return int(_load().get("done", 0))


def budget_for_call(model_id: str, n: int) -> float:
    """Seconds this call may use (load + features + generation)."""
    left_cases = max(1, config.EXPECTED_CASES - problems_done())
    fair = max(0.0, remaining()) * min(n, left_cases) / left_cases
    return max(0.0, min(n * config.cost(config.CASE_TARGET_S, model_id), fair))


def share_for_this_call() -> float:
    """Legacy: fair share per model batch (kept for older callers/tests)."""
    calls_left = max(1, config.EXPECTED_MODEL_CALLS - _calls + 1)
    return max(0.0, remaining() / calls_left)
