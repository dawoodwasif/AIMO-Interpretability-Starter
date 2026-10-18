"""Contract tests: these MUST pass before any bundle is built or uploaded."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))
os.environ["AIMO_DISABLE_INTERNALS"] = "1"

import solution  # noqa: E402
from aimo_core import linear, text_features  # noqa: E402


@dataclass(frozen=True)
class P:
    original_problem: str
    permutation_type: list


# The harness passes plain strings (getting-started ingestion.py).
CASES = ["Find $1+1$.", "Let $n$ be prime. Find n mod 3.", "", "x" * 20000]


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "openai/gpt-oss-120b",
                                      "unknown/model", ""])
@pytest.mark.parametrize("effort", ["default", "low", "medium"])
def test_returns_python_bools_in_order(model_id, effort):
    # exactly how ingestion.py calls a solution that names reasoning_effort
    out = solution.are_robust(model_id, reasoning_effort=effort, problems=CASES)
    assert type(out) is list and len(out) == len(CASES)
    assert all(type(x) is bool for x in out)


def test_signature_is_detected_by_ingestion():
    import inspect

    names = [p.name for p in inspect.signature(solution.are_robust).parameters.values()]
    assert names[:3] == ["model_id", "reasoning_effort", "problems"]


def test_legacy_positional_call():
    out = solution.are_robust("openai/gpt-oss-120b", CASES)
    assert type(out) is list and len(out) == len(CASES) and all(type(x) is bool for x in out)


def test_empty_batch():
    assert solution.are_robust("openai/gpt-oss-120b", reasoning_effort="low", problems=[]) == []


def test_dict_and_object_inputs():
    out = solution.are_robust("x", "default", [{"problem": "a"}, P("b", ["rename"])])
    assert len(out) == 2 and all(type(x) is bool for x in out)


def test_problem_text_extraction():
    from aimo_core.runtime import problem_text

    assert problem_text("abc") == "abc"
    assert problem_text({"original_problem": "q"}) == "q"
    assert problem_text(P("z", [])) == "z"


def test_deterministic():
    assert solution.are_robust("m", "default", CASES) == solution.are_robust("m", "default", CASES)


def test_text_features_stable_dim():
    a = text_features.featurize("short")
    b = text_features.featurize("a much longer problem $x^2$ " * 50)
    assert a.shape == b.shape and a.dtype == b.dtype


def test_batch_rank_labels():
    import numpy as np
    from aimo_core.runtime import to_labels

    s = np.array([0.1, 0.9, 0.5, 0.7])
    assert to_labels(s, {"batch_rank": True}, 0.5) == [False, True, False, True]
    assert to_labels(s, {}, 0.6) == [False, True, False, True]
    assert to_labels(np.ones(4), {"batch_rank": True}, 0.5) == [True] * 4  # ties -> threshold


def test_linear_head_roundtrip(tmp_path):
    import numpy as np

    d = 7
    h = linear.LinearHead("t", "meta", np.zeros(d), np.ones(d), np.arange(d, dtype=float), 0.3,
                          proj=None, meta={"k": 1})
    p = tmp_path / "h.npz"
    h.save(str(p))
    h2 = linear.LinearHead.load(str(p))
    X = np.random.RandomState(0).randn(4, d)
    assert np.allclose(h.proba(X), h2.proba(X))


def test_internal_head_without_torch_falls_back(tmp_path, monkeypatch):
    """A manifest with only an internal head must still answer (prior) when internals are off."""
    import numpy as np
    from aimo_core import artifacts, config

    linear.LinearHead("deep", "internal:last:L3", np.zeros(4), np.ones(4), np.ones(4), 0.0).save(
        str(tmp_path / "deep.npz"))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"default_prior": 0.2, "threshold": 0.5,
         "models": {"m": {"prior": 0.9, "heads": [{"file": "deep.npz", "weight": 1}]}}}))
    monkeypatch.setattr(config, "ARTIFACT_DIR", str(tmp_path))
    artifacts.manifest.cache_clear()
    artifacts.load_head.cache_clear()
    try:
        assert solution.are_robust("m", "default", CASES) == [True] * len(CASES)       # prior 0.9
        assert solution.are_robust("other", "default", CASES) == [False] * len(CASES)  # default 0.2
    finally:
        artifacts.manifest.cache_clear()
        artifacts.load_head.cache_clear()


def test_manifest_heads_exist():
    m = json.loads((ROOT / "solution" / "artifacts" / "manifest.json").read_text())
    for e in m["models"].values():
        for h in e.get("heads", []):
            assert (ROOT / "solution" / "artifacts" / h["file"]).exists(), h["file"]


def test_effort_specific_manifest_entry(tmp_path, monkeypatch):
    from aimo_core import artifacts, config

    (tmp_path / "manifest.json").write_text(json.dumps(
        {"default_prior": 0.2, "threshold": 0.5,
         "models": {"g": {"prior": 0.9}, "g@low": {"prior": 0.1}}}))
    monkeypatch.setattr(config, "ARTIFACT_DIR", str(tmp_path))
    artifacts.manifest.cache_clear()
    try:
        assert solution.are_robust("g", "medium", CASES) == [True] * len(CASES)
        assert solution.are_robust("g", "low", CASES) == [False] * len(CASES)
        assert solution.are_robust("qwen3-8b:low", "default", CASES) == [False] * len(CASES)  # alias -> '*'/default
    finally:
        artifacts.manifest.cache_clear()


@pytest.mark.parametrize("small", [False, True])
def test_official_ingestion_end_to_end(small):
    """Run the vendored Codabench ingestion+scoring on the fixture for both tracks."""
    import subprocess

    cmd = [sys.executable, str(ROOT / "scripts" / "run_local.py"), str(ROOT / "solution"),
           "--cases", str(ROOT / "data" / "fixtures" / "cases.jsonl"), "--no-internals", "--time-limit", "300"]
    if small:
        cmd.append("--small")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=320)
    assert r.returncode == 0, r.stderr[-2000:]
    res = json.loads(r.stdout)
    assert res["coverage"] == 1.0 and res["invalid_predictions"] == 0
    assert all(("gpt-oss" in m) is (not small) or not small for m in res["per_model"])
    if small:
        assert not any("gpt-oss" in m for m in res["per_model"])
