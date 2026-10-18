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


CASES = [P("Find $1+1$.", ["typos"]), P("Let $n$ be prime. Find n mod 3.", ["rename", "distract"]),
         P("", []), P("x" * 20000, ["expert_no_solution"])]


@pytest.mark.parametrize("model_id", ["deepseek-ai/DeepSeek-R1-0528-Qwen3-8B", "openai/gpt-oss-120b",
                                      "unknown/model", ""])
def test_returns_python_bools_in_order(model_id):
    out = solution.are_robust(model_id, CASES)
    assert isinstance(out, list) and len(out) == len(CASES)
    assert all(type(x) is bool for x in out)


def test_empty_batch():
    assert solution.are_robust("openai/gpt-oss-120b", []) == []


def test_dict_and_string_inputs():
    out = solution.are_robust("x", [{"original_problem": "a", "permutation_type": '["typos"]'},
                                    P("b", "rename")])
    assert len(out) == 2 and all(type(x) is bool for x in out)


def test_deterministic():
    assert solution.are_robust("m", CASES) == solution.are_robust("m", CASES)


def test_text_features_stable_dim():
    a = text_features.featurize("short", ["typos"])
    b = text_features.featurize("a much longer problem $x^2$ " * 50, ["rename", "domain"])
    assert a.shape == b.shape and a.dtype == b.dtype


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
        assert solution.are_robust("m", CASES) == [True] * len(CASES)       # prior 0.9
        assert solution.are_robust("other", CASES) == [False] * len(CASES)  # default 0.2
    finally:
        artifacts.manifest.cache_clear()
        artifacts.load_head.cache_clear()


def test_manifest_heads_exist():
    m = json.loads((ROOT / "solution" / "artifacts" / "manifest.json").read_text())
    for e in m["models"].values():
        for h in e.get("heads", []):
            assert (ROOT / "solution" / "artifacts" / h["file"]).exists(), h["file"]
