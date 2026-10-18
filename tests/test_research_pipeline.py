"""CPU tests of the GPU-side research scripts with a fake generator (no vLLM).

Catches bugs before a paid GPU run: two-stage sampling, checkpoint/resume,
labelling rule, per-effort file names, LLM-variant cleaning + filtering.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "solution"))


def _fake_generate(model_id, prompts, n=1, detail=False, **kw):
    """Original problems: always right. Variants containing 'Note'/'student'/'calculator'
    (distractors) are answered wrongly for problem p1 only."""
    out = []
    for m in prompts:
        text = m[0]["content"]
        ans = "7" if "seven" in text else "3"
        if "p1prob" in text and any(w in text for w in ("Note", "student", "calculator", "context", "booklet")):
            ans = "999"
        resp = [{"text": f"... so \\boxed{{{ans}}}", "n_tokens": 10, "truncated": False, "forced": False}] * n
        out.append(resp if detail else [d["text"] for d in resp])
    return out


@pytest.fixture
def env(tmp_path, monkeypatch):
    import common
    import gen_labels

    for mod in (common, gen_labels):
        monkeypatch.setattr(mod, "DATA", tmp_path, raising=False)
        monkeypatch.setattr(mod, "LABELED", tmp_path / "labeled", raising=False)
    monkeypatch.setattr(gen_labels, "generate", _fake_generate)
    probs = [{"id": "p0", "problem": "Alice has seven apples. How many apples? p0prob", "answer": 7},
             {"id": "p1", "problem": "Find $x$ if $x=3$. Bob agrees. p1prob", "answer": 3}]
    pool = tmp_path / "problems.jsonl"
    pool.write_text("".join(json.dumps(p) + "\n" for p in probs))
    return tmp_path, pool


def test_gen_labels_two_stage_and_resume(env, monkeypatch):
    import gen_labels
    import perturb

    tmp, pool = env
    var = tmp / "variants.jsonl"
    monkeypatch.setattr(sys, "argv", ["perturb.py", "--problems", str(pool), "--out", str(var), "--k", "2",
                                      "--types", "rename", "typos", "distract"])
    perturb.main()
    vs = [json.loads(l) for l in open(var)]
    assert {v["type"] for v in vs} >= {"rename", "distract"}
    argv = ["gen_labels.py", "--model", "org/m", "--problems", str(pool), "--variants", str(var),
            "--n-base", "4", "--n-var", "2", "--min-families", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    gen_labels.main()
    rows = {r["problem_id"]: r for r in map(json.loads, open(tmp / "labeled" / "self_org_m.jsonl"))}
    assert len(rows) == 2
    labs = {json.loads(l)["original_problem"][-6:]: json.loads(l)["label"] for l in open(tmp / "labeled" / "self_org_m.jsonl")}
    assert labs["p0prob"] is True and labs["p1prob"] is False  # distract breaks p1
    raw = tmp / "selfgen" / "org_m.jsonl"
    n_raw = len(raw.read_text().splitlines())
    with pytest.raises(SystemExit):  # refuses to overwrite without --resume
        gen_labels.main()
    monkeypatch.setattr(sys, "argv", argv + ["--resume"])
    gen_labels.main()  # nothing left to do
    assert len(raw.read_text().splitlines()) == n_raw
    monkeypatch.setattr(sys, "argv", argv + ["--effort", "low", "--limit", "1"])
    gen_labels.main()
    assert (tmp / "selfgen" / "org_m_low.jsonl").exists()
    r = json.loads(open(tmp / "labeled" / "self_org_m_low.jsonl").readline())
    assert r["reasoning_effort"] == "low"


def test_llm_variant_cleaning_and_filter():
    import perturb

    harmony = "<|channel|>analysis<|message|>hmm<|end|><|start|>assistant<|channel|>final<|message|>A farmer has 12 cows."
    assert perturb.clean_llm(harmony) == "A farmer has 12 cows."
    assert perturb.clean_llm("<think>x</think>**Rewritten problem:** Find 5.") == "Find 5."
    assert perturb.plausible("Find 12 plus 5.", "Compute the sum of 12 and 5.")
    assert not perturb.plausible("Find 12 plus 5.", "Compute the sum of 12 and 6.")  # number lost
    assert not perturb.plausible("Find 12 plus 5.", "Find 12 plus 5. The answer is 17 \\boxed{17}")


def test_llm_variants_flow_with_verification(env, monkeypatch):
    import generate as gen_mod
    import perturb

    tmp, pool = env

    def fake(model_id, prompts, n=1, **kw):
        out = []
        for m in prompts:
            c = m[0]["content"]
            if c.startswith("Solve"):  # verification: right only for the apples problem
                out.append(["\\boxed{7}" if "apples" in c else "\\boxed{0}"] * n)
            else:  # writer: harmony output keeping the numbers
                body = c.split("Problem:\n", 1)[1]
                out.append([f"<|channel|>final<|message|>In a new story: {body}"] * n)
        return out

    monkeypatch.setattr(gen_mod, "generate", fake)
    var = tmp / "v.jsonl"
    argv = ["perturb.py", "--problems", str(pool), "--out", str(var), "--types", "rephrase", "--k", "2", "--verify"]
    monkeypatch.setattr(sys, "argv", argv)
    perturb.main()
    vs = [json.loads(l) for l in open(var)]
    assert len(vs) == 2 and all(v["parent_id"] == "p0" and v["type"] == "rephrase" for v in vs)
    assert vs[0]["problem"].startswith("In a new story:") and "gpt-oss" in vs[0]["writer"]
    monkeypatch.setattr(sys, "argv", argv + ["--append"])
    perturb.main()  # p0 done, p1 retried and rejected again
    assert len([json.loads(l) for l in open(var)]) == 2
