"""End-to-end check of aimo_core.internals on a tiny random Llama (CPU, seconds).

Skipped when torch/transformers are not installed. Verifies shapes, the
problem-span NLL, shift features, determinism and the runtime path that loads
a model and applies an internal head.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "solution"))

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

PROBLEMS = [
    "Let $n$ be the number of ordered pairs $(a,b)$ of integers with $a+b=10$. Find n.",
    "Triangle $ABC$ has $AB=13$ and $BC=14$. Point D lies on side BC. Find $BD$.",
    "Alice has $x$ apples and Bob has $y$ apples. If $x+y=7$, what is x?",
]
CHAT = ("{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}\n{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}")


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

    d = tmp_path_factory.mktemp("tiny")
    tk = Tokenizer(models.BPE(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tr = trainers.BpeTrainer(vocab_size=400, special_tokens=["<unk>", "<pad>", "<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tk.train_from_iterator(PROBLEMS * 20 + ["Solve the following math problem. boxed"], tr)
    fast = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", pad_token="<pad>",
                                   eos_token="<eos>")
    fast.chat_template = CHAT
    fast.save_pretrained(d)
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=fast.vocab_size, hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                      pad_token_id=1, eos_token_id=2)
    LlamaForCausalLM(cfg).save_pretrained(d)
    return str(d)


def test_extract_shapes_and_determinism(tiny_model_dir):
    from aimo_core import internals

    tok, model = internals.load_model(tiny_model_dir, device="cpu")
    f1 = internals.extract(tok, model, PROBLEMS, layers=None, batch_size=2, shift=True)
    f2 = internals.extract(tok, model, PROBLEMS, layers=None, batch_size=3, shift=True)
    assert f1["internal:last:L4"].shape == (3, 32)
    assert f1["internal:nll"].shape == (3, len(internals.NLL_NAMES))
    assert f1["internal:logits"].shape == (3, 4)
    assert f1["internal:shift"].shape[0] == 3 and np.isfinite(f1["internal:shift"]).all()
    # padding / batch size must not change features (left padding, masked mean)
    for k in ("internal:nll", "internal:last:L2", "internal:shift"):
        assert np.allclose(f1[k], f2[k], atol=1e-4), k
    # NLL is over the problem span only: positive and finite
    assert (f1["internal:nll"][:, 0] > 0).all()


def test_features_for_concat(tiny_model_dir):
    from aimo_core import internals

    tok, model = internals.load_model(tiny_model_dir, device="cpu")
    f = internals.features_for(["internal:nll+internal:shift", "internal:mean:L2"], tok, model, PROBLEMS)
    assert f["internal:nll+internal:shift"].shape[1] == f["internal:nll"].shape[1] + f["internal:shift"].shape[1]
    assert "internal:mean:L2" in f


def test_runtime_uses_internal_head(tiny_model_dir, tmp_path, monkeypatch):
    from aimo_core import artifacts, config, internals, linear, runtime

    tok, model = internals.load_model(tiny_model_dir, device="cpu")
    d = internals.extract(tok, model, PROBLEMS, layers=[], shift=False)["internal:nll"].shape[1]
    linear.LinearHead("h", "internal:nll", np.zeros(d), np.ones(d), -np.eye(d)[0], 0.0).save(str(tmp_path / "h.npz"))
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"threshold": 0.5, "models": {tiny_model_dir: {"prior": 0.0, "batch_rank": True,
                                                       "heads": [{"file": "h.npz", "weight": 1}]}}}))
    monkeypatch.setattr(config, "ARTIFACT_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DISABLE_INTERNALS", False)
    monkeypatch.setattr(internals, "load_model", lambda mid, device=None: (tok, model))
    artifacts.manifest.cache_clear()
    artifacts.load_head.cache_clear()
    try:
        out = runtime.predict_batch(tiny_model_dir, "default", PROBLEMS + PROBLEMS[:1])
        assert len(out) == 4 and all(type(x) is bool for x in out)
        assert sum(out) == 2  # batch_rank with rank_frac 0.5
    finally:
        artifacts.manifest.cache_clear()
        artifacts.load_head.cache_clear()
