"""Test-time self-consistency (H9): does the model give the same answer on the
original problem and on answer-preserving perturbed copies?

For each problem we decode (greedy, deterministic) the original plus the shipped
rule-based copies (aimo_core.perturb: rename / typos / distract). A trace that
hits the token cap is closed and forced to answer (a short greedy continuation).
Features per problem:
  agree   share of completions giving the modal answer (gold-free)
  parsed  share of completions with a parseable answer
  len     mean generated length / max_new_tokens
  trunc   share of truncated traces
Score (no training needed): agree - LEN_PENALTY * len, in [~-0.3, 1].
Offline evidence (EXP-002): on byeolki samples, agreement over a few samples
on the original ranks robustness within a problem (pairwise .82-.91), and
shorter reasoning is more often robust.

Time: the caller passes a deadline; max_new_tokens is chosen from it and from the
measured per-sequence decode speed (config.DECODE_TOK_S). Memory: sequences are
chunked so the KV cache fits in the currently free VRAM (the GPU is shared).
"""
from __future__ import annotations

import re
import sys
import time
from collections import Counter

import numpy as np

from . import config, internals, perturb

FORCE_SUFFIX = "\n\n</think>\n\nConsidering the limited time, my final answer is \\boxed{"
# gpt-oss (harmony format): close the analysis channel, open the final channel
FORCE_SUFFIX_HARMONY = "<|end|><|start|>assistant<|channel|>final<|message|>The final answer is \\boxed{"
FORCE_TOKENS = 16
LEN_PENALTY = 0.25
FEATURE_NAMES = ["agree", "parsed", "len", "trunc"]


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


_BOX = "\\boxed{"


def extract_answer(text: str):
    i = text.rfind(_BOX)
    if i < 0:
        return None
    j, depth = i + len(_BOX), 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    a = text[i + len(_BOX): j - 1 if depth == 0 else j]
    a = a.strip().strip("$").replace(",", "").replace(" ", "").replace("\\!", "").rstrip(".")
    a = re.sub(r"\\text\{(.*?)\}", r"\1", a)
    try:
        f = float(a)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return a or None


def kv_bytes_per_token(model) -> float:
    cfg = getattr(model.config, "text_config", None) or model.config
    L = int(getattr(cfg, "num_hidden_layers", 32))
    kvh = int(getattr(cfg, "num_key_value_heads", None) or getattr(cfg, "num_attention_heads", 32))
    hd = int(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    return 2.0 * L * kvh * hd * 2  # K and V, bf16


def _generate(tok, model, prompts: list[str], max_new: int, chunk: int, deadline: float | None = None):
    import torch

    outs = []
    for s in range(0, len(prompts), chunk):
        enc = tok(prompts[s:s + chunk], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        kw = {}
        if deadline is not None:  # hard wall-clock stop: a slow GPU yields shorter traces, never a timeout
            left = deadline - time.monotonic()
            kw["max_time"] = max(1.0, left / max(1, -(-(len(prompts) - s) // chunk)))
        with torch.inference_mode():
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id, **kw)
        L = enc["input_ids"].shape[1]
        for row in g[:, L:]:
            ids = row.tolist()
            n = len(ids)
            while n and ids[n - 1] == tok.pad_token_id:
                n -= 1
            outs.append((tok.decode(ids[:n], skip_special_tokens=True), n, tok.decode(ids[:n], skip_special_tokens=False)))
        del g, enc
    return outs


def features_from_raw(raw: list[dict], kinds=None) -> np.ndarray:
    """[agree, parsed, len, trunc] per problem from per-copy results; `kinds` selects
    which perturbed copies to use (the original is always included)."""
    out = []
    for r in raw:
        sel = [i for i, k in enumerate(r["kinds"]) if k == "original" or kinds is None or k in kinds]
        ans = [r["answers"][i] for i in sel]
        got = [a for a in ans if a is not None]
        per = len(sel)
        agree = Counter(got).most_common(1)[0][1] / per if got else 0.0
        out.append([agree, len(got) / per, float(np.mean([r["lens"][i] for i in sel])) / r["max_new"],
                    float(np.mean([r["trunc"][i] for i in sel]))])
    return np.asarray(out, dtype=np.float64)


def run(tok, model, problems: list[str], deadline: float, kinds=("rename", "typos", "distract"),
        min_tokens: int = 512, max_tokens: int = 16384, model_id: str = "",
        effort: str | None = None, return_raw: bool = False):
    """Return features [n, 4] (or per-copy raw dicts if return_raw) or None if there is
    not enough time for min_tokens."""
    import torch

    texts = [p for prob in problems for p in [prob] + [v for _, v in perturb.variants(prob, kinds, k=1)]]
    per = 1 + len(kinds)
    prompts = [internals.format_prompt(tok, t, effort) for t in texts]
    left = deadline - time.monotonic()
    tps = config.cost(config.DECODE_TOK_S, model_id)
    # decode speed per sequence is ~flat in batch size on a large GPU; keep 35% slack
    max_new = int(min(max_tokens, 0.65 * left * tps))
    if max_new < min_tokens:
        _log(f"behaviour: skip, only {left:.0f}s left -> {max_new} tokens < {min_tokens}")
        return None
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 8e9
    plen = max(len(tok(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    per_seq = kv_bytes_per_token(model) * (plen + max_new + FORCE_TOKENS) * 1.3
    chunk = int(max(1, min(len(prompts), (0.6 * free) // per_seq)))
    t0 = time.monotonic()
    gens = _generate(tok, model, prompts, max_new, chunk, deadline=deadline - 15.0)
    # a trace is "truncated" if it hit the cap or the wall-clock stop (no answer yet)
    trunc = [g[1] >= max_new or extract_answer(g[0]) is None for g in gens]
    todo = [i for i, t in enumerate(trunc) if t]
    suffix = FORCE_SUFFIX_HARMONY if "gpt-oss" in model_id else FORCE_SUFFIX
    if todo and time.monotonic() < deadline:
        forced = _generate(tok, model, [prompts[i] + gens[i][2] + suffix for i in todo], FORCE_TOKENS, chunk)
        for i, f in zip(todo, forced):
            gens[i] = (gens[i][0] + "\\boxed{" + f[0], gens[i][1], "")
    _log(f"behaviour: {len(prompts)} seqs, max_new={max_new}, chunk={chunk}, {time.monotonic() - t0:.0f}s, "
         f"trunc={np.mean(trunc):.2f}")
    raw = []
    for i in range(len(problems)):
        g = gens[i * per:(i + 1) * per]
        raw.append({"kinds": ["original"] + list(kinds), "answers": [extract_answer(x[0]) for x in g],
                    "lens": [int(x[1]) for x in g], "trunc": [bool(t) for t in trunc[i * per:(i + 1) * per]],
                    "max_new": int(max_new)})
    return raw if return_raw else features_from_raw(raw)


def score(feats: np.ndarray) -> np.ndarray:
    """Robustness score mapped to [0, 1]."""
    return (feats[:, 0] - LEN_PENALTY * feats[:, 2] + LEN_PENALTY) / (1.0 + LEN_PENALTY)
