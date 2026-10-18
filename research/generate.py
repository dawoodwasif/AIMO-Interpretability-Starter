"""Sampling backend for label generation (GPU machine). vLLM if present, else HF.

Decoding defaults mirror the organizers' label generation (website FAQ):
temperature=1.0, top_k=40, top_p=0.95. They allowed up to 100k tokens; we cap
lower for cost and, for a truncated sample, run one short greedy "forced
answer" continuation (close the reasoning, ask for \\boxed{}). This choice is
recorded in every raw row (`truncated`, `forced`) and must be reported.
"""
from __future__ import annotations

import re

_LLM = {}
FORCE_SUFFIX = "\n\n</think>\n\nConsidering the limited time, my final answer is \\boxed{"
FORCE_TOKENS = 24


def _chat(tok, messages, thinking: bool | None, reasoning_effort: str | None) -> str:
    kw = {}
    if thinking is not None:
        kw["enable_thinking"] = thinking
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    except Exception:  # noqa: BLE001
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _vllm(model_id: str, max_len: int):
    from vllm import LLM

    if model_id not in _LLM:
        _LLM.clear()  # one model on the GPU at a time
        # NOTE: Skywork-OR1-Math-7B loops with fp8 KV cache -> keep kv_cache_dtype="auto" (bf16).
        _LLM[model_id] = LLM(model=model_id, trust_remote_code=True, max_model_len=max_len,
                             gpu_memory_utilization=0.92, kv_cache_dtype="auto", seed=0)
    return _LLM[model_id]


def generate(model_id: str, prompts: list[list[dict]], n: int = 1, temperature: float = 1.0,
             top_p: float = 0.95, top_k: int = 40, max_tokens: int = 8192,
             thinking: bool | None = None, reasoning_effort: str | None = None,
             force_answer: bool = True, detail: bool = False):
    """Returns, per prompt, a list of n completions (str), or dicts when detail=True:
    {text, n_tokens, truncated, forced}."""
    try:
        import vllm  # noqa: F401
    except Exception:  # noqa: BLE001
        res = _generate_hf(model_id, prompts, n, temperature, top_p, top_k, max_tokens, thinking, reasoning_effort)
        return res if detail else [[d["text"] for d in r] for r in res]

    from vllm import SamplingParams

    llm = _vllm(model_id, max_tokens + 4096)
    tok = llm.get_tokenizer()
    texts = [_chat(tok, m, thinking, reasoning_effort) for m in prompts]
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, seed=0)
    outs = llm.generate(texts, sp)
    res = [[{"text": o.text, "n_tokens": len(o.token_ids), "truncated": o.finish_reason == "length",
             "forced": False} for o in r.outputs] for r in outs]
    if force_answer:
        todo = [(i, j) for i, r in enumerate(res) for j, d in enumerate(r) if d["truncated"]]
        if todo:
            fp = [texts[i] + res[i][j]["text"] + FORCE_SUFFIX for i, j in todo]
            fo = llm.generate(fp, SamplingParams(n=1, temperature=0.0, max_tokens=FORCE_TOKENS))
            for (i, j), o in zip(todo, fo):
                res[i][j]["text"] += FORCE_SUFFIX + o.outputs[0].text
                res[i][j]["forced"] = True
    return res if detail else [[d["text"] for d in r] for r in res]


def _generate_hf(model_id, prompts, n, temperature, top_p, top_k, max_tokens, thinking, reasoning_effort):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto").eval()
    res = []
    for m in prompts:
        enc = tok(_chat(tok, m, thinking, reasoning_effort), return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            g = model.generate(**enc, do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k,
                               max_new_tokens=max_tokens, num_return_sequences=n)
        L = enc["input_ids"].shape[1]
        res.append([{"text": tok.decode(x[L:], skip_special_tokens=True), "n_tokens": int((x[L:] != tok.pad_token_id).sum()),
                     "truncated": int((x[L:] != tok.pad_token_id).sum()) >= max_tokens, "forced": False} for x in g])
    return res


_BOX = "\\boxed{"


def extract_answer(text: str) -> str | None:
    """Content of the LAST \\boxed{...} (brace-matched), normalised."""
    i = text.rfind(_BOX)
    if i < 0:
        m = re.findall(r"(?i)final answer\s*(?:is)?\s*[:=]?\s*\$?([-\d,./]+)", text)
        return _norm(m[-1]) if m else None
    j, depth = i + len(_BOX), 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return _norm(text[i + len(_BOX): j - 1])


def _norm(a: str) -> str:
    a = a.strip().strip("$").replace(",", "").replace(" ", "")
    a = re.sub(r"\\text\{(.*?)\}", r"\1", a)
    a = a.replace("\\!", "").rstrip(".")
    try:
        f = float(a)
        if f.is_integer():
            return str(int(f))
    except ValueError:
        pass
    return a


def is_correct(response: str, answer) -> bool:
    pred = extract_answer(response)
    if answer is None:  # unanswerable variant: credit explicit refusal
        return bool(re.search(r"(?i)no (valid )?solution|cannot be determined|not enough information|ill-posed", response))
    return pred is not None and pred == _norm(str(answer))
