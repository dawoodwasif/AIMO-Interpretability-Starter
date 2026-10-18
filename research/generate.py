"""Sampling backend for label generation (GPU machine). vLLM if present, else HF.

Decoding defaults mirror the organizers' label generation (website FAQ):
temperature=1.0, top_k=40, top_p=0.95. They allowed up to 100k tokens; we cap
lower for cost and force a final answer on truncation (document this choice).
"""
from __future__ import annotations

import re


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


def generate(model_id: str, prompts: list[list[dict]], n: int = 1, temperature: float = 1.0,
             top_p: float = 0.95, top_k: int = 40, max_tokens: int = 8192,
             thinking: bool | None = None, reasoning_effort: str | None = None) -> list[list[str]]:
    try:
        from vllm import LLM, SamplingParams
    except Exception:  # noqa: BLE001
        return _generate_hf(model_id, prompts, n, temperature, top_p, top_k, max_tokens, thinking, reasoning_effort)

    # NOTE: Skywork-OR1-Math-7B loops with fp8 KV cache -> keep kv_cache_dtype="auto" (bf16).
    llm = LLM(model=model_id, trust_remote_code=True, max_model_len=max_tokens + 4096,
              gpu_memory_utilization=0.92, kv_cache_dtype="auto")
    tok = llm.get_tokenizer()
    texts = [_chat(tok, m, thinking, reasoning_effort) for m in prompts]
    sp = SamplingParams(n=n, temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens, seed=0)
    outs = llm.generate(texts, sp)
    return [[o.text for o in r.outputs] for r in outs]


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
        res.append([tok.decode(x[enc["input_ids"].shape[1]:], skip_special_tokens=True) for x in g])
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
