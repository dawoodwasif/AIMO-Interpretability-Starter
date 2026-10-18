"""White-box features from a single prompt-only forward pass.

Feature-set names produced here:
  internal:last:L{i}   hidden state of the final prompt token after layer i
  internal:mean:L{i}   mean of hidden states over prompt tokens after layer i
  internal:logits      [entropy, top1 prob, top5 mass, logit margin] of the
                       next-token distribution at the end of the prompt

The SAME function is used offline (research/extract_features.py) and inside
the bundle, so trained heads see identical inputs.

torch/transformers are imported lazily so the module imports fine on CPU-only
machines without them (the runtime then skips internals).
"""
from __future__ import annotations

import gc
import sys
import time

import numpy as np

from . import config


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


def available() -> bool:
    if config.DISABLE_INTERNALS:
        return False
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def load_model(model_id: str, device: str | None = None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = dict(local_files_only=True, device_map=device, low_cpu_mem_usage=True)
    try:  # transformers >= 5 uses `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", **kwargs)
    model.eval()
    _log(f"loaded {model_id} on {device} in {time.monotonic() - t0:.1f}s")
    return tok, model


def unload(model) -> None:
    try:
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def format_prompt(tok, problem: str) -> str:
    msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": problem}]
    for m in (msgs, [{"role": "user", "content": config.SYSTEM_PROMPT + "\n\n" + problem}]):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        except Exception:  # noqa: BLE001 - template may reject system role / be missing
            continue
    return config.SYSTEM_PROMPT + "\n\n" + problem + "\n"


def extract(tok, model, problems: list[str], layers: list[int] | None = None,
            batch_size: int | None = None, deadline: float | None = None) -> dict[str, np.ndarray]:
    """Return {feature_set_name: array[n, d]} for every requested layer.

    layers=None -> all layers (use offline to cache everything once).
    deadline: time.monotonic() value after which we stop and raise TimeoutError.
    """
    import torch

    batch_size = batch_size or config.FORWARD_BATCH
    texts = [format_prompt(tok, p) for p in problems]
    out: dict[str, list[np.ndarray]] = {}
    dev = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("internals extraction exceeded its time share")
        chunk = texts[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=config.MAX_PROMPT_TOKENS, add_special_tokens=False)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            res = model(**enc, output_hidden_states=True, use_cache=False)
        hs = res.hidden_states  # tuple(len = n_layers + 1)
        idx = range(len(hs)) if layers is None else [i for i in layers if 0 <= i < len(hs)]
        mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
        for i in idx:
            h = hs[i]
            last = h[:, -1, :].float().cpu().numpy()  # left padding -> last pos is real
            mean = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
            out.setdefault(f"internal:last:L{i}", []).append(last)
            out.setdefault(f"internal:mean:L{i}", []).append(mean)
        logits = res.logits[:, -1, :].float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        top = torch.topk(p, 5, dim=-1).values
        tl = torch.topk(logits, 2, dim=-1).values
        stats = torch.stack([ent, top[:, 0], top.sum(-1), tl[:, 0] - tl[:, 1]], -1)
        out.setdefault("internal:logits", []).append(stats.cpu().numpy())
        del res, hs
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in out.items()}


def layers_needed(feature_sets: list[str]) -> list[int]:
    idx = set()
    for fs in feature_sets:
        if fs.startswith("internal:") and ":L" in fs:
            idx.add(int(fs.rsplit(":L", 1)[1]))
    return sorted(idx)
