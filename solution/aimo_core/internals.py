"""White-box features from prompt-only forward passes (no generation).

Feature-set names produced here:
  internal:last:L{i}   hidden state of the final prompt token after layer i
  internal:mean:L{i}   mean of hidden states over prompt tokens after layer i
  internal:logits      [entropy, top1 prob, top5 mass, logit margin] of the
                       next-token distribution at the end of the prompt
  internal:nll         statistics of the model's per-token NLL of the PROBLEM
                       text (memorisation / familiarity signal; Min-K%-style)
  internal:shift       how much the model's state moves when the problem is
                       perturbed by aimo_core.perturb (rename/typos/distract):
                       NLL deltas + cosine distances at fixed relative depths
  internal:shift_frame same statistics for the domain-style "frame" copy (H10)

The SAME functions run offline (research/extract_features.py) and inside the
bundle, so trained heads see identical inputs. torch/transformers are imported
lazily so this module imports on CPU-only machines (the runtime then skips it).
"""
from __future__ import annotations

import gc
import sys
import time
import zlib

import numpy as np

from . import config, perturb


def _log(msg: str) -> None:
    print(f"[aimo] {msg}", file=sys.stderr, flush=True)


SHIFT_KINDS = ("rename", "typos", "distract")
SHIFT_DEPTHS = (0.25, 0.5, 0.75, 1.0)
NLL_NAMES = ["nll_mean", "nll_median", "nll_top10", "nll_top20", "nll_bot20", "nll_std",
             "nll_last32", "log_ntok", "zlib_ratio"]


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


def format_prompt(tok, problem: str, effort: str | None = None) -> str:
    msgs = [{"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": problem}]
    kw = {"reasoning_effort": effort} if effort and effort != "default" else {}
    for m in (msgs, [{"role": "user", "content": config.SYSTEM_PROMPT + "\n\n" + problem}]):
        try:
            return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
        except Exception:  # noqa: BLE001 - template may reject system role / be missing
            continue
    return config.SYSTEM_PROMPT + "\n\n" + problem + "\n"


def nll_stats(nll: np.ndarray, problem: str) -> np.ndarray:
    """Summary of per-token NLL over the problem span (higher = less familiar)."""
    nll = np.asarray(nll, dtype=np.float64)
    if nll.size == 0:
        return np.zeros(len(NLL_NAMES), dtype=np.float32)
    s = np.sort(nll)[::-1]  # descending: hardest tokens first

    def top(f):
        return float(s[: max(1, int(round(f * len(s))))].mean())

    zbytes = max(1, len(zlib.compress(problem.encode("utf-8"))))
    v = [nll.mean(), np.median(nll), top(0.10), top(0.20), float(s[-max(1, int(round(0.2 * len(s)))):].mean()),
         nll.std(), nll[-32:].mean(), np.log1p(len(nll)), nll.sum() / zbytes]
    return np.asarray(v, dtype=np.float32)


def _forward(tok, model, problems, layers, batch_size, deadline, want_states=True, effort=None):
    """Run prompts; return per-layer last/mean states, logit stats and problem-span NLL stats."""
    import torch

    texts = [format_prompt(tok, p, effort) for p in problems]
    out: dict[str, list[np.ndarray]] = {}
    dev = next(model.parameters()).device
    for start in range(0, len(texts), batch_size):
        if deadline is not None and time.monotonic() > deadline:
            raise TimeoutError("internals extraction exceeded its time share")
        chunk = texts[start:start + batch_size]
        probs = problems[start:start + batch_size]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=config.MAX_PROMPT_TOKENS, add_special_tokens=False,
                  return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping").tolist()
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.inference_mode():
            res = model(**enc, output_hidden_states=want_states, use_cache=False)
        ids = enc["input_ids"]
        # per-token NLL of token t given <t (positions 1..T-1)
        logp_all = torch.log_softmax(res.logits[:, :-1, :].float(), -1)
        tok_nll = -logp_all.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1).cpu().numpy()
        del logp_all
        for b, (txt, prob) in enumerate(zip(chunk, probs)):
            c0 = txt.find(prob) if prob else -1
            c1 = c0 + len(prob)
            span = [j for j, (s, e) in enumerate(offsets[b])
                    if c0 >= 0 and e > s and s >= c0 and e <= c1 and j >= 1]
            out.setdefault("internal:nll", []).append(nll_stats(tok_nll[b, [j - 1 for j in span]], prob)[None])
        if want_states:
            hs = res.hidden_states  # tuple(len = n_layers + 1)
            idx = range(len(hs)) if layers is None else [i for i in layers if 0 <= i < len(hs)]
            mask = enc["attention_mask"].unsqueeze(-1).to(hs[0].dtype)
            for i in idx:
                h = hs[i]
                last = h[:, -1, :].float().cpu().numpy()  # left padding -> last pos is real
                mean = ((h * mask).sum(1) / mask.sum(1).clamp(min=1)).float().cpu().numpy()
                out.setdefault(f"internal:last:L{i}", []).append(last)
                out.setdefault(f"internal:mean:L{i}", []).append(mean)
            del hs
        logits = res.logits[:, -1, :].float()
        logp = torch.log_softmax(logits, -1)
        p = logp.exp()
        ent = -(p * logp).sum(-1)
        top = torch.topk(p, 5, dim=-1).values
        tl = torch.topk(logits, 2, dim=-1).values
        stats = torch.stack([ent, top[:, 0], top.sum(-1), tl[:, 0] - tl[:, 1]], -1)
        out.setdefault("internal:logits", []).append(stats.cpu().numpy())
        del res
    return {k: np.concatenate(v, 0).astype(np.float32) for k, v in out.items()}


def n_layers(model) -> int:
    cfg = getattr(model, "config", None)
    cfg = getattr(cfg, "text_config", None) or cfg
    return int(getattr(cfg, "num_hidden_layers", 0) or 0)


def shift_layers(model) -> list[int]:
    L = n_layers(model)
    return sorted({max(1, min(L, int(round(d * L)))) for d in SHIFT_DEPTHS})


def _cos_dist(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return 1.0 - (a * b).sum(-1)


def extract(tok, model, problems: list[str], layers: list[int] | None = None,
            batch_size: int | None = None, deadline: float | None = None,
            shift: bool = False, effort: str | None = None) -> dict[str, np.ndarray]:
    """Return {feature_set_name: array[n, d]}.

    layers=None -> all layers (use offline to cache everything once).
    shift=True  -> also run perturbed copies and add 'internal:shift'.
    deadline: time.monotonic() value after which we stop and raise TimeoutError.
    """
    batch_size = batch_size or config.FORWARD_BATCH
    sl = shift_layers(model) if shift else []
    need = None if layers is None else sorted(set(layers) | set(sl))
    feats = _forward(tok, model, problems, need, batch_size, deadline, effort=effort)
    if shift:
        var = [perturb.variants(p, SHIFT_KINDS, k=1) for p in problems]
        flat = [v for vs in var for _, v in vs]
        pf = _forward(tok, model, flat, sl, batch_size, deadline, effort=effort)
        K = len(SHIFT_KINDS)
        rows = []
        for i in range(len(problems)):
            sel = slice(i * K, (i + 1) * K)
            d_nll = pf["internal:nll"][sel] - feats["internal:nll"][i][None]
            r = list(d_nll[:, [0, 2, 8]].reshape(-1))            # per-kind Δmean, Δtop10, Δzlib
            r += [float(d_nll[:, 0].mean()), float(d_nll[:, 0].max())]
            for li in sl:
                for pool in ("last", "mean"):
                    key = f"internal:{pool}:L{li}"
                    cd = _cos_dist(pf[key][sel], feats[key][i][None])
                    r += [float(cd.mean()), float(cd.max())]
            lg = pf["internal:logits"][sel] - feats["internal:logits"][i][None]
            r += list(np.abs(lg).mean(0))
            rows.append(np.asarray(r, dtype=np.float32))
        feats["internal:shift"] = np.stack(rows)
        # H10: domain-style frame copy (separate set so H10 can be ablated)
        fr = [perturb.variants(p, ("frame",), k=1)[0][1] for p in problems]
        ff = _forward(tok, model, fr, sl, batch_size, deadline, effort=effort)
        rows = []
        for i in range(len(problems)):
            r = list(ff["internal:nll"][i, [0, 2, 8]] - feats["internal:nll"][i, [0, 2, 8]])
            for li in sl:
                for pool in ("last", "mean"):
                    key = f"internal:{pool}:L{li}"
                    r.append(float(_cos_dist(ff[key][i:i + 1], feats[key][i][None])[0]))
            r += list(np.abs(ff["internal:logits"][i] - feats["internal:logits"][i]))
            rows.append(np.asarray(r, dtype=np.float32))
        feats["internal:shift_frame"] = np.stack(rows)
        if layers is not None:  # drop helper layers the caller did not ask for
            for li in sl:
                if li not in layers:
                    feats.pop(f"internal:last:L{li}", None)
                    feats.pop(f"internal:mean:L{li}", None)
    return feats


def layers_needed(feature_sets: list[str]) -> list[int]:
    idx = set()
    for fs in feature_sets:
        for part in fs.split("+"):
            if part.startswith("internal:") and ":L" in part:
                idx.add(int(part.rsplit(":L", 1)[1]))
    return sorted(idx)


def features_for(feature_sets: list[str], tok, model, problems: list[str], deadline=None, effort=None) -> dict:
    """Compute exactly what a set of heads needs (used by the runtime)."""
    feats = extract(tok, model, problems, layers=layers_needed(feature_sets), deadline=deadline,
                    shift=any(p in ("internal:shift", "internal:shift_frame") for fs in feature_sets
                              for p in fs.split("+")), effort=effort)
    # concatenated sets, e.g. "internal:nll+internal:shift"
    for fs in feature_sets:
        if "+" in fs:
            parts = fs.split("+")
            if all(p in feats for p in parts):
                feats[fs] = np.concatenate([feats[p] for p in parts], 1)
    return feats
