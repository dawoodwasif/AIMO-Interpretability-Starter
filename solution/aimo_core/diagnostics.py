"""One-shot environment probe for a DIAGNOSTIC submission (stderr only).

Enabled only when artifacts/diagnostic.json exists (build_bundle.py --diagnostic).
On the first are_robust call it logs: GPU/VRAM, library versions, which models
are in the offline HF cache, and per cached model: load time, VRAM used, the
cost of one prompt-only forward (incl. hidden states) on a SYNTHETIC prompt,
and greedy generate() throughput at several batch sizes.

It never logs evaluation problem text: only model ids, efforts and batch sizes
(the latter two are logged by runtime.py anyway). All work is capped by
`budget_s` so the run stays far below the 3600 s limit.
"""
from __future__ import annotations

import json
import os
import sys
import time

from . import config

_DONE = False
SYNTH = ("Let $n$ be the smallest positive integer such that $n^2+n+41$ is composite. "
         "A sequence $a_1, a_2, \\ldots$ satisfies $a_{k+1} = a_k^2 - a_k + 1$ with $a_1 = 2$. "
         "Find the remainder when $a_{10}$ is divided by $1000$, and add $n$ to it.")


def _log(msg: str) -> None:
    print(f"[diag] {msg}", file=sys.stderr, flush=True)


def settings() -> dict | None:
    p = os.path.join(config.ARTIFACT_DIR, "diagnostic.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def cached_models() -> list[str]:
    roots = [os.environ.get("HF_HUB_CACHE"), os.path.join(os.environ.get("HF_HOME", ""), "hub"),
             os.environ.get("HF_HOME"), os.path.expanduser("~/.cache/huggingface/hub")]
    found = set()
    for r in roots:
        if r and os.path.isdir(r):
            for d in os.listdir(r):
                if d.startswith("models--"):
                    found.add(d[len("models--"):].replace("--", "/"))
    return sorted(found)


def _gpu() -> None:
    import torch

    _log(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()} "
         f"devices={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        _log(f"gpu={torch.cuda.get_device_name(0)} cc={torch.cuda.get_device_capability(0)} "
             f"free={free / 2**30:.1f}GiB total={total / 2**30:.1f}GiB")
    try:
        import transformers

        _log(f"transformers={transformers.__version__}")
    except Exception as exc:  # noqa: BLE001
        _log(f"transformers import failed: {exc!r}")
    _log(f"cpus={os.cpu_count()} HF_HOME={os.environ.get('HF_HOME')} "
         f"offline={os.environ.get('HF_HUB_OFFLINE')}/{os.environ.get('TRANSFORMERS_OFFLINE')}")


def _probe_model(mid: str, deadline: float, gen_batches, gen_tokens: int) -> None:
    import torch

    from . import internals

    if time.monotonic() > deadline:
        _log(f"{mid}: skipped (diagnostic budget exhausted)")
        return
    torch.cuda.empty_cache()
    free0 = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else 0
    t0 = time.monotonic()
    try:
        tok, model = internals.load_model(mid)
    except Exception as exc:  # noqa: BLE001
        _log(f"{mid}: LOAD FAILED after {time.monotonic() - t0:.1f}s: {exc!r}")
        return
    load_s = time.monotonic() - t0
    used = (free0 - torch.cuda.mem_get_info()[0]) / 2**30 if torch.cuda.is_available() else 0
    _log(f"{mid}: load={load_s:.1f}s vram_used={used:.1f}GiB layers={internals.n_layers(model)} "
         f"dtype={getattr(model, 'dtype', '?')} class={type(model).__name__}")
    try:
        for n, shift in ((1, False), (8, False), (8, True)):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.monotonic()
            internals.extract(tok, model, [SYNTH] * n, layers=None, shift=shift)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            _log(f"{mid}: extract n={n} shift={shift} all-layers: {time.monotonic() - t:.2f}s")
        for b in gen_batches:
            if time.monotonic() > deadline:
                break
            texts = [internals.format_prompt(tok, SYNTH)] * b
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t = time.monotonic()
            with torch.inference_mode():
                out = model.generate(**enc, max_new_tokens=gen_tokens, min_new_tokens=gen_tokens, do_sample=False,
                                     pad_token_id=tok.pad_token_id)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt = time.monotonic() - t
            new = (out.shape[1] - enc["input_ids"].shape[1]) * b
            peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            _log(f"{mid}: generate batch={b} new_tokens={gen_tokens}: {dt:.1f}s -> {new / dt:.0f} tok/s "
                 f"(peak_alloc={peak:.1f}GiB)")
        if time.monotonic() < deadline:
            from . import behaviour

            t = time.monotonic()
            f = behaviour.run(tok, model, [SYNTH] * 4, time.monotonic() + 300, min_tokens=1, max_tokens=1024,
                              model_id=mid, effort="low" if "gpt-oss" in mid else None)
            _log(f"{mid}: behaviour.run 4 problems x 4 copies @1024 tok: {time.monotonic() - t:.1f}s "
                 f"feats={None if f is None else f.round(2).tolist()}")
    except Exception as exc:  # noqa: BLE001
        _log(f"{mid}: probe error: {exc!r}")
    finally:
        internals.unload(model)
        del tok


def run_once(model_id: str, effort: str, n: int) -> None:
    """Called by runtime at every batch; the heavy probe runs on the first call only."""
    global _DONE
    s = settings()
    if s is None:
        return
    _log(f"batch model_id={model_id!r} effort={effort!r} n={n}")
    if _DONE:
        return
    _DONE = True
    t0 = time.monotonic()
    deadline = t0 + float(s.get("budget_s", 1500))
    try:
        _gpu()
        cached = cached_models()
        _log(f"cached models ({len(cached)}): {cached}")
        if not s.get("probe_models", True):
            return
        wanted = s.get("models", config.KNOWN_MODELS)
        # if cache discovery finds nothing, still try every model (local_files_only fails fast)
        targets = [m for m in wanted if m in cached] if cached else list(wanted)
        for mid in targets:
            _probe_model(mid, deadline, s.get("gen_batches", [1, 16, 64]), int(s.get("gen_tokens", 256)))
    except Exception as exc:  # noqa: BLE001
        _log(f"diagnostic error: {exc!r}")
    finally:
        _log(f"diagnostic probe done in {time.monotonic() - t0:.1f}s")
