"""Runtime knobs. Values here are estimates; MEASURE (diagnostic bundle) and update them."""
from __future__ import annotations

import os

_BUNDLE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Families used to build the official labels. NOT passed at eval time (the
# harness gives only the problem string); kept for research/label building.
PERTURBATION_TYPES = [
    "rephrase",
    "rename",
    "domain",
    "distract",
    "typos",
    "expert_perturbations",
    "expert_no_solution",
]

# Competition-phase model ids, passed verbatim (getting-started README, 2026-09-17).
SMALL_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
]
KNOWN_MODELS = SMALL_MODELS + ["openai/gpt-oss-120b"]
# Dataset aliases seen in public data (never passed in the Competition phase).
MODEL_ALIASES = {"qwen3-8b:low": "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}

# Track: the Small-track ZIP carries small.txt at its root (ingestion enforces it).
TRACK = os.environ.get("AIMO_TRACK") or (
    "small" if any(n.casefold() == "small.txt" for n in os.listdir(_BUNDLE_DIR)) else "main")

# ---- time budget -----------------------------------------------------------
TOTAL_BUDGET_S = float(os.environ.get("AIMO_BUDGET_S", "3600"))
# Keep a large margin: a timeout scores ZERO. Our own target is <= 2400 s used.
SAFETY_MARGIN_S = float(os.environ.get("AIMO_SAFETY_MARGIN_S", "1200"))
# are_robust() calls per run = one per (model_id, reasoning_effort) group.
# Small: 4 models. Main: 4 + gpt-oss at up to 2 efforts (README: default/low/medium).
EXPECTED_MODEL_CALLS = int(os.environ.get("AIMO_EXPECTED_MODEL_CALLS", "4" if TRACK == "small" else "6"))

# Seconds to load each model. README: ~2 s (4-8B) / ~4 s (gpt-oss-120b) from a warm
# cache; cold disk reads are slower, so stay conservative until the diagnostic
# bundle measures them on the worker.
LOAD_COST_S = {
    "openai/gpt-oss-120b": 120.0,
    "_default": 30.0,
}
# Estimated seconds per problem for one prompt-only forward pass (incl. perturbed copies).
PER_ITEM_COST_S = {
    "openai/gpt-oss-120b": 3.0,
    "_default": 1.0,
}

DISABLE_INTERNALS = os.environ.get("AIMO_DISABLE_INTERNALS", "0") == "1"

# ---- prompt used for internal features (must match training!) -------------
SYSTEM_PROMPT = (
    "Solve the following math problem. Reason step by step and put the final "
    "answer in \\boxed{}."
)
MAX_PROMPT_TOKENS = 2048
FORWARD_BATCH = int(os.environ.get("AIMO_FORWARD_BATCH", "8"))

ARTIFACT_DIR = os.environ.get("AIMO_ARTIFACT_DIR", os.path.join(_BUNDLE_DIR, "artifacts"))


def cost(table: dict[str, float], model_id: str) -> float:
    return float(table.get(model_id, table["_default"]))
