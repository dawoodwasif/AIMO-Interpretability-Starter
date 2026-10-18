"""Runtime knobs. Values here are estimates; MEASURE and update them."""
from __future__ import annotations

import os

PERTURBATION_TYPES = [
    "rephrase",
    "rename",
    "domain",
    "distract",
    "typos",
    "expert_perturbations",
    "expert_no_solution",
]

# Main track models (Small track = same minus gpt-oss-120b). Verify against the
# website FAQ and the real model_ids Codabench passes before trusting this.
KNOWN_MODELS = [
    "Qwen/Qwen3.5-4B",
    "Skywork/Skywork-OR1-Math-7B",
    "allenai/Olmo-3-7B-Think",
    "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
    "openai/gpt-oss-120b",
]

# ---- time budget -----------------------------------------------------------
TOTAL_BUDGET_S = float(os.environ.get("AIMO_BUDGET_S", "3600"))
# Keep a large margin: model loads vary and a timeout scores ZERO.
SAFETY_MARGIN_S = float(os.environ.get("AIMO_SAFETY_MARGIN_S", "420"))
# How many are_robust() calls (one per model) we expect in one run.
EXPECTED_MODEL_CALLS = int(os.environ.get("AIMO_EXPECTED_MODEL_CALLS", "5"))

# Estimated seconds to load each model from the offline cache (update after
# measuring on Codabench; README says ~60 s for the 8B model).
LOAD_COST_S = {
    "openai/gpt-oss-120b": 420.0,
    "_default": 90.0,
}
# Estimated seconds per problem for one prompt-only forward pass.
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

ARTIFACT_DIR = os.environ.get(
    "AIMO_ARTIFACT_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts"),
)


def cost(table: dict[str, float], model_id: str) -> float:
    return float(table.get(model_id, table["_default"]))
