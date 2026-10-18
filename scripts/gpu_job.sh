#!/usr/bin/env bash
# One-command GPU pipeline for a human to run on a rented GPU box.
# Recommended: 1x RTX PRO 6000 96 GB (same card as Codabench -> real timings, fits
# gpt-oss-120b) or 1x H100 80 GB. 4 GPUs: run one model per GPU (see PARALLEL below).
#
#   git clone <this repo> && cd <repo> && git checkout claude/relaxed-ride-xr6uqg
#   bash scripts/gpu_job.sh smoke          # ~30-45 min: every model on 10 problems, timings
#   bash scripts/gpu_job.sh full           # ~20-28 GPU-hours on one card (see docs/PLAN.md)
#   git add -A data && git add -f data/cache/features && git commit -m "GPU job results" && git push
#
# Stages (env STAGES, default all): setup variants labels features
#   STAGES="labels" MODELS="allenai/Olmo-3-7B-Think" bash scripts/gpu_job.sh full
# Everything is resumable: re-running continues from data/selfgen/*.jsonl checkpoints.
set -euo pipefail
MODE=${1:-smoke}
STAGES=${STAGES:-"setup variants labels features"}
SMALL="Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B allenai/Olmo-3-7B-Think"
MODELS=${MODELS:-"$SMALL openai/gpt-oss-120b"}
GPT_EFFORTS=${GPT_EFFORTS:-"low medium"}
MAXTOK=${MAXTOK:-16384}
if [ "$MODE" = smoke ]; then LIMIT="--limit 10"; else LIMIT=""; fi
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd "$ROOT"
LOG=logs/gpu_job_$(date +%m%d_%H%M).log; mkdir -p logs
exec > >(tee -a "$LOG") 2>&1
echo "mode=$MODE stages=$STAGES models=$MODELS maxtok=$MAXTOK"; nvidia-smi --query-gpu=name,memory.total --format=csv

has() { [[ " $STAGES " == *" $1 "* ]]; }

if has setup; then
  # vLLM env for sampling (brings its own torch/transformers)
  python3 -m venv .venv-vllm && .venv-vllm/bin/pip install -q -U pip vllm pandas scikit-learn datasets
  # eval-parity env for feature extraction: same transformers/accelerate as the Codabench image
  python3 -m venv .venv-eval && .venv-eval/bin/pip install -q -U pip \
      "torch" "transformers==5.13.0" "accelerate==1.13.0" "scikit-learn==1.8.0" pandas
  [ -d third_party/baselines ] || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/aimo-interp/baselines third_party/baselines
  [ -d third_party/byeolki ]   || GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 https://github.com/byeolki/aimo-interp third_party/byeolki
  [ -d third_party/qwen25math ] || (GIT_LFS_SKIP_SMUDGE=1 git clone -q --depth 1 --filter=blob:none --sparse \
      https://github.com/QwenLM/Qwen2.5-Math third_party/qwen25math && cd third_party/qwen25math && git sparse-checkout set evaluation/data)
  .venv-vllm/bin/python research/import_external.py
  .venv-vllm/bin/python research/fetch_data.py || echo "HF official sets skipped (set HF_TOKEN for val-sample)"
  for M in $MODELS; do .venv-vllm/bin/huggingface-cli download "$M" >/dev/null || .venv-vllm/bin/hf download "$M" >/dev/null; done
fi

if has variants; then
  # LLM families written by gpt-oss-120b @ low (the organizers' writer), verified at medium
  # resumable: skips (problem, family) pairs that already have LLM variants
  (cd research && ../.venv-vllm/bin/python perturb.py --types rephrase domain --k 2 --append --verify $LIMIT)
fi

if has labels; then
  for M in $MODELS; do
    if [ "$M" = openai/gpt-oss-120b ]; then
      for E in $GPT_EFFORTS; do
        echo "=== labels $M @ $E ==="
        (cd research && ../.venv-vllm/bin/python gen_labels.py --model "$M" --effort "$E" --max-tokens "$MAXTOK" --resume $LIMIT)
      done
    else
      echo "=== labels $M ==="
      (cd research && ../.venv-vllm/bin/python gen_labels.py --model "$M" --max-tokens "$MAXTOK" --resume $LIMIT)
    fi
  done
fi

if has features; then
  for M in $MODELS; do
    echo "=== features $M ==="
    if [ "$M" = openai/gpt-oss-120b ]; then
      for E in $GPT_EFFORTS; do (cd research && HF_HUB_OFFLINE=1 ../.venv-eval/bin/python extract_features.py --model "$M" --effort "$E"); done
    else
      (cd research && HF_HUB_OFFLINE=1 ../.venv-eval/bin/python extract_features.py --model "$M")
    fi
    # H9: shipped test-time self-consistency on every labeled problem (resumable), ~1 h/model
    if [ "$M" = openai/gpt-oss-120b ]; then EFF_LIST="$GPT_EFFORTS"; else EFF_LIST="default"; fi
    for E in $EFF_LIST; do
      EA=""; [ "$E" != default ] && EA="--effort $E"
      (cd research && HF_HUB_OFFLINE=1 ../.venv-eval/bin/python extract_behaviour.py --model "$M" $EA --max-tokens ${BEHAV_TOK:-4096})
    done
  done
fi
echo "Done ($MODE). Log: $LOG. Commit and push:"
echo "  git add -A data logs && git add -f data/cache/features && git commit -m 'GPU job ($MODE)' && git push"
