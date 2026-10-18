#!/usr/bin/env bash
# One-command GPU pipeline for a human to run on a rented GPU box (>=48 GB for
# the 7-8B models; ~80 GB+ for gpt-oss-120b). Produces self-labels and cached
# internals, then commits them back so the cloud Claude session can train.
#
#   git clone <this repo> && cd <repo>
#   bash scripts/gpu_job.sh "Qwen/Qwen3.5-4B deepseek-ai/DeepSeek-R1-0528-Qwen3-8B" 8 8192
#
# args: "<space separated model ids>" <samples per prompt> <max new tokens>
set -euo pipefail
MODELS=${1:-"Qwen/Qwen3.5-4B Skywork/Skywork-OR1-Math-7B allenai/Olmo-3-7B-Think deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"}
N=${2:-8}
MAXTOK=${3:-8192}

pip install -q -r requirements-research.txt -r requirements-gpu.txt
python research/fetch_data.py
test -f data/pool/problems.jsonl || { echo "data/pool/problems.jsonl missing (see research/README.md)"; exit 1; }
test -f data/pool/variants.jsonl || python research/perturb.py --k 5 --types rename typos distract

for M in $MODELS; do
  echo "=== $M ==="
  huggingface-cli download "$M" >/dev/null
  python research/gen_labels.py --model "$M" --n "$N" --max-tokens "$MAXTOK"
  python research/extract_features.py --model "$M" --all-texts
done

git add data/labeled data/selfgen || true
git add -f data/cache/features || true   # gitignored by default; fp16 caches are small
git commit -m "GPU job: self-labels + internals for: $MODELS" || true
echo "Done. Push this branch (git push) so the Claude session can use the results."
