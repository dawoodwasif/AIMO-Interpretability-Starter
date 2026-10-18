# Research pipeline

```
fetch_data.py ─► data/labeled/official_*.jsonl          (CPU)
problem pool  ─► data/pool/problems.jsonl {id, problem, answer}
perturb.py    ─► data/pool/variants.jsonl {id, parent_id, type, problem, answer}
gen_labels.py ─► data/selfgen/<model>.jsonl (raw) + data/labeled/self_<model>.jsonl   (GPU)
extract_features.py ─► data/cache/features/<model>.npz  (GPU; uses shipped aimo_core code)
train.py      ─► solution/artifacts/{manifest.json,*.npz} + results/EXP-*.json        (CPU)
```

## Problem pool
Build `data/pool/problems.jsonl` from public competition problems with
**integer/short exact answers** (AIME, AMC, HMMT, public AIMO reference sets).
`python research/fetch_data.py --pool` downloads candidate sources into
`data/pool/raw/`; inspect their columns and write a small converter. Dedupe
against the official sample problems (keep them for validation only). Aim for
150–400 problems of mixed difficulty; the label signal comes from problems the
model *solves* on the original (base acc ≥ 0.6), so include enough easier ones.

## Cost guide (8 samples × 8k tokens, vLLM, one H100-class GPU)
Roughly 1–3 GPU-hours per 7–8B model for ~200 problems × (1 + 15 variants).
Start with `--limit 20` to measure, then scale. gpt-oss-120b needs ≥80 GB.

## Label-rule sensitivity
Labels are recomputable: `gen_labels.py --relabel-only --robust-max X --spurious-min Y`.
Train on several rules and check the method's ranking is stable.
