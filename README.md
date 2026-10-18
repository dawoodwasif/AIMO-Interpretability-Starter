# AIMO Interpretability Challenge — entry starter

Starter repo for the [AIMO Interpretability Challenge](https://aimo-interp.github.io)
(NeurIPS 2026): given an olympiad problem and one AIMO 3 model, predict whether the
model solves it **robustly** (holds under logic-preserving perturbations) or
**spuriously**.

Built to be driven by Claude Code: `CLAUDE.md` is the operating manual,
`docs/PLAN.md` + `docs/EXPERIMENTS.md` are its memory, `/iterate` and `/ship`
are slash commands for the research loop and for preparing a submission.

## What's here
- `solution/` — the Codabench bundle. Contract-safe (always returns valid
  bools, never raises), shares a 3600 s budget across models, degrades from
  internal-activation heads → text heads → priors.
- `research/` — data fetch, perturbation generation, self-labelling on the
  competition models (GPU), all-layer activation caching, grouped-CV training
  with an honest holdout, artifact export.
- `scripts/` — Codabench-like local scorer with hard timeout, bundle builder
  with import allowlist + smoke + determinism checks, one-command GPU job.

## Quick start
```bash
make setup          # research deps (CPU)
make check          # tests + fixture eval + bundle build
make fetch          # official public datasets -> data/labeled/
python research/train.py --exp EXP-001
python scripts/build_bundle.py --name main-v1   # -> dist/main-v1.zip
```
GPU steps: `bash scripts/gpu_job.sh "<model ids>" 8 8192` on a GPU box, then push.

## Submit
Upload `dist/<name>.zip` on [Codabench](https://www.codabench.org/competitions/16180/)
and record the score in `docs/COMPETITION.md`.

## Reproduction
See `report/REPORT.md` §6 (filled in for the final submission).
