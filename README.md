# AIMO Interpretability Challenge — entry starter

Starter repo for the [AIMO Interpretability Challenge](https://aimo-interp.github.io)
(NeurIPS 2026): given an olympiad problem and one AIMO 3 model, predict whether the
model solves it **robustly** (holds under logic-preserving perturbations) or
**spuriously**.

Built to be driven by Claude Code: `CLAUDE.md` is the operating manual,
`docs/PLAN.md` + `docs/EXPERIMENTS.md` are its memory, `/iterate` and `/ship`
(`.claude/commands/`) are slash commands for the research loop and for preparing a submission.

Interface (verified against the official ingestion code):
`are_robust(model_id, reasoning_effort, problems: list[str]) -> list[bool]`.

## What's here
- `solution/` — the Codabench bundle. Contract-safe (always returns valid
  bools, never raises), shares a 3600 s budget across models, degrades from
  internal-activation heads → text heads → priors.
- `research/` — data fetch, perturbation generation, self-labelling on the
  competition models (GPU), all-layer activation caching, grouped-CV training
  with an honest holdout, artifact export.
- `scripts/` — local scorer that runs the official Codabench ingestion+scoring
  (`third_party/getting-started/components`) with a hard timeout, per-track bundle builder
  with import allowlist + smoke + determinism checks, one-command GPU job.

## Quick start
```bash
make setup          # research deps (CPU)
make check          # tests + official-ingestion fixture eval (both tracks) + bundle build
# reference repos for public data (see third_party/README.md), then:
python research/import_external.py   # organizer CSVs + byeolki labels + leak-filtered pool
make fetch          # HF datasets (needs huggingface.co access; HF_TOKEN for val-sample)
cd research && python train.py --exp EXP-001 --dry-run
python scripts/build_bundle.py --name v1        # -> dist/v1-main.zip, dist/v1-small.zip
```
GPU steps: `bash scripts/gpu_job.sh smoke`, then `bash scripts/gpu_job.sh full` on a GPU box
(see docs/PLAN.md "NEEDS HUMAN" for cost), then commit + push the results.

## Submit
Upload `submissions/<name>-main.zip` (Main) or `submissions/<name>-small.zip` (Small, contains
`small.txt`) on [Codabench](https://www.codabench.org/competitions/16180/) and record the score
in `docs/COMPETITION.md`.

## Reproduction
See `report/REPORT.md` §6 (filled in for the final submission).
