# PLAN (living document — update every iteration)

## §0 Current state (10 lines max)
- Starter repo: contract-safe bundle, priors/text/internal heads, grouped-CV trainer.
- No real labels yet. Next: fetch official data, inspect schemas, build problem pool.

## Phases
1. **Ground truth** — fetch data, inspect schemas, fill docs/COMPETITION.md open
   questions (incl. a diagnostic Codabench submission prepared for the human).
2. **Labels** — problem pool (public AIME/AMC/HMMT etc. with integer answers) →
   variants → self-labels for each competition model (GPU job).
3. **Baselines** — priors, text/meta heads; record in EXPERIMENTS.md.
4. **White-box** — cached internals → per-model probes; then perturbation-shift
   features (roadmap item 4 in CLAUDE.md).
5. **Budget & robustness** — measure runtimes, tune margins, final bundles.
6. **Report** — report/REPORT.md.

## Hypotheses queue
| id | hypothesis | test | status |
|----|-----------|------|--------|
| H1 | per-(model, family) priors beat 50% | train.py on self-labels | open |
| H2 | probes on prompt states mostly encode solve-rate | correlate probe score with base_accuracy | open |
| H3 | representation shift under cheap perturbations predicts spuriousness | new feature set | open |

## NEEDS HUMAN
(commands for the human to run on a GPU box or upload to Codabench, with expected
time/cost and what to send back)
