# <Title> — AIMO Interpretability Challenge, <Main|Small> track

*Team, contact. Code: <repo URL>. Bundle sha256: <...>.*

## 1. Summary
One paragraph: method, offline holdout accuracy vs prior baseline, Codabench score, runtime.

## 2. Task and data
Contract, tracks, what labels mean. Data used: official public sets (rows, models),
self-generated labels (problem sources, perturbation families, sampling params,
label rule and margins), all with counts.

## 3. Method
Features (exact layers / statistics), heads, priors, blending, time budgeting and
fallbacks. Train/serve parity.

## 4. Experiments
Table from docs/EXPERIMENTS.md: grouped CV and holdout per model, baselines,
ablations. Label-rule sensitivity.

## 5. Negative results and analysis
What did not work and why (e.g. probes tracking solve rate). Per-family results.

## 6. Reproduction
Exact commands from a fresh clone to the submitted ZIP. Compute used.
