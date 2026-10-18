# Competition facts, sources and open questions

Researched 2026-10-15. Website is authoritative; update this file as facts change.

## Sources
- Website + FAQ: https://aimo-interp.github.io/
- Proposal paper (partly outdated): https://arxiv.org/abs/2607.13899
- Submission kit + runtime: https://github.com/aimo-interp/getting-started
  (`Dockerfile.competition`, `components/` = real ingestion/scoring programs)
- Baselines: https://github.com/aimo-interp/baselines
- Codabench: https://www.codabench.org/competitions/16180/
- Key issue threads: getting-started #3 (test task composition, logs), #6 (label rule, data rule)
- HF datasets: https://huggingface.co/aimo-interp
- Public participant entry (negative results): https://github.com/byeolki/aimo-interp

## Prizes & dates
Main $3000/$1000/$600, Small $2000/$600/$300, + $5000 writeup prizes (scientific
contribution; negative results / efficient methods welcome). Deadline 2026-11-01;
report + code 2026-11-15; must peer-review up to 3 reports (Nov 15–30); workshop
Dec 12, Paris.

## Environment
RTX PRO 6000 96 GB (sm_120), 192 cores, 3600 s per prediction run (all models),
offline, HF cache at /app/data/huggingface (HF_HOME set). Loading 8B ≈ 1 min.

## Baseline numbers known
- Organizers (DeepSeek, 10-fold CV on their sample): last-token probe 58.4%,
  CoT-uncertainty classifier 69.2%.
- byeolki (Small track): layer probe 0.643 on 14-case Codabench val; test-time
  counterfactual 0.50; "prompt probes read solve rate, not robustness".
- Early test task: always-True 0.53, all-False 0.47 (17 cases).

## Open questions (answer and date each)
- [ ] Exact model_id strings Codabench passes (log them in a diagnostic submission)
- [ ] Which models are cached on the worker right now
- [ ] gpt-oss-120b load time / memory in transformers 5.13 on sm_120 (MXFP4?)
- [ ] Test-set size per track and label balance (always-True score tells balance)
- [ ] Scoring of expert_no_solution variants

## Codabench submission log
| date | track | bundle | what it tests | acc | runtime | notes |
|------|-------|--------|---------------|-----|---------|-------|
