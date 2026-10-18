# Experiment log (append-only)

Primary metric: holdout accuracy on unseen problems (grouped). Always list the
prior baseline. Raw numbers live in results/EXP-*.json.

| exp | date | data (rows/models) | change | model | prior | CV acc | holdout acc | bal acc | decision |
|-----|------|--------------------|--------|-------|-------|--------|-------------|---------|----------|
| EXP-001 | 2026-10-18 | 251 rows (DeepSeek 157 incl. 137 MATH, Qwen 49, Skywork 45), leak-free; 9 mixed problems | priors + problem-only `meta`/`text` heads per model and shared | LogReg (balanced), PCA16, GroupKFold5, 30% problem holdout | majority per model | Qwen text 0.65 / Skywork meta 0.81 / DS meta 0.71 / shared meta 0.64 | Qwen 0.93 vs maj 0.80 (n=15); Skywork 0.79 vs 0.79; DS 0.57 vs 0.72; shared 0.675 vs model-prior 0.649 (n=77); mixed 0.60 (n=10) | shared 0.69 | **reject**: per-model no gain; shared gain = dataset-source confound (MATH vs competition), removed by the balanced test. Ship priors only. |
| EXP-002 | 2026-10-18 | byeolki raw samples, 130 rows leak-free (Qwen 55, Skywork 47, DS 28), 11 mixed problems | H6 behavioural: features from 4 of 8 original samples, label from the other 4 + variants (no shared samples) | AUC / within-problem pairwise | — | — | AUC acc_A 0.76, answer-agreement 0.74 (gold-free), length 0.69 (inverted); mixed pairwise acc_A/agree 0.91 (17 pairs); with leak rows 0.83 (21 pairs) | — | **keep as direction**: solve-rate/self-consistency on the original ranks robustness within a problem. Next: estimate it at test time (H8 probe, H9 sampling). |
