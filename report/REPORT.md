# Solve-rate, Self-Consistency and the Balanced Test: Predicting Robust Reasoning in AIMO-3 Models — AIMO Interpretability Challenge, Main + Small tracks

*Draft, 2026-10-18. Team/contact: dawoodwasif. Code: this repository. Bundle sha256: TBD (final).*

## 1. Summary
*(to be finalised)* We predict, for one olympiad problem and one AIMO-3 model, whether the
model's accuracy survives logic-preserving perturbations. Two facts shape everything: (i) the
harness passes **only the problem string** (no perturbation family), and (ii) the validation and
test sets are **balanced per problem and per model**, so problem difficulty and model priors
carry no signal. Offline, the strongest signal we found is the model's **own reliability on the
original problem** (solve rate / self-consistency): from 4 held-out samples it ranks the robust
model above the spurious one within a problem in 0.91 of pairs (EXP-002, small n). We turn it into
two test-time estimators that fit the 3600 s budget: a white-box solve-rate probe on prompt
activations (H8) and a deterministic self-consistency check across shipped perturbations (H9),
labelled by within-batch rank (EXP-003). Final holdout and Codabench numbers: TBD.

## 2. Task and data
**Contract.** `are_robust(model_id, reasoning_effort, problems: list[str]) -> list[bool]`,
one call per (model, effort) group (verified against the official ingestion code, which we run
verbatim in `scripts/run_local.py`). Main track: Qwen3.5-4B, Skywork-OR1-Math-7B, Olmo-3-7B-Think,
DeepSeek-R1-0528-Qwen3-8B, gpt-oss-120b (low/medium effort); Small: the first four.

**Labels.** Organizers sample each model ~10× (T=1.0, top-k 40, top-p 0.95, ≤100k tokens) on the
original and on variants from seven families (rephrase, rename, domain, distract, typos, expert
perturbations, expert no-solution); a case is spurious if any family clearly degrades accuracy.
The exact rule is unpublished; public sets show a strict rule (robust ⇔ no detrimental variant).

**Data we used (counts after the organizers' leakage advice: no AIME 2024–2026).**

| source | models | rows | notes |
|---|---|---|---|
| organizers' `math-robust-agg.csv` (baselines repo) | DeepSeek-R1-0528-Qwen3-8B | 137 | MATH problems, 26% robust |
| byeolki/aimo-interp generations (MIT), relabelled by us | Qwen3.5-4B, Skywork, DeepSeek | 114 | 8 orig + 3×6 variant samples, 8k budget; base ≥ 0.6, robust if max drop ≤ 0.10, spurious ≥ 0.25 |
| self-generated (GPU job) | all 5 | TBD | 311-problem pool, 8 base samples, rule + gpt-oss-written variants, 16k budget |

Only **9 problems** in the public-derived data have mixed labels across models (the regime of the
test), which is why self-labels on all competition checkpoints are the main lever.

## 3. Method
*(details finalised after the GPU results)*
- **Routes per (model, effort) batch**, in order of availability within the time share:
  1. white-box heads on prompt-only forward passes (hidden states at ~12 depths, next-token
     logit stats, problem-span NLL statistics [Min-K%-style], and representation **shift** between
     the original and three shipped perturbations: cosine distance at 4 relative depths, ΔNLL);
  2. test-time self-consistency (greedy answers on original + rename/typos/distract copies,
     forced answer on truncation; score = agreement − 0.25·relative length);
  3. per-model prior.
- **Labelling:** within-batch rank (top half robust) when the batch has ≥ 4 problems, else threshold.
- **Budget:** each call gets remaining/calls-left of a 2400 s envelope (1200 s margin to the 3600 s
  limit); generation has a hard `max_time` stop and a KV-cache-aware chunk size for the shared GPU.
- **Parity:** every feature is computed by `solution/aimo_core` both offline and in the bundle.

## 4. Experiments
Primary metric: accuracy on held-out problems (grouped by problem), reported with majority and
per-model-prior baselines and separately on mixed problems.

| exp | what | result | decision |
|---|---|---|---|
| EXP-001 | priors + problem-only text/meta heads | per model ≤ majority; shared head +2.6 pts over model prior from dataset-source confound | reject |
| EXP-002 | solve-rate/self-consistency from 4 held-out samples | AUC 0.74–0.76; within-problem pairwise 0.91 (17 pairs) | pursue (H8, H9) |
| EXP-002b | same signal on the test-shaped metric (test_sim) | 0.68 (n=28 mixed rows, SE≈.09) vs chance 0.50 | current best |
| EXP-003 | batch rank vs threshold on balanced batches | +9–15 pts (Qwen, DeepSeek), −2 (Skywork) | keep |
| EXP-004 | which official families cause spuriousness (DeepSeek/MATH) | domain detrimental in 88% of problems, sole cause in 54/105 spurious cases; typos 5%, rename 9% | add domain-style copy (H10), gpt-oss-written domain variants in self-labels |
| EXP-005 | metric control: probe on a *random* network | within-model AUC 0.73 but test_sim 0.29 | decide on test_sim only |
| … | white-box probes, NLL, shift, ablations | TBD | |

**White-Box ablation (for the award):** TBD — full system vs. the same system with the white-box
route removed, per model, on holdout and Codabench.

## 5. Negative results and analysis
- **Problem-only signals do not survive the balanced design.** Text/meta heads appear to work on
  pooled public data (0.68 holdout vs 0.42 majority) purely by separating dataset sources (easy MATH
  rows mostly spurious, competition rows mostly robust); per model they do not beat majority (EXP-001).
- **Why "two-step" solve-rate routes failed for others.** Labelling never-solved problems as robust
  makes robustness non-monotone in solve rate; the official cases only contain solved originals
  (base ≥ ~0.6), where higher solve rate monotonically favours robustness (EXP-002).
- **Rule-based perturbations ≠ official families.** The organizers' detrimental variants are mostly
  LLM-written `domain`/`rephrase` (written by gpt-oss-120b at low effort); rule-based rename/typos/
  distract rarely change answers. Our self-labels add gpt-oss-written families for that reason.
- **Within-model metrics are confounded by difficulty.** A solve-rate probe on a randomly initialised
  network reaches within-model AUC 0.73 (problem length/difficulty predict robustness *within* a model),
  yet falls to 0.29 on the test-shaped metric (EXP-005). We therefore evaluate every method on mixed
  problems with per-model batch-rank labelling, pooled over models (`test_sim`).
- **Official spuriousness is mostly a domain-transfer failure** (EXP-004), so cheap rule-based
  perturbations probe the wrong axis; we add a domain-style framing copy and gpt-oss-written domain
  variants.
- TBD: probe-vs-solve-rate confound on real activations, shift features, gpt-oss cost/benefit.

## 6. Reproduction
```bash
make setup && make check                          # CPU: tests + official-ingestion smoke + bundles
bash scripts/gpu_job.sh smoke && bash scripts/gpu_job.sh full    # GPU: labels + feature caches
cd research && python import_external.py && python train.py --exp EXP-XXX
python scripts/build_bundle.py --name final       # dist/final-main.zip, dist/final-small.zip
```
Third-party data: organizers' baselines CSVs (Apache-2.0), byeolki/aimo-interp generations (MIT),
Qwen2.5-Math eval sets (AIME 2024 texts used only to *exclude* leakage-risk problems; AMC 2023 and
MATH level-5 problems in the pool). Compute: TBD GPU-hours.
