# PLAN (living document — update every iteration)

## §0 Current state (2026-10-18, end of session 1)
- Contract fixed to the OFFICIAL interface (`are_robust(model_id, reasoning_effort, problems: list[str])`,
  no perturbation type at eval time); run_local.py runs the vendored official ingestion+scoring;
  build_bundle.py makes `-main` / `-small` (small.txt) ZIPs. `make check` green (31 tests).
- Ground truth re-verified (docs/COMPETITION.md): balanced-per-problem-and-model test, ~30–50
  cases, loads ~2–4 s, White-Box Award needs an ablation, AIME 2024–26 leakage advice.
- Data: official DeepSeek/MATH labels (137) + byeolki self-labels relabelled (153; only 9
  mixed problems leak-free) + pool of 311 leak-filtered problems + 1499 rule variants.
- No GPU here; HF and GitHub push blocked from this session (see NEEDS HUMAN).
- CPU experiments EXP-001..: see EXPERIMENTS.md. Diagnostic bundle ready: submissions/diag-v1-*.zip.
- **Next:** human runs diag-v1 upload + GPU smoke job; then train internals/shift heads on
  returned caches (train.py --sources self byeolki official_math).

## Phases
1. **Ground truth** — done 2026-10-18 except worker measurements (diag-v1).
2. **Labels** — pool + variants done (rule-based); LLM families + self-labels for all 4 small
   models and gpt-oss (low, medium) = GPU job.
3. **Baselines** — priors/majority, text heads, behavioural features from byeolki samples (CPU).
4. **White-box** — cached internals → per-model probes; NLL (memorisation) and representation
   shift under shipped perturbations; solve-rate-confound control on every probe.
5. **Budget & robustness** — measured runtimes from diag-v1, ≤ 2400 s worst case, final bundles.
6. **Report** — report/REPORT.md incl. negative results and the White-Box ablation.

## Key design facts (drive every decision)
- Test labels are balanced per problem AND per model ⇒ problem-only features and per-model
  priors ≈ chance. Primary offline metric: **holdout accuracy on mixed problems** (problems
  with both labels across models) + overall holdout, grouped by problem; always next to the
  majority and prior baselines.
- Because labels are balanced per model, a calibrated threshold matters less than ranking:
  `batch_rank` (label the top half of each batch robust) is a legitimate inference-time rule
  that uses only the batch's own inputs. Test it as a separate switch (H5).

## Hypotheses queue
| id | hypothesis | test | decision rule | status |
|----|-----------|------|---------------|--------|
| H1 | per-model priors beat majority on mixed problems | EXP-001 | — (expected ≈ chance) | EXP-001 |
| H2 | prompt probes mostly encode solve-rate | probe score vs base_accuracy partial corr | report | needs GPU cache |
| H3 | representation shift under cheap perturbations predicts spuriousness | `internal:shift` head | keep if mixed-holdout ≥ prior+2 & CV ↑ | needs GPU cache |
| H4 | memorisation ⇒ spurious: low problem NLL (Min-K%) and large NLL increase under rename/typos | `internal:nll(+shift)` heads | same | needs GPU cache |
| H5 | batch-rank labelling beats a fixed threshold when the test is balanced per model | simulate on holdout | keep if ≥ +2 pts on balanced resamples | EXP-003 |
| H6 | behavioural signal (own base accuracy / answer agreement from a few samples) ranks models within a problem | byeolki samples, split-sample | report; decides test-time sampling | **supported EXP-002** (AUC .74–.76, mixed pairwise .91, n small) |
| H7 | text-only heads add nothing under balance | EXP-001 | drop if ≤ prior | **confirmed EXP-001** (shared gain = source confound) |
| H8 | white-box solve-rate probe: ridge on own hidden states → predicted base accuracy p̂ (trained on ALL pool problems × model, solved or not); robust ⇔ high p̂ (batch rank) | self labels + caches | keep if mixed-holdout and holdout ≥ prior+2 | needs GPU |
| H9 | test-time self-consistency: greedy answers on the original + 3 shipped perturbations (deterministic), score = agreement, tie-break shorter reasoning | bundle `behav` route, timing from diag-v1 | keep if mixed-holdout ≥ prior+2 within budget | implementing |

## NEEDS HUMAN (commands, expected time/cost, what to send back)

**A. Fix GitHub push access (blocks everything that comes back via git).**
`git push` from this session returns 403 "Claude doesn't have GitHub access to
dawoodwasif/AIMO-Interpretability-Starter". Reconnect GitHub at https://claude.ai/connect-github
and install the Claude GitHub App on the repo. Until then commits stay local in the session.

**B. Upload the diagnostic bundle (2 min of your time; ~10–25 min run).**
Main track: `submissions/diag-v1-main.zip` (sha256 9ad713a8b9fe79e5). Optional Small:
`submissions/diag-v1-small.zip`. It predicts all-True (expect ≈ 0.50) and prints `[diag]`
lines to stderr: GPU, cached models, per-model load time + VRAM, forward cost, generate tok/s.
Send back: the score, runtime, and the full stderr from the submission detail panel.

**C. GPU job (the main lever: labels for the actual competition checkpoints).**
Box: 1× RTX PRO 6000 96 GB (same as Codabench; ~$1–1.5/h on vast.ai/RunPod) or 1× H100 80 GB.
```bash
git clone <repo> && cd <repo> && git checkout claude/relaxed-ride-xr6uqg
export HF_TOKEN=...            # optional: lets fetch_data.py pull val-sample / train-main-v2
bash scripts/gpu_job.sh smoke  # ~45 min, ~$1: 10 problems/model, prints tok/s and truncation
git add -A data logs && git add -f data/cache/features && git commit -m "GPU smoke" && git push
bash scripts/gpu_job.sh full   # resumable; see estimate below
git add -A data logs && git add -f data/cache/features && git commit -m "GPU full" && git push
```
Estimate for `full` (16k-token cap, 8 base samples, 10 variants × 4 samples for solved problems,
311 problems): per 4–8B model ≈ 4–6 h (Olmo-3-7B-Think may be ~2× slower), gpt-oss-120b low ≈ 1 h,
medium ≈ 2–3 h, LLM variants ≈ 1 h. **Total ≈ 22–30 GPU-h ≈ $30–45** on one card, or ~6–8 h wall
on 4 cards: `MODELS=<one id> STAGES="labels features" bash scripts/gpu_job.sh full` per GPU
(`CUDA_VISIBLE_DEVICES=i`) after one shared `STAGES="setup variants"` run.
Priority if budget is short: `STAGES="setup variants labels features" MODELS="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B Qwen/Qwen3.5-4B"` first.
Send back: the pushed commit (data/labeled/self_*.jsonl, data/selfgen/*, data/cache/features/*.npz, logs/).

**D. (Optional) Allow `huggingface.co` in this environment's network settings** so this session
can fetch `aimo-interp/train-main-v2` and `val-sample` directly (val-sample also needs HF_TOKEN).
