# CLAUDE.md — operating manual for this repo

You are the research engineer on a NeurIPS 2026 competition entry. Read this
file fully at the start of every session, then `docs/PLAN.md` and the tail of
`docs/EXPERIMENTS.md` to recover state. Keep both files current: they are the
memory between sessions.

## 1. The task (what wins)

AIMO Interpretability Challenge — <https://aimo-interp.github.io> (authoritative
over the arXiv paper). Submit a ZIP whose `solution.py` implements

```python
are_robust(model_id: str, problems: list[Problem]) -> list[bool]   # True = robust
# Problem has ONLY: original_problem: str, permutation_type: list[str]
```

A (problem, model) case is **robust** if the model's accuracy holds under every
listed perturbation family; **spurious** if at least one family clearly degrades
it. Metric: plain accuracy on a hidden test set. Ties broken by faster runtime.

- **Main track** models: `Qwen/Qwen3.5-4B`, `Skywork/Skywork-OR1-Math-7B`,
  `allenai/Olmo-3-7B-Think`, `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`,
  `openai/gpt-oss-120b`. **Small track**: same minus gpt-oss-120b.
- Perturbation families: `rephrase, rename, domain, distract, typos,
  expert_perturbations, expert_no_solution`. A case lists one family or all.
- **Deadline: submissions close 2026-11-01 (treat 2026-10-31 as real).**
  Report + open-source code due 2026-11-15.

## 2. Hard constraints (violating any = zero score)

1. **3600 s total** for the whole prediction run, all models together, on one
   RTX PRO 6000 (96 GB, Blackwell sm_120), 192 CPU cores. A timeout scores 0.
   Keep `SAFETY_MARGIN_S` ≥ 300 until runtimes are measured on Codabench.
2. **Offline.** No network. Models only from the read-only HF cache
   (`local_files_only=True`). Never download at eval time.
3. **Only these packages exist**: Python stdlib, torch 2.12.1 (CUDA 13.2),
   transformers 5.13.0, accelerate 1.13.0, huggingface_hub 1.22.0, joblib 1.5.3,
   pandas 3.0.3, safetensors 0.8.0, scikit-learn 1.8.0, tokenizers 0.22.2 (+numpy).
   **No vLLM in the eval container.** `scripts/build_bundle.py` enforces this.
4. Return exactly `len(problems)` real Python `bool`s, same order, never raise.
5. Deterministic predictions (no sampling at eval time unless seeded).
6. No attempts to extract/leak evaluation data → disqualification.

## 3. What is known about the data (verify, then update this section)

- Official labels come from sampling each model ~10× (temperature 1.0, top_k 40,
  top_p 0.95, ≤100k tokens) on the original and on variants. Qwen3.5-4B uses
  thinking; gpt-oss-120b used low/medium/high effort.
- Public data: label is a **conjunction** — robust only if *no* variant degrades
  (all 13 public rows with exactly one detrimental variant are non-robust). No
  public row has decay in (0, 0.5): labels have a wide margin. The exact private
  rule is deliberately unpublished.
- Only originals the model solves (base accuracy ≥ ~0.6) appear.
- Public labeled data is tiny (val-sample = 28 rows, mostly API models NOT in the
  competition). The 2026-09-19 rule change **allows extra labeled data**, so
  self-generated labels on the competition checkpoints are legal and are the
  main lever. Leaking eval data is still forbidden.
- Live test set is tiny (~17–21 cases per track ⇒ one case ≈ 5 accuracy
  points). **Do not overfit the leaderboard.** Trust grouped CV on many labels.
- Known pitfall (participant report): prompt-only probes mostly predict *solve
  rate*, not robustness. Always compare against a "base accuracy" baseline.
- Open questions to resolve early (record answers in docs/COMPETITION.md):
  exact model_ids passed by Codabench; which models are actually cached;
  whether gpt-oss-120b loads within budget in transformers 5.13 on sm_120;
  how `expert_no_solution` variants are scored.

## 4. Repo map

```
solution/                 <- EXACTLY what ships (zip root)
  solution.py             contract entry point (tiny, never raises)
  aimo_core/              runtime, time budget, text + internal features, heads
  artifacts/              manifest.json + *.npz heads (written by research/train.py)
research/                 offline only (never imported by solution/)
  fetch_data.py           official HF datasets -> data/labeled/official_*.jsonl
  perturb.py              variants of pool problems (rule-based + LLM)
  generate.py             vLLM/HF sampling + answer checking (GPU)
  gen_labels.py           self-labels for a competition model (GPU)
  extract_features.py     cache ALL-layer internals with the shipped code (GPU)
  train.py                grouped-CV head selection + holdout + export
scripts/
  run_local.py            Codabench-like scorer (subprocess + hard timeout)
  build_bundle.py         zip + import allowlist + smoke + determinism checks
  gpu_job.sh              one-command GPU pipeline for a human to run
tests/                    contract tests (must stay green)
docs/PLAN.md              current plan + hypotheses (keep updated)
docs/EXPERIMENTS.md       append-only experiment log
docs/COMPETITION.md       facts, sources, open questions, Codabench results
report/REPORT.md          technical report draft (prize requirement)
```

Train/serve parity: features used by heads MUST come from `solution/aimo_core`
(same code offline and in the bundle). Never re-implement a feature in research/.

## 5. Working loop (plan → implement → execute → test → iterate)

Every iteration:
1. **Plan**: write the hypothesis, the exact change, the metric and the
   decision rule in `docs/PLAN.md` *before* coding.
2. **Implement** the smallest change that tests it. Commit.
3. **Execute** `python research/train.py --exp EXP-NNN ...` (and GPU jobs if needed).
4. **Test**: `make check` (pytest + fixture eval + bundle build) must pass.
5. **Record** a row in `docs/EXPERIMENTS.md`: id, change, grouped-CV acc,
   balanced acc, holdout acc, prior baseline, decision. Commit with the EXP id.
6. **Decide**: keep only if holdout ≥ prior baseline + 2 pts and CV improves;
   otherwise revert artifacts (`git checkout solution/artifacts`).

Rules of evidence:
- Primary offline metric = **holdout accuracy on unseen problems**, grouped by
  problem_id. Never shuffle rows of the same problem across folds.
- Always report the prior baseline and an "always-majority" baseline next to it.
- Evaluate per model; a shared head must not hide a model where it loses.
- A Codabench score is one noisy sample of ~20 cases; never pick a method on
  leaderboard deltas < 10 points.

## 6. Method roadmap (ordered by expected value per hour)

1. **Priors that are actually right** — P(robust | model, perm family) from
   self-labels. Cheap, zero runtime, likely already beats 50%.
2. **Text/meta heads** (CPU, ms) — perm family × problem features.
3. **White-box prompt probes** per model: last-token / mean-pooled hidden
   states at mid layers + next-token logit stats, PCA + strong L2. One forward
   pass per problem fits the budget for ≤8B models.
4. **Robustness-specific internal signals** (likely the winning idea): run the
   forward pass on the original AND on 1–3 cheap self-made perturbations
   (rename/typos/distract via `research/perturb.py` rules — pure Python, ship
   them in aimo_core if used) and use representation *shift* (cosine distance,
   change in logit stats, change in probe score) as features. Only prompt
   forwards — no generation.
5. Short-budget behavioural checks for small models only if timing allows
   (e.g. ≤64 greedy tokens of the answer after a forced "\\boxed{" prefix).
6. gpt-oss-120b: first verify it loads in time. If not, ship priors/text heads
   for it and spend the budget on the four small models.

Time budget sketch (measure, then edit `config.py`): load ≈ 60–90 s per 7–8B
model, gpt-oss-120b unknown; cases per model are few, so forward passes are cheap.

## 7. Compute reality (read this)

- This cloud session likely has **no GPU**. Do everything CPU-feasible here:
  data fetching, schema inspection, perturbation code, training heads from
  cached features, bundling, tests, report.
- GPU work (sampling labels, extracting internals) runs on a GPU box via
  `scripts/gpu_job.sh`. When you need it: finish and commit the scripts, write
  the exact command and expected runtime/cost in `docs/PLAN.md` under
  "NEEDS HUMAN", and keep working on CPU tasks. Results come back as commits
  (`data/labeled/self_*.jsonl`, `data/cache/features/*.npz` via git LFS or a
  release asset — small enough to commit if fp16 and few problems).
- Codabench uploads require the human. Prepare `dist/latest.zip` plus a line in
  `docs/COMPETITION.md` stating what it tests; the human pastes back the score.
- Diagnostic submissions are valuable: e.g. a bundle that logs model_ids,
  cached models, load times and GPU name to stderr (visible in the submission
  detail panel) while returning priors.

## 8. Commands

```bash
make setup        # pip install -r requirements-research.txt
make test         # pytest
make check        # test + fixture eval + bundle build (run before every commit)
make fetch        # official datasets -> data/labeled/
python research/train.py --exp EXP-001
python scripts/run_local.py solution --cases data/labeled/official_val-sample.jsonl --no-internals
python scripts/build_bundle.py --name main-v1
```

## 9. Definition of done (by 2026-10-30)

- [ ] `dist/main-final.zip` and `dist/small-final.zip` built by build_bundle.py,
      measured runtime on Codabench < 2400 s, coverage 1.0.
- [ ] docs/EXPERIMENTS.md shows the chosen method beats priors on holdout per model.
- [ ] report/REPORT.md complete (method, data, negative results, reproduction).
- [ ] Repo reproducible from README alone (prize rule).
