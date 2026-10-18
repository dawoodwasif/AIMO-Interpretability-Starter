# Competition facts, sources and open questions

Re-verified **2026-10-18** against the website source (aimo-interp.github.io repo,
commit of 2026-10-16), getting-started @ de79405 (2026-09-17, incl. the actual
`components/` ingestion + scoring code), its issues #3/#6/#11, and the public
participant repo byeolki/aimo-interp (2026-10-18). Website > repo > paper.

Network note: this cloud session cannot reach `huggingface.co` or
`aimo-interp.github.io` directly (DNS/proxy block); GitHub git clones and
github.com pages work. HF datasets therefore need a human (see PLAN.md NEEDS HUMAN).

## Sources
- Website + FAQ: https://aimo-interp.github.io/ (source: github.com/aimo-interp/aimo-interp.github.io)
- Proposal paper (outdated where it differs): https://arxiv.org/abs/2607.13899
- Submission kit + runtime: https://github.com/aimo-interp/getting-started
  (`Dockerfile.competition`, `components/` vendored in `third_party/getting-started/`)
- Baselines: https://github.com/aimo-interp/baselines (Apache-2.0; ships `data/math-robust-{final,agg}.csv`)
- Codabench: https://www.codabench.org/competitions/16180/
- Issues: getting-started #3 (warm-up test composition, logs), #6 (organizer answers), #11 (other models)
- HF datasets: https://huggingface.co/aimo-interp (`val-sample` is private/auth; `train-main-v2`)
- Public participant entry (negative results, MIT data): https://github.com/byeolki/aimo-interp

## The contract (VERIFIED from ingestion.py, 2026-10-18)
```python
are_robust(model_id: str, reasoning_effort: str, problems: list[str]) -> list[bool]
```
- `problems` are **plain strings** (case field `problem`). **No permutation_type is
  passed.** A case = (model_id, problem, reasoning_effort); the official importer
  collapses all per-family rows of one (model, problem, effort) into ONE label.
- Ingestion inspects the signature: if a parameter is named `reasoning_effort`, it
  calls `are_robust(model_id, reasoning_effort=e, problems=[...])` once per
  (model_id, effort) group; otherwise legacy `are_robust(model_id, problems)` per model.
- Effort values: `"default"`, `"low"`, `"medium"` (README); FAQ says gpt-oss labels
  used low/medium/high.
- Result must be `type(x) is list`, right length, every `type(v) is bool`; else the
  whole batch is invalid (scored wrong). Exceptions are swallowed per batch.
- **Track marker:** Small-track ZIP must contain `small.txt` at root; Main must NOT.
  A wrong marker fails the submission before import.
- Score = accuracy over all labels (invalid = wrong). Ties: faster *average runtime
  over three repeated trials*.
- `model_id`s are passed verbatim; no aliases in the Competition phase
  (`qwen3-8b:low` = DeepSeek-R1-0528-Qwen3-8B appears only in public data).

## Prizes & dates (website, 2026-10-16)
Main $2500/$1000/$500 + White-Box Award $1000; Small $1500/$700/$300 + White-Box
Award $1000; Writeup prizes $4000 total (scientific contribution; generalisation of
interpretability methods, actionable interp, **negative results**, efficient methods).
**White-Box Award** needs white-box signals AND a dedicated ablation in the report
showing the signal has a substantial effect on performance.
Deadline 2026-11-01 (interface closes). Report + open code by 2026-11-15; peer-review up
to 3 reports Nov 15–30; workshop Dec 12, Paris. Multiple submissions can be nominated
for the final secret-test evaluation (#6, 2026-10-17; announced on Discord).

## Environment (README, verified)
RTX PRO 6000 Blackwell 96 GB (cc 12.0), 1 visible GPU, **shared with other
submissions (free VRAM may be lower)**; 192 cores, no RAM cap; 3600 s per prediction
run (all models). Offline, `HF_HOME=/app/data/huggingface`, `HF_HUB_OFFLINE=1`.
Load from warm cache: **~2 s for 4–8B, ~4 s for gpt-oss-120b** (MXFP4, ~62 GiB, Triton
kernels shipped). Only the 5 competition models are cached ("nothing [else] is cached",
#11). Using a different cached model than the judged one is allowed (#11, 2026-10-17).
Packages: torch 2.12.1 (CUDA 13.2), transformers 5.13.0, accelerate 1.13.0,
huggingface-hub 1.22.0, joblib 1.5.3, pandas 3.0.3, safetensors 0.8.0, scikit-learn
1.8.0, tokenizers 0.22.2 (+numpy via torch). No vLLM.

## Data and labels (what is known)
- Labels: sampled ~10× at T=1.0, top_k 40, top_p 0.95, ≤100k tokens incl. thinking;
  responses without a parseable answer = wrong. Qwen3.5-4B thinking on; gpt-oss at
  low/medium/high. Exact robust/spurious rule **unpublished on purpose**; "every case
  in the validation sets is supported by a substantially large boundary".
- **Balanced design (Discord via byeolki, 2026-09-15/16):** after top solutions were
  found to separate labels by problem features, the val set was rebuilt so labels
  are **balanced per problem and per model** (each problem robust for some models,
  spurious for others); the test set is built the same way. ⇒ problem-only features
  and per-model priors carry ~no signal; only model×problem interaction does.
- **Leakage advice (Discord via byeolki, 2026-10-09):** exclude AIME 2024/2025/2026,
  AIMO 2/3 public reference sets, JMO and problems-public from training pools.
- Extra labeled data allowed since 2026-09-19. Large-scale test-time perturbation
  generation is "disincentivised through the time limit, not disallowed".
- Public labeled data reachable without HF: `baselines/data/math-robust-agg.csv` (141
  rows) / `-final.csv` (675 per-family rows): DeepSeek-R1-0528-Qwen3-8B (`qwen3-8b:low`)
  on 137 MATH problems; robust ⇔ zero detrimental variants (strict). byeolki
  `data/self_labels.jsonl`: 450 self-labels (Qwen3.5-4B, Skywork, DeepSeek × 150
  problems; 8k-token budget, rule-based families; never-solved labeled robust).

## Test set
- Warm-up test (Aug): 17 cases, 4 models (#3). Codabench val in Oct: 14 cases, balanced
  (always-true 0.50, byeolki). **Final secret test: ~30–50 questions; organizers will
  adjust the time limit to keep ~170 s per sample** (#6, 2026-10-17). We still plan for
  3600 s total until a new limit is announced.

## Baseline numbers known
- Organizers (DeepSeek, 10-fold CV on their sample): last-token probe 58.4%,
  CoT-uncertainty classifier 69.2% (pre-rebalance).
- byeolki (Small): layer-probe 0.643 and self-probe 0.643 on the 14-case val (9/14);
  test-time counterfactual 0.50; on per-problem-balanced self labels, probes without
  model identity are at chance (0.42–0.49) and "prompt probes read solve rate"
  (r ≈ 0.7 with own solve rate). Leakage-free rerun: no probe clearly beats majority.

## Open questions (answer and date each)
- [ ] Which models are cached / load times / VRAM on the worker → **diag-v1** bundle
- [ ] gpt-oss-120b load + forward + generate cost in transformers 5.13 → **diag-v1**
- [ ] Generation throughput (tok/s at batch 1/16/64) → decides test-time behaviour checks
- [x] Exact model_id strings: verbatim HF ids (README, 2026-09-17); diag-v1 logs them
- [x] Test size: ~30–50 (#6, 2026-10-17)
- [ ] Final time limit (may scale to ~170 s/case)
- [ ] Label balance of the final test (val is balanced)
- [ ] Scoring of expert_no_solution variants (unknown; not needed: family not passed)

## Codabench submission log
| date | track | bundle (submissions/) | what it tests | acc | runtime | notes |
|------|-------|-----------------------|---------------|-----|---------|-------|
| — | main | diag-v1-main.zip (sha256 46388f2b73041801) | env probe: cache, load+VRAM, forward, generate tok/s @ batch 1/16/64, behaviour.run timing, ids/efforts/batch sizes; predicts prior (all True) | | | expect ≈0.50; read stderr |
| — | small | diag-v1-small.zip (sha256 4a82f77902d52e43) | same, small track (does the small worker cache gpt-oss?) | | | optional |
