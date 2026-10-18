Prepare Codabench submissions for BOTH tracks from the current solution/ (or from a manifest given in $ARGUMENTS).

1. Run `make check`; stop if anything fails.
2. Confirm the method beats the previous best on HOLDOUT (docs/EXPERIMENTS.md), per model, next to the prior and majority baselines. If it does not, say so and stop unless the user asked for a diagnostic bundle.
3. Check worst-case runtime: measured LOAD_COST_S / PER_ITEM_COST_S / DECODE_TOK_S in solution/aimo_core/config.py must come from diag runs (docs/COMPETITION.md). Worst case (6 calls main, 4 small, ~50 cases) must be <= 2400 s; SAFETY_MARGIN_S >= 1200.
4. Build: `python scripts/build_bundle.py --name <exp-id> [--manifest configs/<file>.json]` -> dist/<name>-main.zip and dist/<name>-small.zip (small.txt only in small).
5. Copy both ZIPs and their .json metadata to submissions/ (tracked; dist/ is gitignored).
6. Append one row per track to the Codabench submission log in docs/COMPETITION.md: date, track, file, sha256, what it tests, expected score.
7. Commit ("ship <exp-id>") and push. Tell the user exactly which file to upload to which track.
