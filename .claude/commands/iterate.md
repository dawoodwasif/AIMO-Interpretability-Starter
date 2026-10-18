Run one research iteration following CLAUDE.md §5 for the hypothesis in $ARGUMENTS (or the top open row of docs/PLAN.md's hypotheses queue).

1. Plan: write hypothesis, exact change, metric (holdout acc on unseen problems, incl. mixed-problem acc), and keep/revert rule in docs/PLAN.md BEFORE coding.
2. Implement the smallest change; features must live in solution/aimo_core (train/serve parity). Commit.
3. Execute: `cd research && python train.py --exp EXP-NNN [...]` (add --dry-run first).
4. Test: `make check`.
5. Record a row in docs/EXPERIMENTS.md: id, data, change, model, prior/majority, CV, holdout (+mixed), bal acc, decision. Check the solve-rate confound (base_acc AUC vs head AUC, corr with base accuracy).
6. Run `python research/ablate.py --exp EXP-NNN` and read test_sim (mixed problems, per-model batch rank) — the binding metric (CLAUDE.md §5).
7. Decide: keep only if test_sim >= previous best + 2 pts (and > 0.5), per-model holdout >= prior; else `git checkout solution/artifacts`. Commit with the EXP id. If it beats the previous best, run /ship.
