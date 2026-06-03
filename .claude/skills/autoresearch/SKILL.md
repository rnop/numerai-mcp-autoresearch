---
name: autoresearch
description: >-
  Run an autoresearch campaign to finetune the Numerai main strategy in this repo:
  branch, reproduce the current champion as a baseline, then loop — edit
  autoresearch-src/train.py, run walk-forward validation, keep changes that raise
  val_mmc_mean and revert the rest — and finally promote a new winner into the live
  submission path. Use this whenever the user wants to improve, finetune, or
  experiment on the model/strategy: trying new features, targets, hyperparameters,
  feature counts, or model types, or asks to "do some research", "beat the current
  model", or "run experiments". This is the R&D loop, NOT the weekly submission. For
  the routine weekly retrain-and-submit of the existing strategy, use the
  weekly-submission skill instead.
---

# Numerai AutoResearch (finetuning)

**Follow the canonical playbook: [`playbooks/autoresearch.md`](../../../playbooks/autoresearch.md).**
Read it now and run the campaign it describes. The full procedure lives there (and not in
this file) so the same instructions are shared across agents — Codex and others read the
same playbook via `AGENTS.md`. Keep procedural edits in the playbook, not here. For deep
rationale, `program.md` at the repo root is the underlying reference the playbook distills.

Non-negotiables to keep front of mind while you read it:

- **This is finetuning an already-deployed champion**, not a greenfield search. Reconstruct
  the champion as the baseline — `train.py`'s current constants are last-experiment scratch,
  not the champion.
- **Edit only `train.py`; never `prepare.py`.** Keep `CORR_TARGET` and `MMC_BENCHMARK_COLUMN`
  fixed. GPU required.
- **Promotion to live is the user's call**, and it's a *reconciliation* of
  `make_submission.py`, not a copy-paste — regime-specific knobs (`TOP_K_FEATURES`,
  `TRAILING_ERAS`, purge) must not be blindly mirrored from the validation regime.
- **Two results files:** the untracked root `results.tsv` is your working log; the committed
  `experiments/results.tsv` is what drives the leaderboard.
