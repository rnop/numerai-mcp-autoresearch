# Experiment History

This folder contains the full flat-file experiment ledger from the working
repository.

## Why this exists

`results.tsv` is the fastest way to understand how the research loop evolved:

- baseline model comparisons
- walk-forward validation adoption
- dynamic feature selection gains
- runtime and performance tradeoffs
- public-repo MLflow validation of the same strategy
- the full sequence of retained benchmark runs

## How to read it

- `run`: short public-facing experiment identifier
- `research_score`: blended optimization score combining MMC and CORR
- `notes`: the hypothesis, model family, or workflow change being demonstrated

## Relationship to MLflow

This file is intentionally complementary to MLflow:

- `results.tsv` gives a complete flat-file record of the experimentation loop
- MLflow in `mlflow/runs/mlruns/` captures run lineage, nested trials, params,
  metrics, and artifacts for the public tuning workflow

Together they show both the agent-driven research process and the supporting
MLOps discipline.
