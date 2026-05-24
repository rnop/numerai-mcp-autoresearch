# MLflow Workspace

This folder centralizes the local MLflow and Bayesian tuning state for the public autoresearch repo.

## Layout

- `mlflow.db`
  Optional local SQLite tracking database retained from earlier runs.
- `runs/mlruns/`
  Local file-backed MLflow tracking store used by default by `autoresearch-src/bayesian_tune.py`.
- `logs/`
  Long-running MLflow and Bayesian tuning logs.
- `results/`
  Flat-file Bayesian tuning outputs such as `bayesian_tune_results.csv` and `bayesian_tune_summary.json`.

## Default behavior

`autoresearch-src/bayesian_tune.py` now defaults to a file-based MLflow tracking URI rooted at `mlflow/runs/mlruns/`.

If you want to use a different backend, pass `--mlflow-tracking-uri` explicitly.
