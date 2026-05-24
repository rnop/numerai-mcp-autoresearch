# Numerai Autoresearch Showcase Instructions

## Python interpreter

Prefer running scripts with the active virtual environment, or by setting:

```powershell
$env:NUMERAI_PYTHON = "C:\path\to\python.exe"
```

The public repo is designed so tooling can fall back to `sys.executable` when
`NUMERAI_PYTHON` is not set.

## GPU requirement

The XGBoost research workflow is intended to run on GPU. `autoresearch-src/train.py` and
`autoresearch-src/bayesian_tune.py` validate CUDA availability at startup.

## Public-repo scope

- `autoresearch-src/train.py` - core autoresearch training loop
- `autoresearch-src/prepare.py` - data loading, metrics, and evaluation helpers
- `autoresearch-src/bayesian_tune.py` - Optuna + MLflow experiment tracking
- `custom_mcp/make_submission.py` - live model packaging
- `custom_mcp/server.py` - weekly automation pipeline
- `program.md` - operating manual for the research agent

## Showcase expectations

- Keep secrets out of the repo. Use `.mcp.example.json` as a template only.
- Prefer reproducible, walk-forward experiments over ad hoc one-off runs.
- Preserve `results.tsv`-style logging if you add MLflow so both quick review
  and richer experiment lineage remain available.
