# Numerai MCP + Autoresearch Project

GitHub Pages: <a href="https://rnop.github.io/numerai-mcp-autoresearch/" target="_blank">View Live Deployment and Autoresearch HTML Report</a>

## Overview

This is an agentic autoresearch and deployment harness for Numerai classic tournament that combines:

- Orchestrating agents with instructions in `program.md` (agent-neutral - Claude, Codex, etc.)
- Autoresearch inspired by Karpathy in `autoresearch-src/prepare.py`, `autoresearch-src/train.py`, and `program.md` for feature analysis, experimentation loop, and logging results. 
- Feature analysis with dynamic per-era feature selection
- Bayesian optimization with Optuna and MLflow-backed experiment tracking
- Walk-forward training and time-series cross-validation
- Custom MCP server in both Python and TypeScript + Numerai's official MCP server with agent skills for weekly retraining, data drift analysis, predictions, and submissions
- Structured weekly submissions with generated HTML summary reports

## Weekly MCP orchestration prompt

Agent skills, tools, and instructions are written in the playbook located at `playbooks/weekly-submission.md`. 

The playbook connects the agent to the custom MCP + Numerai's official MCP with skills for weekly retraining, validation, data drift analysis, feature comparison, report generation, and model uploads.

Agent-neutral (Claude, Codex, etc.) weekly prompt: 

```md
Follow the instructions in `playbooks/weekly-submission.md` for weekly retraining, validation, data drift analysis, feature comparison, report generation, and model uploads.
```

## Main Files

**Agent instructions:**
- `AGENTS.md` and `CLAUDE.md`:
  Collaboration instructions used to steer agent behavior during research

**Autoresearch for data analysis, experimentation, and machine learning:**
- `autoresearch-src/train.py`:
  Main research loop for validation runs. Supports walk-forward evaluation and
  dynamic feature selection
- `autoresearch-src/prepare.py`:
  Data loading and preprocessing, Numerai metrics, and evaluation setup
- `program.md`:
  Agent-readable research manual that defines the optimization loop and constraints
- `autoresearch-src/feature_analysis.py`:
  Exploratory feature analysis workflow for dynamic feature selection
- `autoresearch-src/bayesian_tune.py`: 
  Setup Bayesian optimization with Optuna + MLflow experiment tracking

**MCP with skills for weekly orchestration and reporting:**
- `custom_mcp/make_submission.py`:
  Operational live-model packaging and weekly retrain entrypoint
- `custom_mcp/server.py`:
  Operational Python MCP layer for weekly retraining, feature-diffing, summaries, and report generation
- `custom_mcp/server.ts` / `custom_mcp/server.js`:
  Alternative TypeScript MCP layer that mirrors the weekly operational tools while reusing Python helpers for model-specific work
- `custom_mcp/site_builder.py`:
  HTML report and dashboard generator for weekly and research outputs
- `docs/index.html`:
  A browser-friendly HTML home page organizing experiment summaries, weekly reports, and feature analysis
- `docs/example_weekly_report.html`: Weekly operations report covering the currently deployed model, feature changes, and training configuration.
- `docs/feature_analysis_report.html`: Interactive feature and feature-set evaluation metrics across validation eras.


## System architecture

```mermaid
flowchart TD
    A["AGENT INSTRUCTIONS<br/>AGENTS.md / CLAUDE.md / program.md"]

    subgraph Research["RESEARCH"]
        direction TB
        C["Feature Research + Feature Selection<br/>autoresearch-src/feature_analysis.py"]
        D["Data Setup for Time-Series Cross-Validation<br/>autoresearch-src/prepare.py"]
        B["Research loop<br/>autoresearch-src/train.py"]
        H["Optuna + MLflow<br/>autoresearch-src/bayesian_tune.py / tracked trials"]
        F["Experiment tracking + Save Artifacts<br/>results.tsv / metrics.json / validation_per_era.csv"]
        D --> B
        H --> B
        H --> F
        B --> F
        F --> B
        B --> C
    end

    subgraph Reports["REPORTS"]
        direction TB
        N["Feature Analysis<br/>feature_analysis_report.html"]
        P["Weekly Submission<br/>example_weekly_report.html"]
        Q["Research Experiments Overview<br/>index.html"]
        C --> N
        N --> Q
        P --> Q
    end

    subgraph Live["LIVE DEPLOYMENT"]
        direction TB
        L["Custom MCP server<br/>custom_mcp/server.py or custom_mcp/server.js"]
        J["Weekly Retrain Pipeline<br/>retrain / packaging / report generation / submission artifact"]
        R["Official Numerai MCP Server<br/>model upload + tournament operations"]
        S["Numerai Tournament<br/>Submissions"]
        L --> B
        L --> J
        J --> P
        R --> S
    end

    A --> C
    A --> D
    A --> L
    C --> B
    F --> J
    L --> R
    J --> R
```

## Live Deployed Strategy

The live deployed strategy showcased here centers on:

- Target: `target_ender_60`
- Model: XGBoost trained on GPU
- Evaluation CORR target: `target_ender_20`
- Walk-forward regime: 142-era lookback with a 4-era purge
- Feature strategy: dynamic top-K ranking over a 699-feature candidate pool
- Benchmark neutralization: 10% vs `v52_lgbm_ender20`
- Validation Sharpe: `val_sharpe = 1.582`
- Validation evals: `val_corr_mean = 0.01545`, `val_mmc_mean = 0.00270`
- Baseline evals: `val_corr_mean = 0.00932`, `val_mmc_mean = 0.00144`

### Follow the models here:
- [ANGOSTURA](https://numer.ai/angostura)
- [PIXELATED](https://numer.ai/pixelated)
- [TAILSPIN](https://numer.ai/tailspin)