"""
Bayesian hyperparameter search for the XGBoost walkforward model.

Fixed config (experiment: wf_dyn_xgboost_ender60 — the live champion regime):
  target        : target_ender_60
  feature groups: faith+wisdom+strength+intelligence + rain/sunshine extras
  feature select: trailing=20, top_k=60
  walkforward   : lookback=142, purge=4, per-step dynamic features
  neutralization: 0.10

Results are written to mlflow/results/bayesian_tune_results.csv after every trial
and mirrored into MLflow for richer experiment lineage.
"""

import argparse
import gc
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Environment & hardware guards
# ---------------------------------------------------------------------------

_REQUIRED_ENV = "numerai_rag_env"
if _REQUIRED_ENV not in sys.executable:
    raise EnvironmentError(
        f"Wrong Python interpreter: {sys.executable}\n"
        "Run with the intended environment for this project."
    )

_build = xgb.build_info()
if not _build.get("USE_CUDA", _build.get("cuda", False)):
    raise RuntimeError(
        "XGBoost was not built with CUDA support. "
        "Reinstall xgboost with GPU support inside numerai_rag_env."
    )

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from prepare import (
    ARTIFACTS_DIR,
    DATA_VERSION,
    check_validation_era_freshness,
    era_stats,
    ensure_data,
    get_feature_set,
    neutralize,
    per_era_bmc,
    per_era_corr,
    read_benchmarks,
    read_split_custom,
)

# ---------------------------------------------------------------------------
# Fixed experiment config (mirrors wf_dyn_xgboost_ender60 — live champion)
# ---------------------------------------------------------------------------

CANDIDATE_GROUPS = ["faith", "wisdom", "strength", "intelligence"]

EXTRA_FEATURES = [
    # Rain
    "feature_tonal_illuminating_porgy",
    "feature_stalworth_rotund_inflammability",
    "feature_imminent_unobserved_lengthening",
    "feature_northumbrian_outflowing_connie",
    "feature_gravitational_xeromorphic_myxoma",
    "feature_depressing_punitive_recuperation",
    "feature_crimpy_amnesiac_desalinization",
    "feature_unchecked_parented_ngultrum",
    "feature_gilbertian_heliconian_perpendicular",
    "feature_different_wilier_burweed",
    "feature_tempered_devouring_izzard",
    "feature_readier_reversed_accusal",
    # Sunshine
    "feature_bridal_fingered_pensioner",
    "feature_twaddly_eleven_fustet",
    "feature_unacted_fore_folia",
    "feature_estranging_stylish_liker",
    "feature_millennial_uncanonical_sunna",
]

TRAILING_ERAS      = 20    # champion regime — feature signal averaged over last 20 train eras
TOP_K_FEATURES     = 60
MAIN_TARGET        = "target_ender_60"
CORR_TARGET        = "target_ender_20"
MMC_BENCHMARK_COL  = "v52_lgbm_ender20"
VALIDATION_ERA_COUNT     = 100
BENCHMARK_NEUTRALIZATION = 0.10
LOOKBACK_ERAS      = 142
PURGE_ERAS         = 4
EARLY_STOPPING_ERAS   = 10
EARLY_STOPPING_ROUNDS = 50
NUM_BOOST_ROUNDS   = 2000
SEED               = 42

# Known-good baseline (used to seed Optuna as trial 0 via enqueue_trial)
BASELINE_PARAMS = {
    # Only the 6 tunable params — gamma and reg_alpha are fixed at 0 in the
    # objective and must NOT appear here (Optuna validates against suggest_*).
    "max_depth":         5,
    "learning_rate":     0.01,
    "subsample":         0.85,
    "colsample_bytree":  0.40,
    "reg_lambda":        5.0,
    "min_child_weight":  20,
    "max_bin":           128,
}

ROOT = Path(__file__).resolve().parent.parent
MLFLOW_DIR = ROOT / "mlflow"
MLFLOW_RUNS_DIR = MLFLOW_DIR / "runs" / "mlruns"
MLFLOW_RESULTS_DIR = MLFLOW_DIR / "results"
DEFAULT_MLFLOW_TRACKING_URI = MLFLOW_RUNS_DIR.resolve().as_uri()
RESULTS_CSV = MLFLOW_RESULTS_DIR / "bayesian_tune_results.csv"


def log_parent_run_metadata(args: argparse.Namespace) -> None:
    mlflow.log_params(
        {
            "data_version": DATA_VERSION,
            "main_target": MAIN_TARGET,
            "corr_target": CORR_TARGET,
            "benchmark_column": MMC_BENCHMARK_COL,
            "benchmark_neutralization": BENCHMARK_NEUTRALIZATION,
            "lookback_eras": LOOKBACK_ERAS,
            "purge_eras": PURGE_ERAS,
            "trailing_eras": TRAILING_ERAS,
            "top_k_features": TOP_K_FEATURES,
            "validation_era_count": VALIDATION_ERA_COUNT,
            "num_boost_round": NUM_BOOST_ROUNDS,
            "early_stopping_eras": EARLY_STOPPING_ERAS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "feature_pool_groups": ",".join(CANDIDATE_GROUPS),
            "extra_feature_count": len(EXTRA_FEATURES),
            "optuna_study_name": args.study_name,
            "optuna_trials_requested": args.trials,
        }
    )
    mlflow.set_tags(
        {
            "project": "numerai-autoresearch",
            "workflow": "walkforward-bayesian-tuning",
            "tracker": "optuna+mlflow+csv",
        }
    )

# ---------------------------------------------------------------------------
# Feature selection helpers (identical logic to train.py)
# ---------------------------------------------------------------------------

def build_feature_pool() -> list[str]:
    pool: set[str] = set()
    for group in CANDIDATE_GROUPS:
        pool |= set(get_feature_set(group))
    pool |= set(EXTRA_FEATURES)
    return sorted(pool)


def precompute_era_correlations(
    df: pd.DataFrame, feature_pool: list[str], target_col: str
) -> dict[str, np.ndarray]:
    df = df.reset_index(drop=True)
    feat_arr = df[feature_pool].to_numpy(dtype=np.float64)
    tgt_arr  = df[target_col].to_numpy(dtype=np.float64)
    era_arr  = df["era"].to_numpy()
    result: dict[str, np.ndarray] = {}
    for era in np.unique(era_arr):
        mask = era_arr == era
        X, y = feat_arr[mask], tgt_arr[mask]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            result[str(era)] = np.zeros(len(feature_pool), dtype=np.float32)
            continue
        X, y = X[valid], y[valid]
        y_c   = y - y.mean()
        y_std = y_c.std()
        if y_std == 0:
            result[str(era)] = np.zeros(len(feature_pool), dtype=np.float32)
            continue
        X_c   = X - X.mean(axis=0)
        X_std = X_c.std(axis=0)
        num   = (y_c @ X_c) / len(y)
        corr  = np.where(X_std > 0, num / (X_std * y_std), 0.0)
        result[str(era)] = corr.astype(np.float32)
    return result


def select_features_dynamic(
    train_eras: list[str],
    era_corrs: dict[str, np.ndarray],
    feature_pool: list[str],
    top_k: int,
    trailing: int,
) -> list[str]:
    recent = [e for e in train_eras[-trailing:] if e in era_corrs]
    if not recent:
        return feature_pool[:top_k]
    arr      = np.stack([era_corrs[e] for e in recent])
    mean_corr = arr.mean(axis=0)
    top_idx  = np.argsort(np.abs(mean_corr))[-top_k:]
    return [feature_pool[i] for i in top_idx]


def make_era_balanced_weights(eras: pd.Series) -> np.ndarray:
    counts = eras.value_counts()
    return (1.0 / eras.map(counts)).to_numpy(dtype=np.float32)


def ordered_eras(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["era"].astype(str).unique().tolist())


def build_era_index(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(era): idx
        for era, idx in frame.groupby("era", sort=False).indices.items()
    }


def concat_indices(index_map: dict[str, np.ndarray], eras: list[str]) -> np.ndarray:
    return np.concatenate([index_map[era] for era in eras]).astype(np.int64)


def get_walkforward_train_eras(all_eras: list[str], eval_era: str) -> list[str]:
    eval_pos   = all_eras.index(eval_era)
    train_end  = eval_pos - PURGE_ERAS
    train_start = train_end - LOOKBACK_ERAS
    if train_start < 0:
        raise ValueError(
            f"Not enough prior eras for eval era {eval_era}. "
            f"Need {LOOKBACK_ERAS} lookback + {PURGE_ERAS} purge."
        )
    return all_eras[train_start:train_end]

# ---------------------------------------------------------------------------
# Single-step XGBoost trainer
# ---------------------------------------------------------------------------

def train_xgb_step(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    xgb_params: dict,
) -> np.ndarray:
    params = dict(xgb_params)
    params["seed"] = SEED

    clean = train_df.loc[train_df[MAIN_TARGET].notna()].copy()
    era_list  = sorted(clean["era"].astype(str).unique().tolist())
    es_era_set = set(era_list[-EARLY_STOPPING_ERAS:])

    fit_data = clean.loc[~clean["era"].astype(str).isin(es_era_set)].copy()
    es_data  = clean.loc[ clean["era"].astype(str).isin(es_era_set)].copy()

    fit_data["sample_weight"] = make_era_balanced_weights(fit_data["era"])
    es_data["sample_weight"]  = make_era_balanced_weights(es_data["era"])

    max_bin = params.get("max_bin", 256)
    dtrain  = xgb.QuantileDMatrix(
        data=fit_data[features], label=fit_data[MAIN_TARGET],
        weight=fit_data["sample_weight"], missing=np.nan, max_bin=max_bin,
    )
    deval   = xgb.QuantileDMatrix(
        data=es_data[features], label=es_data[MAIN_TARGET],
        weight=es_data["sample_weight"], missing=np.nan, max_bin=max_bin,
        ref=dtrain,
    )
    dpred   = xgb.QuantileDMatrix(
        data=predict_df[features], missing=np.nan, max_bin=max_bin, ref=dtrain,
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(deval, "eval")],
        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)],
    )
    preds = booster.predict(dpred)
    # Explicitly free GPU/CPU memory held by XGBoost objects
    del dtrain, deval, dpred, booster
    return preds

# ---------------------------------------------------------------------------
# Walkforward evaluation
# ---------------------------------------------------------------------------

def run_walkforward(xgb_params: dict, data: dict) -> dict[str, float]:
    """
    Run the full walkforward with dynamic features and return evaluation metrics.
    `data` is the pre-loaded data bundle returned by load_data().
    """
    history_df      = data["history_df"]
    history_eras    = data["history_eras"]
    history_index   = data["history_index"]
    history_corrs   = data["history_corrs"]
    feature_pool    = data["feature_pool"]
    eval_df_base    = data["eval_df_base"]
    eval_eras       = data["eval_eras"]

    collected_preds: list[np.ndarray] = []

    for eval_era in eval_eras:
        step_train_eras = get_walkforward_train_eras(history_eras, eval_era)

        step_features = select_features_dynamic(
            train_eras=step_train_eras,
            era_corrs=history_corrs,
            feature_pool=feature_pool,
            top_k=TOP_K_FEATURES,
            trailing=TRAILING_ERAS,
        )

        step_train_idx   = concat_indices(history_index, step_train_eras)
        step_predict_idx = history_index[eval_era]

        # Only copy the columns actually needed — avoids pulling ~300 pool
        # features into RAM when only 60 are used for training.
        needed_cols = list({MAIN_TARGET, CORR_TARGET, MMC_BENCHMARK_COL, "era", "id"}
                           | set(step_features))
        step_train_df   = history_df.iloc[step_train_idx][needed_cols].reset_index(drop=True)
        step_predict_df = history_df.iloc[step_predict_idx][needed_cols].reset_index(drop=True)

        preds = train_xgb_step(step_train_df, step_predict_df, step_features, xgb_params)

        # Benchmark neutralization
        bmark = step_predict_df[MMC_BENCHMARK_COL].to_numpy()
        preds = neutralize(preds, bmark, proportion=BENCHMARK_NEUTRALIZATION)

        collected_preds.append(preds)

        # Release step frames immediately; GC every 10 steps
        del step_train_df, step_predict_df
        if (len(collected_preds) % 10) == 0:
            gc.collect()

    eval_df = eval_df_base.copy()
    eval_df["prediction"] = np.concatenate(collected_preds)

    # Metrics
    corr_by_era = per_era_corr(
        eval_df[["era", "prediction", CORR_TARGET]].dropna(subset=[CORR_TARGET]),
        "prediction", target_col=CORR_TARGET,
    )
    mmc_frame = eval_df.dropna(subset=[MMC_BENCHMARK_COL, CORR_TARGET]).copy()
    mmc_by_era = per_era_bmc(
        mmc_frame[["era", "prediction", MMC_BENCHMARK_COL, CORR_TARGET]],
        "prediction", benchmark_col=MMC_BENCHMARK_COL, target_col=CORR_TARGET,
    )

    metrics: dict[str, float] = {}
    metrics.update(era_stats(corr_by_era, "val_corr"))
    metrics.update(era_stats(mmc_by_era, "val_mmc"))
    metrics["research_score"] = float(
        0.65 * metrics["val_mmc_mean"] + 0.35 * metrics["val_corr_mean"]
    )
    return metrics

# ---------------------------------------------------------------------------
# Data loading (done once before the study)
# ---------------------------------------------------------------------------

def load_data() -> dict:
    ensure_data(download=True, include_live=False)
    check_validation_era_freshness()

    feature_pool = build_feature_pool()
    targets_to_load = [MAIN_TARGET, CORR_TARGET]

    train_df      = read_split_custom("train",      features=feature_pool, targets=targets_to_load)
    validation_df = read_split_custom("validation", features=feature_pool, targets=targets_to_load)

    val_benchmarks = read_benchmarks("validation")
    if MMC_BENCHMARK_COL not in val_benchmarks.columns:
        raise KeyError(f"Benchmark column `{MMC_BENCHMARK_COL}` not found.")
    validation_df = validation_df.merge(
        val_benchmarks[["id", "era", MMC_BENCHMARK_COL]], on=["id", "era"], how="left",
    )

    validation_eras = ordered_eras(validation_df)
    eval_eras       = validation_eras[-VALIDATION_ERA_COUNT:]
    eval_era_set    = set(eval_eras)
    eval_df_base    = validation_df.loc[
        validation_df["era"].astype(str).isin(eval_era_set)
    ].copy().reset_index(drop=True)

    history_df    = pd.concat([train_df, validation_df], ignore_index=True)
    # Free the split frames immediately — history_df is the single source of truth.
    del train_df, validation_df
    gc.collect()

    history_eras  = ordered_eras(history_df)
    history_index = build_era_index(history_df)

    print(f"Precomputing per-era correlations on full history ({len(history_eras)} eras × {len(feature_pool)} features)...")
    t0 = time.time()
    history_corrs = precompute_era_correlations(history_df, feature_pool, MAIN_TARGET)
    print(f"  Done in {time.time() - t0:.1f}s")

    return {
        "history_df":    history_df,
        "history_eras":  history_eras,
        "history_index": history_index,
        "history_corrs": history_corrs,
        "feature_pool":  feature_pool,
        "eval_df_base":  eval_df_base,
        "eval_eras":     eval_eras,
    }

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def objective(trial: optuna.Trial, data: dict, mlflow_enabled: bool) -> float:
    xgb_params = {
        "objective":        "reg:squarederror",
        "tree_method":      "hist",
        "device":           "cuda",
        "verbosity":        0,
        # --- Search space: 6 active params, gamma and reg_alpha fixed at 0 ---
        # Baseline: depth=5, lr=0.01, sub=0.85, col=0.40, lam=5.0, mcw=20, max_bin=128
        #
        # gamma and reg_alpha are FIXED at 0 and excluded from search.
        # Era-balanced weights (~1/5000 per row) make split gains tiny
        # (≈0.003–0.006). Any gamma > ~0.004 blocks ALL splits → constant
        # predictions → identical degenerate score every time. reg_alpha has
        # the same effect via leaf-score penalization. Both must stay at 0.
        "max_depth":        trial.suggest_int("max_depth", 4, 6),
        "learning_rate":    trial.suggest_float("learning_rate", 0.005, 0.02, log=True),
        "subsample":        trial.suggest_float("subsample", 0.70, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.25, 0.60),
        "reg_lambda":       trial.suggest_float("reg_lambda", 2.0, 10.0, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 40),
        "max_bin":          trial.suggest_categorical("max_bin", [64, 128, 256]),
        # Fixed — not tuned
        "reg_alpha":        0.0,
        "gamma":            0.0,
    }

    t_trial = time.time()
    print(f"\n[Trial {trial.number}] params: {json.dumps({k: v for k, v in xgb_params.items() if k not in ('objective','tree_method','device','verbosity')}, indent=None)}")

    run_context = (
        mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True)
        if mlflow_enabled
        else nullcontext()
    )

    with run_context:
        if mlflow_enabled:
            mlflow.log_params(
                {
                    "trial_number": trial.number,
                    **{
                        k: v
                        for k, v in xgb_params.items()
                        if k not in ("objective", "tree_method", "device", "verbosity")
                    },
                }
            )

        try:
            metrics = run_walkforward(xgb_params, data)
        except Exception as exc:
            print(f"[Trial {trial.number}] FAILED: {exc}")
            if mlflow_enabled:
                mlflow.set_tag("trial_status", "pruned")
                mlflow.log_param("failure_reason", str(exc))
            raise optuna.exceptions.TrialPruned()

        elapsed = time.time() - t_trial
        score = metrics["research_score"]

        print(
            f"[Trial {trial.number}] research_score={score:.5f}  "
            f"corr={metrics['val_corr_mean']:.5f}  mmc={metrics['val_mmc_mean']:.5f}  "
            f"sharpe={metrics['val_corr_sharpe']:.3f}  elapsed={elapsed:.0f}s"
        )

        row = {
            "trial": trial.number,
            "research_score": score,
            "val_corr_mean": metrics["val_corr_mean"],
            "val_mmc_mean": metrics["val_mmc_mean"],
            "val_corr_sharpe": metrics["val_corr_sharpe"],
            "val_corr_max_dd": metrics["val_corr_max_drawdown"],
            "elapsed_s": round(elapsed, 1),
            **{
                k: v
                for k, v in xgb_params.items()
                if k not in ("objective", "tree_method", "device", "verbosity")
            },
        }
        _append_result(row)

        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "research_score": score,
                    "val_corr_mean": metrics["val_corr_mean"],
                    "val_mmc_mean": metrics["val_mmc_mean"],
                    "val_corr_sharpe": metrics["val_corr_sharpe"],
                    "val_mmc_sharpe": metrics["val_mmc_sharpe"],
                    "val_corr_max_drawdown": metrics["val_corr_max_drawdown"],
                    "elapsed_s": elapsed,
                }
            )
            mlflow.set_tag("trial_status", "finished")

        return score


def _append_result(row: dict) -> None:
    df_row = pd.DataFrame([row])
    if RESULTS_CSV.exists():
        df_row.to_csv(RESULTS_CSV, mode="a", header=False, index=False)
    else:
        df_row.to_csv(RESULTS_CSV, index=False)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian XGBoost hyperparameter search")
    parser.add_argument("--trials",      type=int, default=25,                  help="Number of Optuna trials (default 25)")
    parser.add_argument("--study-name",  type=str, default="xgb_ender60_k60_t20", help="Optuna study name")
    parser.add_argument("--storage",     type=str, default=None,                help="Optuna storage URI (e.g. sqlite:///tune.db)")
    parser.add_argument("--skip-baseline", action="store_true",                 help="Skip enqueuing the known-good baseline as trial 0")
    parser.add_argument("--mlflow-experiment", type=str, default="numerai-autoresearch", help="MLflow experiment name")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="Optional MLflow tracking URI")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow and keep CSV logging only")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    MLFLOW_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)
    mlflow_enabled = not args.no_mlflow

    tracking_uri = args.mlflow_tracking_uri or DEFAULT_MLFLOW_TRACKING_URI
    if mlflow_enabled:
        mlflow.set_tracking_uri(tracking_uri)
    if mlflow_enabled:
        mlflow.set_experiment(args.mlflow_experiment)

    print(f"=== Bayesian XGBoost Tuning ===")
    print(f"Study : {args.study_name}")
    print(f"Trials: {args.trials}")
    print(f"Target: {MAIN_TARGET} | Corr eval: {CORR_TARGET}")
    print(f"WF    : lookback={LOOKBACK_ERAS}, purge={PURGE_ERAS}, top_k={TOP_K_FEATURES}, trailing={TRAILING_ERAS}")
    print(f"MLflow: {'enabled' if mlflow_enabled else 'disabled'}")
    if mlflow_enabled:
        print(f"URI   : {tracking_uri}")
    print()

    parent_context = (
        mlflow.start_run(run_name=args.study_name)
        if mlflow_enabled
        else nullcontext()
    )

    with parent_context:
        if mlflow_enabled:
            log_parent_run_metadata(args)

        data = load_data()

        sampler = optuna.samplers.TPESampler(seed=SEED, n_startup_trials=5)
        study = optuna.create_study(
            study_name=args.study_name,
            direction="maximize",
            sampler=sampler,
            storage=args.storage,
            load_if_exists=True,
        )

        if not args.skip_baseline and len(study.trials) == 0:
            study.enqueue_trial(BASELINE_PARAMS)
            print(f"Enqueued baseline params as trial 0: {BASELINE_PARAMS}")

        study.optimize(
            lambda trial: objective(trial, data, mlflow_enabled=mlflow_enabled),
            n_trials=args.trials,
            catch=(Exception,),
        )

        best = study.best_trial
        print(f"\n{'='*60}")
        print(f"Best trial: #{best.number}  research_score={best.value:.5f}")
        print(f"Best params: {json.dumps(best.params, indent=2)}")

        summary = {
            "best_trial": best.number,
            "best_research_score": best.value,
            "best_params": best.params,
            "n_trials": len(study.trials),
            "study_name": args.study_name,
            "mlflow_enabled": mlflow_enabled,
        }
        summary_path = MLFLOW_RESULTS_DIR / "bayesian_tune_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        if mlflow_enabled:
            mlflow.log_metrics(
                {
                    "best_research_score": best.value,
                    "completed_trials": len(study.trials),
                }
            )
            mlflow.log_dict(summary, "bayesian_tune_summary.json")
            if RESULTS_CSV.exists():
                mlflow.log_artifact(str(RESULTS_CSV))
            mlflow.log_artifact(str(summary_path))

        print(f"\nSummary saved to {summary_path}")
        print(f"All trial results: {RESULTS_CSV}")


if __name__ == "__main__":
    main()
