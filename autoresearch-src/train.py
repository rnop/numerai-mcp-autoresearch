"""
Numerai autoresearch training script.

This script supports both a fast single-pass validation run and the stronger
walk-forward workflow used in the public autoresearch showcase.

Default training target : target_ender_60
CORR eval target        : target_ender_20  (fixed per ../program.md)
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

# ---------------------------------------------------------------------------
# Environment & hardware guards — must run before any heavy imports or work
# ---------------------------------------------------------------------------

_REQUIRED_ENV = "numerai_rag_env"
if _REQUIRED_ENV not in sys.executable:
    raise EnvironmentError(
        f"Wrong Python interpreter: {sys.executable}\n"
        f"Activate the correct environment first:\n"
        f"  conda activate {_REQUIRED_ENV}\n"
        f"Then run:  python autoresearch-src/train.py"
    )

_build = xgb.build_info()
if not _build.get("USE_CUDA", _build.get("cuda", False)):
    raise RuntimeError(
        "XGBoost was not built with CUDA support. "
        "Reinstall xgboost with GPU support inside numerai_rag_env."
    )

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

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Editable research configuration
# ---------------------------------------------------------------------------

# Dynamic feature pool: named groups whose features form the candidate pool.
# Agility/dexterity excluded (reliably negative group-level signal).
# Serenity/charisma excluded (negligible unique signal vs. overlap cost).
CANDIDATE_GROUPS = ["faith", "wisdom", "strength", "intelligence"]

# Hand-picked rain + sunshine features: not in medium set (max MMC novelty),
# strong validation pct_pos, regime-emergent recent signal.
EXTRA_FEATURES = [
    # Rain (0% in medium, strong recent val signal)
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
    # Sunshine (stable, not in medium)
    "feature_bridal_fingered_pensioner",
    "feature_twaddly_eleven_fustet",
    "feature_unacted_fore_folia",
    "feature_estranging_stylish_liker",
    "feature_millennial_uncanonical_sunna",
]

# Feature ranking: rank pool features by mean absolute Pearson corr over the
# full training window (TRAILING_ERAS = LOOKBACK_ERAS), keep top TOP_K_FEATURES.
TRAILING_ERAS = 20
TOP_K_FEATURES = 60

# Training target — drives what the model learns to predict.
MAIN_TARGET = "target_ender_60"

# CORR evaluation target — must stay "target_ender_20" per ../program.md.
CORR_TARGET = "target_ender_20"

MMC_BENCHMARK_COLUMN = "v52_lgbm_ender20"

# Synthetic averaged target: simple mean of the three 60-day targets.
# Computed at load time — not a real Numerai column.
AVG60_TARGET = "target_avg60"
AVG60_SOURCES = ["target_ender_60", "target_teager2b_60", "target_jasper_60"]

VALIDATION_ERA_COUNT = 100
BENCHMARK_NEUTRALIZATION = 0.1

# Walkforward regime (activated by --walkforward flag)
WALKFORWARD = False
LOOKBACK_ERAS = 142
PURGE_ERAS = 4
DYNAMIC_WF_FEATURES = False  # per-step feature selection in walkforward; enabled by --dynamic-features
CUSTOM_FEATURE_CSV: str | None = None
CUSTOM_FEATURE_SET_LABEL: str | None = None

# Experiment (jun21): per-step dynamic target selection. At each walkforward
# step, rank the v5.2 LGBM benchmark models by numerai_corr to CORR_TARGET over
# the step's lookback eras and train on the winning model's target. Feature
# ranking then follows the chosen target. Enabled by --dynamic-target; xgboost
# + walkforward only. CORR_TARGET / MMC_BENCHMARK_COLUMN scoring stays fixed.
DYNAMIC_TARGET_SELECT = False
BENCHMARK_MODEL_TO_TARGET = {
    "v52_lgbm_cyrusd20": "target_cyrusd_20",
    "v52_lgbm_teager2b20": "target_teager2b_20",
    "v52_lgbm_ender20": "target_ender_20",
    "v52_lgbm_jasper20": "target_jasper_20",
    "v52_lgbm_cyrusd60": "target_cyrusd_60",
    "v52_lgbm_teager2b60": "target_teager2b_60",
    "v52_lgbm_ender60": "target_ender_60",
    "v52_lgbm_jasper60": "target_jasper_60",
}

# Early stopping: hold out the last EARLY_STOPPING_ERAS of training data as
# an internal eval set. All held-out eras come from train.parquet — no leakage
# from validation.
EARLY_STOPPING_ERAS = 10
EARLY_STOPPING_ROUNDS = 50

SEED = 42

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "device": "cuda",
    "verbosity": 0,
    "max_depth": 5,
    "learning_rate": 0.01,
    "subsample": 0.85,
    "colsample_bytree": 0.40,
    "min_child_weight": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 5.0,
    "gamma": 0.0,
    "max_bin": 128,
    # Seed lives in the params dict (mirrors make_submission.py) so promoting a
    # tuned XGB_PARAMS to the live builder carries the seed instead of silently
    # dropping it. The training loops still override via params["seed"] = seed
    # for per-run control, so this default is harmless when an explicit seed is
    # passed.
    "seed": SEED,
}

NUM_BOOST_ROUNDS = 2000
SAVE_LIVE_PREDICTIONS = False

LGBM_PARAMS = {
    "objective": "regression",
    "device": "cpu",
    "verbosity": -1,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.4,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "reg_lambda": 5.0,
    "n_jobs": -1,
}
LGBM_NUM_BOOST_ROUNDS = 600
LGBM_EARLY_STOPPING_ROUNDS = 30

MLP_CONFIG = {
    "hidden_sizes": [256, 128],  # 2 hidden layers
    "dropout": 0.3,
    "batch_norm": True,
    "lr": 1e-3,
    "lr_min": 1e-5,
    "weight_decay": 1e-5,
    "batch_size": 4096,
    "max_epochs": 100,
    "es_patience": 10,
}

MODEL = "xgboost"  # overridden by --model arg


# ---------------------------------------------------------------------------
# Feature pool + dynamic selection
# ---------------------------------------------------------------------------

def build_feature_pool() -> list[str]:
    """Merge named groups + hand-picked extras into a sorted deduplicated pool."""
    pool: set[str] = set()
    for group in CANDIDATE_GROUPS:
        pool |= set(get_feature_set(group))
    pool |= set(EXTRA_FEATURES)
    return sorted(pool)


def load_custom_feature_pool(csv_path: str) -> list[str]:
    """Load a deduplicated feature list from a CSV with a `feature` column."""
    frame = pd.read_csv(csv_path)
    if "feature" not in frame.columns:
        raise KeyError(f"Custom feature CSV `{csv_path}` must include a `feature` column.")
    features = [str(feature) for feature in frame["feature"].dropna().tolist()]
    deduped = list(dict.fromkeys(features))
    if not deduped:
        raise ValueError(f"Custom feature CSV `{csv_path}` did not contain any features.")
    return deduped


def precompute_era_correlations(
    df: pd.DataFrame,
    feature_pool: list[str],
    target_col: str,
) -> dict[str, np.ndarray]:
    """
    Compute per-era Pearson correlation of every pool feature with target_col.
    Returns mapping era_str -> float32 array of length len(feature_pool).
    """
    df = df.reset_index(drop=True)
    feat_arr = df[feature_pool].to_numpy(dtype=np.float64)
    tgt_arr = df[target_col].to_numpy(dtype=np.float64)
    era_arr = df["era"].to_numpy()

    result: dict[str, np.ndarray] = {}
    for era in np.unique(era_arr):
        mask = era_arr == era
        X = feat_arr[mask]
        y = tgt_arr[mask]
        valid = ~np.isnan(y)
        if valid.sum() < 10:
            result[str(era)] = np.zeros(len(feature_pool), dtype=np.float32)
            continue
        X, y = X[valid], y[valid]
        y_c = y - y.mean()
        y_std = y_c.std()
        if y_std == 0:
            result[str(era)] = np.zeros(len(feature_pool), dtype=np.float32)
            continue
        X_c = X - X.mean(axis=0)
        X_std = X_c.std(axis=0)
        num = (y_c @ X_c) / len(y)
        corr = np.where(X_std > 0, num / (X_std * y_std), 0.0)
        result[str(era)] = corr.astype(np.float32)
    return result


def select_features_dynamic(
    train_eras: list[str],
    era_corrs: dict[str, np.ndarray],
    feature_pool: list[str],
    top_k: int,
    trailing: int,
) -> list[str]:
    """
    Rank pool features by mean absolute Pearson corr over the last `trailing`
    train eras and return the top_k feature names.
    Falls back to the first top_k pool features if no era data is available.
    """
    recent = [e for e in train_eras[-trailing:] if e in era_corrs]
    if not recent:
        return feature_pool[:top_k]
    arr = np.stack([era_corrs[e] for e in recent])  # (trailing, n_features)
    mean_corr = arr.mean(axis=0)
    top_idx = np.argsort(np.abs(mean_corr))[-top_k:]
    return [feature_pool[i] for i in top_idx]


def precompute_model_era_corrs(
    bench_df: pd.DataFrame,
    model_cols: list[str],
    target_col: str,
) -> dict[str, np.ndarray]:
    """
    Per-era numerai_corr of each benchmark-model column with target_col.
    Returns mapping era_str -> float64 array aligned to model_cols (NaN where a
    model has no coverage for that era).
    """
    frame = bench_df.copy()
    frame["era"] = frame["era"].astype(str)
    per_model: dict[str, pd.Series] = {}
    for col in model_cols:
        sub = frame[["era", col, target_col]].dropna(subset=[col, target_col])
        per_model[col] = per_era_corr(sub, col, target_col=target_col)
    corr_df = pd.DataFrame(per_model)
    corr_df.index = corr_df.index.astype(str)
    return {
        era: corr_df.loc[era, model_cols].to_numpy(dtype=np.float64)
        for era in corr_df.index
    }


def select_target_dynamic(
    train_eras: list[str],
    model_era_corrs: dict[str, np.ndarray],
    model_cols: list[str],
    model_to_target: dict[str, str],
    fallback_target: str,
) -> str:
    """
    Pick the training target whose benchmark model has the highest mean per-era
    corr to CORR_TARGET over `train_eras`. Falls back when no era data exists.
    """
    rows = [model_era_corrs[e] for e in train_eras if e in model_era_corrs]
    if not rows:
        return fallback_target
    mean_corr = np.nanmean(np.stack(rows), axis=0)
    if not np.isfinite(mean_corr).any():
        return fallback_target
    best = model_cols[int(np.nanargmax(mean_corr))]
    return model_to_target[best]


def make_era_balanced_weights(eras: pd.Series) -> np.ndarray:
    counts = eras.value_counts()
    return (1.0 / eras.map(counts)).to_numpy(dtype=np.float32)


def ordered_eras(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["era"].astype(str).unique().tolist())


def build_era_index(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {str(era): idx for era, idx in frame.groupby("era", sort=False).indices.items()}


def concat_indices(index_map: dict[str, np.ndarray], eras: list[str]) -> np.ndarray:
    return np.concatenate([index_map[era] for era in eras]).astype(np.int64)


def get_walkforward_train_eras(all_eras: list[str], eval_era: str) -> list[str]:
    eval_pos = all_eras.index(eval_era)
    train_end = eval_pos - PURGE_ERAS
    train_start = train_end - LOOKBACK_ERAS
    if train_start < 0:
        raise ValueError(
            f"Not enough prior eras for eval era {eval_era}. "
            f"Need {LOOKBACK_ERAS} lookback + {PURGE_ERAS} purge eras."
        )
    return all_eras[train_start:train_end]


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    seed: int,
    target: str | None = None,
) -> np.ndarray:
    """
    Train a single XGBoost model on the full training dataset with an internal
    early-stopping holdout (last EARLY_STOPPING_ERAS of train_df), then
    predict on predict_df. Trains on `target` (defaults to MAIN_TARGET).
    """
    target = target or MAIN_TARGET
    params = dict(XGB_PARAMS)
    params["seed"] = seed

    clean_train = train_df.loc[train_df[target].notna()].copy()
    if clean_train.empty:
        raise ValueError("Training data has no labeled rows.")

    era_list = sorted(clean_train["era"].astype(str).unique().tolist())
    es_era_set = set(era_list[-EARLY_STOPPING_ERAS:])

    fit_mask = ~clean_train["era"].astype(str).isin(es_era_set)
    es_mask = clean_train["era"].astype(str).isin(es_era_set)

    fit_data = clean_train.loc[fit_mask].copy()
    es_data = clean_train.loc[es_mask].copy()

    fit_data["sample_weight"] = make_era_balanced_weights(fit_data["era"])
    es_data["sample_weight"] = make_era_balanced_weights(es_data["era"])

    print(
        f"  Fit eras: {len(era_list) - EARLY_STOPPING_ERAS}  "
        f"ES eras: {EARLY_STOPPING_ERAS}  "
        f"Fit rows: {len(fit_data):,}  ES rows: {len(es_data):,}"
    )

    dtrain = xgb.QuantileDMatrix(
        data=fit_data[features],
        label=fit_data[target],
        weight=fit_data["sample_weight"],
        missing=np.nan,
        max_bin=params.get("max_bin", 256),
    )
    deval = xgb.QuantileDMatrix(
        data=es_data[features],
        label=es_data[target],
        weight=es_data["sample_weight"],
        missing=np.nan,
        max_bin=params.get("max_bin", 256),
        ref=dtrain,
    )
    dpredict = xgb.QuantileDMatrix(
        data=predict_df[features],
        missing=np.nan,
        max_bin=params.get("max_bin", 256),
        ref=dtrain,
    )

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(deval, "eval")],
        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)],
    )
    return booster.predict(dpredict)


def train_model_lgbm(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    seed: int,
) -> np.ndarray:
    """
    Train a single LightGBM model on the full training dataset with an internal
    early-stopping holdout (last EARLY_STOPPING_ERAS of train_df), then
    predict on predict_df.
    """
    import lightgbm as lgb

    params = dict(LGBM_PARAMS)
    params["seed"] = seed

    clean_train = train_df.loc[train_df[MAIN_TARGET].notna()].copy()
    if clean_train.empty:
        raise ValueError("Training data has no labeled rows.")

    era_list = sorted(clean_train["era"].astype(str).unique().tolist())
    es_era_set = set(era_list[-EARLY_STOPPING_ERAS:])

    fit_mask = ~clean_train["era"].astype(str).isin(es_era_set)
    es_mask = clean_train["era"].astype(str).isin(es_era_set)

    fit_data = clean_train.loc[fit_mask].copy()
    es_data = clean_train.loc[es_mask].copy()

    fit_data["sample_weight"] = make_era_balanced_weights(fit_data["era"])
    es_data["sample_weight"] = make_era_balanced_weights(es_data["era"])

    print(
        f"  Fit eras: {len(era_list) - EARLY_STOPPING_ERAS}  "
        f"ES eras: {EARLY_STOPPING_ERAS}  "
        f"Fit rows: {len(fit_data):,}  ES rows: {len(es_data):,}"
    )

    dtrain = lgb.Dataset(
        data=fit_data[features],
        label=fit_data[MAIN_TARGET],
        weight=fit_data["sample_weight"],
    )
    deval = lgb.Dataset(
        data=es_data[features],
        label=es_data[MAIN_TARGET],
        weight=es_data["sample_weight"],
        reference=dtrain,
    )

    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=LGBM_NUM_BOOST_ROUNDS,
        valid_sets=[deval],
        callbacks=[
            lgb.early_stopping(stopping_rounds=LGBM_EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    return booster.predict(predict_df[features])


def train_model_mlp(
    train_df: pd.DataFrame,
    predict_df: pd.DataFrame,
    features: list[str],
    seed: int,
) -> np.ndarray:
    """
    Train a 2-hidden-layer MLP on GPU with era-balanced weighted MSE, BatchNorm,
    Dropout, Adam + cosine LR decay, and epoch-level early stopping.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MLP_CONFIG

    clean_train = train_df.loc[train_df[MAIN_TARGET].notna()].copy()
    if clean_train.empty:
        raise ValueError("Training data has no labeled rows.")

    era_list = sorted(clean_train["era"].astype(str).unique().tolist())
    es_era_set = set(era_list[-EARLY_STOPPING_ERAS:])

    fit_data = clean_train.loc[~clean_train["era"].astype(str).isin(es_era_set)].copy()
    es_data = clean_train.loc[clean_train["era"].astype(str).isin(es_era_set)].copy()

    fit_data["sample_weight"] = make_era_balanced_weights(fit_data["era"])
    es_data["sample_weight"] = make_era_balanced_weights(es_data["era"])

    print(
        f"  Fit eras: {len(era_list) - EARLY_STOPPING_ERAS}  "
        f"ES eras: {EARLY_STOPPING_ERAS}  "
        f"Fit rows: {len(fit_data):,}  ES rows: {len(es_data):,}"
    )

    # Standardize features (statistics from fit split only)
    X_fit_np = fit_data[features].to_numpy(dtype=np.float32)
    feat_mean = X_fit_np.mean(axis=0)
    feat_std = X_fit_np.std(axis=0)
    feat_std[feat_std == 0] = 1.0

    def normalize(X: np.ndarray) -> np.ndarray:
        return (X - feat_mean) / feat_std

    X_fit = normalize(X_fit_np)
    X_es = normalize(es_data[features].to_numpy(dtype=np.float32))
    X_pred = normalize(predict_df[features].to_numpy(dtype=np.float32))

    y_fit = fit_data[MAIN_TARGET].to_numpy(dtype=np.float32)
    y_es = es_data[MAIN_TARGET].to_numpy(dtype=np.float32)
    w_fit = fit_data["sample_weight"].to_numpy(dtype=np.float32)
    w_es = es_data["sample_weight"].to_numpy(dtype=np.float32)

    t_X_es = torch.from_numpy(X_es).to(device)
    t_y_es = torch.from_numpy(y_es).to(device)
    t_w_es = torch.from_numpy(w_es).to(device)

    train_ds = TensorDataset(
        torch.from_numpy(X_fit),
        torch.from_numpy(y_fit),
        torch.from_numpy(w_fit),
    )
    loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, drop_last=True)

    # Build model
    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = len(features)
            for h in cfg["hidden_sizes"]:
                layers.append(nn.Linear(prev, h))
                if cfg["batch_norm"]:
                    layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(cfg["dropout"]))
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    model = MLP().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["max_epochs"], eta_min=cfg["lr_min"]
    )

    def weighted_mse(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return (weight * (pred - target) ** 2).sum() / weight.sum()

    best_loss = float("inf")
    best_state: dict = {}
    patience_counter = 0

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        for X_b, y_b, w_b in loader:
            X_b, y_b, w_b = X_b.to(device), y_b.to(device), w_b.to(device)
            optimizer.zero_grad()
            weighted_mse(model(X_b), y_b, w_b).backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            es_loss = weighted_mse(model(t_X_es), t_y_es, t_w_es).item()

        if epoch % 10 == 0:
            print(f"  [epoch {epoch:03d}]  eval-wmse: {es_loss:.6f}  lr: {scheduler.get_last_lr()[0]:.2e}")

        if es_loss < best_loss:
            best_loss = es_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["es_patience"]:
                print(f"  Early stopping at epoch {epoch} (best epoch: {epoch - patience_counter})")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_pred).to(device)).cpu().numpy()

    return preds


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(scored_df: pd.DataFrame) -> tuple[dict[str, float], pd.DataFrame]:
    corr_by_era = per_era_corr(
        scored_df[["era", "prediction", CORR_TARGET]].dropna(subset=[CORR_TARGET]),
        "prediction",
        target_col=CORR_TARGET,
    )

    mmc_frame = scored_df.dropna(subset=[MMC_BENCHMARK_COLUMN, CORR_TARGET]).copy()
    mmc_by_era = per_era_bmc(
        mmc_frame[["era", "prediction", MMC_BENCHMARK_COLUMN, CORR_TARGET]],
        "prediction",
        benchmark_col=MMC_BENCHMARK_COLUMN,
        target_col=CORR_TARGET,
    )

    metrics = {}
    metrics.update(era_stats(corr_by_era, "val_corr"))
    metrics.update(era_stats(mmc_by_era, "val_mmc"))
    metrics["drawdown"] = metrics["val_corr_max_drawdown"]
    metrics["sharpe"] = metrics["val_corr_sharpe"]
    metrics["research_score"] = float(
        0.65 * metrics["val_mmc_mean"] + 0.35 * metrics["val_corr_mean"]
    )
    metrics["corr_era_count"] = float(len(corr_by_era))
    metrics["mmc_era_count"] = float(len(mmc_by_era))

    per_era = pd.DataFrame({"corr": corr_by_era}).join(
        pd.DataFrame({"mmc": mmc_by_era}), how="left"
    )
    return metrics, per_era


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse
    _parser = argparse.ArgumentParser(add_help=False)
    _parser.add_argument("--target", default=None)
    _parser.add_argument("--model", default=None, choices=["xgboost", "lgbm", "mlp"])
    _parser.add_argument("--walkforward", action="store_true", default=False)
    _parser.add_argument("--dynamic-features", action="store_true", default=False)
    _parser.add_argument("--dynamic-target", action="store_true", default=False)
    _parser.add_argument("--top-k", type=int, default=None)
    _parser.add_argument("--trailing-eras", type=int, default=None)
    _parser.add_argument("--feature-csv", default=None)
    _parser.add_argument("--feature-set-label", default=None)
    _args, _ = _parser.parse_known_args()
    global MAIN_TARGET, MODEL, WALKFORWARD, DYNAMIC_WF_FEATURES, TOP_K_FEATURES
    global TRAILING_ERAS, CUSTOM_FEATURE_CSV, CUSTOM_FEATURE_SET_LABEL
    global DYNAMIC_TARGET_SELECT
    if _args.target is not None:
        MAIN_TARGET = _args.target
    if _args.model is not None:
        MODEL = _args.model
    if _args.walkforward:
        WALKFORWARD = True
    if _args.dynamic_features:
        DYNAMIC_WF_FEATURES = True
    if _args.top_k is not None:
        TOP_K_FEATURES = _args.top_k
    if _args.trailing_eras is not None:
        TRAILING_ERAS = _args.trailing_eras
    if _args.feature_csv is not None:
        CUSTOM_FEATURE_CSV = _args.feature_csv
    if _args.feature_set_label is not None:
        CUSTOM_FEATURE_SET_LABEL = _args.feature_set_label
    if _args.dynamic_target:
        DYNAMIC_TARGET_SELECT = True

    if DYNAMIC_TARGET_SELECT:
        if MODEL != "xgboost":
            raise ValueError("--dynamic-target supports only --model xgboost")
        if not WALKFORWARD:
            raise ValueError("--dynamic-target requires --walkforward")

    t0 = time.time()
    np.random.seed(SEED)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    ensure_data(download=True, include_live=False)
    check_validation_era_freshness()

    # -----------------------------------------------------------------------
    # Build candidate feature pool
    # -----------------------------------------------------------------------
    if CUSTOM_FEATURE_CSV:
        feature_pool = load_custom_feature_pool(CUSTOM_FEATURE_CSV)
        feature_set_name = CUSTOM_FEATURE_SET_LABEL or Path(CUSTOM_FEATURE_CSV).stem
        print(f"Feature pool: {len(feature_pool)} features from custom CSV `{CUSTOM_FEATURE_CSV}`")
    else:
        feature_pool = build_feature_pool()
        feature_set_name = "dynamic_pool"
        print(f"Feature pool: {len(feature_pool)} features from {CANDIDATE_GROUPS} + {len(EXTRA_FEATURES)} extras")
    if DYNAMIC_TARGET_SELECT:
        targets_to_load = sorted(
            set(BENCHMARK_MODEL_TO_TARGET.values()) | {MAIN_TARGET, CORR_TARGET}
        )
    elif MAIN_TARGET == AVG60_TARGET:
        targets_to_load = list({CORR_TARGET} | set(AVG60_SOURCES))
    else:
        targets_to_load = list({MAIN_TARGET, CORR_TARGET})

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    train_df = read_split_custom("train", features=feature_pool, targets=targets_to_load)
    validation_df = read_split_custom("validation", features=feature_pool, targets=targets_to_load)

    # Compute synthetic averaged target if requested
    if MAIN_TARGET == AVG60_TARGET:
        for df in [train_df, validation_df]:
            df[AVG60_TARGET] = df[AVG60_SOURCES].mean(axis=1)

    validation_benchmarks = read_benchmarks("validation")
    if MMC_BENCHMARK_COLUMN not in validation_benchmarks.columns:
        raise KeyError(
            f"Benchmark column `{MMC_BENCHMARK_COLUMN}` not found in validation benchmarks."
        )
    validation_df = validation_df.merge(
        validation_benchmarks[["id", "era", MMC_BENCHMARK_COLUMN]],
        on=["id", "era"],
        how="left",
    )

    # Subset validation to the last VALIDATION_ERA_COUNT eras for evaluation
    validation_eras = ordered_eras(validation_df)
    eval_era_set = set(validation_eras[-VALIDATION_ERA_COUNT:])
    eval_df = validation_df.loc[
        validation_df["era"].astype(str).isin(eval_era_set)
    ].copy()

    train_eras = ordered_eras(train_df)
    print(
        f"Train: {len(train_df):,} rows ({len(train_eras)} eras)  "
        f"Eval: {len(eval_df):,} rows ({len(eval_era_set)} eras)"
    )

    # -----------------------------------------------------------------------
    # Precompute per-era Pearson corr on training data only
    # -----------------------------------------------------------------------
    print(f"Precomputing per-era Pearson correlations ({len(train_eras)} eras x {len(feature_pool)} features)...")
    t_corr = time.time()
    era_corrs = precompute_era_correlations(train_df, feature_pool, MAIN_TARGET)
    print(f"  Done in {time.time() - t_corr:.1f}s")

    # -----------------------------------------------------------------------
    # Select top-K features once using trailing era signal on training data
    # -----------------------------------------------------------------------
    features = select_features_dynamic(
        train_eras=train_eras,
        era_corrs=era_corrs,
        feature_pool=feature_pool,
        top_k=TOP_K_FEATURES,
        trailing=TRAILING_ERAS,
    )
    print(f"Selected {len(features)} features (trailing={TRAILING_ERAS} eras, pool={len(feature_pool)})")

    # -----------------------------------------------------------------------
    # Train model and predict on eval eras
    # -----------------------------------------------------------------------
    def _run_model(step_train: pd.DataFrame, step_predict: pd.DataFrame, step_feats: list[str], target: str | None = None) -> np.ndarray:
        if MODEL == "lgbm":
            return train_model_lgbm(step_train, step_predict, step_feats, SEED)
        elif MODEL == "mlp":
            return train_model_mlp(step_train, step_predict, step_feats, SEED)
        else:
            return train_model(step_train, step_predict, step_feats, SEED, target=target)

    if WALKFORWARD:
        feat_mode = "dynamic" if DYNAMIC_WF_FEATURES else "fixed"
        tgt_mode = "dynamic_select" if DYNAMIC_TARGET_SELECT else MAIN_TARGET
        print(
            f"\nWalkforward: {VALIDATION_ERA_COUNT} steps  "
            f"lookback={LOOKBACK_ERAS}  purge={PURGE_ERAS}  "
            f"features={feat_mode}  model={MODEL}  target={tgt_mode}"
        )
        history_df = pd.concat([train_df, validation_df], ignore_index=True)
        history_eras = ordered_eras(history_df)
        history_index = build_era_index(history_df)

        # Per-target feature-corr cache: feature ranking follows the chosen
        # target, but we only pay the precompute for targets actually selected.
        feature_corr_cache: dict[str, dict[str, np.ndarray]] = {}

        def feature_corrs_for(target: str) -> dict[str, np.ndarray]:
            if target not in feature_corr_cache:
                print(f"Precomputing per-era feature corrs vs {target} ({len(history_eras)} eras)...")
                t_fc = time.time()
                feature_corr_cache[target] = precompute_era_correlations(
                    history_df, feature_pool, target
                )
                print(f"  Done in {time.time() - t_fc:.1f}s")
            return feature_corr_cache[target]

        if DYNAMIC_WF_FEATURES and not DYNAMIC_TARGET_SELECT:
            history_era_corrs = feature_corrs_for(MAIN_TARGET)

        if DYNAMIC_TARGET_SELECT:
            model_cols = list(BENCHMARK_MODEL_TO_TARGET.keys())
            bench_all = pd.concat(
                [read_benchmarks("train"), read_benchmarks("validation")],
                ignore_index=True,
            ).merge(
                history_df[["id", "era", CORR_TARGET]], on=["id", "era"], how="inner"
            )
            print(f"Precomputing benchmark-model era corrs vs {CORR_TARGET} ({len(model_cols)} models)...")
            t_mc = time.time()
            model_era_corrs = precompute_model_era_corrs(bench_all, model_cols, CORR_TARGET)
            print(f"  Done in {time.time() - t_mc:.1f}s")

        eval_eras = validation_eras[-VALIDATION_ERA_COUNT:]
        collected_preds: list[np.ndarray] = []
        selected_targets: list[str] = []

        for i, eval_era in enumerate(eval_eras):
            step_train_eras = get_walkforward_train_eras(history_eras, eval_era)

            if DYNAMIC_TARGET_SELECT:
                step_target = select_target_dynamic(
                    train_eras=step_train_eras,
                    model_era_corrs=model_era_corrs,
                    model_cols=model_cols,
                    model_to_target=BENCHMARK_MODEL_TO_TARGET,
                    fallback_target=MAIN_TARGET,
                )
            else:
                step_target = MAIN_TARGET
            selected_targets.append(step_target)

            if DYNAMIC_WF_FEATURES:
                era_corrs = (
                    feature_corrs_for(step_target)
                    if DYNAMIC_TARGET_SELECT
                    else history_era_corrs
                )
                step_features = select_features_dynamic(
                    train_eras=step_train_eras,
                    era_corrs=era_corrs,
                    feature_pool=feature_pool,
                    top_k=TOP_K_FEATURES,
                    trailing=TRAILING_ERAS,
                )
            else:
                step_features = features

            step_train_idx = concat_indices(history_index, step_train_eras)
            step_predict_idx = history_index[eval_era]

            step_train_df = history_df.iloc[step_train_idx].reset_index(drop=True)
            step_predict_df = history_df.iloc[step_predict_idx].reset_index(drop=True)

            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  Step {i+1:3d}/{len(eval_eras)}: era={eval_era}  "
                    f"train_eras={len(step_train_eras)}  rows={len(step_train_df):,}  "
                    f"target={step_target}"
                )

            step_preds = _run_model(step_train_df, step_predict_df, step_features, target=step_target)

            if BENCHMARK_NEUTRALIZATION > 0:
                bmark = step_predict_df[MMC_BENCHMARK_COLUMN].to_numpy()
                step_preds = neutralize(step_preds, bmark, proportion=BENCHMARK_NEUTRALIZATION)

            collected_preds.append(step_preds)

        preds = np.concatenate(collected_preds)
        eval_df["prediction"] = preds

        if DYNAMIC_TARGET_SELECT:
            from collections import Counter
            dist = Counter(selected_targets)
            print(
                "Dynamic target selection distribution: "
                f"{dict(sorted(dist.items(), key=lambda kv: -kv[1]))}"
            )
    else:
        if MODEL == "lgbm":
            print(
                f"Training LightGBM (target={MAIN_TARGET}, "
                f"lr={LGBM_PARAMS['learning_rate']}, max_rounds={LGBM_NUM_BOOST_ROUNDS}, "
                f"es_patience={LGBM_EARLY_STOPPING_ROUNDS})..."
            )
            preds = _run_model(train_df, eval_df, features)
        elif MODEL == "mlp":
            print(
                f"Training MLP (target={MAIN_TARGET}, "
                f"hidden={MLP_CONFIG['hidden_sizes']}, lr={MLP_CONFIG['lr']}, "
                f"max_epochs={MLP_CONFIG['max_epochs']}, es_patience={MLP_CONFIG['es_patience']})..."
            )
            preds = _run_model(train_df, eval_df, features)
        else:
            print(
                f"Training XGBoost (target={MAIN_TARGET}, "
                f"lr={XGB_PARAMS['learning_rate']}, max_rounds={NUM_BOOST_ROUNDS}, "
                f"es_patience={EARLY_STOPPING_ROUNDS})..."
            )
            preds = _run_model(train_df, eval_df, features)

        if BENCHMARK_NEUTRALIZATION > 0:
            preds = neutralize(
                preds,
                eval_df[MMC_BENCHMARK_COLUMN].to_numpy(),
                proportion=BENCHMARK_NEUTRALIZATION,
            )

        eval_df["prediction"] = preds

    # -----------------------------------------------------------------------
    # Evaluate and report
    # -----------------------------------------------------------------------
    metrics, per_era = evaluate_predictions(eval_df)
    metrics.update(
        {
            "data_version": DATA_VERSION,
            "model": MODEL,
            "walkforward": WALKFORWARD,
            "dynamic_wf_features": DYNAMIC_WF_FEATURES if WALKFORWARD else None,
            "lookback_eras": LOOKBACK_ERAS if WALKFORWARD else None,
            "purge_eras": PURGE_ERAS if WALKFORWARD else None,
            "feature_set": feature_set_name,
            "feature_pool_size": len(feature_pool),
            "feature_count": TOP_K_FEATURES,
            "candidate_groups": ",".join(CANDIDATE_GROUPS),
            "trailing_eras": TRAILING_ERAS,
            "target": "dynamic_select" if DYNAMIC_TARGET_SELECT else MAIN_TARGET,
            "corr_target": CORR_TARGET,
            "mmc_benchmark_column": MMC_BENCHMARK_COLUMN,
            "validation_era_count": VALIDATION_ERA_COUNT,
            "benchmark_neutralization": BENCHMARK_NEUTRALIZATION,
            "wall_clock_seconds": round(time.time() - t0, 3),
        }
    )

    eval_df.to_csv(ARTIFACTS_DIR / "validation_predictions.csv", index=False)
    per_era.to_csv(ARTIFACTS_DIR / "validation_per_era.csv", index_label="era")
    with open(ARTIFACTS_DIR / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    print("Validation metrics:")
    for key in sorted(metrics):
        print(f"  {key}: {metrics[key]}")
    print(f"RESULT_JSON: {json.dumps(metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
