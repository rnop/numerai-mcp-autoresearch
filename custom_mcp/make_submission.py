"""
Build and pickle the live submission model.

Strategy: walkforward per-step dynamic feature selection, applied to the live window.
  - Feature pool : faith + wisdom + strength + intelligence + quantum
                   + 17 rain/sunshine extras (~1506 features, v5.3 Quantum data)
  - Training eras: last LOOKBACK_ERAS of combined train+validation data (no purge — we predict the next era)
  - Feature selection: top TOP_K_FEATURES by mean abs Pearson corr over trailing TRAILING_ERAS of that window
  - Model: XGBoost (GPU), same hyperparams as the walkforward experiment
  - Neutralization: 10% against v53_lgbm_ender20 applied inside predict()

Output: submission_model.pkl  (cloudpickle of the predict function)

IMPORTANT — upload with Python 3.11 docker image (ID 4d39918c-a82b-42ea-8dc7-ed5a30e676c5).
The pickle contains Python 3.11 bytecode; Numerai's default (3.12) will fail with "unknown opcode 0".
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
import cloudpickle

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "autoresearch-src"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

# ---------------------------------------------------------------------------
# Environment guard
# ---------------------------------------------------------------------------
_REQUIRED_ENV = "numerai_rag_env"
if _REQUIRED_ENV not in sys.executable:
    raise EnvironmentError(
        f"Wrong Python interpreter: {sys.executable}\n"
        "Run with the intended environment, or set NUMERAI_PYTHON to the "
        "interpreter you want the automation layer to use."
    )

from prepare import (
    DATA_VERSION,
    ensure_data,
    check_validation_era_freshness,
    get_feature_set,
    refresh_data,
    read_split_custom,
)

# ---------------------------------------------------------------------------
# Configuration — must match the winning walkforward experiment
# ---------------------------------------------------------------------------
MAIN_TARGET = "target_ender_60"
CORR_TARGET = "target_ender_20"
MMC_BENCHMARK_COLUMN = "v53_lgbm_ender20"
LIVE_FALLBACK_TARGET = CORR_TARGET

# jul18 promotion (research commit 26b95bb): quantum added to the pool.
CANDIDATE_GROUPS = ["faith", "wisdom", "strength", "intelligence", "quantum"]
EXTRA_FEATURES = [
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
    "feature_bridal_fingered_pensioner",
    "feature_twaddly_eleven_fustet",
    "feature_unacted_fore_folia",
    "feature_estranging_stylish_liker",
    "feature_millennial_uncanonical_sunna",
]

LOOKBACK_ERAS = 142
TRAILING_ERAS = 20
# top_k 60->120 with the pool growing 699->1506: keeps the live filter at its
# historical ~8% selectivity (120/1506 vs the old 60/699) and matches the
# validated champion, whose selection regime (trailing-20 ranking) is the
# same one used here. Keeping 60 would have tightened the cut to 4%.
TOP_K_FEATURES = 120
EARLY_STOPPING_ERAS = 10
EARLY_STOPPING_ROUNDS = 50
BENCHMARK_NEUTRALIZATION = 0.1
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
    "seed": SEED,
}
NUM_BOOST_ROUNDS = 2000

SUBMISSIONS_DIR = "submissions"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_feature_pool() -> list[str]:
    pool: set[str] = set()
    for group in CANDIDATE_GROUPS:
        pool |= set(get_feature_set(group))
    pool |= set(EXTRA_FEATURES)
    return sorted(pool)


def ordered_eras(frame: pd.DataFrame) -> list[str]:
    return sorted(frame["era"].astype(str).unique().tolist())


def make_era_balanced_weights(eras: pd.Series) -> np.ndarray:
    counts = eras.value_counts()
    return (1.0 / eras.map(counts)).to_numpy(dtype=np.float32)


def build_live_training_target(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["live_training_target"] = frame[MAIN_TARGET].where(
        frame[MAIN_TARGET].notna(),
        frame[LIVE_FALLBACK_TARGET],
    )
    frame["live_training_target_source"] = np.where(
        frame[MAIN_TARGET].notna(),
        MAIN_TARGET,
        LIVE_FALLBACK_TARGET,
    )
    return frame


def precompute_era_correlations(df: pd.DataFrame, feature_pool: list[str], target_col: str) -> dict[str, np.ndarray]:
    # Slices one era at a time (mirrors train.py) — the 1506-feature v5.3 pool
    # times the full live window would need ~9GB as a single dense matrix.
    result: dict[str, np.ndarray] = {}
    for era, chunk in df.groupby("era", sort=False):
        X = chunk[feature_pool].to_numpy(dtype=np.float64)
        y = chunk[target_col].to_numpy(dtype=np.float64)
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


def select_features_dynamic(train_eras: list[str], era_corrs: dict[str, np.ndarray],
                             feature_pool: list[str], top_k: int, trailing: int) -> list[str]:
    recent = [e for e in train_eras[-trailing:] if e in era_corrs]
    if not recent:
        return feature_pool[:top_k]
    arr = np.stack([era_corrs[e] for e in recent])
    mean_corr = arr.mean(axis=0)
    top_idx = np.argsort(np.abs(mean_corr))[-top_k:]
    return [feature_pool[i] for i in top_idx]


# ---------------------------------------------------------------------------
# Build and train the live model
# ---------------------------------------------------------------------------

def main() -> None:
    t0 = time.time()
    np.random.seed(SEED)

    refresh_data(include_live=False)
    check_validation_era_freshness()
    ensure_data(download=True, include_live=False)

    feature_pool = build_feature_pool()
    targets_to_load = [MAIN_TARGET, CORR_TARGET]
    print(f"Feature pool: {len(feature_pool)} features")

    # Load all available labeled data
    train_df = read_split_custom("train", features=feature_pool, targets=targets_to_load)
    validation_df = read_split_custom("validation", features=feature_pool, targets=targets_to_load)

    # Keep recent validation eras when ender_60 is still missing but ender_20 is available.
    train_df = build_live_training_target(train_df)
    validation_df = build_live_training_target(validation_df)
    validation_df = validation_df.dropna(subset=["live_training_target"])

    history_df = pd.concat([train_df, validation_df], ignore_index=True)
    all_eras = ordered_eras(history_df)
    print(f"Total eras available: {len(all_eras)}  (train={len(ordered_eras(train_df))}, val_labeled={len(ordered_eras(validation_df))})")

    # Take the last LOOKBACK_ERAS for live training (no purge — next era is live)
    if len(all_eras) < LOOKBACK_ERAS:
        raise ValueError(f"Need {LOOKBACK_ERAS} eras, only have {len(all_eras)}")
    window_eras = all_eras[-LOOKBACK_ERAS:]
    window_df = history_df[history_df["era"].astype(str).isin(set(window_eras))].copy()
    era_start, era_end = window_eras[0], window_eras[-1]
    submissions_dir = Path(SUBMISSIONS_DIR)
    submissions_dir.mkdir(exist_ok=True)
    output_path = submissions_dir / f"submission_model_train_{era_start}_{era_end}.pkl"
    meta_path = submissions_dir / f"submission_model_train_{era_start}_{era_end}_meta.json"
    print(f"Training window: eras {era_start}–{era_end}  rows={len(window_df):,}")
    fallback_era_list = sorted(
        window_df.loc[
            window_df["live_training_target_source"].eq(LIVE_FALLBACK_TARGET),
            "era",
        ].astype(str).unique().tolist()
    )
    print(
        "Fallback target eras: "
        f"{len(fallback_era_list)} using {LIVE_FALLBACK_TARGET}"
        + (f" -> {fallback_era_list[0]}–{fallback_era_list[-1]}" if fallback_era_list else "")
    )

    # Compute era correlations and select features from trailing window
    print(f"Computing era correlations ({len(window_eras)} eras x {len(feature_pool)} features)...")
    era_corrs = precompute_era_correlations(window_df, feature_pool, "live_training_target")
    selected_features = select_features_dynamic(
        train_eras=window_eras,
        era_corrs=era_corrs,
        feature_pool=feature_pool,
        top_k=TOP_K_FEATURES,
        trailing=TRAILING_ERAS,
    )
    print(f"Selected {len(selected_features)} features from trailing {TRAILING_ERAS} eras")

    # Early stopping: hold out last EARLY_STOPPING_ERAS of the window
    clean = window_df.loc[window_df["live_training_target"].notna()].copy()
    era_list = sorted(clean["era"].astype(str).unique().tolist())
    es_era_set = set(era_list[-EARLY_STOPPING_ERAS:])

    fit_data = clean.loc[~clean["era"].astype(str).isin(es_era_set)].copy()
    es_data = clean.loc[clean["era"].astype(str).isin(es_era_set)].copy()

    fit_data["sample_weight"] = make_era_balanced_weights(fit_data["era"])
    es_data["sample_weight"] = make_era_balanced_weights(es_data["era"])

    print(f"  Fit eras: {len(era_list) - EARLY_STOPPING_ERAS}  ES eras: {EARLY_STOPPING_ERAS}  "
          f"Fit rows: {len(fit_data):,}  ES rows: {len(es_data):,}")

    dtrain = xgb.QuantileDMatrix(
        data=fit_data[selected_features],
        label=fit_data["live_training_target"],
        weight=fit_data["sample_weight"],
        missing=np.nan,
        max_bin=XGB_PARAMS["max_bin"],
    )
    deval = xgb.QuantileDMatrix(
        data=es_data[selected_features],
        label=es_data["live_training_target"],
        weight=es_data["sample_weight"],
        missing=np.nan,
        max_bin=XGB_PARAMS["max_bin"],
        ref=dtrain,
    )

    print("Training XGBoost...")
    booster = xgb.train(
        params=XGB_PARAMS,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        evals=[(deval, "eval")],
        callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)],
    )
    print(f"  Best iteration: {booster.best_iteration}")

    # Capture everything needed for prediction
    _features = selected_features
    _bmark_col = MMC_BENCHMARK_COLUMN
    _neutralization = BENCHMARK_NEUTRALIZATION

    def predict(live_features: pd.DataFrame, live_benchmark_models: pd.DataFrame) -> pd.DataFrame:
        dmat = xgb.DMatrix(data=live_features[_features], missing=np.nan)
        preds = booster.predict(dmat)

        if _neutralization > 0 and _bmark_col in live_benchmark_models.columns:
            exposures = live_benchmark_models[_bmark_col].to_numpy(dtype=np.float64)
            p = preds.reshape(-1, 1).astype(np.float64)
            e = exposures.reshape(-1, 1)
            design = np.hstack([e, np.ones((len(e), 1))])
            coeffs, *_ = np.linalg.lstsq(design, p, rcond=None)
            preds = (p - _neutralization * (design @ coeffs)).reshape(-1)

        return pd.Series(preds, index=live_features.index).to_frame("prediction")

    # Pickle
    print(f"Pickling predict function -> {output_path}")
    p = cloudpickle.dumps(predict)
    with open(output_path, "wb") as f:
        f.write(p)

    # Write metadata so retrains are traceable
    from datetime import date
    meta = {
        "built_date": str(date.today()),
        "target": MAIN_TARGET,
        "model": "xgboost",
        "era_window_start": window_eras[0],
        "era_window_end": window_eras[-1],
        "era_count": len(window_eras),
        "fit_eras": len(era_list) - EARLY_STOPPING_ERAS,
        "es_eras": EARLY_STOPPING_ERAS,
        "best_iteration": int(booster.best_iteration),
        "top_k_features": TOP_K_FEATURES,
        "trailing_eras": TRAILING_ERAS,
        "lookback_eras": LOOKBACK_ERAS,
        "feature_pool_size": len(feature_pool),
        "benchmark_neutralization": BENCHMARK_NEUTRALIZATION,
        "benchmark_col": MMC_BENCHMARK_COLUMN,
        "selected_features": _features,
        "live_training_target": "live_training_target",
        "fallback_target": LIVE_FALLBACK_TARGET,
        "fallback_eras": fallback_era_list,
        "fallback_era_count": len(fallback_era_list),
        "fallback_row_count": int(
            window_df["live_training_target_source"].eq(LIVE_FALLBACK_TARGET).sum()
        ),
        "pickle_size_mb": round(len(p) / 1_000_000, 2),
        "wall_clock_seconds": round(time.time() - t0, 1),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    size_mb = len(p) / 1_000_000
    print(f"Done in {time.time() - t0:.0f}s  |  pickle size: {size_mb:.1f} MB")
    print(f"Era window: {window_eras[0]} – {window_eras[-1]}  ({len(window_eras)} eras)")
    print(f"Features used: {_features[:5]} ... (total {len(_features)})")
    print(f"Metadata written -> {meta_path}")


if __name__ == "__main__":
    main()
