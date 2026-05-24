import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from numerapi import NumerAPI
from scipy.stats import norm


ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT_OVERRIDE = os.environ.get("NUMERAI_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT_OVERRIDE) if _DATA_ROOT_OVERRIDE else (ROOT / "data" / "numerai")
DATA_VERSION = "v5.2"
DATA_DIR = DATA_ROOT / DATA_VERSION
ARTIFACTS_DIR = ROOT / "artifacts"

TIME_BUDGET_SECONDS = 300
DEFAULT_FEATURE_SET = "small"
DEFAULT_TARGET = "target"
DEFAULT_AUX_TARGETS = ["target_cyrusd_20", "target_teager2b_20"]

DATASET_FILES = {
    "features": "features.json",
    "train": "train.parquet",
    "validation": "validation.parquet",
    "live": "live.parquet",
    "meta_model": "meta_model.parquet",
    "train_benchmarks": "train_benchmark_models.parquet",
    "validation_benchmarks": "validation_benchmark_models.parquet",
    "live_benchmarks": "live_benchmark_models.parquet",
}


def get_data_path(name: str) -> Path:
    return DATA_DIR / DATASET_FILES[name]


def _restore_index_column(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.index.name and frame.index.name not in frame.columns:
        return frame.reset_index()
    return frame


def _build_split_filters(split: str, era_filter: list | None = None) -> list[tuple[str, str, object]] | None:
    filters: list[tuple[str, str, object]] = []
    if split == "validation":
        # validation.parquet also contains live test eras; exclude them from
        # any labeled-history workflow so era windows only advance on real
        # validation rows.
        filters.append(("data_type", "!=", "test"))
    if era_filter:
        filters.append(("era", "in", era_filter))
    return filters or None


ERA_TRACKER_PATH = DATA_DIR / "era_tracker.json"


def check_validation_era_freshness() -> None:
    """Read max era from validation.parquet, compare to era_tracker.json, and warn if stale."""
    val_path = get_data_path("validation")
    max_era = pd.read_parquet(
        val_path,
        columns=["era"],
        filters=_build_split_filters("validation"),
    )["era"].max()

    if ERA_TRACKER_PATH.exists():
        with open(ERA_TRACKER_PATH, "r") as f:
            tracker = json.load(f)
        prev_era = tracker.get("max_era")
        if max_era == prev_era:
            print(
                f"WARNING: validation max era is still {max_era} — "
                "data may not have updated yet. Check that the new round has released."
            )
            return
        print(f"Validation data updated: era {prev_era} -> {max_era}")
    else:
        print(f"Initializing era tracker at max era {max_era}")

    with open(ERA_TRACKER_PATH, "w") as f:
        json.dump({"max_era": max_era}, f)


def ensure_data(download: bool = True, include_live: bool = True) -> None:
    required = [
        "features",
        "train",
        "validation",
        "meta_model",
        "train_benchmarks",
        "validation_benchmarks",
    ]
    if include_live:
        required.extend(["live", "live_benchmarks"])

    missing = [name for name in required if not get_data_path(name).exists()]
    if not missing:
        return
    if not download:
        missing_files = ", ".join(DATASET_FILES[name] for name in missing)
        raise FileNotFoundError(
            f"Missing Numerai data files: {missing_files}. Run `python autoresearch-src/prepare.py` first."
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    napi = NumerAPI()
    for name in missing:
        remote = f"{DATA_VERSION}/{DATASET_FILES[name]}"
        local = str(get_data_path(name))
        print(f"Downloading {remote} -> {local}")
        napi.download_dataset(remote, local)


def load_feature_metadata(require_all: bool = False) -> dict:
    if require_all:
        ensure_data(download=True, include_live=False)
    elif not get_data_path("features").exists():
        ensure_data(download=True, include_live=False)
    with open(get_data_path("features"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def get_feature_set(name: str = DEFAULT_FEATURE_SET) -> list[str]:
    metadata = load_feature_metadata()
    feature_sets = metadata["feature_sets"]
    if name not in feature_sets:
        valid = ", ".join(sorted(feature_sets))
        raise KeyError(f"Unknown feature set `{name}`. Valid sets: {valid}")
    return feature_sets[name]


def get_targets() -> list[str]:
    metadata = load_feature_metadata()
    return metadata["targets"]


def read_split(
    split: str,
    feature_set: str = DEFAULT_FEATURE_SET,
    targets: list[str] | None = None,
) -> pd.DataFrame:
    if split not in {"train", "validation", "live"}:
        raise ValueError(f"Unsupported split `{split}`")

    ensure_data(download=True, include_live=(split == "live"))
    features = get_feature_set(feature_set)
    columns = ["era", "id", *features]
    if split != "live":
        for target in targets or [DEFAULT_TARGET]:
            if target not in columns:
                columns.append(target)

    return _restore_index_column(
        pd.read_parquet(
            get_data_path(split),
            columns=columns,
            filters=_build_split_filters(split),
        )
    )


def read_split_custom(
    split: str,
    features: list[str],
    targets: list[str] | None = None,
    era_filter: list | None = None,
) -> pd.DataFrame:
    """Load a split with an explicit feature list instead of a named feature set.

    Use this when train.py needs a custom feature pool that does not correspond
    to any Numerai-defined set — e.g. the dynamic pool built from faith + wisdom
    + strength + intelligence + hand-picked rain/sunshine features.

    Args:
        split:       "train", "validation", or "live"
        features:    explicit list of feature column names to load
        targets:     target columns to include; defaults to [DEFAULT_TARGET]
        era_filter:  optional list of era values to row-filter (passed to parquet)
    """
    if split not in {"train", "validation", "live"}:
        raise ValueError(f"Unsupported split `{split}`")
    ensure_data(download=True, include_live=(split == "live"))
    columns = ["era", *features]
    if split != "live":
        for target in targets or [DEFAULT_TARGET]:
            if target not in columns:
                columns.append(target)
    return _restore_index_column(
        pd.read_parquet(
            get_data_path(split),
            columns=columns,
            filters=_build_split_filters(split, era_filter),
        )
    )


def read_benchmarks(split: str) -> pd.DataFrame:
    mapping = {
        "train": "train_benchmarks",
        "validation": "validation_benchmarks",
        "live": "live_benchmarks",
    }
    if split not in mapping:
        raise ValueError(f"Unsupported benchmark split `{split}`")
    ensure_data(download=True, include_live=(split == "live"))
    return _restore_index_column(pd.read_parquet(get_data_path(mapping[split])))


def read_meta_model() -> pd.DataFrame:
    ensure_data(download=True, include_live=False)
    return _restore_index_column(pd.read_parquet(get_data_path("meta_model")))


def _rank_uniform(values: np.ndarray) -> np.ndarray:
    series = pd.Series(values)
    ranked = (series.rank(method="average").to_numpy(dtype=np.float64) - 0.5) / len(series)
    return np.clip(ranked, 1e-6, 1.0 - 1e-6)


def gaussianize(values: np.ndarray) -> np.ndarray:
    return norm.ppf(_rank_uniform(values))


def numerai_corr(preds: np.ndarray, target: np.ndarray) -> float:
    preds_g = gaussianize(preds)
    target_centered = target.astype(np.float64) - float(np.mean(target))
    preds_p15 = np.sign(preds_g) * np.abs(preds_g) ** 1.5
    target_p15 = np.sign(target_centered) * np.abs(target_centered) ** 1.5
    corr = np.corrcoef(preds_p15, target_p15)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def normalize_for_mmc(values: np.ndarray) -> np.ndarray:
    normalized = gaussianize(values)
    std = np.std(normalized)
    if std == 0 or np.isnan(std):
        return normalized
    return normalized / std


def neutralize(preds: np.ndarray, exposures: np.ndarray, proportion: float = 1.0) -> np.ndarray:
    preds = np.asarray(preds, dtype=np.float64).reshape(-1, 1)
    exposures = np.asarray(exposures, dtype=np.float64)
    if exposures.ndim == 1:
        exposures = exposures.reshape(-1, 1)
    design = np.hstack([exposures, np.ones((len(exposures), 1), dtype=np.float64)])
    coeffs, *_ = np.linalg.lstsq(design, preds, rcond=None)
    adjustment = design @ coeffs
    result = preds - proportion * adjustment
    return result.reshape(-1)


def per_era_corr(df: pd.DataFrame, pred_col: str, target_col: str = DEFAULT_TARGET) -> pd.Series:
    rows = []
    for era, chunk in df.groupby("era", sort=True):
        rows.append((era, numerai_corr(chunk[pred_col].to_numpy(), chunk[target_col].to_numpy())))
    return pd.Series(dict(rows), name="corr")


def per_era_mmc(
    df: pd.DataFrame,
    pred_col: str,
    meta_col: str = "numerai_meta_model",
    target_col: str = DEFAULT_TARGET,
) -> pd.Series:
    rows = []
    for era, chunk in df.groupby("era", sort=True):
        pred = normalize_for_mmc(chunk[pred_col].to_numpy())
        meta = normalize_for_mmc(chunk[meta_col].to_numpy())
        target = chunk[target_col].to_numpy(dtype=np.float64)
        neutral_pred = neutralize(pred, meta, proportion=1.0)
        centered_target = target - target.mean()
        mmc = float(np.dot(centered_target, neutral_pred) / len(chunk))
        rows.append((era, mmc))
    return pd.Series(dict(rows), name="mmc")


def per_era_bmc(
    df: pd.DataFrame,
    pred_col: str,
    benchmark_col: str,
    target_col: str = DEFAULT_TARGET,
) -> pd.Series:
    rows = []
    for era, chunk in df.groupby("era", sort=True):
        pred = normalize_for_mmc(chunk[pred_col].to_numpy())
        bench = normalize_for_mmc(chunk[benchmark_col].to_numpy())
        target = chunk[target_col].to_numpy(dtype=np.float64)
        neutral_pred = neutralize(pred, bench, proportion=1.0)
        centered_target = target - target.mean()
        bmc = float(np.dot(centered_target, neutral_pred) / len(chunk))
        rows.append((era, bmc))
    return pd.Series(dict(rows), name="bmc")


def era_stats(series: pd.Series, prefix: str) -> dict[str, float]:
    values = series.astype(float)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    sharpe = mean / std if std > 0 else 0.0
    cumulative = values.cumsum()
    drawdown = cumulative - cumulative.cummax()
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_sharpe": float(sharpe),
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_max_drawdown": float(drawdown.min()),
    }


def research_score(metrics: dict[str, float]) -> float:
    return float(
        0.60 * metrics["val_mmc_mean"]
        + 0.30 * metrics["val_corr_mean"]
        + 0.10 * metrics["val_bmc_mean"]
    )


def evaluate_validation(
    validation: pd.DataFrame,
    predictions: pd.Series,
    benchmark_col: str | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    frame = validation[["era", "id", DEFAULT_TARGET, "numerai_meta_model"]].copy()
    frame["prediction"] = predictions.to_numpy(dtype=np.float64)
    if benchmark_col is not None:
        frame[benchmark_col] = validation[benchmark_col].to_numpy(dtype=np.float64)

    corr_by_era = per_era_corr(frame, "prediction")
    mmc_by_era = per_era_mmc(frame, "prediction")
    metrics = {}
    metrics.update(era_stats(corr_by_era, "val_corr"))
    metrics.update(era_stats(mmc_by_era, "val_mmc"))

    if benchmark_col is not None:
        bmc_by_era = per_era_bmc(frame, "prediction", benchmark_col)
        metrics.update(era_stats(bmc_by_era, "val_bmc"))
    else:
        zero_bmc = pd.Series(np.zeros(len(corr_by_era)), index=corr_by_era.index)
        metrics.update(era_stats(zero_bmc, "val_bmc"))

    metrics["research_score"] = research_score(metrics)
    metrics["era_count"] = float(len(corr_by_era))

    per_era = pd.DataFrame(
        {
            "corr": corr_by_era,
            "mmc": mmc_by_era,
        }
    )
    if benchmark_col is not None:
        per_era["bmc"] = per_era_bmc(frame, "prediction", benchmark_col)
    return metrics, per_era


def pick_primary_benchmark(validation_benchmarks: pd.DataFrame) -> str:
    benchmark_cols = [col for col in validation_benchmarks.columns if col not in {"id", "era"}]
    if not benchmark_cols:
        raise ValueError("No benchmark columns found.")
    return benchmark_cols[0]


def print_summary() -> None:
    print(f"Data directory: {DATA_DIR}")
    print(f"Version: {DATA_VERSION}")
    if get_data_path("features").exists():
        metadata = load_feature_metadata()
        print(f"Feature sets: {', '.join(sorted(metadata['feature_sets']))}")
        print(f"Targets: {len(metadata['targets'])}")
    else:
        print("Feature metadata: missing")
    for split in ["train", "validation", "live"]:
        path = get_data_path(split)
        print(f"{split}: {'present' if path.exists() else 'missing'} -> {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Numerai data for autoresearch.")
    parser.add_argument("--check", action="store_true", help="Only verify files exist.")
    parser.add_argument(
        "--skip-live",
        action="store_true",
        help="Skip downloading live files when preparing locally.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print a brief dataset summary after checking data.",
    )
    args = parser.parse_args()

    try:
        ensure_data(download=not args.check, include_live=not args.skip_live)
        print("Numerai data is ready.")
    except FileNotFoundError as exc:
        print(str(exc))
        if args.check or args.summary:
            print_summary()
        sys.exit(1)

    if args.summary or args.check:
        print_summary()


if __name__ == "__main__":
    main()
