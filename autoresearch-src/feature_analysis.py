"""
Numerai v5.3 — Feature Group Analysis

Analyses per-era Spearman feature-target correlations for all 13 feature groups
across 6 targets in both train and validation splits.

Groups: intelligence, charisma, strength, dexterity, constitution, wisdom,
        agility, serenity, sunshine, rain, midnight, faith, quantum

Outputs saved to artifacts/feature_analysis/:
  {group}_feature_ranks_{train|validation}.csv  — per-feature stats per target
  uniqueness_report.csv                          — faith/rain overlap with small/medium
  target_intercorr.csv                           — target inter-correlations (train)
  combined_all_groups.csv                        — deduplicated cross-group ranking
  recommended_unique_{target}.csv                — stable + not-in-medium per target

Checkpointed: per-group CSVs are skipped if they already exist.
Estimated runtime: ~25 min for a cold run.
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT_OVERRIDE = os.environ.get("NUMERAI_DATA_ROOT")
DATA_ROOT = Path(_DATA_ROOT_OVERRIDE) if _DATA_ROOT_OVERRIDE else (ROOT / "data" / "numerai")
DATA_DIR = DATA_ROOT / "v5.3"
OUT_DIR = ROOT / "artifacts" / "feature_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGETS = [
    "target_ender_20",
    "target_ender_60",
    "target_teager2b_20",
    "target_teager2b_60",
    "target_jasper_20",
    "target_jasper_60",
]

ALL_GROUPS = [
    "intelligence", "charisma", "strength", "dexterity", "constitution",
    "wisdom", "agility", "serenity", "sunshine", "rain", "midnight", "faith",
    "quantum",
]

VALIDATION_TAIL = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_meta() -> dict:
    with open(DATA_DIR / "features.json") as f:
        return json.load(f)


def spearman_corr_matrix_fast(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vectorised Spearman corr of each column of X with y.
    Uses pandas rank (average tie-breaking) for ordinal features.
    Returns (n_features,) float32 array. Caller filters NaN rows in y.
    """
    n, m = X.shape
    y_ranked = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    y_c = y_ranked - y_ranked.mean()
    y_std = y_c.std()
    if y_std == 0:
        return np.zeros(m, dtype=np.float32)
    y_c /= y_std

    X_ranked = pd.DataFrame(X).rank(method="average", axis=0).to_numpy(dtype=np.float64)
    X_c = X_ranked - X_ranked.mean(axis=0)
    X_std = X_c.std(axis=0)
    X_std[X_std == 0] = np.nan

    return ((y_c @ X_c) / (n * X_std)).astype(np.float32)


def per_era_feature_corr(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> pd.DataFrame:
    """Per-era Spearman corr matrix. Returns (eras x features) DataFrame."""
    df = df.reset_index(drop=True)
    feat_arr = df[feature_cols].to_numpy(dtype=np.float32)
    tgt_arr = df[target_col].to_numpy(dtype=np.float32)
    era_arr = df["era"].to_numpy()

    rows = {}
    for era in np.unique(era_arr):
        mask = era_arr == era
        X = feat_arr[mask]
        y = tgt_arr[mask]
        valid_rows = ~np.isnan(y)
        if valid_rows.sum() < 10:
            continue
        rows[era] = spearman_corr_matrix_fast(X[valid_rows], y[valid_rows])

    return pd.DataFrame(rows, index=feature_cols).T


def era_stats(corr_df: pd.DataFrame) -> pd.DataFrame:
    mean = corr_df.mean()
    std = corr_df.std(ddof=0)
    sharpe = mean / std.replace(0, np.nan)
    pct_pos = (corr_df > 0).mean()
    return pd.DataFrame({
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "pct_pos": pct_pos,
        "n_eras": (~corr_df.isna()).sum(),
    })


def load_split_filtered(
    split: str,
    feature_cols: list[str],
    targets: list[str],
    era_filter: list | None = None,
) -> pd.DataFrame:
    columns = ["era"] + feature_cols + [t for t in targets if t not in feature_cols]
    path = DATA_DIR / f"{split}.parquet"
    filters = [("era", "in", era_filter)] if era_filter else None
    return pd.read_parquet(path, columns=columns, filters=filters).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-group analysis
# ---------------------------------------------------------------------------

def analyze_group(
    group_name: str,
    feature_cols: list[str],
    small_set: set,
    medium_set: set,
    val_era_filter: list,
) -> None:
    """Compute and save per-feature stats for one group. Skips if CSVs exist."""
    for split, era_filter in [("train", None), ("validation", val_era_filter)]:
        csv_path = OUT_DIR / f"{group_name}_feature_ranks_{split}.csv"
        if csv_path.exists():
            print(f"  [{group_name}] {split}: cached ({csv_path.name})")
            continue

        t0 = time.time()
        print(f"  [{group_name}] {split}: loading {len(feature_cols)} features...", flush=True)
        df = load_split_filtered(split, feature_cols, TARGETS, era_filter)
        print(f"    rows={len(df):,} eras={df['era'].nunique()}", flush=True)

        all_stats = {}
        for target in TARGETS:
            if target not in df.columns:
                continue
            era_corr = per_era_feature_corr(df, feature_cols, target)
            stats = era_stats(era_corr)
            stats["target"] = target
            all_stats[target] = stats

        combined = pd.concat(all_stats.values())
        combined.index.name = "feature"
        combined = combined.reset_index()
        combined["group"] = group_name
        combined["in_small"] = combined["feature"].isin(small_set)
        combined["in_medium"] = combined["feature"].isin(medium_set)
        combined.to_csv(csv_path, index=False)
        print(f"    saved {csv_path.name} in {time.time()-t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Supplementary analyses
# ---------------------------------------------------------------------------

def save_uniqueness_report(feature_sets: dict, small_set: set, medium_set: set) -> None:
    """Report overlap of faith and rain with small/medium sets."""
    rows = []
    for group in ["faith", "rain"]:
        for feat in feature_sets[group]:
            rows.append({
                "feature": feat,
                "group": group,
                "in_small": feat in small_set,
                "in_medium": feat in medium_set,
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "uniqueness_report.csv", index=False)

    print("\nUniqueness report (faith + rain vs small/medium):")
    print(df.groupby("group")[["in_small", "in_medium"]].mean().round(3).to_string())


def save_target_intercorr() -> None:
    """Spearman inter-correlation of the 6 targets on the last 50 train eras."""
    train_tgt = pd.read_parquet(DATA_DIR / "train.parquet", columns=["era"] + TARGETS)
    all_train_eras = sorted(train_tgt["era"].astype(str).unique())
    recent_eras = set(all_train_eras[-50:])
    train_tgt = train_tgt[train_tgt["era"].astype(str).isin(recent_eras)]
    corr = train_tgt[TARGETS].corr(method="spearman")
    corr.to_csv(OUT_DIR / "target_intercorr.csv")
    print("\nTarget inter-correlation (Spearman, last 50 train eras):")
    print(corr.round(3).to_string())


# ---------------------------------------------------------------------------
# Combined ranking
# ---------------------------------------------------------------------------

def build_combined_ranking(feature_sets: dict, small_set: set, medium_set: set) -> pd.DataFrame:
    """Load all per-group CSVs and build a deduplicated combined ranking."""
    frames = []
    for group in ALL_GROUPS:
        for split in ["train", "validation"]:
            csv = OUT_DIR / f"{group}_feature_ranks_{split}.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                df["split"] = split
                frames.append(df)

    if not frames:
        print("No group CSVs found.")
        return pd.DataFrame()

    all_data = pd.concat(frames, ignore_index=True)
    train_data = all_data[all_data["split"] == "train"].copy()
    val_data = all_data[all_data["split"] == "validation"].copy()

    # For features in multiple groups, keep the record with the best train sharpe
    train_dedup = (
        train_data.sort_values("sharpe", ascending=False)
        .drop_duplicates(subset=["feature", "target"])
        .reset_index(drop=True)
    )
    val_dedup = (
        val_data.sort_values("sharpe", ascending=False)
        .drop_duplicates(subset=["feature", "target"])
        .reset_index(drop=True)
    )

    combined = train_dedup.merge(
        val_dedup[["feature", "target", "mean", "sharpe", "pct_pos"]],
        on=["feature", "target"],
        suffixes=("_train", "_val"),
    )
    combined["combined_score"] = (combined["sharpe_train"] + combined["sharpe_val"]) / 2
    return combined


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def print_group_summary(combined: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("GROUP-LEVEL SUMMARY (train, mean across all features in group)")
    print("="*70)
    for target in TARGETS:
        sub = combined[combined["target"] == target]
        grp = sub.groupby("group")[["mean_train", "sharpe_train", "pct_pos_train"]].mean()
        grp = grp.sort_values("sharpe_train", ascending=False)
        print(f"\n{target}:")
        print(grp.round(4).to_string())


def print_top_features_per_target(combined: pd.DataFrame, n: int = 15) -> None:
    print("\n" + "="*70)
    print(f"TOP-{n} FEATURES PER TARGET (ranked by combined train+val sharpe)")
    print("="*70)
    for target in TARGETS:
        sub = combined[combined["target"] == target].nlargest(n, "combined_score")
        cols = ["feature", "group", "in_medium",
                "mean_train", "pct_pos_train", "sharpe_train",
                "mean_val", "pct_pos_val", "sharpe_val"]
        print(f"\n{target}:")
        print(sub[cols].round(4).to_string(index=False))


def print_mmc_unique_top(combined: pd.DataFrame, n: int = 15) -> None:
    print("\n" + "="*70)
    print(f"TOP-{n} MMC-UNIQUE FEATURES PER TARGET (not in medium, best combined sharpe)")
    print("="*70)
    unique = combined[~combined["in_medium"]]
    for target in TARGETS:
        sub = unique[unique["target"] == target].nlargest(n, "combined_score")
        cols = ["feature", "group",
                "mean_train", "pct_pos_train", "sharpe_train",
                "mean_val", "pct_pos_val", "sharpe_val"]
        print(f"\n{target}:")
        print(sub[cols].round(4).to_string(index=False))


def print_rain_spotlight(combined: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("RAIN GROUP SPOTLIGHT (0% in medium = max MMC novelty)")
    print("="*70)
    rain = combined[combined["group"] == "rain"]
    for target in TARGETS:
        sub = rain[rain["target"] == target].nlargest(10, "combined_score")
        cols = ["feature", "mean_train", "pct_pos_train", "sharpe_train",
                "mean_val", "pct_pos_val", "sharpe_val"]
        print(f"\n{target}:")
        print(sub[cols].round(4).to_string(index=False))


def print_stable_universal_features(combined: pd.DataFrame) -> None:
    print("\n" + "="*70)
    print("STABLE UNIVERSAL FEATURES (pct_pos>0.60 both splits, 4+ of 6 targets)")
    print("="*70)
    stable = combined[
        (combined["mean_train"] > 0) &
        (combined["mean_val"] > 0) &
        (combined["pct_pos_train"] > 0.60) &
        (combined["pct_pos_val"] > 0.60)
    ]
    freq = stable.groupby("feature").agg(
        n_targets=("target", "count"),
        group=("group", "first"),
        in_medium=("in_medium", "first"),
        avg_train_sharpe=("sharpe_train", "mean"),
        avg_val_sharpe=("sharpe_val", "mean"),
    ).reset_index()
    freq = freq[freq["n_targets"] >= 4].sort_values(
        ["n_targets", "avg_val_sharpe"], ascending=[False, False]
    )
    print(freq.round(4).to_string(index=False))


def save_combined_csvs(combined: pd.DataFrame) -> None:
    combined.to_csv(OUT_DIR / "combined_all_groups.csv", index=False)
    for target in TARGETS:
        sub = combined[combined["target"] == target]
        stable_unique = sub[
            (~sub["in_medium"]) &
            (sub["mean_train"] > 0) & (sub["mean_val"] > 0) &
            (sub["pct_pos_train"] > 0.60) & (sub["pct_pos_val"] > 0.60)
        ].sort_values("combined_score", ascending=False)
        stable_unique.to_csv(OUT_DIR / f"recommended_unique_{target}.csv", index=False)

    # Legacy-schema artifacts consumed by site_builder's feature-analysis
    # report (group_summary_validation.csv + recommended_features_{target}.csv).
    # Summary: one aggregate row per target over all analyzed features on the
    # validation split. Recommended: stable features (positive mean and
    # pct_pos>0.60 on both splits), top 60 by validation sharpe.
    summary_rows = []
    for target in TARGETS:
        sub = combined[combined["target"] == target]
        summary_rows.append({
            "group": "other",
            "mean": sub["mean_val"].mean(),
            "sharpe": sub["sharpe_val"].mean(),
            "pct_pos": sub["pct_pos_val"].mean(),
            "target": target,
            "split": "validation",
        })
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "group_summary_validation.csv", index=False)

    legacy_cols = {
        "mean_train": "tr_mean", "sharpe_train": "tr_sharpe",
        "pct_pos_train": "tr_pct_pos", "mean_val": "val_mean",
        "sharpe_val": "val_sharpe", "pct_pos_val": "val_pct_pos",
    }
    for target in TARGETS:
        sub = combined[combined["target"] == target]
        stable = sub[
            (sub["mean_train"] > 0) & (sub["mean_val"] > 0) &
            (sub["pct_pos_train"] > 0.60) & (sub["pct_pos_val"] > 0.60)
        ].sort_values("sharpe_val", ascending=False).head(60)
        stable = stable.rename(columns=legacy_cols)
        stable[[
            "feature", "tr_mean", "std", "tr_sharpe", "tr_pct_pos", "n_eras",
            "target", "split", "group", "in_small", "in_medium",
            "val_mean", "val_sharpe", "val_pct_pos",
        ]].to_csv(OUT_DIR / f"recommended_features_{target}.csv", index=False)
    print(f"\nAll CSVs saved to {OUT_DIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    t_start = time.time()

    print("=" * 70)
    print("Numerai v5.3 - Feature Group Analysis")
    print("=" * 70)

    meta = load_meta()
    feature_sets = meta["feature_sets"]
    small_set = set(feature_sets["small"])
    medium_set = set(feature_sets["medium"])

    # Validation era filter (last VALIDATION_TAIL eras, computed once)
    print("\nIdentifying validation tail eras...")
    era_only = pd.read_parquet(DATA_DIR / "validation.parquet", columns=["era"])
    all_val_eras = sorted(era_only["era"].unique())
    val_era_filter = list(all_val_eras[-VALIDATION_TAIL:])
    print(f"  Last {VALIDATION_TAIL} val eras: {val_era_filter[0]} -> {val_era_filter[-1]}")

    # Per-group Spearman analysis with checkpointing
    print("\nProcessing groups...")
    for group in ALL_GROUPS:
        feats = feature_sets[group]
        print(f"\n[{group}] {len(feats)} features")
        analyze_group(group, feats, small_set, medium_set, val_era_filter)

    # Supplementary analyses
    print("\n" + "="*70)
    print("Supplementary analyses...")
    save_uniqueness_report(feature_sets, small_set, medium_set)
    save_target_intercorr()

    # Combined cross-group ranking
    print("\n" + "="*70)
    print("Building combined cross-group ranking...")
    combined = build_combined_ranking(feature_sets, small_set, medium_set)
    print(f"Combined: {len(combined)} (feature, target) pairs, {combined['feature'].nunique()} unique features")

    print_group_summary(combined)
    print_top_features_per_target(combined, n=15)
    print_mmc_unique_top(combined, n=15)
    print_rain_spotlight(combined)
    print_stable_universal_features(combined)
    save_combined_csvs(combined)

    print(f"\nTotal runtime: {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
