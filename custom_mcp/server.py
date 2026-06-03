"""
Numerai Weekly Submission MCP Server

Tools for the full weekly workflow:
  run_weekly_retrain       — run custom_mcp/make_submission.py, download fresh data, train
  check_retrain_status     — poll the background training job
  get_training_summary     — read latest metadata JSON, return config snapshot
  compare_weekly_features  — diff this week's features vs last week by group
  generate_weekly_report   — write docs/YYYY-WW_weekly_report.md

Upload is handled by the official Numerai MCP (mcp__numerai__upload_model).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_MCP_DIR = Path(__file__).resolve().parent
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
REPORTS_DIR = PROJECT_ROOT / "docs"
PYTHON_EXE = os.environ.get("NUMERAI_PYTHON", sys.executable)
SOURCE_DIR = PROJECT_ROOT / "autoresearch-src"
MAKE_SUBMISSION = str(CUSTOM_MCP_DIR / "make_submission.py")
FEATURES_JSON = PROJECT_ROOT / "data" / "numerai" / "v5.2" / "features.json"
CANDIDATE_GROUPS = ["faith", "wisdom", "strength", "intelligence"]

EXTRA_FEATURES: set[str] = {
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
}

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))

from custom_mcp.site_builder import build_dashboard, build_report_html
from prepare import refresh_data

mcp = FastMCP("numerai-weekly")

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_feature_group_cache: dict[str, str] | None = None


def _load_feature_group_lookup() -> dict[str, str]:
    """Build feature_name → group lookup from features.json + EXTRA_FEATURES."""
    global _feature_group_cache
    if _feature_group_cache is not None:
        return _feature_group_cache
    lookup: dict[str, str] = {}
    if FEATURES_JSON.exists():
        with open(FEATURES_JSON, encoding="utf-8") as f:
            data = json.load(f)
        feature_sets = data.get("feature_sets", {})
        for group in CANDIDATE_GROUPS:
            for feat in feature_sets.get(group, []):
                lookup[feat] = group
    for feat in EXTRA_FEATURES:
        lookup[feat] = "extra"
    _feature_group_cache = lookup
    return lookup


def _classify_feature(name: str) -> str:
    return _load_feature_group_lookup().get(name, "other")


def _group_features(features: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for f in features:
        g = _classify_feature(f)
        groups.setdefault(g, []).append(f)
    return groups


def _sorted_metas() -> list[Path]:
    """Return all *_meta.json files sorted oldest → newest by mtime."""
    return sorted(SUBMISSIONS_DIR.glob("*_meta.json"), key=lambda p: p.stat().st_mtime)


def _load_meta(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _format_metric(value: object, digits: int = 5) -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "n/a"


def _safe_table_cell(value: object) -> str:
    text = str(value)
    return " ".join(text.replace("|", "").split())


def _format_metric_block_rows(title: str, rows: list[tuple[str, str]]) -> str:
    if not rows:
        return ""
    lines = [f"**{title}**", "", "| Metric | Value |", "| --- | --- |"]
    for label, value in rows:
        lines.append(f"| {label} | {value} |")
    return "\n".join(lines)


def _current_live_prediction_era(meta: dict) -> str | None:
    try:
        current_max = _current_max_labeled_era()
        if current_max is not None:
            width = len(str(current_max))
            return f"{int(current_max) + 1:0{width}d}"
    except Exception:
        pass

    try:
        era_end = str(meta.get("era_window_end"))
        return f"{int(era_end) + 1:0{len(era_end)}d}"
    except Exception:
        return None


def _compute_live_report_metrics(meta_path: Path, meta: dict) -> dict[str, object]:
    cached = meta.get("report_metrics")
    if isinstance(cached, dict) and cached:
        return cached

    pkl_path = SUBMISSIONS_DIR / (meta_path.stem.replace("_meta", "") + ".pkl")
    if not pkl_path.exists():
        return {}

    try:
        import cloudpickle
        import pandas as pd
        from prepare import era_stats, get_data_path, per_era_bmc, per_era_corr, read_split_custom

        selected_features = list(meta.get("selected_features", []))
        main_target = str(meta.get("target", "target_ender_60"))
        corr_target = str(meta.get("fallback_target", "target_ender_20"))
        benchmark_col = str(meta.get("benchmark_col", "v52_lgbm_ender20"))
        es_eras = int(meta.get("es_eras", 10))
        era_start = int(meta.get("era_window_start"))
        era_end = int(meta.get("era_window_end"))
        era_width = max(len(str(meta.get("era_window_start"))), len(str(meta.get("era_window_end"))))
        window_eras = [f"{era:0{era_width}d}" for era in range(era_start, era_end + 1)]

        with pkl_path.open("rb") as handle:
            predict_fn = cloudpickle.load(handle)

        def build_split(split: str) -> pd.DataFrame:
            feature_frame = read_split_custom(
                split,
                features=selected_features,
                targets=[main_target, corr_target],
                era_filter=window_eras,
            )
            benchmark_name = "train_benchmarks" if split == "train" else "validation_benchmarks"
            benchmark_frame = pd.read_parquet(
                get_data_path(benchmark_name),
                columns=["id", "era", benchmark_col],
                filters=[("era", "in", window_eras)],
            )
            merged = feature_frame.merge(benchmark_frame, on=["id", "era"], how="left")
            merged["live_training_target"] = merged[main_target].where(
                merged[main_target].notna(),
                merged[corr_target],
            )
            merged["live_training_target_source"] = main_target
            merged.loc[
                merged[main_target].isna(),
                "live_training_target_source",
            ] = corr_target
            return merged

        train_df = build_split("train")
        validation_df = build_split("validation")
        history_df = pd.concat([train_df, validation_df], ignore_index=True)
        history_df = history_df.loc[history_df["live_training_target"].notna()].copy()

        era_list = sorted(history_df["era"].astype(str).unique().tolist())
        if len(era_list) <= es_eras:
            return {}
        es_era_set = set(era_list[-es_eras:])

        fit_data = history_df.loc[~history_df["era"].astype(str).isin(es_era_set)].copy()
        eval_data = history_df.loc[history_df["era"].astype(str).isin(es_era_set)].copy()

        def score_frame(frame: pd.DataFrame) -> dict[str, float]:
            predictions = predict_fn(frame[selected_features], frame[[benchmark_col]])
            scored = frame[["era", corr_target, benchmark_col]].copy()
            scored["prediction"] = predictions["prediction"].to_numpy()
            corr_by_era = per_era_corr(
                scored[["era", "prediction", corr_target]].dropna(subset=[corr_target]),
                "prediction",
                target_col=corr_target,
            )
            mmc_by_era = per_era_bmc(
                scored[["era", "prediction", benchmark_col, corr_target]].dropna(
                    subset=[benchmark_col, corr_target]
                ),
                "prediction",
                benchmark_col=benchmark_col,
                target_col=corr_target,
            )
            metrics: dict[str, float] = {}
            metrics.update(era_stats(corr_by_era, "corr"))
            metrics.update(era_stats(mmc_by_era, "mmc"))
            return metrics

        fit_metrics = score_frame(fit_data)
        eval_metrics = score_frame(eval_data)
        report_metrics = {
            "training_target": main_target,
            "validation_target": corr_target,
            "benchmark_col": benchmark_col,
            "fit_corr_mean": fit_metrics.get("corr_mean"),
            "fit_corr_sharpe": fit_metrics.get("corr_sharpe"),
            "fit_mmc_mean": fit_metrics.get("mmc_mean"),
            "fit_mmc_sharpe": fit_metrics.get("mmc_sharpe"),
            "val_corr_mean": eval_metrics.get("corr_mean"),
            "val_corr_sharpe": eval_metrics.get("corr_sharpe"),
            "val_mmc_mean": eval_metrics.get("mmc_mean"),
            "val_mmc_sharpe": eval_metrics.get("mmc_sharpe"),
        }
        meta["report_metrics"] = report_metrics
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return report_metrics
    except Exception:
        return {}


def _compute_live_prediction_diagnostics(meta_path: Path, meta: dict) -> dict[str, object]:
    cached = meta.get("live_diagnostics")
    if isinstance(cached, dict) and cached:
        return cached

    pkl_path = SUBMISSIONS_DIR / (meta_path.stem.replace("_meta", "") + ".pkl")
    if not pkl_path.exists():
        return {"status": "error", "message": f"Missing pickle artifact: {pkl_path.name}"}

    try:
        import cloudpickle
        import numpy as np
        import pandas as pd
        from prepare import ensure_data, get_data_path, read_split_custom

        selected_features = list(meta.get("selected_features", []))
        benchmark_col = str(meta.get("benchmark_col", "v52_lgbm_ender20"))

        ensure_data(download=True, include_live=True)
        live_features = read_split_custom("live", features=selected_features)
        live_benchmarks = pd.read_parquet(
            get_data_path("live_benchmarks"),
            columns=["id", benchmark_col],
        )

        if "id" in live_features.columns:
            scoring = live_features.merge(
                live_benchmarks,
                on="id",
                how="left",
                validate="one_to_one",
            )
        else:
            scoring = live_features.copy()
            scoring[benchmark_col] = live_benchmarks[benchmark_col].to_numpy()

        with pkl_path.open("rb") as handle:
            predict_fn = cloudpickle.load(handle)
        predictions = predict_fn(scoring[selected_features], scoring[[benchmark_col]])
        scoring["prediction"] = predictions["prediction"].to_numpy(dtype=np.float64)

        pred = scoring["prediction"].astype(float)
        bench = scoring[benchmark_col].astype(float)
        live_eras = sorted(scoring["era"].astype(str).unique().tolist()) if "era" in scoring.columns else []
        p01 = float(pred.quantile(0.01))
        p99 = float(pred.quantile(0.99))
        spread_99_1 = p99 - p01
        duplicate_fraction = float(1.0 - (pred.nunique() / len(pred)))
        benchmark_corr = float(pred.corr(bench))

        summary: dict[str, object] = {
            "status": "pass",
            "model_era_window": f"{meta.get('era_window_start')}-{meta.get('era_window_end')}",
            "built_date": meta.get("built_date"),
            "live_eras": live_eras,
            "row_count": int(len(scoring)),
            "prediction_mean": float(pred.mean()),
            "prediction_std": float(pred.std(ddof=0)),
            "prediction_min": float(pred.min()),
            "prediction_p01": p01,
            "prediction_p05": float(pred.quantile(0.05)),
            "prediction_median": float(pred.median()),
            "prediction_p95": float(pred.quantile(0.95)),
            "prediction_p99": p99,
            "prediction_max": float(pred.max()),
            "prediction_skew": float(pred.skew()),
            "prediction_kurtosis": float(pred.kurt()),
            "prediction_spread_p99_p01": float(spread_99_1),
            "unique_predictions": int(pred.nunique()),
            "duplicate_fraction": duplicate_fraction,
            "benchmark_col": benchmark_col,
            "benchmark_corr": benchmark_corr,
            "artifacts": {},
            "checks": [],
        }

        checks: list[dict[str, str]] = []

        def add_check(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        if summary["row_count"] == 0:
            add_check("row_count", "fail", "Live split produced zero rows.")
        else:
            add_check("row_count", "pass", f"Scored {summary['row_count']:,} live rows.")

        pred_std = float(summary["prediction_std"])
        if not np.isfinite(pred_std):
            add_check("dispersion", "fail", "Prediction standard deviation is not finite.")
        elif pred_std < 0.001:
            add_check("dispersion", "fail", f"Prediction std is only {pred_std:.6f}.")
        elif pred_std < 0.002:
            add_check("dispersion", "warn", f"Prediction std is narrow at {pred_std:.6f}.")
        else:
            add_check("dispersion", "pass", f"Prediction std is {pred_std:.6f}.")

        if not np.isfinite(spread_99_1):
            add_check("tail_spread", "fail", "Prediction p99-p01 spread is not finite.")
        elif spread_99_1 < 0.005:
            add_check("tail_spread", "fail", f"Prediction p99-p01 spread is only {spread_99_1:.6f}.")
        elif spread_99_1 < 0.010:
            add_check("tail_spread", "warn", f"Prediction p99-p01 spread is narrow at {spread_99_1:.6f}.")
        else:
            add_check("tail_spread", "pass", f"Prediction p99-p01 spread is {spread_99_1:.6f}.")

        if duplicate_fraction > 0.25:
            add_check("duplicates", "fail", f"Duplicate prediction fraction is {duplicate_fraction:.3%}.")
        elif duplicate_fraction > 0.05:
            add_check("duplicates", "warn", f"Duplicate prediction fraction is {duplicate_fraction:.3%}.")
        else:
            add_check("duplicates", "pass", f"Duplicate prediction fraction is {duplicate_fraction:.3%}.")

        abs_benchmark_corr = abs(benchmark_corr) if np.isfinite(benchmark_corr) else np.inf
        if not np.isfinite(benchmark_corr):
            add_check("benchmark_corr", "fail", "Correlation to the benchmark model is not finite.")
        elif abs_benchmark_corr > 0.8:
            add_check("benchmark_corr", "fail", f"abs corr(pred, {benchmark_col}) is {abs_benchmark_corr:.3f}.")
        elif abs_benchmark_corr > 0.5:
            add_check("benchmark_corr", "warn", f"abs corr(pred, {benchmark_col}) is {abs_benchmark_corr:.3f}.")
        else:
            add_check("benchmark_corr", "pass", f"abs corr(pred, {benchmark_col}) is {abs_benchmark_corr:.3f}.")

        severity_order = {"pass": 0, "warn": 1, "fail": 2}
        summary["checks"] = checks
        summary["status"] = max(checks, key=lambda item: severity_order[item["status"]])["status"]
        summary["ready_for_submission"] = summary["status"] != "fail"

        artifacts_dir = PROJECT_ROOT / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)
        stem = f"train_{meta.get('era_window_start')}_{meta.get('era_window_end')}"
        plot_path = artifacts_dir / f"live_prediction_distribution_{stem}.png"
        csv_path = artifacts_dir / f"live_predictions_{stem}.csv"
        json_path = artifacts_dir / f"live_prediction_distribution_{stem}_summary.json"

        export_cols = [col for col in ["id", "era", benchmark_col, "prediction"] if col in scoring.columns]
        scoring[export_cols].to_csv(csv_path, index=False)

        summary["artifacts"] = {
            "plot_path": str(plot_path),
            "csv_path": str(csv_path),
            "summary_path": str(json_path),
        }
        summary["plot_generated"] = False

        try:
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            rank_pct = pred.rank(method="average", pct=True)
            sorted_pred = np.sort(pred.to_numpy(dtype=np.float64))
            live_era_label = live_eras[0] if len(live_eras) == 1 else ", ".join(live_eras[:3])

            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            fig.suptitle(
                "Live prediction distribution | "
                f"model train eras {meta.get('era_window_start')}-{meta.get('era_window_end')} | "
                f"live era {live_era_label or 'unknown'}",
                fontsize=14,
            )

            ax = axes[0, 0]
            ax.hist(pred, bins=50, color="#7aa6c2", edgecolor="white", linewidth=0.4)
            ax.axvline(p01, color="#d1495b", linestyle="--", linewidth=1)
            ax.axvline(float(summary["prediction_median"]), color="#222222", linestyle="--", linewidth=1)
            ax.axvline(p99, color="#d1495b", linestyle="--", linewidth=1)
            ax.set_title("Raw prediction histogram")
            ax.set_xlabel("prediction")
            ax.set_ylabel("count")

            ax = axes[0, 1]
            ax.plot(
                np.linspace(0, 1, len(sorted_pred), endpoint=False),
                sorted_pred,
                color="#1d3557",
                linewidth=1.6,
            )
            ax.set_title("Sorted prediction curve")
            ax.set_xlabel("rank percentile")
            ax.set_ylabel("prediction")

            ax = axes[1, 0]
            sample = scoring[[benchmark_col, "prediction"]].dropna()
            if len(sample) > 12000:
                sample = sample.sample(12000, random_state=42)
            ax.scatter(
                sample[benchmark_col],
                sample["prediction"],
                s=8,
                alpha=0.25,
                color="#457b9d",
                edgecolors="none",
            )
            ax.set_title(f"Prediction vs {benchmark_col} (corr={benchmark_corr:.3f})")
            ax.set_xlabel(benchmark_col)
            ax.set_ylabel("prediction")

            ax = axes[1, 1]
            ax.hist(rank_pct, bins=20, color="#8d99ae", edgecolor="white", linewidth=0.4)
            ax.set_title("Percentile-ranked predictions")
            ax.set_xlabel("rank percentile")
            ax.set_ylabel("count")
            text = (
                f"rows: {summary['row_count']:,}\n"
                f"mean/std: {summary['prediction_mean']:.5f} / {pred_std:.5f}\n"
                f"p01/p50/p99: {p01:.5f} / {summary['prediction_median']:.5f} / {p99:.5f}\n"
                f"spread(99-1): {spread_99_1:.5f}\n"
                f"dup frac: {duplicate_fraction:.6f}\n"
                f"bench corr: {benchmark_corr:.3f}"
            )
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
            )

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            fig.savefig(plot_path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            summary["plot_generated"] = True
        except Exception as exc:
            summary["plot_error"] = str(exc)

        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        meta["live_diagnostics"] = summary
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return summary
    except Exception as exc:
        return {"status": "error", "message": str(exc), "ready_for_submission": False}




# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------

def _current_max_labeled_era() -> str | None:
    """Return the latest usable labeled era across train + validation, excluding validation test rows."""
    try:
        import pyarrow.parquet as pq
        data_dir = PROJECT_ROOT / "data" / "numerai" / "v5.2"
        eras: list[str] = []
        train_path = data_dir / "train.parquet"
        if train_path.exists():
            tbl = pq.read_table(str(train_path), columns=["era"])
            eras.extend(tbl.column("era").to_pylist())

        validation_path = data_dir / "validation.parquet"
        if validation_path.exists():
            tbl = pq.read_table(
                str(validation_path),
                columns=["era"],
                filters=[("data_type", "!=", "test")],
            )
            eras.extend(tbl.column("era").to_pylist())
        if not eras:
            return None
        return str(sorted(set(eras))[-1])
    except Exception:
        return None


@mcp.tool()
def run_weekly_retrain(force: bool = False) -> dict:
    """
    Run custom_mcp/make_submission.py with the correct Python interpreter.

    Downloads the latest Numerai data, retrains the live model on the last
    LOOKBACK_ERAS of history using dynamic per-step feature selection, and
    writes a pkl + metadata JSON to submissions/.

    Skips the retrain and returns status="skipped" if the era window has not
    advanced since the last submission (i.e. no new data).  Pass force=True
    to override this guard and retrain unconditionally.

    This call returns immediately — the training job runs in the background.
    Call check_retrain_status() to poll for completion and results.

    Returns: pid, log_path, and status="running" (or status="skipped").
    """
    SUBMISSIONS_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # Refresh labeled data before checking whether the era window advanced.
    refresh_data(include_live=False, dataset_names=["validation"])

    # Era-window guard: skip if data hasn't advanced since the last retrain.
    if not force:
        metas = _sorted_metas()
        if metas:
            last_meta = _load_meta(metas[-1])
            last_end = str(last_meta.get("era_window_end", ""))
            current_max = _current_max_labeled_era()
            if current_max is not None and current_max == last_end:
                return {
                    "status": "skipped",
                    "reason": f"Era window unchanged — last submission already covers up to era {last_end}. "
                              "Pass force=True to retrain anyway.",
                    "last_era_window": f"{last_meta.get('era_window_start')} – {last_end}",
                    "last_built_date": last_meta.get("built_date"),
                }

    log_path = REPORTS_DIR / "retrain_latest.log"
    pid_path = REPORTS_DIR / "retrain_latest.pid"

    # Terminate any previous run that is still alive.
    if pid_path.exists():
        try:
            old_pid = int(pid_path.read_text().strip())
            import psutil
            p = psutil.Process(old_pid)
            p.terminate()
        except Exception:
            pass

    log_file = open(log_path, "w", encoding="utf-8")
    env = {**__import__("os").environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(
        [PYTHON_EXE, "-u", MAKE_SUBMISSION],
        cwd=str(PROJECT_ROOT),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
    )
    log_file.close()

    pid_path.write_text(str(proc.pid), encoding="utf-8")

    return {
        "status": "running",
        "pid": proc.pid,
        "log_path": str(log_path),
        "message": "Training started in background. Call check_retrain_status() to poll for results.",
    }


@mcp.tool()
def check_retrain_status() -> dict:
    """
    Poll the background training job launched by run_weekly_retrain().

    Returns the job status (running / completed / failed), the tail of the
    log, and — when finished — the same result fields as the old blocking
    run_weekly_retrain: era_window, best_iteration, wall_clock_seconds, etc.
    """
    pid_path = REPORTS_DIR / "retrain_latest.pid"
    log_path = REPORTS_DIR / "retrain_latest.log"

    if not pid_path.exists():
        return {"status": "no_job", "message": "No retrain job found. Call run_weekly_retrain() first."}

    pid = int(pid_path.read_text().strip())

    # Check if the process is still alive.
    running = False
    try:
        import psutil
        running = psutil.pid_exists(pid)
    except Exception:
        running = False

    log_tail = ""
    if log_path.exists():
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        log_tail = "\n".join(ln for ln in lines[-30:] if ln.strip())

    if running:
        return {
            "status": "running",
            "pid": pid,
            "log_tail": log_tail,
        }

    # Process finished — determine success by checking for a new metadata file.
    metas = _sorted_metas()
    if not metas:
        return {
            "status": "failed",
            "message": "Process exited but no metadata JSON found in submissions/.",
            "log_tail": log_tail,
        }

    meta = _load_meta(metas[-1])
    live_diagnostics = _compute_live_prediction_diagnostics(metas[-1], meta)

    # Detect failure via returncode hint in log (make_submission prints non-zero exit).
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    if "Traceback" in log_text or "Error" in log_text.split("RESULT_JSON")[0]:
        # Still return the meta if it exists — the run may have partially succeeded.
        pass

    return {
        "status": "completed",
        "era_window": f"{meta['era_window_start']} – {meta['era_window_end']}",
        "era_count": meta["era_count"],
        "best_iteration": meta["best_iteration"],
        "features_selected": meta["top_k_features"],
        "wall_clock_seconds": meta["wall_clock_seconds"],
        "pickle_size_mb": meta["pickle_size_mb"],
        "meta_path": str(metas[-1]),
        "live_diagnostics": live_diagnostics,
        "log_tail": log_tail,
    }


@mcp.tool()
def get_training_summary() -> dict:
    """
    Read the latest submission metadata JSON and return a structured snapshot
    of the full training configuration: era window, lookback, trailing eras,
    top-k features, target, neutralization, best iteration, and pkl size.
    """
    metas = _sorted_metas()
    if not metas:
        return {"error": "No metadata JSON found in submissions/. Run run_weekly_retrain first."}

    meta = _load_meta(metas[-1])
    live_diagnostics = _compute_live_prediction_diagnostics(metas[-1], meta)

    return {
        "built_date": meta.get("built_date"),
        "target": meta.get("target"),
        "era_window": f"{meta.get('era_window_start')} – {meta.get('era_window_end')}",
        "era_count": meta.get("era_count"),
        "lookback_eras": meta.get("lookback_eras"),
        "trailing_eras": meta.get("trailing_eras"),
        "top_k_features": meta.get("top_k_features"),
        "feature_pool_size": meta.get("feature_pool_size"),
        "fit_eras": meta.get("fit_eras"),
        "early_stopping_eras": meta.get("es_eras"),
        "best_iteration": meta.get("best_iteration"),
        "benchmark_neutralization": meta.get("benchmark_neutralization"),
        "benchmark_col": meta.get("benchmark_col"),
        "pickle_size_mb": meta.get("pickle_size_mb"),
        "wall_clock_seconds": meta.get("wall_clock_seconds"),
        "pkl_file": metas[-1].stem.replace("_meta", "") + ".pkl",
        "pkl_path": str(SUBMISSIONS_DIR / (metas[-1].stem.replace("_meta", "") + ".pkl")),
        "live_diagnostics": live_diagnostics,
    }


@mcp.tool()
def check_live_predictions() -> dict:
    """
    Score the current live split with the latest packaged submission model and
    write distribution QA artifacts into artifacts/.

    Returns a pass / warn / fail verdict, summary distribution stats, and
    artifact paths for the plot, CSV, and cached JSON summary.
    """
    metas = _sorted_metas()
    if not metas:
        return {"error": "No metadata JSON found in submissions/. Run run_weekly_retrain first."}

    meta_path = metas[-1]
    meta = _load_meta(meta_path)
    diagnostics = _compute_live_prediction_diagnostics(meta_path, meta)
    diagnostics["meta_path"] = str(meta_path)
    diagnostics["pkl_path"] = str(SUBMISSIONS_DIR / (meta_path.stem.replace("_meta", "") + ".pkl"))
    return diagnostics


@mcp.tool()
def compare_weekly_features() -> dict:
    """
    Diff this week's selected features against the previous week's submission.

    Groups features by candidate group (faith, wisdom, strength, intelligence,
    extra) and reports added, removed, and retained counts + feature lists.
    Returns a no-op message if fewer than two training runs exist.
    """
    metas = _sorted_metas()

    if not metas:
        return {"error": "No metadata JSON found in submissions/."}

    if len(metas) == 1:
        meta = _load_meta(metas[0])
        curr_feats = meta.get("selected_features", [])
        return {
            "note": "Only one training run found — no previous week to compare against.",
            "current_features_total": len(curr_feats),
            "current_by_group": {k: len(v) for k, v in _group_features(curr_feats).items()},
        }

    prev_meta = _load_meta(metas[-2])
    curr_meta = _load_meta(metas[-1])

    prev_set = set(prev_meta.get("selected_features", []))
    curr_set = set(curr_meta.get("selected_features", []))

    added = sorted(curr_set - prev_set)
    removed = sorted(prev_set - curr_set)
    retained = sorted(curr_set & prev_set)

    def by_group(feats: list[str]) -> dict[str, int]:
        return {k: len(v) for k, v in _group_features(feats).items()}

    return {
        "previous_build": prev_meta.get("built_date"),
        "current_build": curr_meta.get("built_date"),
        "previous_era_window": f"{prev_meta.get('era_window_start')} – {prev_meta.get('era_window_end')}",
        "current_era_window": f"{curr_meta.get('era_window_start')} – {curr_meta.get('era_window_end')}",
        "summary": {
            "total_features": len(curr_set),
            "added": len(added),
            "removed": len(removed),
            "retained": len(retained),
        },
        "added_by_group": by_group(added),
        "removed_by_group": by_group(removed),
        "retained_by_group": by_group(retained),
        "added_features": added,
        "removed_features": removed,
    }


@mcp.tool()
def generate_weekly_report() -> dict:
    """
    Build a full markdown + HTML report for the current ISO week and save it to
    docs/YYYY-WW_weekly_report.md and docs/YYYY-WW_weekly_report.html.

    The report covers: training config, feature changes vs last week (grouped),
    target info (target_ender_60 default), and model stats.

    Returns the report paths and full markdown/HTML content.
    """
    REPORTS_DIR.mkdir(exist_ok=True)

    metas = _sorted_metas()
    if not metas:
        return {"error": "No metadata found. Run run_weekly_retrain first."}

    curr_meta = _load_meta(metas[-1])
    report_metrics = _compute_live_report_metrics(metas[-1], curr_meta)
    live_diagnostics = _compute_live_prediction_diagnostics(metas[-1], curr_meta)
    live_prediction_era = _current_live_prediction_era(curr_meta)
    now = datetime.now()
    iso = now.isocalendar()
    week_label = f"{iso.year}-W{iso.week:02d}"
    report_path = REPORTS_DIR / f"{week_label}_weekly_report.md"
    html_report_path = REPORTS_DIR / f"{week_label}_weekly_report.html"

    # --- Feature comparison section ---
    if len(metas) >= 2:
        prev_meta = _load_meta(metas[-2])
        prev_set = set(prev_meta.get("selected_features", []))
        curr_set = set(curr_meta.get("selected_features", []))
        added = sorted(curr_set - prev_set)
        removed = sorted(prev_set - curr_set)
        retained = sorted(curr_set & prev_set)
        added_grp = _group_features(added)
        removed_grp = _group_features(removed)
        retained_grp = _group_features(retained)

        def grp_table(title: str, grp: dict[str, list[str]]) -> str:
            if not any(grp.values()):
                return f"**{title}:** none\n\n"
            total = sum(len(v) for v in grp.values())
            rows = ["| Group | Count | Sample Features |", "| --- | --- | --- |"]
            for g, feats in sorted(grp.items()):
                sample = ", ".join(f"`{f}`" for f in feats[:4])
                suffix = f", +{len(feats) - 4} more" if len(feats) > 4 else ""
                rows.append(f"| {g} | {len(feats)} | {sample}{suffix} |")
            return f"**{title}** ({total} features)\n\n" + "\n".join(rows) + "\n\n"

        feat_section = (
            f"## Feature Changes vs Previous Week\n\n"
            f"> Previous build: **{prev_meta.get('built_date')}** "
            f"— era window {prev_meta.get('era_window_start')} – {prev_meta.get('era_window_end')}\n\n"
            f"| | Count |\n| --- | --- |\n"
            f"| Total features (current) | {len(curr_set)} |\n"
            f"| Added this week | {len(added)} |\n"
            f"| Removed this week | {len(removed)} |\n"
            f"| Retained | {len(retained)} |\n\n"
            + grp_table("Added", added_grp)
            + grp_table("Removed", removed_grp)
            + grp_table("Retained", retained_grp)
        )
    else:
        curr_grp = _group_features(curr_meta.get("selected_features", []))
        grp_lines = "\n".join(f"- **{g}**: {len(v)}" for g, v in sorted(curr_grp.items()))
        feat_section = (
            "## Feature Changes\n\n"
            "_No previous week to compare — this is the first recorded training run._\n\n"
            f"**Selected features by group:**\n\n{grp_lines}\n\n"
        )

    training_sharpe = report_metrics.get("fit_corr_sharpe")
    validation_sharpe = report_metrics.get("val_corr_sharpe")
    top_snapshot_rows = [
        ("Live training target", f"`{report_metrics.get('training_target', curr_meta.get('target'))}`"),
        ("Validation target", f"`{report_metrics.get('validation_target', curr_meta.get('fallback_target', 'target_ender_20'))}`"),
        ("MMC benchmark", f"`{report_metrics.get('benchmark_col', curr_meta.get('benchmark_col'))}`"),
        ("Training: CORR mean", _format_metric(report_metrics.get("fit_corr_mean"))),
        ("Training: MMC mean", _format_metric(report_metrics.get("fit_mmc_mean"))),
        ("Training Sharpe", _format_metric(training_sharpe, 3)),
        ("Validation: CORR mean", _format_metric(report_metrics.get("val_corr_mean"))),
        ("Validation: MMC mean", _format_metric(report_metrics.get("val_mmc_mean"))),
        ("Validation Sharpe", _format_metric(validation_sharpe, 3)),
    ]
    top_snapshot = _format_metric_block_rows("Model Snapshot", top_snapshot_rows)
    title_suffix = f" | Live Era {live_prediction_era}" if live_prediction_era else ""
    live_era_line = f" | **Live submission era:** {live_prediction_era}" if live_prediction_era else ""
    live_diag_rows = [
        ("Verdict", str(live_diagnostics.get("status", "n/a")).upper()),
        ("Ready for submission", "yes" if live_diagnostics.get("ready_for_submission") else "no"),
        ("Rows scored", str(live_diagnostics.get("row_count", "n/a"))),
        ("Prediction std", _format_metric(live_diagnostics.get("prediction_std"))),
        ("Prediction p99-p01 spread", _format_metric(live_diagnostics.get("prediction_spread_p99_p01"))),
        ("Duplicate fraction", _format_metric(live_diagnostics.get("duplicate_fraction"))),
        ("Benchmark corr", _format_metric(live_diagnostics.get("benchmark_corr"))),
    ]
    live_diag_checks = "\n".join(
        f"| {_safe_table_cell(row.get('name', 'check'))} | {_safe_table_cell(str(row.get('status', 'n/a')).upper())} | {_safe_table_cell(row.get('detail', ''))} |"
        for row in live_diagnostics.get("checks", [])
        if isinstance(row, dict)
    ) or "| n/a | n/a | No live diagnostic checks were generated. |"
    live_diag_artifacts = live_diagnostics.get("artifacts", {}) if isinstance(live_diagnostics.get("artifacts"), dict) else {}
    plot_path = live_diag_artifacts.get("plot_path")
    plot_rel = (
        Path(os.path.relpath(str(plot_path), REPORTS_DIR)).as_posix()
        if plot_path
        else None
    )
    csv_path = live_diag_artifacts.get("csv_path")
    csv_rel = (
        Path(os.path.relpath(str(csv_path), REPORTS_DIR)).as_posix()
        if csv_path
        else None
    )
    summary_path = live_diag_artifacts.get("summary_path")
    summary_rel = (
        Path(os.path.relpath(str(summary_path), REPORTS_DIR)).as_posix()
        if summary_path
        else None
    )
    live_visualization_block = (
        "\n### Visualization\n\n"
        f"![Live prediction QA plot]({plot_rel})\n\n"
        "The chart combines the raw histogram, sorted prediction curve, benchmark exposure scatter, "
        "and percentile-ranked distribution for the current live batch.\n"
        if plot_rel
        else ""
    )

    # --- Report assembly ---
    report = f"""# Numerai Weekly Report — {week_label}{title_suffix}

**Model:** tailspin | **Built:** {curr_meta.get("built_date")} | **Era window:** {curr_meta.get("era_window_start")} – {curr_meta.get("era_window_end")}{live_era_line}

---

{feat_section}
---

## Target Analysis

**Current target:** `{curr_meta.get("target")}`

This model is trained on `target_ender_60` as established by the v5.2 feature analysis. This target
was selected because it provides the best generalization for MMC in walk-forward testing.

> A dynamic target recommendation system is planned for a future update. Until then,
> `target_ender_60` remains the fixed default.

---

## Top Statistics

{top_snapshot}

---

## Live Prediction QA

{live_visualization_block}

{_format_metric_block_rows("Distribution Check", live_diag_rows)}

| Check | Status | Details |
| --- | --- | --- |
{live_diag_checks}

| Artifact | Path |
| --- | --- |
| Distribution plot | `{plot_rel or live_diag_artifacts.get("plot_path", "n/a")}` |
| Scored CSV | `{csv_rel or live_diag_artifacts.get("csv_path", "n/a")}` |
| Summary JSON | `{summary_rel or live_diag_artifacts.get("summary_path", "n/a")}` |

---

## Artifact Details

| Metric | Value |
| --- | --- |
| Built date | {curr_meta.get("built_date")} |
| Model type | XGBoost (GPU) |
| Best iteration | {curr_meta.get("best_iteration")} |
| Wall clock time | {curr_meta.get("wall_clock_seconds")}s |
| Pickle size | {curr_meta.get("pickle_size_mb")} MB |

---

## Training Configuration

| Parameter | Value |
| --- | --- |
| Target | `{curr_meta.get("target")}` |
| Era window | {curr_meta.get("era_window_start")} – {curr_meta.get("era_window_end")} |
| Era count | {curr_meta.get("era_count")} |
| Lookback eras | {curr_meta.get("lookback_eras")} |
| Trailing eras (feature ranking) | {curr_meta.get("trailing_eras")} |
| Top-K features selected | {curr_meta.get("top_k_features")} |
| Feature pool size | {curr_meta.get("feature_pool_size")} |
| Fit eras | {curr_meta.get("fit_eras")} |
| Early stopping eras | {curr_meta.get("es_eras")} |
| Best iteration | {curr_meta.get("best_iteration")} |
| Benchmark neutralization | {curr_meta.get("benchmark_neutralization")} vs `{curr_meta.get("benchmark_col")}` |

---

_Generated by numerai-weekly MCP on {now.strftime("%Y-%m-%d %H:%M")}._
"""

    report_html = build_report_html(f"Numerai Weekly Report - {week_label}", report)
    report_path.write_text(report, encoding="utf-8")
    html_report_path.write_text(report_html, encoding="utf-8")
    build_dashboard()

    return {
        "report_path": str(report_path),
        "html_report_path": str(html_report_path),
        "dashboard_path": str(REPORTS_DIR / "index.html"),
        "week": week_label,
        "live_diagnostics": live_diagnostics,
        "content": report,
        "html_content": report_html,
    }


if __name__ == "__main__":
    mcp.run()
