"""Scoring functions that decide whether an autoresearch experiment is kept."""
import numpy as np
import pandas as pd
import pytest

from prepare import era_stats, gaussianize, neutralize, numerai_corr, per_era_corr, research_score


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_gaussianize_is_rank_based_and_centered(rng):
    values = rng.normal(size=500)
    out = gaussianize(values)
    assert np.argsort(out).tolist() == np.argsort(values).tolist()
    assert abs(out.mean()) < 1e-9
    # A monotone transform of the input must not change the output.
    np.testing.assert_allclose(gaussianize(np.exp(values)), out)


def test_numerai_corr_sign_and_bounds(rng):
    target = rng.choice([0.0, 0.25, 0.5, 0.75, 1.0], size=2000)
    signal = target + rng.normal(scale=0.3, size=target.size)
    assert numerai_corr(signal, target) > 0.3
    assert numerai_corr(-signal, target) < -0.3
    assert -1.0 <= numerai_corr(signal, target) <= 1.0


def test_numerai_corr_constant_target_returns_zero(rng):
    with np.errstate(invalid="ignore", divide="ignore"):
        assert numerai_corr(rng.normal(size=100), np.full(100, 0.5)) == 0.0


def test_neutralize_removes_exposure(rng):
    exposure = rng.normal(size=1000)
    preds = 2.0 * exposure + rng.normal(size=1000)
    neutral = neutralize(preds, exposure, proportion=1.0)
    assert abs(np.corrcoef(neutral, exposure)[0, 1]) < 1e-8
    np.testing.assert_allclose(neutralize(preds, exposure, proportion=0.0), preds)


def test_per_era_corr_groups_by_era(rng):
    frame = pd.DataFrame(
        {
            "era": np.repeat(["0001", "0002"], 200),
            "target": rng.choice([0.0, 0.5, 1.0], size=400),
        }
    )
    frame["pred"] = frame["target"] + rng.normal(scale=0.1, size=400)
    result = per_era_corr(frame, "pred", "target")
    assert list(result.index) == ["0001", "0002"]
    assert (result > 0.5).all()


def test_era_stats_values():
    stats = era_stats(pd.Series([0.02, -0.01, 0.03, -0.04]), "val_corr")
    assert stats["val_corr_mean"] == pytest.approx(0.0)
    assert stats["val_corr_min"] == pytest.approx(-0.04)
    assert stats["val_corr_max"] == pytest.approx(0.03)
    # Cumulative path 0.02, 0.01, 0.04, 0.00 -> worst drop from peak is 0.04.
    assert stats["val_corr_max_drawdown"] == pytest.approx(-0.04)


def test_era_stats_zero_std_has_zero_sharpe():
    assert era_stats(pd.Series([0.01, 0.01, 0.01]), "x")["x_sharpe"] == 0.0


def test_research_score_weights():
    metrics = {"val_mmc_mean": 1.0, "val_corr_mean": 0.0, "val_bmc_mean": 0.0}
    assert research_score(metrics) == pytest.approx(0.60)
    metrics = {"val_mmc_mean": 0.01, "val_corr_mean": 0.02, "val_bmc_mean": 0.03}
    assert research_score(metrics) == pytest.approx(0.006 + 0.006 + 0.003)
