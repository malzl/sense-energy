"""SARIMA baseline: Fourier regressors, transforms and a small end-to-end run."""

from __future__ import annotations

import numpy as np
import pytest

from sense_energy.experiments import poc, sarima
from test_poc import CFG, _panel

SPEC = {
    "order": [1, 0, 0],
    "seasonal_order": [0, 1, 0, 48],
    "weekly_fourier_k": 1,
    "transform": "log",
    "refit": "origin",
    "init_weeks": 1,
    "maxiter": 10,
    "maxiter_refit": 5,
    "n_jobs": 2,
}


def test_fourier_week_is_weekly_periodic_and_paired():
    hw = np.arange(0, 336 * 2)
    X = sarima.fourier_week(hw, 3)
    assert X.shape == (672, 6)
    assert np.allclose(X[:336], X[336:])
    assert np.allclose(X[:, 0] ** 2 + X[:, 1] ** 2, 1.0)
    assert sarima.fourier_week(hw, 0).shape == (672, 0)


def test_log_transform_floors_at_one_percent_of_the_median():
    y = np.array([0.0, 10.0, 10.0, 12.0, np.nan])
    z = sarima.transform(y, "log")
    assert np.isclose(np.exp(z[0]), 0.1)  # floor = 1% of the median 10
    assert np.allclose(sarima.back_transform(z[1:4], "log"), y[1:4])
    assert np.isnan(z[4])
    assert np.array_equal(sarima.transform(y[:4], "none"), y[:4])
    with pytest.raises(ValueError):
        sarima.transform(y, "sqrt")


def test_forecast_task_gives_monotone_gaussian_quantiles():
    rng = np.random.default_rng(1)
    t = np.arange(48 * 8)
    y = np.log(100 + 30 * np.sin(2 * np.pi * t / 48) + rng.normal(0, 1, len(t)))
    X = sarima.fourier_week(t % 336, 1)
    init = sarima.fit_task(y, X, SPEC, None, 10)
    assert init["params"] is not None
    z = np.array([-1.2816, 0.0, 1.2816])
    r = sarima.forecast_task(y, X, X[:63], 63, SPEC, init["params"], True, 5, z)
    assert r["error"] is None and r["q"].shape == (63, 3)
    assert (np.diff(r["q"], axis=1) > 0).all()
    assert np.allclose(r["q"][:, 1], r["mu"])
    truth = np.log(100 + 30 * np.sin(2 * np.pi * (t[-1] + 1 + np.arange(63)) / 48))
    assert np.abs(r["mu"] - truth).mean() < 0.05


def test_forecast_task_reports_failures_instead_of_raising():
    r = sarima.forecast_task(
        np.full(96, np.nan),
        np.zeros((96, 0)),
        np.zeros((5, 0)),
        5,
        SPEC,
        None,
        True,
        5,
        np.zeros(3),
    )
    assert r["q"] is None and r["error"] is not None


def test_run_sarima_recovers_the_daily_pattern():
    panel = _panel(days=30)
    origins = poc.make_origins(panel, CFG, "2025-01-20", "2025-01-26", 6)
    cfg = {**CFG, "context_periods": 48 * 7, "train_end": "2025-01-18", "sarima": SPEC}
    fc = sarima.run_sarima(panel, origins, cfg)
    assert set(fc["model"]) == {"sarima"} and len(fc) == 2 * 2 * 48
    assert fc[["q10", "q50", "q90"]].notna().all().all()
    per, pooled, _ = poc.score(fc, panel, CFG["quantiles"], naive_model="sarima")
    assert pooled.set_index("model").loc["sarima", "nmae"] < 0.05
