"""Thermometer comparison: common keys and the per-site agreement statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import thermometers as th


def _table(seed: int, noise: float, days: int = 120, bias: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-03-01", periods=days)
    truth = 10 + 8 * np.sin(2 * np.pi * np.arange(days) / 365) + rng.normal(0, 3, days)
    rows = []
    for site in ("A", "B"):
        t = truth + bias + rng.normal(0, noise, days)
        rows.append(
            pd.DataFrame(
                {"site_code": site, "date": dates, "t_mean": t, "t_max": t + 5, "t_eve": t + 1}
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_common_keys_intersect_site_days():
    a, b = _table(0, 0.1), _table(1, 0.1).iloc[10:]
    keys = th.common_keys({"x": a, "y": b})
    assert len(keys) == len(b)


def test_agreement_ranks_the_closer_thermometer_and_reports_bias():
    rng = np.random.default_rng(3)
    station = _table(0, 0.0)
    station["distance_km"] = 12.0
    near = station.copy()
    near[["t_mean", "t_max", "t_eve"]] += rng.normal(0, 0.3, (len(near), 1))
    far = station.copy()
    far[["t_mean", "t_max", "t_eve"]] += 1.5 + rng.normal(0, 1.0, (len(far), 1))
    a = th.agreement({"stations": station, "near": near, "far": far}).set_index(
        ["thermometer", "site_code"]
    )
    assert a.loc[("near", "A"), "rmse_mean_c"] < a.loc[("far", "A"), "rmse_mean_c"]
    assert a.loc[("far", "A"), "bias_mean_c"] == pytest.approx(1.5, abs=0.3)
    assert a.loc[("near", "A"), "anomaly_corr"] > a.loc[("far", "A"), "anomaly_corr"]
    assert a.loc[("near", "A"), "station_km"] == 12.0
