"""Hot-day forecast scoring, causal-ML helpers and the ERIC change detector."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import estates_changes as ec
from sense_energy.analysis import forecast_heat_days as fh
from sense_energy.analysis import heat_causal_ml as hcm


def test_safe_names_are_unique_and_lightgbm_safe():
    out = hcm.safe_names(["use_inpatient, acute", "use_inpatient acute", "summer °C", "a", "a"])
    assert out == ["use_inpatient_acute", "use_inpatient_acute_2", "summer_C", "a", "a_2"]
    assert all(n.replace("_", "").isalnum() for n in out)


def test_demean_removes_fixed_effect_means():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "fe_month": rng.choice(["m1", "m2", "m3"], n),
            "fe_dow": rng.choice(["d1", "d2"], n),
            "date_id": rng.choice([f"t{i}" for i in range(20)], n),
        }
    )
    df["x"] = (
        df["fe_month"].map({"m1": 1.0, "m2": 5.0, "m3": -2.0})
        + df["date_id"].str[1:].astype(float) * 0.3
        + rng.normal(size=n)
    )
    r = hcm.demean_fe(df, ["x"])["x"]
    assert abs(r.groupby(df["fe_month"]).mean()).max() < 1e-6
    assert abs(r.groupby(df["date_id"]).mean()).max() < 1e-6


def test_r_learner_recovers_planted_heterogeneity():
    rng = np.random.default_rng(1)
    n_sites, n_days = 40, 300
    sites = np.repeat([f"S{i:02d}" for i in range(n_sites)], n_days)
    z = np.repeat(rng.normal(size=n_sites), n_days)  # the one covariate that matters
    t_res = rng.normal(size=n_sites * n_days)
    tau_true = 0.02 + 0.02 * z
    y_res = tau_true * t_res + rng.normal(0, 0.05, n_sites * n_days)
    x = pd.DataFrame({"z": z, "noise": rng.normal(size=len(z))})
    tau = hcm.r_learner(x, y_res, t_res, sites)
    site_tau = pd.Series(tau).groupby(sites).mean()
    site_true = pd.Series(tau_true).groupby(sites).mean()
    assert site_tau.corr(site_true) > 0.8


def test_day_classes_split_on_daily_maximum_and_season(tmp_path, monkeypatch):
    w = pd.DataFrame(
        {
            "site_code": "A",
            "date": pd.to_datetime(
                ["2025-07-01", "2025-07-02", "2025-07-03", "2025-01-10", "2025-03-01"]
            ),
            "t_max": [29.0, 26.0, 20.0, 4.0, 10.0],
            "t_mean": [22.0, 19.0, 15.0, 1.0, 6.0],
        }
    )
    w.to_parquet(tmp_path / "weather_daily.parquet", index=False)
    monkeypatch.setattr(fh, "INTERIM_DIR", tmp_path)
    out = fh.day_classes()
    assert out["day_class"].tolist() == [
        "28 °C and above",
        "25–28 °C",
        "other June–September",
        "at or below 2 °C",
        "rest of year",
    ]


def test_steps_in_series_finds_a_planted_level_shift():
    idx = pd.date_range("2024-01-01", periods=400, freq="D")
    rng = np.random.default_rng(2)
    kw = pd.Series(np.exp(np.log(50.0) + rng.normal(0, 0.03, 400)), index=idx)
    kw.iloc[250:] *= 0.7  # a 30 % drop on day 250
    steps = ec.steps_in_series(kw)
    assert len(steps) == 1
    assert abs((steps[0]["date"] - idx[250]).days) <= 3
    assert steps[0]["change"] == pytest.approx(-0.3, abs=0.05)


def test_match_steps_picks_the_largest_step_in_the_window():
    changes = pd.DataFrame(
        {"any_change": [True, True]}, index=pd.Index(["A", "B"], name="site_code")
    )
    steps = pd.DataFrame(
        {
            "site_code": ["A", "A", "B"],
            "date": pd.to_datetime(["2024-06-01", "2024-12-01", "2022-01-01"]),
            "change": [0.2, -0.5, 0.9],
        }
    )
    out = ec.match_steps(changes, steps)
    assert out.loc["A", "steps_in_window"] == 2 and out.loc["A", "step_change"] == -0.5
    assert out.loc["B", "steps_in_window"] == 0 and pd.isna(out.loc["B", "step_date"])
