"""Binned fixed-effect temperature response: bin naming, run counters and coefficient recovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import weather_response as wr


def test_bin_terms_round_trip_and_reference():
    assert wr.bin_term(wr.REF_BIN) == f"tb{wr.EDGES.index(14)}"
    assert wr.bin_label(-np.inf) == "<-2"
    assert wr.bin_label(24.0) == ">=24"
    assert wr.bin_label(6.0) == "6"


def test_runs_day_counts_consecutive_days():
    flag = pd.Series([False, True, True, True, False, True])
    assert wr._runs_day(flag).tolist() == [0, 1, 2, 3, 0, 1]


def _synthetic(effect_hot: float, seed: int = 0) -> pd.DataFrame:
    """Four sites, 120 days: log demand = site x month level + weekday + effect on >= 24 degC days."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-06-01", periods=120)
    rows = []
    for i in range(4):
        level = rng.normal(5, 1, size=len(dates))
        month_level = pd.Series(level, index=dates).groupby(dates.month).transform("mean")
        t = rng.uniform(12, 27, size=len(dates))
        log_kw = (
            month_level.to_numpy()
            + 0.1 * (dates.dayofweek >= 5)
            + effect_hot * (t >= 24)
            + rng.normal(0, 0.01, len(dates))
        )
        rows.append(
            pd.DataFrame({"site_code": f"S{i}", "date": dates, "t_mean": t, "kw": np.exp(log_kw)})
        )
    df = pd.concat(rows, ignore_index=True)
    df["tbin"] = pd.cut(df["t_mean"], wr.EDGES, right=False).map(lambda iv: iv.left).astype(float)
    df["holiday"] = df["christmas"] = 0.0
    df["school_summer"] = df["date"].dt.month.isin([8]).astype(float) * (df["date"].dt.day > 15)
    df["fe_month"] = df["site_code"] + "|" + df["date"].dt.strftime("%Y-%m")
    df["fe_dow"] = df["site_code"] + "|" + df["date"].dt.dayofweek.astype(str)
    df["week"] = df["date"].dt.strftime("%G-%V")
    return df


def test_curve_recovers_a_known_hot_day_effect():
    c = wr.curve(_synthetic(np.log(1.2)), "kw").set_index("bin")
    assert c.loc[">=24", "coef_pct"] == pytest.approx(20.0, abs=1.0)
    assert c.loc["16", "coef_pct"] == pytest.approx(0.0, abs=1.0)
    assert c.loc["14", "coef_pct"] == 0.0  # reference bin
    assert (c["lo_pct"] <= c["coef_pct"]).all() and (c["coef_pct"] <= c["hi_pct"]).all()


def test_daily_maximum_bands_have_their_own_labels_and_reference():
    assert wr.bin_label(-np.inf, "daily maximum") == "<2"
    assert wr.bin_label(32.0, "daily maximum") == ">=32"
    assert wr.bin_label(10.0, "daily maximum") == "10"
    assert wr.bin_term(wr.REF_BIN_MAX, "daily maximum") == f"tb{wr.EDGES_MAX.index(18)}"


def test_curve_on_the_daily_maximum_recovers_a_30_plus_effect():
    df = _synthetic(0.0)
    df["t_max"] = df["t_mean"] + 6  # 18-33 degC maxima: the 18-20 reference band is populated
    df["tbin_max"] = (
        pd.cut(df["t_max"], wr.EDGES_MAX, right=False).map(lambda iv: iv.left).astype(float)
    )
    df.loc[df["t_max"] >= 30, "kw"] *= 1.15
    c = wr.curve(df, "kw", "daily maximum").set_index("bin")
    assert c.loc[">=32", "coef_pct"] == pytest.approx(15.0, abs=2.0)
    assert c.loc["30", "coef_pct"] == pytest.approx(15.0, abs=2.0)
    assert c.loc["18", "coef_pct"] == 0.0
    assert c.loc["30", "bin_mid"] == 31.0 and c.loc[">=32", "bin_mid"] == 33.0
