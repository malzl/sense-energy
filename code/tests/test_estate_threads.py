"""Hourly response engine, warming and heatwave counterfactuals, installation savings, solar watch
and CHP dispatch: the pure parts, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import chp_dispatch as cd
from sense_energy.analysis import estate_counterfactuals as ec
from sense_energy.analysis import hourly_response as hr
from sense_energy.analysis import installation_savings as ins
from sense_energy.analysis import solar_watch as sw


def _wide(days=3, sites=("A", "B")):
    idx = pd.date_range(
        "2025-07-01", periods=48 * days, freq="30min", tz="Europe/London"
    ).tz_convert("UTC")
    hours = idx.tz_convert("Europe/London").hour.to_numpy()
    return pd.DataFrame({s: 10.0 + hours + k for k, s in enumerate(sites)}, index=idx)


def test_hour_means_average_the_two_half_hours():
    w = _wide()
    w.iloc[5, 0] = np.nan  # one half-hour of hour 2 on day 1 missing at site A
    out = hr.hour_means(w)
    a = out[(out["site_code"] == "A") & (out["date"] == pd.Timestamp("2025-07-01"))].set_index(
        "hour"
    )["hour_kw"]
    assert 2 not in a.index  # needs both half-hours
    assert a.loc[17] == pytest.approx(27.0)


def test_shrinkage_pulls_noisy_hours_to_the_pool():
    rows = []
    for h in (0, 1):
        for i in range(10):
            rows.append(
                {
                    "site_code": f"S{i}",
                    "term": "cdd20",
                    "hour": h,
                    "slope": 0.02 + 0.001 * i,
                    "se": 0.001,
                }
            )
        rows.append({"site_code": "noisy", "term": "cdd20", "hour": h, "slope": 0.2, "se": 0.1})
    post, pooled = hr.shrink_hourly(pd.DataFrame(rows))
    n = post[post["site_code"] == "noisy"]
    assert (n["posterior"] < 0.05).all() and len(pooled) == 2


def test_doses_and_multiplier():
    c, h = ec.doses(np.array([10.0, 17.0, 23.0]))
    assert c.tolist() == [4.0, 0.0, 0.0] and h.tolist() == [0.0, 0.0, 3.0]
    prof = {
        "heat": pd.DataFrame(0.02, index=["A"], columns=range(24)),
        "cold": pd.DataFrame(0.01, index=["A"], columns=range(24)),
    }
    long = pd.DataFrame({"site_code": ["A", "A"], "hour": [15, 3], "t_mean": [22.0, 10.0]})
    m = ec.multiplier(long, prof, np.array([27.0, 12.0]))
    assert m[0] == pytest.approx(np.exp(0.02 * 5))  # heat dose 2 -> 7
    assert m[1] == pytest.approx(np.exp(0.01 * -2))  # cold dose 4 -> 2


def test_smoothed_profiles_are_circular(tmp_path):
    rows = [
        {"site_code": "A", "term": t, "hour": h, "posterior": (1.0 if h == 0 else 0.0)}
        for t in ("cdd20", "hdd14")
        for h in range(24)
    ]
    path = tmp_path / "h.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    p = ec.smoothed_profiles(path)["heat"].loc["A"]
    assert (
        p[0] == pytest.approx(0.5) and p[1] == pytest.approx(0.25) and p[23] == pytest.approx(0.25)
    )


def test_nac_is_unchanged_when_only_the_weather_differs():
    rng = np.random.default_rng(0)
    rows = []
    for year, shift in (("2023", 0.0), ("2025", 3.0)):  # a warmer second year, same building
        dates = pd.date_range(f"{year}-04-01", periods=365)
        t = (
            11
            + 7 * np.sin(2 * np.pi * (dates.dayofyear - 110) / 365)
            + shift
            + rng.normal(0, 2, 365)
        )
        hdd, cdd = np.maximum(14 - t, 0), np.maximum(t - 20, 0)
        kw = np.exp(4 + 0.02 * hdd + 0.03 * cdd + rng.normal(0, 0.02, 365))
        rows.append(
            pd.DataFrame(
                {
                    "site_code": "A",
                    "date": dates,
                    "kw": kw,
                    "hdd14": hdd,
                    "cdd20": cdd,
                    "solar_kwh_m2": 3.0,
                    "holiday": 0.0,
                    "christmas": 0.0,
                    "school_summer": 0.0,
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    nac = ins.nac_table(df).set_index("year")
    raw = 100 * (nac.loc["2025/26", "raw_mean_kw"] / nac.loc["2023/24", "raw_mean_kw"] - 1)
    norm = 100 * (nac.loc["2025/26", "nac_kw"] / nac.loc["2023/24", "nac_kw"] - 1)
    assert abs(norm) < 1.0 and abs(raw) > abs(norm)


def test_bootstrap_difference_has_the_right_sign():
    rng = np.random.default_rng(1)
    est, lo, hi = ins.bootstrap_diff(
        np.array([-10.0, -12, -9, -11]), np.array([0.0, 1, -1, 0.5, -0.5]), rng
    )
    assert est < 0 and hi < 0


def test_daytime_means_keep_06_to_18():
    w = _wide(days=1)
    out = sw.daytime_means(w).set_index("site_code")["day_kw"]
    assert out["A"] == pytest.approx(10 + np.mean(range(6, 18)))


def test_chp_classification_against_the_placebo(tmp_path, monkeypatch):
    pd.DataFrame({"site_code": ["C1", "C2"], "site_name": ["One", "Two"]}).to_parquet(
        tmp_path / "sites.parquet"
    )
    monkeypatch.setattr(cd, "INTERIM_DIR", tmp_path)
    monkeypatch.setattr(cd, "PROJECT_ROOT", tmp_path)
    dly = pd.DataFrame(
        {
            "site_code": ["C1", "C2", "no CHP (pooled)"] * 3,
            "term": ["hdd14"] * 3 + ["cdd20"] * 3 + ["price"] * 3,
            "coef": [-0.02, 0.013, 0.013, 0.01, 0.01, 0.01, 0.0, 0.0, 0.0],
            "se": [0.002] * 9,
            "p": [0.0] * 9,
        }
    )
    hrly = pd.DataFrame(
        {
            "site_code": ["C1", "C2", "no CHP (pooled)"] * 2,
            "term": ["price"] * 3 + ["red_workday"] * 3,
            "coef": [-0.05, 0.13, 0.13, -0.1, 0.14, 0.14],
            "se": [0.01] * 6,
            "p": [0.0] * 6,
        }
    )
    out = cd.classify(dly, hrly, ["C1", "C2"])
    assert out.loc["C1", ["heat_led", "price_following", "red_band_avoiding"]].tolist() == [
        True,
        True,
        True,
    ]
    assert out.loc["C2", ["heat_led", "price_following", "red_band_avoiding"]].tolist() == [
        False,
        False,
        False,
    ]
    assert out.loc["C1", "site_name"] == "One"
