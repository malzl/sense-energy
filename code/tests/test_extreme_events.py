"""Heat waves and cold spells at the sites: the pure parts, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.analysis import extreme_events as ee


def _daily(days: int = 120, sites: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    idx = pd.date_range("2025-03-03", periods=days, freq="D")
    week = 100.0 + 10.0 * (idx.dayofweek < 5)
    return pd.DataFrame({s: week * (k + 1) for k, s in enumerate(sites)}, index=idx)


def _weather(kw: pd.DataFrame, hot: slice) -> pd.DataFrame:
    rows = []
    for s in kw.columns:
        t = pd.DataFrame(
            {"date": kw.index, "site_code": s, "t_max": 20.0, "t_mean": 15.0, "t_min": 10.0}
        )
        t.loc[t.index[hot], "t_max"] = 30.0
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _masks(kw: pd.DataFrame, hot: slice) -> dict[str, pd.DataFrame]:
    heat = pd.DataFrame(False, index=kw.index, columns=kw.columns)
    heat.iloc[hot] = True
    return {"heatwave": heat, "cold spell": heat & False}


def test_reference_is_the_same_weekday_and_skips_unusable_days():
    kw = _daily()
    usable = np.ones(kw.shape, dtype=bool)
    ref = ee.reference(kw.to_numpy(), usable)
    assert np.allclose(ref[30:60], kw.to_numpy()[30:60])  # a weekly pattern is its own reference
    usable[:] = False
    assert np.isnan(ee.reference(kw.to_numpy(), usable)).all()


def test_reaction_recovers_a_known_rise_and_leaves_event_days_out_of_the_reference():
    kw = _daily()
    hot = slice(56, 61)  # five days, Monday to Friday
    kw.iloc[hot, 0] *= 1.2  # site A rises by 20 %, site B does not react
    r = ee.reactions(kw, _weather(kw, hot), _masks(kw, hot))
    ev = ee.event_days(r)
    assert set(ev["type"]) == {"heatwave"} and len(ev) == 10
    assert np.allclose(ev["degrees"], 10.0)
    per = ee._ratios(ev, ["site_code"])["pct"]
    assert np.isclose(per["A"], 100 * np.log(1.2)) and np.isclose(per["B"], 0.0)
    # the week after the event has the event days among its candidate references: unchanged
    after = r[(r["site_code"] == "A") & (r["date"] == kw.index[64])]
    assert np.isclose(after["ref_kw"].iloc[0], after["kw"].iloc[0])


def test_event_days_number_the_events_of_a_site():
    kw = _daily()
    heat = pd.DataFrame(False, index=kw.index, columns=kw.columns)
    heat.iloc[40:43, 0] = True
    heat.iloc[70:74, 0] = True
    masks = {"heatwave": heat, "cold spell": heat & False}
    ev = ee.event_days(ee.reactions(kw, _weather(kw, slice(40, 43)), masks))
    assert ev.groupby("run").size().tolist() == [3, 4]


def test_ratio_of_sums_survives_a_day_with_no_import():
    assert np.isclose(ee.ratio_pct([0.0, 220.0], [100.0, 100.0]), 100 * np.log(1.1))
    assert np.isnan(ee.ratio_pct([np.nan], [1.0]))


def test_morans_i_sees_a_gradient_and_not_a_shuffle():
    rng = np.random.default_rng(0)
    lat, lon = rng.uniform(50, 55, 80), rng.uniform(-4, 1, 80)
    smooth = ee.morans_i(lat + 0.1 * rng.normal(size=80), lat, lon)
    noise = ee.morans_i(rng.normal(size=80), lat, lon)
    assert smooth["morans_i"] > 0.5 and smooth["p"] < 0.01
    assert abs(noise["morans_i"]) < 0.2 and noise["p"] > 0.05


def test_region_share_is_high_for_a_regional_quantity_only():
    rng = np.random.default_rng(1)
    region = pd.Series(np.repeat(list("abcd"), 20))
    regional = region.map({"a": 0.0, "b": 1.0, "c": 2.0, "d": 3.0}) + 0.1 * rng.normal(size=80)
    own = pd.Series(rng.normal(size=80))
    assert ee.region_share(regional, region)["share"] > 0.9
    got = ee.region_share(own, region)
    assert got["share"] < 0.2 and got["p"] > 0.05


def test_excluded_days_cover_bank_holidays_and_the_christmas_fortnight():
    d = pd.DatetimeIndex(["2025-12-20", "2025-12-21", "2026-01-03", "2026-01-04", "2025-08-25"])
    assert ee.excluded_days(d).tolist() == [False, True, True, False, True]
