"""Extreme-weather event detection: runs, site events, national events, heat thresholds."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.analysis import weather_events as we


def _daily(site: str, t_max, start="2025-07-01", **cols) -> pd.DataFrame:
    n = len(t_max)
    base = {
        "t_max": t_max,
        "t_mean": np.full(n, 15.0),
        "gust_max": np.full(n, 10.0),
        "precip_mm": np.zeros(n),
        "snow_mm_we": np.zeros(n),
    }
    base.update(cols)
    return pd.DataFrame({"site_code": site, "date": pd.date_range(start, periods=n), **base})


def test_runs_respects_minimum_length_and_gaps():
    idx = pd.date_range("2025-07-01", periods=8)
    flag = pd.Series([True, True, False, True, True, True, False, True], index=idx)
    assert we.runs(flag, 3) == [(idx[3], idx[5])]
    assert len(we.runs(flag, 1)) == 3


def test_heatwave_needs_three_days_at_the_site_threshold():
    thr = pd.DataFrame({"site_code": ["A", "B"], "heat_threshold_c": [25.0, 28.0]})
    d = pd.concat(
        [_daily("A", [26, 26, 26, 20, 26, 26]), _daily("B", [26, 26, 26, 20, 26, 26])],
        ignore_index=True,
    )
    ev = we.site_events(d, thr)
    hw = ev[ev["type"] == "heatwave"]
    assert list(hw["site_code"]) == ["A"]  # B's 28 degC threshold is never reached
    assert int(hw["days"].iloc[0]) == 3
    # One degree lower thresholds: still nothing at B (26 < 27).
    assert (we.site_events(d, thr, heat_offset=1.0)["type"] == "heatwave").sum() == 1


def test_single_day_events_and_cold_spell_length():
    thr = pd.DataFrame({"site_code": ["A"], "heat_threshold_c": [25.0]})
    d = _daily(
        "A",
        [20] * 5,
        t_mean=np.array([1.0, 1.5, 5.0, 1.0, 5.0]),
        gust_max=np.array([10, 23, 10, 10, 10.0]),
        precip_mm=np.array([0, 0, 30, 0, 0.0]),
    )
    ev = we.site_events(d, thr).set_index("type")
    assert ev.loc["cold spell", "days"] == 2  # the lone cold day on day 4 is not a spell
    assert ev.loc["high wind", "start"] == pd.Timestamp("2025-07-02")
    assert ev.loc["heavy rain", "peak"] == 30


def test_national_events_need_five_percent_of_sites_and_bridge_one_day_gaps():
    rows = []
    for i in range(20):  # 20 sites; 5 % is one site
        rows.append(
            {
                "site_code": f"S{i}",
                "type": "heatwave",
                "start": pd.Timestamp("2025-07-01"),
                "end": pd.Timestamp("2025-07-03"),
                "days": 3,
                "threshold": 25,
                "peak": 30,
            }
        )
    rows.append(
        {
            "site_code": "S0",
            "type": "heatwave",
            "start": pd.Timestamp("2025-07-05"),
            "end": pd.Timestamp("2025-07-07"),
            "days": 3,
            "threshold": 25,
            "peak": 30,
        }
    )
    nat, _ = we.national_events(pd.DataFrame(rows), n_sites=20)
    assert len(nat) == 1  # 4 July has no site in an event but is bridged
    e = nat.iloc[0]
    assert (e["start"], e["end"], e["sites"], e["peak_sites_on_one_day"]) == (
        pd.Timestamp("2025-07-01"),
        pd.Timestamp("2025-07-07"),
        20,
        20,
    )
    nat40, _ = we.national_events(pd.DataFrame(rows), n_sites=40)  # 5 % = 2 sites
    assert nat40["end"].iloc[0] == pd.Timestamp("2025-07-03")
