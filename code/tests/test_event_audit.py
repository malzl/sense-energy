"""Audit of the event findings and the whole-record forecast scores: the pure parts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.analysis import event_audit as ea
from sense_energy.analysis import forecast_record as fr


def test_leave_one_out_mean_leaves_the_row_out():
    v = pd.Series([1.0, 3.0, 5.0, 10.0])
    g = pd.Series(["a", "a", "a", "b"])
    out = ea._loo_mean(v, g)
    assert out.iloc[:3].tolist() == [4.0, 3.0, 2.0] and np.isnan(out.iloc[3])


def _site_events(n_sites: int = 12, n_events: int = 6, seed: int = 0) -> pd.DataFrame:
    """Reactions = event effect + site effect + small noise; regions carry nothing."""
    rng = np.random.default_rng(seed)
    site_effect = rng.normal(0, 5, n_sites)
    event_effect = rng.normal(8, 2, n_events)
    rows = []
    for s in range(n_sites):
        for e in range(n_events):
            rows.append(
                {
                    "site_code": f"S{s}",
                    "event_id": f"E{e}",
                    "region": "north" if s % 2 else "south",
                    "reaction_pct": event_effect[e] + site_effect[s] + rng.normal(0, 0.5),
                }
            )
    return pd.DataFrame(rows)


def test_a_site_effect_is_found_by_the_sites_other_events_and_not_by_the_region():
    p = ea.predictions(_site_events())
    err = {c: (p["reaction_pct"] - p[c]).abs().mean() for c in p.columns if c.startswith("p_")}
    assert err["p_event_site"] < 1.5 < err["p_event"]
    assert err["p_region"] > err["p_event_site"]
    assert err["p_zero"] > err["p_event"]


def test_grid_summary_counts_signs_and_placebos():
    grid = pd.DataFrame(
        {
            "metric": "reaction, no plant",
            "type": "heatwave",
            "value": [5.0, 4.0, -1.0, 6.0],
            "lo": [2.0, 1.0, -3.0, -1.0],
            "hi": [8.0, 7.0, 1.0, 13.0],
            "baseline": [True, False, False, False],
        }
    )
    plc = pd.DataFrame({"metric": "reaction, no plant", "type": "heatwave", "value": [0.5, -6.0]})
    s = ea.grid_summary(grid, plc).iloc[0]
    assert s["baseline"] == 5.0 and s["specifications"] == 4
    assert s["same_sign_share"] == 0.75 and s["interval_excludes_zero_share"] == 0.5
    assert s["placebo_as_large"] == 1 and s["placebo_min"] == -6.0


def test_a_shifted_mask_moves_the_event_days():
    idx = pd.date_range("2025-01-01", periods=10, freq="D")
    m = pd.DataFrame({"A": [False] * 10}, index=idx)
    m.iloc[2:4] = True
    out = ea._shift(m, 5)
    assert out["A"].tolist() == [False] * 7 + [True, True, False]


def test_error_and_captured_share_from_site_sums():
    per = pd.DataFrame(
        {"ae": [1.0, 4.0], "a": [10.0, 20.0], "cut_point_kw": [1.0, 2.0], "perfect_kw": [2.0, 4.0]}
    )
    assert np.isclose(fr.nmae(per), 15.0)  # mean of 10 % and 20 %
    assert np.isclose(fr.captured(per), 50.0)
    stat, lo, hi = fr._boot(pd.concat([per] * 5, ignore_index=True), fr.captured)
    assert np.isclose(stat, 50.0) and lo <= 50.0 <= hi


def test_a_perfect_forecast_gets_the_whole_peak_cut_for_every_battery():
    t = np.arange(48)
    act = np.tile(100.0 + 40.0 * np.exp(-((t - 30) ** 2) / 20.0), (3, 1))
    mx = {"key": pd.DataFrame({"site_code": ["A", "B", "C"]}), "act": act, "point": act.copy()}
    mean_kw = pd.Series({"A": 100.0, "B": 100.0, "C": 100.0})
    for share, hours in fr.BATTERIES:
        v = fr.value_columns(mx, mean_kw, share, hours)
        assert (v["perfect_kw"] > 0).all()
        assert np.allclose(v["cut_point_kw"], v["perfect_kw"], atol=1e-6)
    low = {**mx, "point": act * 0.8}  # a forecast that is too low plans too low a threshold
    v = fr.value_columns(low, mean_kw, 0.2, 2.0)
    assert (v["cut_point_kw"] < v["perfect_kw"]).all()
    assert np.allclose(v["cut_level_kw"], v["perfect_kw"], atol=1e-6)  # the true level repairs it


def test_seasons_put_january_into_the_winter_that_started_the_year_before():
    d = pd.Series(pd.to_datetime(["2024-01-10", "2024-12-10", "2025-07-01"]))
    assert fr._season(d, "cold spell").tolist() == [
        "winter 2023/24",
        "winter 2024/25",
        "winter 2025/26",
    ]
    assert fr._season(d, "heatwave").tolist() == ["summer 2024", "summer 2024", "summer 2025"]
