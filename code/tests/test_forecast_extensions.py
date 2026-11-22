"""Extensions of the day-ahead forecast (correction within the day, split weather inputs,
re-training, the forecast range): the pure parts, on synthetic data."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import forecast_extensions as fe
from sense_energy.analysis import forecast_range as rg
from sense_energy.experiments import forecast_extensions as fx
from sense_energy.experiments import gbm_rolling


def test_quarters_cover_the_test_without_a_gap():
    q = gbm_rolling.quarters("2024-01-01", "2024-08-15")
    assert q == [
        ("2024-01-01", "2024-03-31"),
        ("2024-04-01", "2024-06-30"),
        ("2024-07-01", "2024-08-15"),
    ]


def test_block_starts_are_two_hour_blocks_of_the_chosen_days():
    days = 10
    tod = np.tile(np.arange(48), days)
    local_date = np.repeat(np.arange("2025-01-01", "2025-01-11", dtype="datetime64[D]"), 48)
    pick = np.array(["2025-01-09", "2025-01-10"], dtype="datetime64[D]")
    starts = fx.block_starts(tod, local_date, pick)
    assert len(starts) == 24 and (tod[starts] % 4 == 0).all()
    assert set(local_date[starts]) == set(pick)
    # a day with too little history before it gives no block
    assert (
        len(fx.block_starts(tod, local_date, np.array(["2025-01-02"], dtype="datetime64[D]"))) == 0
    )


def test_windows_come_start_by_start_and_site_by_site():
    a = np.arange(40, dtype="float32").reshape(20, 2)  # 20 periods, 2 sites
    starts = np.array([5, 10])
    before = fx.windows_before(a, starts, 3)
    after = fx.windows_from(a, starts, 2)
    assert before.shape == (4, 3) and after.shape == (4, 2)
    assert before[0].tolist() == [4.0, 6.0, 8.0]  # site 0, periods 2..4
    assert before[1].tolist() == [5.0, 7.0, 9.0]  # site 1, the same periods
    assert after[2].tolist() == [20.0, 22.0]  # second start, site 0, periods 10..11


def test_fill_rows_leaves_complete_rows_alone():
    x = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]], dtype="float32")
    out = fx.fill_rows(x.copy())
    assert np.isfinite(out).all() and out[1].tolist() == [4.0, 5.0, 6.0] and out[0, 1] == 1.0


def _scores(days: int = 900, sites: int = 3, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (days, sites, 48)).astype("float32")


def test_quantile_tracking_reaches_the_nominal_miss_rate():
    s = _scores()
    for integrator in (False, True):
        used = rg.track(s, integrator=integrator)
        miss = (s > used)[100:].mean()
        assert miss == pytest.approx(rg.ALPHA, abs=0.015)
    # the offset approaches the 90 % quantile of a standard normal
    assert rg.track(s)[-200:].mean() == pytest.approx(1.28, abs=0.25)


def test_tracking_uses_only_scores_known_at_the_issue():
    s = _scores(days=40)
    base = rg.track(s)
    assert (base[0] == 0).all()
    assert (base[1][:, rg.LATE_SLOT :] == 0).all()  # late half-hours wait one more day
    assert (base[1][:, : rg.LATE_SLOT] != 0).all()
    changed = s.copy()
    changed[20] += 5.0
    out = rg.track(changed)
    assert np.array_equal(out[:21], base[:21])
    assert np.array_equal(out[21][:, rg.LATE_SLOT :], base[21][:, rg.LATE_SLOT :])
    assert not np.array_equal(out[21][:, : rg.LATE_SLOT], base[21][:, : rg.LATE_SLOT])
    assert not np.array_equal(out[22][:, rg.LATE_SLOT :], base[22][:, rg.LATE_SLOT :])


def test_a_missing_score_gives_no_update():
    s = _scores(days=30)
    s[10] = np.nan
    used = rg.track(s)
    assert np.array_equal(used[11][:, : rg.LATE_SLOT], used[10][:, : rg.LATE_SLOT])


def test_rolling_quantile_waits_for_thirty_days_and_respects_the_delay():
    s = _scores(days=200)
    off = rg.rolling_quantile(s)
    assert (off[:30] == 0).all()
    assert off[150:].mean() == pytest.approx(1.28, abs=0.2)
    changed = s.copy()
    changed[100] += 50.0
    out = rg.rolling_quantile(changed)
    assert np.array_equal(out[:101], off[:101])
    assert np.array_equal(out[101][:, rg.LATE_SLOT :], off[101][:, rg.LATE_SLOT :])


def test_site_stats_count_the_half_hours_outside_the_range():
    days, sites = 10, 2
    y = np.full((days, sites, 48), 10.0, "float32")
    lower, upper = np.full_like(y, 8.0), np.full_like(y, 12.0)
    y[:, 0, :12] = 13.0  # site 0: a quarter above
    y[:, 1, :24] = 7.0  # site 1: half below
    c = {"y": y, "sites": ["A", "B"]}
    per = rg.site_stats(c, lower, upper, np.ones((1, 1, 1), bool)).set_index("site_code")
    assert per.loc["A", "above_pct"] == pytest.approx(25.0)
    assert per.loc["B", "below_pct"] == pytest.approx(50.0)
    assert per.loc["A", "inside_pct"] == pytest.approx(75.0)
    assert per.loc["A", "width_pct"] == pytest.approx(100 * 4.0 / y[:, 0].mean())
    # only site 0's first twelve half-hours cost at the upper bound: 0.9 * 1 kWh each, and
    # every half-hour inside costs 0.1 times its distance to each bound
    loss_a = (0.9 * 1.0 * 12 + 0.1 * 5.0 * 12 + 0.1 * 4.0 * 36) / y[0, 0].sum()
    assert per.loc["A", "loss_pct"] == pytest.approx(100 * loss_a)


def test_wider_bounds_after_tracking_cover_a_model_that_is_too_narrow():
    rng = np.random.default_rng(1)
    days, sites = 500, 4
    y = (100 + rng.normal(0, 10, (days, sites, 48))).astype("float32")
    c = {
        "y": y,
        "sites": list("ABCD"),
        "days": pd.date_range("2024-01-01", periods=days),
        "q10": np.full_like(y, 95.0),  # true 10 % and 90 % quantiles are 87 and 113
        "q50": np.full_like(y, 100.0),
        "q90": np.full_like(y, 105.0),
    }
    mask = rg.scored_days(c)
    raw = rg.site_stats(c, *rg.bounds(c, rg.RAW), mask)["inside_pct"].mean()
    for method in (rg.ROLLING, rg.TRACKED, rg.INTEGRATED):
        per = rg.site_stats(c, *rg.bounds(c, method), mask)
        assert per["inside_pct"].mean() == pytest.approx(80.0, abs=3.0)
        assert per["inside_pct"].mean() > raw + 30


def _panel(days: int = 60, sites: tuple[str, ...] = ("A", "B", "C")) -> SimpleNamespace:
    t = pd.date_range("2025-01-01", periods=48 * days, freq="30min", tz="UTC")
    rng = np.random.default_rng(0)
    y = pd.DataFrame({s: 10 + k + rng.normal(0, 0.5, len(t)) for k, s in enumerate(sites)}, index=t)
    local_date = t.tz_localize(None).normalize().to_numpy().astype("datetime64[D]")
    return SimpleNamespace(y_raw=y, sites=list(sites), local_date=local_date, index=t)


def test_site_days_keep_only_what_every_model_has():
    panel = _panel()
    t = panel.index[panel.index >= "2025-02-20"]
    rows = []
    for model, shift in (("a", 0.0), ("b", 1.0)):
        for s in panel.sites:
            rows.append(
                pd.DataFrame(
                    {
                        "model": model,
                        "site_code": s,
                        "target": t,
                        "point": panel.y_raw.loc[t, s].to_numpy() + shift,
                    }
                )
            )
    f = pd.concat(rows, ignore_index=True)
    # model b has no forecast for site C on the last day
    last = t.normalize() == t.normalize().max()
    drop = (f["model"] == "b") & (f["site_code"] == "C") & f["target"].isin(t[last])
    sd = fe.site_days(f[~drop], panel)
    assert sd.groupby(["site_code", "day"])["model"].nunique().eq(2).all()
    assert not ((sd["site_code"] == "C") & (sd["day"] == sd["day"].max())).any()
    e = fe.errors(sd, {"all": pd.Series(True, index=sd.index)}).set_index("model")
    assert e.loc["a", "nmae_pct"] == pytest.approx(0.0, abs=1e-6)
    assert 7.0 < e.loc["b", "nmae_pct"] < 10.5  # one kWh over a demand of 10 to 12


def test_correction_long_drops_unobserved_half_hours():
    wide = pd.DataFrame(
        {
            "site_code": ["A", "A"],
            "target": pd.date_range("2025-01-01", periods=2, freq="30min", tz="UTC"),
            "step": [1, 2],
            "y": [1.0, np.nan],
            **{m: [1.0, 2.0] for m in fe.CORRECTION_MODELS},
        }
    )
    out = fe.correction_long(wide)
    assert len(out) == len(fe.CORRECTION_MODELS) and set(out["model"]) == set(fe.CORRECTION_MODELS)


def test_site_differences_show_when_one_site_carries_the_mean():
    sites = pd.DataFrame(
        {"site_code": list("ABCDE"), "a": [10.0, 10.0, 10.0, 10.0, 110.0], "b": [11.0] * 5}
    )
    d = fe.site_differences(sites, [("a", "b")]).iloc[0]
    assert d["mean_diff_pts"] == pytest.approx(19.0)
    assert d["median_diff_pts"] == pytest.approx(-1.0) and d["sites_a_better"] == 4
    assert d["largest_diff_site"] == "E" and d["mean_diff_without_it_pts"] == pytest.approx(-1.0)
