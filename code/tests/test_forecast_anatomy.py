"""Where the day-ahead error sits: the pure parts, on synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import flexibility as fx
from sense_energy.analysis import forecast_anatomy as fa
from sense_energy.data.strikes import ROUNDS, strike_days
from sense_energy.experiments import forecast_probe as fp
from sense_energy.experiments import poc


def _actual(days: int = 3, sites: tuple[str, ...] = ("A", "B"), seed: int = 0) -> pd.DataFrame:
    """Half-hourly demand in kWh with a daily shape, long, with the local day."""
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-06-02", periods=48 * days, freq="30min", tz="UTC")
    shape = 10.0 + 4.0 * np.sin(np.arange(48 * days) * 2 * np.pi / 48)
    rows = []
    for k, s in enumerate(sites):
        y = (1 + k) * shape + rng.normal(0, 0.1, len(t))
        rows.append(pd.DataFrame({"target": t, "site_code": s, "y": y}))
    out = pd.concat(rows, ignore_index=True)
    out["day"] = out["target"].dt.tz_localize(None).dt.normalize()
    return out


def _forecast(actual: pd.DataFrame, model: str, point) -> pd.DataFrame:
    f = actual[["site_code", "target"]].copy()
    f["model"] = model
    f["point"] = point
    return f


def test_strike_days_cover_the_rounds_once():
    s = strike_days()
    assert s["date"].is_unique and len(s) == 70
    assert len(ROUNDS) == 19 and sum(g == "junior doctors" for *_, g in ROUNDS) == 15
    both = s[s["junior"] & s["consultant"]]["date"].dt.strftime("%Y-%m-%d").tolist()
    assert both == ["2023-09-20", "2023-10-02", "2023-10-03", "2023-10-04"]
    first = s.set_index("date").loc["2024-01-08", "round"]
    assert first == pd.Timestamp("2024-01-03")  # a strike day knows the first day of its round
    assert s["date"].min() >= pd.Timestamp("2022-12-07")  # inside the demand record


def test_near_holidays_marks_christmas_and_the_easter_fortnight():
    d = pd.Series(
        pd.to_datetime(
            [
                "2023-12-20",  # the week before Christmas
                "2024-01-05",  # first week of January
                "2024-01-09",
                "2023-04-11",  # the day after Easter Monday 2023
                "2023-03-13",
                "2025-07-25",
                "2026-04-07",  # the day after Easter Monday 2026
            ]
        )
    )
    assert fa.near_holidays(d).tolist() == [True, True, False, True, False, False, True]


def test_level_oracle_removes_a_pure_level_error_and_keeps_a_shape_error():
    actual = _actual()
    level = _forecast(actual, "level", 1.2 * actual["y"])  # 20 % too high all day
    flat = actual.groupby(["site_code", "day"])["y"].transform("mean")
    shape = _forecast(actual, "shape", flat)  # right level, no shape
    sd = fa.site_day_scores(pd.concat([level, shape]), actual)
    lev, shp = sd[sd["model"] == "level"], sd[sd["model"] == "shape"]
    assert lev["ae_shape"].max() == pytest.approx(0.0, abs=1e-9)
    assert lev["bias_pct"].to_numpy() == pytest.approx(20.0)
    assert (lev["ae"] / lev["a_kwh"]).to_numpy() == pytest.approx(0.2)
    assert shp["ae_shape"].to_numpy() == pytest.approx(shp["ae"].to_numpy())
    assert shp["bias_pct"].abs().max() < 1e-9

    sd["group"] = sd["site_code"].map({"A": "small", "B": "large"})
    t = fa.error_table(sd, ["group"]).set_index(["model", "group"])
    assert t.loc[("level", "small"), "level_share_pct"] == pytest.approx(100.0)
    assert t.loc[("shape", "small"), "level_share_pct"] == pytest.approx(0.0, abs=1e-9)
    assert t.loc[("level", "small"), "nmae_pct"] == pytest.approx(20.0)
    # site B has twice the demand of site A, so twice the kWh of error at the same normalised error
    assert t.loc[("level", "large"), "error_share_pct"] == pytest.approx(200 / 3, rel=1e-2)
    assert t.loc[("level", "large"), "demand_share_pct"] == pytest.approx(200 / 3, rel=1e-2)
    assert t.loc[("level", "large"), "day_share_pct"] == pytest.approx(50.0)


def test_site_days_with_too_few_periods_are_dropped():
    actual = _actual(days=2, sites=("A",))
    f = _forecast(actual, "m", actual["y"]).iloc[:60]  # one full day and 12 periods of the next
    sd = fa.site_day_scores(f, actual)
    assert len(sd) == 1 and sd["n"].iloc[0] == 48
    assert len(fa.site_day_scores(f, actual, min_periods=10)) == 2


def test_unusual_steps_find_the_day_a_plant_switches():
    t = pd.date_range("2025-01-06", periods=48 * 70, freq="30min", tz="UTC")
    rng = np.random.default_rng(1)
    base = 100.0 + 20.0 * np.sin(np.arange(len(t)) * 2 * np.pi / 48)
    y = pd.DataFrame({"A": base + rng.normal(0, 1.0, len(t))}, index=t)
    day = pd.DatetimeIndex(t.tz_localize(None).normalize())
    switch_day = pd.Timestamp("2025-03-05")
    at = (day == switch_day) & (t.hour >= 10)
    y.loc[at, "A"] += (
        60.0  # the plant stops at 10:00: import rises by 60 kW for the rest of the day
    )
    out = fa.unusual_steps(y, day).set_index("day")["step_ratio"]
    assert out.idxmax() == switch_day
    assert out[switch_day] > 5 * fa.SWITCH_RATIO
    ordinary = out.drop([switch_day, switch_day + pd.Timedelta(days=1)]).dropna()
    assert (ordinary > fa.SWITCH_RATIO).mean() < 0.02  # the daily shape itself is not a step


def test_a_perfect_forecast_captures_the_whole_peak_cut():
    actual = _actual(days=4)
    mean_kw = (actual.groupby("site_code")["y"].mean() * 2.0).rename(None)
    perfect = _forecast(actual, "perfect", actual["y"])
    flat = _forecast(actual, "flat", actual.groupby("site_code")["y"].transform("mean"))
    f = pd.concat([perfect, flat])
    for q in ("q70",):
        f[q] = f["point"]
    v = fa.value_site_days(f, actual, mean_kw)
    assert set(v.columns) >= {"perfect_kw", "cut_point_kw", "cut_level_kw"}
    assert "cut_q70_kw" not in v.columns  # a q70 equal to the point is not a second plan
    t = fa.value_table(v.assign(all="all"), ["all"]).set_index("model")
    assert t.loc["perfect", "captured_point_pct"] == pytest.approx(100.0, abs=1e-6)
    assert t.loc["perfect", "captured_level_pct"] == pytest.approx(100.0, abs=1e-6)
    assert t.loc["flat", "captured_point_pct"] < t.loc["perfect", "captured_point_pct"]
    assert (v["perfect_kw"] > 0).all()


def test_quantile_scores_coverage_and_pinball():
    actual = _actual(days=2, sites=("A",))
    f = _forecast(actual, "m", actual["y"])
    for q in range(10, 100, 10):
        f[f"q{q}"] = actual["y"] + (q - 50) / 10.0  # the truth sits at the median
    sd = fa.quantile_scores(f, actual)
    assert sd["in80"].to_numpy() == pytest.approx(1.0)
    assert sd["ae"].to_numpy() == pytest.approx(0.0)
    # pinball of a symmetric fan around the truth: sum over tau of max(tau d, (tau - 1) d)
    d = -(np.arange(10, 100, 10) - 50) / 10.0
    tau = np.arange(10, 100, 10) / 100.0
    per_period = 2 * np.maximum(tau * d, (tau - 1) * d).sum() / 9
    assert sd["crps"].to_numpy() == pytest.approx(48 * per_period)
    off = f.assign(**{f"q{q}": actual["y"] + 5.0 + q / 100.0 for q in range(10, 100, 10)})
    assert fa.quantile_scores(off, actual)["in80"].max() == 0.0


def test_paired_bootstrap_sees_a_real_difference_and_no_false_one(monkeypatch):
    monkeypatch.setattr(fa, "N_BOOT", 200)
    rng = np.random.default_rng(2)
    days = pd.date_range("2024-03-01", periods=84, freq="D")
    rows = []
    for s in ("A", "B"):
        noise = rng.uniform(5.0, 15.0, len(days))
        for model, extra in (("base", 0.0), ("same", 0.0), ("worse", 2.0)):
            rows.append(
                pd.DataFrame(
                    {"model": model, "site_code": s, "day": days, "a": 100.0, "ae": noise + extra}
                )
            )
    sd = pd.concat(rows, ignore_index=True)
    same = fa.paired_bootstrap(sd, "same", "base")
    assert same["diff_pts"] == pytest.approx(0.0, abs=1e-12)
    assert same["lo"] == pytest.approx(0.0, abs=1e-9) and same["hi"] == pytest.approx(0.0, abs=1e-9)
    worse = fa.paired_bootstrap(sd, "worse", "base")
    assert worse["diff_pts"] == pytest.approx(2.0)
    assert worse["lo"] > 1.9 and worse["hi"] < 2.1  # paired: the common noise cancels


def _panel(days: int = 4) -> poc.Panel:
    local = pd.date_range("2025-06-02", periods=48 * days, freq="30min", tz="Europe/London")
    index = local.tz_convert("UTC")
    y = pd.DataFrame({"A": np.arange(len(index), dtype=float)}, index=index)
    return poc.Panel(
        y_raw=y,
        y_imp=y,
        index=index,
        local=local,
        local_date=local.tz_localize(None).normalize().to_numpy().astype("datetime64[D]"),
        tod=(local.hour * 2 + local.minute // 30).to_numpy(),
        dow=local.dayofweek.to_numpy(),
        sites=["A"],
    )


def test_intraday_origins_forecast_the_rest_of_the_target_day():
    panel = _panel()
    config = {"timezone": "Europe/London"}
    o6 = fp.intraday_origins(panel, config, "2025-06-03", "2025-06-05", 1, 6)
    assert [str(o.target_day) for o in o6] == ["2025-06-03", "2025-06-04", "2025-06-05"]
    first = o6[0]
    assert panel.local[first.origin_pos] == pd.Timestamp("2025-06-03 06:00", tz="Europe/London")
    assert len(first.target_pos) == 35  # 06:30 to 23:30
    assert first.target_pos[0] == first.origin_pos + 1
    assert (panel.local_date[first.target_pos] == first.target_day).all()
    assert first.lead_hours[0] == pytest.approx(0.5) and first.lead_hours[-1] == pytest.approx(17.5)
    o0 = fp.intraday_origins(panel, config, "2025-06-03", "2025-06-03", 1, 0)
    assert len(o0[0].target_pos) == 47  # 00:30 to 23:30
    # a day before the record starts has no context and gives no origin
    assert fp.intraday_origins(panel, config, "2025-05-30", "2025-05-30", 1, 6) == []


def test_issue_on_the_day_before_and_on_the_day_share_their_last_periods():
    panel = _panel()
    config = {"timezone": "Europe/London", "issue_time_local": "16:30"}
    day_ahead = poc.make_origins(panel, config, "2025-06-04", "2025-06-04", 1)[0]
    late = fp.intraday_origins(panel, config, "2025-06-04", "2025-06-04", 1, 6)[0]
    assert set(late.target_pos) < set(day_ahead.target_pos)
    assert late.origin_pos - day_ahead.origin_pos == 27  # 16:30 to 06:00 is 13.5 hours


def test_the_plain_footfall_variant_has_no_covariates():
    assert fp.mobility_windows(_panel(), [], {"context_periods": 48}, "plain") is None
    assert set(fp.MOBILITY_VARIANTS) == {"plain", "base", "prev_day", "oracle", "uni_oracle"}


def test_a_day_run_in_two_parts_gives_the_peak_of_one_run():
    rng = np.random.default_rng(3)
    load = 50.0 + 30.0 * rng.random((5, 48))
    p, e = np.full(5, 10.0), np.full(5, 20.0)
    theta = fx.plan_threshold(load, p, e) + 2.0
    whole = fx.realised_peak(theta, load, p, e)
    first, soc = fx.realised_peak(theta, load[:, :13], p, e, return_soc=True)
    second = fx.realised_peak(theta, load[:, 13:], p, e, soc0=soc)
    assert np.maximum(first, second) == pytest.approx(whole)
    assert (soc <= e + 1e-9).all() and (soc >= -1e-9).all()


def test_a_plan_from_an_empty_battery_needs_a_higher_threshold():
    load = np.full((1, 48), 10.0)
    load[0, 0:4] = 20.0  # a two-hour peak at the start: no time to charge first
    p, e = np.array([5.0]), np.array([10.0])
    full = fx.plan_threshold(load, p, e)
    empty = fx.plan_threshold(load, p, e, soc0=np.array([0.0]))
    half = fx.plan_threshold(load, p, e, soc0=np.array([5.0]))
    assert full[0] == pytest.approx(15.0, abs=1e-6)
    assert empty[0] == pytest.approx(20.0, abs=1e-6)
    assert half[0] == pytest.approx(17.5, abs=1e-6)  # 5 kWh hold 2.5 kW for two hours


def test_size_adjustment_separates_the_plant_from_the_size():
    kw = np.array([10.0, 30.0, 100.0, 300.0, 1000.0])
    rate = 20.0 * kw**-0.2  # per cent: larger sites have a lower normalised error
    plain = pd.DataFrame(
        {
            "site_code": [f"P{i}" for i in range(5)],
            "group": "inpatient",
            "mean_kw": kw,
            "nmae_pct": rate,
            "demand_mwh": kw,
            "error_mwh": kw * rate / 100.0,
            "chp_share": np.nan,
        }
    )
    # two CHP sites that import 100 kW of a 200 kW demand: one with the error of a plain site
    # of 200 kW, one with twice that
    expected = 200.0 * 20.0 * 200.0**-0.2 / 100.0
    chp = pd.DataFrame(
        {
            "site_code": ["C0", "C1"],
            "group": "on-site CHP",
            "mean_kw": 100.0,
            "demand_mwh": 100.0,
            "error_mwh": [expected, 2 * expected],
            "chp_share": 0.5,
        }
    )
    chp["nmae_pct"] = 100 * chp["error_mwh"] / chp["demand_mwh"]
    sites = pd.concat([plain, chp], ignore_index=True)
    sites["error_share_pct"] = 100 * sites["error_mwh"] / sites["error_mwh"].sum()
    out, summary = fa.size_adjusted(sites)
    out = out.set_index("site_code")
    assert out.loc["C0", "total_kw"] == pytest.approx(200.0)
    assert out.loc["C0", "plant_added_mwh"] == pytest.approx(0.0, abs=1e-9)
    assert out.loc["C1", "plant_added_mwh"] == pytest.approx(expected)
    assert summary["error_falls_with_size_slope"] == pytest.approx(-0.2)
    assert summary["plant_added_share_of_chp_error_pct"] == pytest.approx(100 / 3, abs=0.1)
    assert summary["chp_share_of_total_demand_pct"] > summary["chp_share_of_import_pct"]


def _site_days(wave: list[str], ae: list[float]) -> pd.DataFrame:
    days = pd.date_range("2025-07-01", periods=len(wave), freq="D")
    return pd.DataFrame(
        {
            "model": "m",
            "site_code": "A",
            "group": "inpatient",
            "day": days,
            "wave": wave,
            "ae": ae,
            "ae_shape": ae,
            "a_kwh": 100.0,
        }
    )


def test_wave_days_are_compared_with_the_same_site_and_month():
    july = _site_days(["heat wave"] * 3 + ["other days"] * 7, [20.0] * 3 + [10.0] * 7)
    august = _site_days(["other days"] * 10, [50.0] * 10).assign(
        day=pd.date_range("2025-08-01", periods=10, freq="D")
    )
    t = fa.wave_matched(pd.concat([july, august]), "heat wave")
    t = t[t["group"] == "all sites"].set_index("wave")
    assert t.loc["heat wave", "nmae_pct"] == pytest.approx(20.0)
    assert t.loc["other days", "nmae_pct"] == pytest.approx(10.0)  # August has no heat day
    assert t.loc["other days", "site_days"] == 7


def test_step_table_shares_cover_the_panel():
    rng = np.random.default_rng(4)
    n = 400
    sd = pd.DataFrame(
        {
            "model": fa.BEST,
            "site_code": rng.choice(["A", "B", "C"], n),
            "day": pd.Timestamp("2025-06-01"),
            "ae": rng.uniform(1, 5, n),
            "a_kwh": rng.uniform(50, 100, n),
            "step_bin": rng.choice(fa.STEP_LABELS, n),
        }
    )
    sd["group"] = sd["site_code"].map({"A": "inpatient", "B": "outpatient", "C": "on-site CHP"})
    t = fa.step_table(sd)
    assert t["error_share_pct"].sum() == pytest.approx(100.0)
    assert t["day_share_pct"].sum() == pytest.approx(100.0)
    assert t.groupby("group")["day_share_of_group_pct"].sum().to_numpy() == pytest.approx(100.0)
    assert list(t[t["group"] == "inpatient"]["step_bin"]) == list(fa.STEP_LABELS)
