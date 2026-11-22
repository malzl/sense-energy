"""The special-day rule and the sensitivity bounds: the pure parts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.analysis import causal_sensitivity as cs
from sense_energy.analysis import special_days as sp


def test_calendar_names_holidays_and_the_days_next_to_them():
    cal = sp.special_calendar("2024-03-25", "2024-04-05").set_index("day")
    assert cal.loc["2024-03-29", "name"] == "Good Friday"
    assert cal.loc["2024-04-01", "name"] == "Easter Monday"
    assert cal.loc["2024-03-28", "kind"] == "day before a bank holiday"
    assert cal.loc["2024-04-02", "kind"] == "day after a bank holiday"
    assert pd.Timestamp("2024-03-30") not in cal.index  # the weekend between is an ordinary one
    xmas = sp.special_calendar("2025-12-20", "2026-01-05").set_index("day")
    assert xmas.loc["2025-12-24", "kind"] == "Christmas period"
    assert xmas.loc["2026-01-02", "name"] == "day after New Year"
    assert xmas.loc["2025-12-25", "kind"] == "bank holiday"


def test_the_corresponding_past_day_is_the_same_holiday_and_prefers_the_same_kind_of_weekday():
    cal = sp.special_calendar("2020-01-01", "2026-01-05")
    past = sp.corresponding_past_day(cal, set(cal["day"])).to_numpy()
    by_day = dict(zip(cal["day"], past, strict=True))
    assert by_day[pd.Timestamp("2025-04-18")] == pd.Timestamp("2024-03-29")  # Good Friday
    # Christmas Day 2023 was a Monday; 2022 and 2021 fell on a weekend, 2020 on a Friday
    assert by_day[pd.Timestamp("2023-12-25")] == pd.Timestamp("2020-12-25")
    # the substitute day of 2022 is no Christmas Day
    names = cal.set_index("day")["name"]
    assert names.loc["2022-12-27"] == "substitute bank holiday"
    assert sp.corresponding_past_day(cal, set()).isna().all()


def test_the_rule_repairs_a_forecast_that_misses_a_holiday_drop():
    point = np.full((1, 48), 100.0)
    past_act, past_point = np.full((1, 48), 70.0), np.full((1, 48), 100.0)
    assert np.allclose(sp.adjust(point, past_act, past_point, "half-hour"), 70.0)
    assert np.allclose(sp.adjust(point, past_act, past_point, "day"), 70.0)
    wild = sp.adjust(point, np.full((1, 48), 1.0), past_point, "day")
    assert np.allclose(wild, 100.0 * sp.RATIO_LIMITS[0])  # the ratio is held within limits


def test_no_hidden_confounding_gives_the_plain_mean():
    y = np.random.default_rng(0).normal(3.0, 2.0, 500)
    lo, hi = cs.mean_bounds(y, 1.0, 0.2)
    assert np.isclose(lo, y.mean()) and np.isclose(hi, y.mean())


def test_bounds_widen_with_gamma_and_tighten_with_the_propensity():
    y = np.random.default_rng(1).normal(0.0, 1.0, 2000)
    widths = [np.diff(cs.mean_bounds(y, g, 0.1))[0] for g in (1.0, 1.5, 2.0, 5.0)]
    assert all(a < b for a, b in zip(widths, widths[1:], strict=False))
    assert np.diff(cs.mean_bounds(y, 2.0, 0.9))[0] < np.diff(cs.mean_bounds(y, 2.0, 0.1))[0]
    lo, hi = cs.mean_bounds(y, 3.0, 0.1)
    assert lo < y.mean() < hi


def test_gmsm_weights_sum_to_one():
    for gamma in (1.0, 1.3, 4.0):
        for q in (0.02, 0.5, 0.97):
            s_minus, s_plus = cs.gmsm_scales(gamma, q)
            upper = gamma / (1 + gamma) / s_plus + 1 / (1 + gamma) / s_minus
            lower = 1 / (1 + gamma) / s_minus + gamma / (1 + gamma) / s_plus
            assert np.isclose(upper, 1.0) and np.isclose(lower, 1.0)
            assert s_minus <= 1.0 <= s_plus


def test_a_constant_shift_between_arms_is_bounded_around_itself():
    rng = np.random.default_rng(2)
    y0, y1 = rng.normal(0, 1, 800), rng.normal(5, 1, 800)
    lo, hi = cs.effect_bounds(y1, y0, 0.05, 1.5)
    assert lo < 5.0 < hi and lo > 0  # a large effect survives a moderate hidden factor
    lo, hi = cs.effect_bounds(y1 - 4.0, y0, 0.05, 50.0)
    assert lo < 0 < hi  # a small effect does not survive a strong one


def test_day_classes_change_only_the_days_inside_the_policy(monkeypatch):
    monkeypatch.setattr(sp.fr, "N_BOOT", 50)
    sites = [f"S{i}" for i in range(6)]
    days = pd.date_range("2024-12-20", "2025-01-10")
    record = pd.DataFrame(
        [
            {"model": "m", "site_code": s, "day": d, "a": 100.0, "ae": 10.0}
            for s in sites
            for d in days
        ]
    )
    kinds = {
        pd.Timestamp("2024-12-25"): ("bank holiday", 4.0),  # the rule helps
        pd.Timestamp("2024-12-27"): ("Christmas period", 14.0),  # the rule hurts
        pd.Timestamp("2025-01-02"): ("day after a bank holiday", 16.0),
        pd.Timestamp("2025-01-01"): ("bank holiday", np.nan),  # no corresponding past day
    }
    sd = pd.DataFrame(
        [
            {"model": "m", "site_code": s, "day": d, "kind": k, "ae_rule_half-hour": e}
            for s in sites
            for d, (k, e) in kinds.items()
        ]
    )
    out = sp.day_classes(sd, record).set_index(["policy", "day_class"])
    assert sp.day_class(sd["kind"]).value_counts().to_dict() == {
        "special days": 18,
        "proximity days": 6,
    }
    for policy in sp.POLICIES:
        assert out.loc[(policy, "normal days"), "rule_minus_forecast_pts"] == 0.0
        assert out.loc[(policy, "normal days"), "days"] == len(days) - 4
    holidays = out.loc[("rule on bank holidays", "bank holidays")]
    assert np.isclose(holidays["nmae_with_rule_pct"], 7.0)  # one of two corrected: (4 + 10) / 2
    assert np.isclose(holidays["rule_applied_pct"], 50.0)
    assert out.loc[("rule on bank holidays", "Christmas period"), "rule_minus_forecast_pts"] == 0.0
    christmas = out.loc[("rule on special days", "Christmas period"), "nmae_with_rule_pct"]
    assert np.isclose(christmas, 14.0)
    assert out.loc[("rule on special days", "proximity days"), "rule_minus_forecast_pts"] == 0.0
    both = out.loc[("rule on special and proximity days", "proximity days")]
    assert np.isclose(both["nmae_with_rule_pct"], 16.0)
    # all days: the three changes add up over the 22 days
    total = out.loc[("rule on special and proximity days", "all days"), "rule_minus_forecast_pts"]
    assert np.isclose(total, (-6.0 + 4.0 + 6.0) / len(days))
