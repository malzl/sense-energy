"""Quasi-experimental weather designs: doses, stacking rules, event-study recovery and the date
effect's role."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import weather_causal as wc


def test_dose_arms():
    hdd, cdd = wc.dose([10.0, 14.0, 17.0, 23.5])
    assert hdd.tolist() == [4.0, 0.0, 0.0, 0.0]
    assert cdd.tolist() == [0.0, 0.0, 0.0, 3.5]


def _panel(
    n_sites=30, n_days=240, b_cold=0.012, b_hot=0.03, national_shock=0.0, seed=1
) -> pd.DataFrame:
    """Sites in two regions; temperature = national swing + site anomaly; an optional day shock to
    demand that follows the national temperature (confounds spec A, not spec B)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_days)
    national = 12 + 8 * np.sin(2 * np.pi * np.arange(n_days) / 365) + rng.normal(0, 3, n_days)
    shock = national_shock * np.maximum(14 - national, 0) + rng.normal(0, 0.01, n_days)
    rows = []
    for i in range(n_sites):
        t = national + rng.normal(0, 1.5, n_days) + (i % 2) * 1.0
        hdd, cdd = wc.dose(t)
        month_level = (
            pd.Series(rng.normal(4, 0.5, n_days), index=dates)
            .groupby(dates.month)
            .transform("mean")
        )
        log_kw = (
            month_level.to_numpy()
            + 0.08 * (dates.dayofweek >= 5)
            + b_cold * hdd
            + b_hot * cdd
            + shock
        )
        log_kw = log_kw + rng.normal(0, 0.02, n_days)
        rows.append(
            pd.DataFrame(
                {
                    "site_code": f"S{i:02d}",
                    "date": dates,
                    "t_mean": t,
                    "t_max": t + 4,
                    "kw": np.exp(log_kw),
                    "region_ons": "North" if i % 2 else "South",
                    "group": "inpatient" if i % 3 else "outpatient",
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["log_kw"] = np.log(df["kw"])
    df["hdd14"], df["cdd20"] = wc.dose(df["t_mean"])
    df["holiday"] = df["christmas"] = df["school_summer"] = 0.0
    for c in wc.OTHER_WEATHER:  # spec D controls, independent noise here
        df[c] = rng.normal(size=len(df))
    df["fe_month"] = df["site_code"] + "|" + df["date"].dt.strftime("%Y-%m")
    df["fe_dow"] = df["site_code"] + "|" + df["date"].dt.dayofweek.astype(str)
    df["date_id"] = df["date"].dt.strftime("%Y-%m-%d")
    df["region_date"] = df["region_ons"] + "|" + df["date_id"]
    return df


def test_date_effects_remove_a_national_confounder():
    df = _panel(national_shock=0.01)
    s = wc.twfe_specs(df).set_index(["group", "spec", "term"])
    a, b = s.loc[("no CHP", "A", "hdd14")], s.loc[("no CHP", "B", "hdd14")]
    assert a["coef"] == pytest.approx(0.022, abs=0.004)  # 1.2 % true + 1 % national confounder
    assert b["coef"] == pytest.approx(0.012, abs=0.004)
    assert s.loc[("no CHP", "B", "cdd20"), "coef"] == pytest.approx(0.03, abs=0.01)


def test_identifying_variation_shrinks_with_date_effects():
    v = wc.identifying_variation(_panel()).set_index(["group", "spec", "dose"])
    assert (
        v.loc[("no CHP", "B", "hdd14"), "within_sd"]
        < 0.6 * v.loc[("no CHP", "A", "hdd14"), "within_sd"]
    )


def _events(
    df: pd.DataFrame,
    onset: str,
    treated: list[str],
    duration: int = 3,
    event_id: str = "heatwave_x",
):
    s = pd.Timestamp(onset)
    nat = pd.DataFrame(
        {
            "event_id": [event_id],
            "type": ["heatwave"],
            "start": [s],
            "end": [s + pd.to_timedelta(duration - 1, "D")],
        }
    )
    se = pd.DataFrame(
        {
            "site_code": treated,
            "type": "heatwave",
            "start": s,
            "end": s + pd.to_timedelta(duration - 1, "D"),
        }
    )
    return nat, se


def test_stacking_assigns_treated_and_clean_controls_and_logs_drops():
    df = _panel(n_sites=20, n_days=60)
    treated = [f"S{i:02d}" for i in range(10)]
    nat, se = _events(df, "2025-02-01", treated)
    # A second event whose onset falls in the first one's pre-period, and a third with too few treated.
    nat2, se2 = _events(df, "2025-01-26", treated[:8], event_id="heatwave_pre")
    nat3, se3 = _events(df, "2025-02-20", treated[:2], event_id="heatwave_small")
    st, log = wc.stacks(df, "heatwave", pd.concat([nat, nat2, nat3]), pd.concat([se, se2, se3]))
    assert log.set_index("event_id")["status"].to_dict() == {
        "heatwave_pre": "kept",
        "heatwave_x": "dropped: onset in pre-period",
        "heatwave_small": "dropped: fewer than 5 treated or clean-control sites",
    }
    kept = st[st["stack"] == "heatwave_pre"]
    n = kept.groupby("treated")["site_code"].nunique()
    assert n[1.0] == 8
    assert n[0.0] == 10  # S08, S09 are treated by the overlapping event x, so not clean controls
    assert kept["k"].min() == -wc.WINDOW and kept["k"].max() == wc.WINDOW


def test_event_study_recovers_a_known_effect_with_flat_pre_trends():
    df = _panel(n_sites=24, n_days=80, b_cold=0.0, b_hot=0.0)
    treated = [f"S{i:02d}" for i in range(12)]
    nat, se = _events(df, "2025-02-10", treated, duration=3)
    hit = df["site_code"].isin(treated) & df["date"].between("2025-02-10", "2025-02-12")
    df.loc[hit, "log_kw"] += np.log(1.10)
    st, _ = wc.stacks(df, "heatwave", nat, se)
    att = wc.event_att(st, "log_kw").set_index("term")
    assert 100 * np.expm1(att.loc["d_event", "coef"]) == pytest.approx(10.0, abs=1.5)
    assert abs(att.loc["d_lead", "coef"]) < 0.03  # no pre-effect of the order of the true one
    dyn, _, p_leads = wc.event_study(st, "log_kw")
    dyn = dyn.set_index("k")
    assert dyn.loc[[0, 1, 2], "coef"].mean() == pytest.approx(np.log(1.10), abs=0.015)
    assert p_leads > 0.05


def test_per_degree_divides_demand_att_by_temperature_att():
    att = pd.DataFrame(
        {
            "term": ["d_event"] * 2,
            "type": ["heatwave"] * 2,
            "group": ["no CHP"] * 2,
            "outcome": ["demand", "temperature"],
            "coef": [np.log(1.06), 3.0],
            "lo": [np.log(1.04), 2.5],
            "hi": [np.log(1.08), 3.5],
            "se": [0.01, 0.25],
        }
    )
    p = wc.per_degree(att).iloc[0]
    assert p["pct_per_c"] == pytest.approx(2.0)
    assert p["demand_att_pct"] == pytest.approx(6.0)


def test_extreme_flags_split_at_28_and_30():
    hot28, hot30 = wc.extreme_flags([27.9, 28.0, 29.9, 30.0, 33.0])
    assert hot28.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]
    assert hot30.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0]


def test_haversine_london_manchester():
    km = wc.haversine_km([51.5074, 53.4808], [-0.1278, -2.2426])
    assert km[0, 1] == pytest.approx(262.0, abs=3.0)
    assert km[0, 0] == 0.0


def test_neighbour_weights_use_only_the_band_and_normalise():
    km = pd.DataFrame(
        [[0.0, 40.0, 200.0], [40.0, 0.0, 100.0], [200.0, 100.0, 0.0]],
        index=list("abc"),
        columns=list("abc"),
    )
    w = wc.neighbour_weights(km, band=(30.0, 150.0))
    assert w.loc["a"].tolist() == [0.0, 1.0, 0.0]  # only b is 30-150 km from a
    assert w.loc["b"].tolist() == pytest.approx([0.5, 0.0, 0.5])
    assert w.loc["c"].tolist() == [0.0, 1.0, 0.0]
    far = wc.neighbour_weights(km, band=(300.0, 400.0))
    assert far.isna().all().all()  # nobody has a neighbour that far


def test_response_curve_has_every_spec_and_variable_with_a_zero_reference():
    from sense_energy.analysis import weather_response as wr

    df = _panel(b_cold=0.012, b_hot=0.03)
    for _, (col, bin_col, edges, _, _) in wr.VARIABLES.items():
        df[bin_col] = pd.cut(df[col], edges, right=False).map(lambda iv: iv.left).astype(float)
    c = wc.response_curve(df)
    assert set(c["spec"]) == {"A", "B", "D"}
    assert set(c["variable"]) == {"daily mean", "daily maximum"}
    ref = c[(c["variable"] == "daily mean") & (c["bin"] == "14")]
    assert len(ref) == 3 and (ref["coef_pct"] == 0).all()
    hot = c[(c["variable"] == "daily mean") & (c["bin"] == ">=24") & (c["spec"] == "B")]
    assert hot["coef_pct"].iloc[0] > 5.0  # 3 % per degree above 20, bins centred near 25 degC
