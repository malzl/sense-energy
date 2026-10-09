"""ERIC and KH03 loaders, the meter coverage flags, the heat-pump model and the scoring subsets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import estates
from sense_energy.data import eric, nhs_beds


def _eric_csv(tmp_path):
    cols = {
        "Site Code": ["S1", "S2"],
        "Site Name": ["One", "Two"],
        "Trust Code": ["T1", "T1"],
        "Trust Name": ["Trust", "Trust"],
        "Trust Type": ["ACUTE - LARGE", "ACUTE - LARGE"],
        "Commissioning Region": ["R", "R"],
        "Site Type": ["General acute hospital", "Non inpatient"],
        "Tenure": ["Freehold", "Leasehold"],
        "Post Code": ["AB1 2CD", "EF3 4GH"],
        "Electrical energy consumption (KWh)": ["", "1,000"],
    }
    for c in eric.ELEC_COMPONENTS:
        cols[c] = ["1,000", ""]
    for c in eric.NUMERIC.values():
        cols[c] = ["10", "0"]
    for c in eric.AGE_BANDS:
        cols[c] = ["0", "0"]
    cols["Age profile - 2015 to 2024 (%)"] = ["50", ""]
    cols["Age profile - pre 1948 (%)"] = ["50", ""]
    p = tmp_path / "site.csv"
    pd.DataFrame(cols).to_csv(p, index=False)
    return p


def test_eric_site_sums_electricity_components_and_build_year(tmp_path):
    site = eric.load_site(_eric_csv(tmp_path))
    s1, s2 = site.set_index("site_code").loc["S1"], site.set_index("site_code").loc["S2"]
    assert s1["elec_kwh"] == 5000  # five components of 1,000
    assert np.isnan(s2["elec_kwh"])  # components empty although the headline field is filled
    assert s1["mean_build_year"] == pytest.approx((2019.5 + 1935.0) / 2)
    assert np.isnan(s2["mean_build_year"])
    assert s1["elec_kwh_per_m2"] == pytest.approx(500)


def test_kh03_loader_merges_overnight_and_day_only(tmp_path, monkeypatch):
    on = pd.DataFrame(
        {
            "Organisation_Code": ["RAA", "RAA"],
            "Sector": ["General & Acute", "Mental illness"],
            "Number_Of_Beds": ["100.5", "20"],
            "Effective_Snapshot_Date": ["31/03/2024", "31/03/2024"],
        }
    )
    day = on.assign(Number_Of_Beds=["7", "0"], Sector=["General & Acute", "Mental Illness"])
    on.to_csv(tmp_path / f"{nhs_beds.STEM}kh03occupied_overnight_only.csv", index=False)
    day.to_csv(tmp_path / f"{nhs_beds.STEM}kh03_occupied_day_only.csv", index=False)
    monkeypatch.setattr(nhs_beds, "BEDS_DIR", tmp_path)
    s = nhs_beds.load("sector").set_index("sector")
    assert set(s.index) == {"General & Acute", "Mental Illness"}  # spelling unified
    assert s.loc["General & Acute", "overnight"] == pytest.approx(100.5)
    assert s.loc["General & Acute", "day_only"] == pytest.approx(7)
    assert s["quarter_end"].iloc[0] == pd.Timestamp("2024-03-31")


def test_coverage_flags(monkeypatch):
    idx = pd.date_range(
        "2023-04-01", "2024-04-01", freq="30min", tz="Europe/London", inclusive="left"
    ).tz_convert("UTC")
    n = len(idx)
    wide = pd.DataFrame(
        {c: np.ones(n) for c in ("ok", "part", "agg", "chp", "gappy", "none")}, index=idx
    )
    wide.iloc[: n // 2, wide.columns.get_loc("gappy")] = np.nan
    monkeypatch.setattr(estates, "_wide", lambda energy, flavour="raw": wide)
    total = float(n)  # 1 kWh per half hour
    site = pd.DataFrame(
        {
            "site_code": ["ok", "part", "agg", "chp", "gappy"],
            "elec_kwh": [total, total / 0.4, total / 2.0, total / 2.0, total],
            "max_demand_kw": [2.0] * 5,
            "capacity_kva": [10.0] * 5,
            "site_type": ["x"] * 5,
            "chp_units": [0, 0, 0, 1, 0],
            "solar_generated_kwh": [0] * 5,
        }
    )
    f = estates.coverage_flags(site).set_index("site_code")["coverage_flag"]
    assert f.to_dict() == {
        "ok": "ok",
        "part": "partial_supply",
        "agg": "aggregate_supply",
        "chp": "chp_mismatch",
        "gappy": "meter_gaps",
        "none": "no_eric",
    }


def test_cop_rises_with_outdoor_temperature_and_matches_staffell():
    t = np.array([-5.0, 0.0, 7.0, 15.0])
    cop = estates.cop_ashp(t, 55.0)
    assert np.all(np.diff(cop) > 0)
    dt = 55.0
    assert cop[1] == pytest.approx(6.81 - 0.121 * dt + 0.000630 * dt**2)
    assert cop.min() >= 1.0


def test_headroom_status_rules():
    head = pd.DataFrame(
        {
            "site_code": ["a", "b", "c", "d"],
            "util_now": [0.5, 1.2, 0.5, 0.5],
            "chp_units": [0, 0, 1, 0],
        }
    )
    flags = pd.DataFrame(
        {"site_code": ["a", "b", "c", "d"], "coverage_flag": ["ok", "ok", "ok", "partial_supply"]}
    )
    s = estates.headroom_status(head, flags).set_index("site_code")["headroom_status"]
    assert s.to_dict() == {
        "a": "included",
        "b": "capacity_implausible",
        "c": "gas_feeds_chp",
        "d": "meter_not_whole_site",
    }


def test_scoring_sensitivity_subsets():
    per = pd.DataFrame(
        {
            "model": ["m"] * 3,
            "site_code": ["a", "b", "c"],
            "nmae": [0.1, 0.2, 0.3],
            "crps_q": [1.0, 2.0, 3.0],
            "coverage_80": [0.8, 0.8, 0.8],
            "crps_skill_vs_naive": [0.5, 0.4, 0.3],
        }
    )
    flags = pd.DataFrame(
        {"site_code": ["a", "b", "c"], "coverage_flag": ["ok", "partial_supply", "no_eric"]}
    )
    s = estates.scoring_sensitivity(per, flags).set_index("subset")
    assert s.loc["all sites", "sites"] == 3 and s.loc["all sites", "nmae"] == pytest.approx(0.2)
    assert s.loc["without flagged supplies", "sites"] == 2
    assert s.loc["verified only", "sites"] == 1 and s.loc["verified only", "nmae"] == pytest.approx(
        0.1
    )
