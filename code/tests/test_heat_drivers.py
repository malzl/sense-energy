"""What moves heat: site designs, block means, features, typology, meta-regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.analysis import heat_drivers as hd
from sense_energy.analysis import weather_causal as wc


def _panel(n_sites=12, n_days=400, seed=3, mechanism=False) -> pd.DataFrame:
    """Summer-heavy panel: heat and sunshine vary independently within site-months; even sites
    carry panels (demand falls with sunshine), odd sites carry chillers (demand rises with heat)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-04-01", periods=n_days)
    national_t = 16 + 7 * np.sin(2 * np.pi * (np.arange(n_days) - 60) / 365)
    national_sun = 3.5 + 2.5 * np.sin(2 * np.pi * (np.arange(n_days) - 80) / 365)
    rows = []
    for i in range(n_sites):
        t = national_t + rng.normal(0, 2.0, n_days)
        sun = np.clip(national_sun + rng.normal(0, 1.2, n_days), 0.2, None)
        hdd, cdd = wc.dose(t)
        swing = rng.uniform(0, 4, n_days)  # diurnal range varies from day to day
        hot_afternoon = np.clip(t + 4 + swing - hd.HOT_AFTERNOON_BASE, 0, None)
        warm_night = np.clip(t - 4 - swing - hd.WARM_NIGHT_BASE, 0, None)
        b_sun = -0.02 if i % 2 == 0 else 0.0
        b_heat = 0.0 if i % 2 == 0 else 0.03
        if mechanism:  # even sites: heating tail on warm nights; odd: cooling on hot afternoons
            b_heat = 0.0
            extra = -0.02 * warm_night if i % 2 == 0 else 0.02 * hot_afternoon
        else:
            extra = 0.0
        log_kw = 4 + 0.01 * hdd + b_heat * cdd + b_sun * sun + extra + rng.normal(0, 0.02, n_days)
        rows.append(
            pd.DataFrame(
                {
                    "site_code": f"S{i:02d}",
                    "date": dates,
                    "t_mean": t,
                    "t_max": t + 4 + swing,
                    "t_min": t - 4 - swing,
                    "solar_kwh_m2": sun,
                    "kw": np.exp(log_kw),
                    "region_ons": "North" if i % 3 else "South",
                    "group": "inpatient" if i % 3 else "outpatient",
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    df["log_kw"] = np.log(df["kw"])
    df["hdd14"], df["cdd20"] = wc.dose(df["t_mean"])
    df["fe_month"] = df["site_code"] + "|" + df["date"].dt.strftime("%Y-%m")
    df["fe_dow"] = df["site_code"] + "|" + df["date"].dt.dayofweek.astype(str)
    df["date_id"] = df["date"].dt.strftime("%Y-%m-%d")
    return df


def test_site_design_builds_one_column_per_site_and_term():
    df = _panel(n_sites=3, n_days=50)
    design, names, sites = hd.site_design(df, hd.TERMS, weekend=True)
    assert sites == ["S00", "S01", "S02"]
    assert design.shape[1] == 3 * (len(hd.TERMS) + 1)
    col = names[("cdd20", "S01")]
    mask = df["site_code"] == "S01"
    assert np.allclose(design.loc[mask, col], df.loc[mask, "cdd20"])
    assert (design.loc[~mask, col] == 0).all()
    we = names[("cdd20we", "S01")]
    weekend = df["date"].dt.dayofweek >= 5
    assert (design.loc[mask & ~weekend, we] == 0).all()


def test_site_terms_separate_sunshine_from_heat():
    df = _panel()
    out = hd.site_terms(df)
    wide = out.pivot(index="site_code", columns="term", values="slope")
    panels = wide.index.str[-1].astype(int) % 2 == 0
    assert wide.loc[panels, "solar_kwh_m2"].mean() == pytest.approx(-0.02, abs=0.006)
    assert wide.loc[~panels, "solar_kwh_m2"].mean() == pytest.approx(0.0, abs=0.006)
    assert wide.loc[~panels, "cdd20"].mean() == pytest.approx(0.03, abs=0.01)
    assert wide.loc[panels, "cdd20"].mean() == pytest.approx(0.0, abs=0.01)
    assert {"cdd20_weekend", "cdd20_weekend_diff"} <= set(out["term"])


def test_block_means_use_local_blocks_and_the_period_floor():
    idx = pd.date_range(
        "2025-06-30 23:00", periods=48 * 2, freq="30min", tz="UTC"
    )  # BST: 00:00 local
    w = pd.DataFrame({"A": 10.0, "B": 20.0}, index=idx)
    w.iloc[:6, 1] = np.nan  # B misses three hours of the first night block -> 6 of 12 periods
    out = hd.block_means(w)
    a = out[out["site_code"] == "A"]
    assert a["date"].nunique() == 2 and sorted(a["block"].unique()) == [0, 1, 2, 3]
    assert (a["block_kw"] == 10.0).all()
    b = out[(out["site_code"] == "B") & (out["date"] == pd.Timestamp("2025-07-01"))]
    assert 0 not in b["block"].tolist()  # below the ten-period floor
    assert len(b) == 3


def test_typology_and_flags():
    f = pd.DataFrame(
        {
            "heat": [0.03, 0.03, 0.0, -0.03, 0.0],
            "heat_se": 0.005,
            "sun_day": [0.0, -0.02, -0.02, 0.0, 0.0],
            "sun_day_se": 0.005,
            "heat_afternoon_excess": [0.02, 0.0, 0.0, 0.0, 0.0],
            "heat_afternoon_excess_se": 0.005,
            "heat_weekend_diff": [-0.03, 0.0, 0.0, 0.0, 0.0],
            "heat_weekend_diff_se": 0.005,
            "sun_night": [0.0, 0.0, 0.0, 0.0, -0.03],
            "sun_night_se": 0.005,
            "hot_afternoon": [0.02, 0.02, 0.0, 0.0, 0.0],
            "hot_afternoon_se": 0.005,
            "warm_night": [0.0, 0.0, 0.0, -0.02, 0.0],
            "warm_night_se": 0.005,
        },
        index=list("abcde"),
    )
    t = hd.typology(f)
    assert t.tolist() == [
        "rises with heat",
        "rises with heat, falls with sunshine",
        "falls with sunshine only",
        "falls with heat",
        "neither",
    ]
    fl = hd.flags(f)
    assert fl["afternoon_peaked"].tolist() == [True, False, False, False, False]
    assert fl["weekday_only"].tolist() == [True, False, False, False, False]
    assert fl["night_sunshine_negative"].tolist() == [False, False, False, False, True]
    assert fl["cooling_afternoon"].tolist() == [True, True, False, False, False]
    assert fl["heating_tail"].tolist() == [False, False, False, True, False]
    assert hd.sign_groups(f).tolist() == [
        "rises with heat",
        "rises with heat",
        "neither",
        "falls with heat",
        "neither",
    ]


def test_meta_regression_explains_a_planted_moderator():
    rng = np.random.default_rng(0)
    x = rng.normal(size=150)
    se = np.full(150, 0.003)
    true = 0.01 + 0.012 * x + rng.normal(0, 0.002, 150)
    y = true + rng.normal(0, se)
    m = hd.meta_regression(y, se, x[:, None])
    assert m["beta"][1] == pytest.approx(0.012, abs=0.002)
    assert m["r2_true"] > 0.9
    noise = hd.meta_regression(y, se, rng.normal(size=(150, 1)))
    assert noise["r2_true"] < 0.1


def test_tenure_group():
    t = pd.Series(
        [
            "Freehold",
            "Leased from NHS Property Services",
            "SLA/lease from public sector",
            "Whole site - Private Finance Initiative (PFI)",
            None,
        ]
    )
    out = hd.tenure_group(t)
    assert out.tolist()[:4] == ["freehold", "leased", "leased", "other"]
    assert pd.isna(out.iloc[4])


def test_night_day_terms_separate_heating_tails_from_cooling():
    df = _panel(mechanism=True)
    out = hd.night_day_terms(df)
    wide = out.pivot(index="site_code", columns="term", values="slope")
    tails = wide.index.str[-1].astype(int) % 2 == 0
    assert wide.loc[tails, "warm_night"].mean() == pytest.approx(-0.02, abs=0.006)
    assert wide.loc[tails, "hot_afternoon"].mean() == pytest.approx(0.0, abs=0.006)
    assert wide.loc[~tails, "hot_afternoon"].mean() == pytest.approx(0.02, abs=0.006)
    assert wide.loc[~tails, "warm_night"].mean() == pytest.approx(0.0, abs=0.006)


def test_sign_groups_and_group_decomposition():
    df = _panel(mechanism=True)
    f = pd.DataFrame(
        {"heat": np.where(np.arange(12) % 2 == 0, -0.02, 0.03), "heat_se": 0.004},
        index=[f"S{i:02d}" for i in range(12)],
    )
    groups = hd.sign_groups(f)
    assert set(groups) == {"rises with heat", "falls with heat"}
    g = hd.group_decomposition(df, groups).set_index(["model", "sign", "term"])
    assert g.loc[("night and day", "falls with heat", "warm_night"), "coef"] == pytest.approx(
        -0.02, abs=0.005
    )
    assert g.loc[("night and day", "rises with heat", "hot_afternoon"), "coef"] == pytest.approx(
        0.02, abs=0.005
    )
    assert g.loc[("daily mean", "rises with heat", "cdd20"), "coef"] > 0
    assert set(g.index.get_level_values("model")) == {"daily mean", "night and day"}


def test_dec_descriptors_classify_air_conditioning_and_handle_missing_table(tmp_path, monkeypatch):
    idx = pd.Index(["A", "B", "C", "D"], name="site_code")
    gia = pd.Series([1000.0, 2000.0, np.nan, 500.0], index=idx)
    monkeypatch.setattr(hd, "INTERIM_DIR", tmp_path)
    empty = hd.dec_descriptors(idx, gia)
    assert empty["dec_ac_class"].isna().all() and empty["dec_cooled_share"].isna().all()
    pd.DataFrame(
        {
            "site_code": ["A", "B", "C"],
            "dec_buildings": [2, 1, 0],
            "dec_floor_area_m2": [800.0, 3000.0, np.nan],
            "dec_ac_share": [0.0, 0.7, np.nan],
            "dec_cooled_share": [0.0, 0.5, np.nan],
            "dec_ac_kw_m2": [np.nan, 0.05, np.nan],
            "dec_electric_heating_share": [1.0, 0.0, np.nan],
            "dec_elec_kwh_m2": [120.0, 90.0, np.nan],
            "dec_operational_rating": [100.0, 80.0, np.nan],
            "dec_main_environment": ["heating, natural ventilation", "air conditioning", None],
        }
    ).to_parquet(tmp_path / "dec_sites.parquet", index=False)
    d = hd.dec_descriptors(idx, gia)
    assert d.loc["A", "dec_ac_class"] == "no air conditioning"
    assert d.loc["B", "dec_ac_class"] == "mostly air-conditioned"
    assert pd.isna(d.loc["C", "dec_ac_class"]) and pd.isna(d.loc["D", "dec_ac_class"])
    assert d.loc["A", "dec_coverage"] == 0.8 and d.loc["B", "dec_coverage"] == 1.5
    assert d.loc["A", "log_dec_elec_kwh_m2"] == pytest.approx(np.log(120.0))
    assert pd.isna(d.loc["C", "dec_main_environment"])
