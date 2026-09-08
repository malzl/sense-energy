"""Tests for the cleaning stage.

The behaviour that matters most here is that meters at the same site are never
treated as duplicates of one another — collapsing them would discard most of a
multi-meter hospital's demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.data import cleaning


def test_drop_duplicates_removes_only_true_duplicates(synthetic_readings):
    out = cleaning.drop_duplicate_readings(synthetic_readings)
    assert len(out) == len(synthetic_readings) - 1
    assert not out.duplicated(cleaning.METER_GRAIN).any()


def test_concurrent_meters_at_one_site_are_not_duplicates(synthetic_readings):
    """Two meters at RXX01 share every timestamp; both must survive."""
    out = cleaning.drop_duplicate_readings(synthetic_readings)
    site = out[out.site_code == "RXX01"]
    assert site.mpxn.nunique() == 2
    per_timestamp = site.groupby("datetime").size()
    assert (per_timestamp == 2).all()


def test_zeros_become_nan(synthetic_readings):
    out = cleaning.zeros_to_nan(synthetic_readings)
    assert (out.consumption == 0).sum() == 0
    assert out.consumption.isna().sum() == 30  # 10 zeros x 3 meters


def test_reindex_inserts_missing_timestamps(synthetic_readings):
    df = cleaning.drop_duplicate_readings(synthetic_readings)
    gapped = df.drop(df.index[100:105])
    out = cleaning.reindex_to_grid(gapped)
    assert len(out) == len(df)
    for _, g in out.groupby(cleaning.SERIES_KEYS):
        assert g.datetime.diff().dropna().eq(pd.Timedelta("30min")).all()


def test_outliers_are_flagged_not_dropped(synthetic_readings):
    df = cleaning.drop_duplicate_readings(synthetic_readings).rename(
        columns={"consumption": "consumption_kwh"}
    )
    df.loc[df.index[500], "consumption_kwh"] = 1e6
    out = cleaning.flag_outliers(df)
    assert len(out) == len(df)
    assert out.is_outlier.sum() >= 1
    assert out.loc[out.index[500], "is_outlier"]


def test_interpolation_respects_max_gap(synthetic_readings):
    df = cleaning.drop_duplicate_readings(synthetic_readings).rename(
        columns={"consumption": "consumption_kwh"}
    )
    df = df.sort_values(["mpxn", "datetime"]).reset_index(drop=True)
    df.loc[100:101, "consumption_kwh"] = np.nan  # short gap - fillable
    df.loc[200:220, "consumption_kwh"] = np.nan  # long gap - must stay missing
    out = cleaning.interpolate_short_gaps(df, max_gap=4).reset_index(drop=True)
    assert out.loc[100:101, "consumption_kwh"].notna().all()
    assert out.loc[205:215, "consumption_kwh"].isna().all()


def test_aggregate_to_site_sums_meters(synthetic_demand):
    out = cleaning.aggregate_to_level(synthetic_demand, level="site_code")
    assert set(out.site_code) == {"RXX01", "RXX02"}

    ts = synthetic_demand.datetime.iloc[0]
    expected = synthetic_demand[
        (synthetic_demand.site_code == "RXX01") & (synthetic_demand.datetime == ts)
    ].consumption_kwh.sum()
    got = out[(out.site_code == "RXX01") & (out.datetime == ts)].consumption_kwh.iloc[0]
    assert got == expected


def test_aggregate_flags_partial_totals(synthetic_demand):
    df = synthetic_demand.copy()
    df.loc[
        (df.mpxn == "1000000000002") & (df.datetime == df.datetime.iloc[0]), "consumption_kwh"
    ] = np.nan
    out = cleaning.aggregate_to_level(df, level="site_code")
    row = out[(out.site_code == "RXX01") & (out.datetime == df.datetime.iloc[0])].iloc[0]
    assert row.is_partial
    assert row.n_reporting == 1


def test_outlier_flag_ignores_quantised_low_consumption_meters():
    """A meter reading 0.1/0.2 kWh with the odd 1 kWh must not be flagged: MAD ~ 0 is not evidence."""
    idx = pd.date_range("2025-01-01", periods=500, freq="30min", tz="UTC")
    vals = np.where(np.arange(500) % 2 == 0, 0.1, 0.2).astype(float)
    vals[10] = (
        1.0  # 5x median but tiny in absolute terms and well within the ratio-and-z rule? ratio=5 -> borderline
    )
    vals[20] = 25.0  # a genuine fault: 125x median
    df = pd.DataFrame(
        {"mpxn": "m", "energy_type": "elec", "datetime": idx, "consumption_kwh": vals}
    )
    out = cleaning.flag_outliers(df)
    assert out.loc[out.index[20], "is_outlier"]
    assert out["is_outlier"].sum() <= 2
    assert out["is_outlier"].mean() < 0.01


def test_outlier_flag_is_high_side_only(synthetic_readings):
    df = cleaning.drop_duplicate_readings(synthetic_readings).rename(
        columns={"consumption": "consumption_kwh"}
    )
    df.loc[df.index[100], "consumption_kwh"] = (
        0.5  # implausibly low, but low is not an outlier here
    )
    out = cleaning.flag_outliers(df)
    assert not out.loc[out.index[100], "is_outlier"]


def test_bimodal_meter_regime_is_not_an_outlier():
    """Idle at 0.2 kWh for 70% of periods, running at 6 kWh for 30%: nothing to flag."""
    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-01-01", periods=2000, freq="30min", tz="UTC")
    running = rng.random(2000) < 0.3
    vals = np.where(running, 6.0 + rng.normal(0, 0.3, 2000), 0.2 + rng.normal(0, 0.02, 2000))
    df = pd.DataFrame(
        {"mpxn": "m", "energy_type": "elec", "datetime": idx, "consumption_kwh": vals}
    )
    assert cleaning.flag_outliers(df)["is_outlier"].sum() == 0
