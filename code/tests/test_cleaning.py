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
