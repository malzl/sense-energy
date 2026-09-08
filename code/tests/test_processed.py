"""Profile-KNN imputation and aggregation rules for the processed datasets."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sense_energy.data import processed

CFG = {
    "k": 3,
    "window_weeks": 8,
    "same_day_type": True,
    "min_observed_fraction_of_day": 0.25,
    "max_gap_periods": 48,
    "level_scaling": True,
}


def _series(days: int = 70, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=days * 48, freq="30min", tz="UTC")
    tod = np.tile(np.arange(48), days)
    profile = 100 + 40 * np.sin(2 * np.pi * (tod - 12) / 48)
    weekend = np.repeat(idx[::48].tz_convert("Europe/London").dayofweek.to_numpy() >= 5, 48)
    return pd.Series(profile * np.where(weekend, 0.7, 1.0) + rng.normal(0, 2, days * 48), index=idx)


def test_short_gap_is_filled_close_to_truth():
    s = _series()
    truth = s.copy()
    s.iloc[20 * 48 + 10 : 20 * 48 + 16] = np.nan  # a 3 h gap on day 20
    filled, flags, stats = processed.impute_series(s, CFG)
    assert flags.sum() == 6 and stats.n_imputed == 6 and stats.n_days_imputed == 1
    err = np.abs(
        filled.iloc[20 * 48 + 10 : 20 * 48 + 16] - truth.iloc[20 * 48 + 10 : 20 * 48 + 16]
    ).mean()
    assert err < 10  # well inside the daily swing of 80


def test_whole_missing_day_is_never_filled():
    s = _series()
    s.iloc[30 * 48 : 31 * 48] = np.nan
    filled, flags, stats = processed.impute_series(s, CFG)
    assert flags.sum() == 0 and stats.n_days_skipped_too_sparse == 1
    assert filled.iloc[30 * 48 : 31 * 48].isna().all()


def test_gap_longer_than_a_day_is_not_bridged():
    s = _series()
    s.iloc[30 * 48 + 20 : 32 * 48 + 20] = np.nan  # 48 h gap spanning three days
    filled, flags, stats = processed.impute_series(s, CFG)
    assert flags.sum() == 0
    assert stats.n_days_skipped_long_gap + stats.n_days_skipped_too_sparse >= 2


def test_observed_values_are_never_altered():
    s = _series()
    s.iloc[10 * 48 + 5 : 10 * 48 + 9] = np.nan
    filled, flags, _ = processed.impute_series(s, CFG)
    observed = s.notna()
    assert np.allclose(filled[observed], s[observed])
    assert not flags[observed].any()


def test_neighbours_respect_day_type():
    """A weekend day's gap must be filled from weekend neighbours, not weekday level."""
    s = _series()
    starts = s.index[::48]
    weekend_days = np.flatnonzero(starts.tz_convert("Europe/London").dayofweek >= 5)
    d = weekend_days[3]
    s.iloc[d * 48 + 20 : d * 48 + 26] = np.nan
    filled, _, _ = processed.impute_series(s, CFG)
    assert (
        filled.iloc[d * 48 + 20 : d * 48 + 26].mean() < 0.85 * 130
    )  # weekend level (~0.7 x peak), not weekday


def test_aggregate_requires_every_member():
    idx = pd.date_range("2024-01-01", periods=4, freq="30min", tz="UTC")
    wide = pd.DataFrame(
        {
            "m1": [1.0, 2.0, np.nan, 4.0],
            "m2": [10.0, 20.0, 30.0, np.nan],
            "m3": [5.0, 5.0, 5.0, 5.0],
        },
        index=idx,
    )
    mapping = pd.Series({"m1": "S1", "m2": "S1", "m3": "S2"})
    totals, n_members = processed.aggregate_wide(wide, mapping)
    assert totals["S1"].tolist()[:2] == [11.0, 22.0]
    assert totals["S1"].iloc[2:].isna().all()  # any missing member -> no total
    assert totals["S2"].tolist() == [5.0] * 4
    assert n_members["S1"] == 2 and n_members["S2"] == 1


def test_member_with_no_data_at_all_does_not_block_totals():
    idx = pd.date_range("2024-01-01", periods=2, freq="30min", tz="UTC")
    wide = pd.DataFrame({"m1": [1.0, 2.0], "ghost": [np.nan, np.nan]}, index=idx)
    totals, n_members = processed.aggregate_wide(wide, pd.Series({"m1": "S1", "ghost": "S1"}))
    assert totals["S1"].tolist() == [1.0, 2.0] and n_members["S1"] == 1


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Homerton Healthcare NHS Foundation Trust ", "HOMERTON_HEALTHCARE_NHS_FOUNDATION_TRUST"),
        ("A & B", "A_B"),
    ],
)
def test_trust_ids_are_stable_identifiers(name, expected):
    assert processed._trust_id(name) == expected
