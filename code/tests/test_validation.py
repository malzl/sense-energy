import pandas as pd
import pytest

from sense_energy.data import validation


def test_validate_schema_accepts_good_frame(synthetic_readings):
    validation.validate_schema(synthetic_readings)


def test_validate_schema_rejects_missing_columns(synthetic_readings):
    with pytest.raises(ValueError, match="Missing required columns"):
        validation.validate_schema(synthetic_readings.drop(columns=["consumption"]))


def test_validate_schema_rejects_naive_timestamps(synthetic_readings):
    df = synthetic_readings.copy()
    df["datetime"] = df["datetime"].dt.tz_localize(None)
    with pytest.raises(TypeError, match="tz-aware"):
        validation.validate_schema(df)


def test_check_regular_index_finds_gaps(synthetic_readings):
    with_gap = synthetic_readings.drop(synthetic_readings.index[10:15])
    report = validation.check_regular_index(with_gap)
    assert report["n_missing"].sum() == 5


def test_summarise_missing_counts_nulls(synthetic_readings):
    df = synthetic_readings.copy()
    df.loc[df.index[:7], "consumption"] = pd.NA
    report = validation.summarise_missing(df)
    assert report.loc["consumption", "n_missing"] == 7
