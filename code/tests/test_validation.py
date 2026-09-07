import pandas as pd
import pytest

from sense_energy.data import validation


def test_validate_schema_accepts_good_frame(synthetic_demand):
    validation.validate_schema(synthetic_demand)


def test_validate_schema_rejects_missing_columns(synthetic_demand):
    with pytest.raises(ValueError, match="Missing required columns"):
        validation.validate_schema(synthetic_demand.drop(columns=["value"]))


def test_validate_schema_rejects_naive_timestamps(synthetic_demand):
    df = synthetic_demand.copy()
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    with pytest.raises(TypeError, match="tz-aware"):
        validation.validate_schema(df)


def test_check_regular_index_finds_gaps(synthetic_demand):
    with_gap = synthetic_demand.drop(synthetic_demand.index[10:15])
    report = validation.check_regular_index(with_gap)
    assert report["n_missing"].sum() == 5


def test_summarise_missing_counts_nulls(synthetic_demand):
    df = synthetic_demand.copy()
    df.loc[df.index[:7], "value"] = pd.NA
    report = validation.summarise_missing(df)
    assert report.loc["value", "n_missing"] == 7
