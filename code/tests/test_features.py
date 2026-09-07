import numpy as np

from sense_energy.features import calendar, lags, weather


def test_calendar_features_use_local_time(synthetic_demand):
    df = calendar.add_calendar_features(synthetic_demand)
    assert df["hour"].between(0, 23).all()
    assert df["day_of_week"].between(0, 6).all()
    assert set(df["is_weekend"].unique()) <= {0, 1}


def test_cyclical_features_are_on_the_unit_circle(synthetic_demand):
    df = calendar.add_cyclical_features(calendar.add_calendar_features(synthetic_demand))
    assert np.allclose(df["tod_sin"] ** 2 + df["tod_cos"] ** 2, 1.0)


def test_degree_days_are_non_negative_and_exclusive(synthetic_demand):
    df = weather.add_degree_days(synthetic_demand)
    assert (df["heating_degrees"] >= 0).all()
    assert (df["cooling_degrees"] >= 0).all()
    assert ((df["heating_degrees"] > 0) & (df["cooling_degrees"] > 0)).sum() == 0


def test_target_lags_do_not_leak_the_present(synthetic_demand):
    df = lags.add_target_lags(synthetic_demand, lags=(1, 48))
    site = df[df["site_id"] == "SITE_A"].sort_values("timestamp").reset_index(drop=True)
    # lag_1 at row i must equal the value at row i-1, never row i.
    assert np.allclose(site["value_lag_1"].iloc[1:], site["value"].iloc[:-1], equal_nan=True)
    assert site["value_lag_48"].iloc[:48].isna().all()


def test_rolling_features_exclude_current_period(synthetic_demand):
    df = lags.add_rolling_features(synthetic_demand, windows=(48,))
    site = df[df["site_id"] == "SITE_A"].sort_values("timestamp")
    assert site["value_roll_mean_48"].iloc[0] != site["value"].iloc[0]
    assert site["value_roll_mean_48"].isna().iloc[0]
