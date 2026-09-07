"""Weather-derived features: degree days, lags, rolling comfort measures."""

from __future__ import annotations

import pandas as pd

#: Standard UK base temperature for degree-day calculations.
BASE_TEMP_C = 15.5


def add_degree_days(
    df: pd.DataFrame, temp_col: str = "temperature_c", base: float = BASE_TEMP_C
) -> pd.DataFrame:
    """Heating and cooling degree hours relative to a base temperature."""
    df = df.copy()
    df["heating_degrees"] = (base - df[temp_col]).clip(lower=0)
    df["cooling_degrees"] = (df[temp_col] - base).clip(lower=0)
    return df


def add_weather_lags(
    df: pd.DataFrame,
    temp_col: str = "temperature_c",
    lags: tuple[int, ...] = (1, 2, 4, 48),
    windows: tuple[int, ...] = (48, 336),
) -> pd.DataFrame:
    """Lagged and rolling-mean temperature — buildings respond with thermal inertia."""
    df = df.sort_values(["mpxn", "datetime"]).copy()
    grouped = df.groupby("mpxn")[temp_col]

    for lag in lags:
        df[f"{temp_col}_lag_{lag}"] = grouped.shift(lag)
    for window in windows:
        df[f"{temp_col}_roll_mean_{window}"] = grouped.transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=max(1, w // 4)).mean()
        )
    return df
