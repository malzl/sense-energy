"""Autoregressive features on the target.

Every lag and rolling window here is shifted by at least one period so that no
feature can see the value it is used to predict. When forecasting *h* steps
ahead, the minimum usable lag is *h* — see ``min_lag_for_horizon``.
"""

from __future__ import annotations

import pandas as pd


def add_target_lags(
    df: pd.DataFrame,
    target: str = "value",
    lags: tuple[int, ...] = (1, 2, 48, 96, 336),
    group_col: str = "site_id",
) -> pd.DataFrame:
    """Add lagged values of the target (48 = one day, 336 = one week at half-hourly)."""
    df = df.sort_values([group_col, "timestamp"]).copy()
    grouped = df.groupby(group_col)[target]
    for lag in lags:
        df[f"{target}_lag_{lag}"] = grouped.shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    target: str = "value",
    windows: tuple[int, ...] = (48, 336),
    group_col: str = "site_id",
) -> pd.DataFrame:
    """Rolling mean/std/min/max of the target, shifted to exclude the current period."""
    df = df.sort_values([group_col, "timestamp"]).copy()
    grouped = df.groupby(group_col)[target]
    for window in windows:
        shifted = grouped.transform(lambda s: s.shift(1))
        roll = shifted.groupby(df[group_col]).rolling(window, min_periods=max(1, window // 4))
        df[f"{target}_roll_mean_{window}"] = roll.mean().reset_index(level=0, drop=True)
        df[f"{target}_roll_std_{window}"] = roll.std().reset_index(level=0, drop=True)
    return df


def min_lag_for_horizon(horizon: int) -> int:
    """The shortest target lag that remains available when forecasting ``horizon`` ahead."""
    return horizon
