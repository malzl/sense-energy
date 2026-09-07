"""Calendar features: time of day, day of week, season, holidays, NHS working patterns.

Hospital demand is strongly driven by the operational calendar — theatre lists,
outpatient clinics, bank holidays — so these are usually the strongest features
after temperature and recent demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import LOCAL_TZ


def add_calendar_features(df: pd.DataFrame, timestamp_col: str = "datetime") -> pd.DataFrame:
    """Add local-time calendar features derived from a tz-aware UTC timestamp."""
    df = df.copy()
    local = df[timestamp_col].dt.tz_convert(LOCAL_TZ)

    df["hour"] = local.dt.hour
    df["minute_of_day"] = local.dt.hour * 60 + local.dt.minute
    df["day_of_week"] = local.dt.dayofweek
    df["day_of_year"] = local.dt.dayofyear
    df["month"] = local.dt.month
    df["year"] = local.dt.year
    df["is_weekend"] = (local.dt.dayofweek >= 5).astype(int)
    return df


def add_cyclical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Sine/cosine encodings so midnight sits next to 23:30, and Dec next to Jan."""
    df = df.copy()
    df["tod_sin"] = np.sin(2 * np.pi * df["minute_of_day"] / 1440)
    df["tod_cos"] = np.cos(2 * np.pi * df["minute_of_day"] / 1440)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    return df


def add_holiday_features(
    df: pd.DataFrame, holidays: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    """Flag England & Wales bank holidays plus the days either side.

    TODO: source the holiday calendar (``holidays`` package or a static CSV in
    ``code/data/external/``) and pass it in.
    """
    df = df.copy()
    if holidays is None:
        df["is_holiday"] = 0
        df["is_day_before_holiday"] = 0
        df["is_day_after_holiday"] = 0
        return df

    local_date = df["datetime"].dt.tz_convert(LOCAL_TZ).dt.normalize().dt.tz_localize(None)
    hset = set(pd.DatetimeIndex(holidays).normalize())
    df["is_holiday"] = local_date.isin(hset).astype(int)
    df["is_day_before_holiday"] = (local_date + pd.Timedelta(days=1)).isin(hset).astype(int)
    df["is_day_after_holiday"] = (local_date - pd.Timedelta(days=1)).isin(hset).astype(int)
    return df
