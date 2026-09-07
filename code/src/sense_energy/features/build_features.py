"""Assemble the model-ready feature table from the processed dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PROCESSED_DIR
from ..logging_utils import get_logger
from . import calendar, lags, weather

logger = get_logger(__name__)


def build_features(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply the configured feature blocks in order."""
    target = config.get("target", "consumption_kwh")

    df = calendar.add_calendar_features(df)
    df = calendar.add_cyclical_features(df)
    df = calendar.add_holiday_features(df)

    if config.get("use_weather", True) and "temperature_c" in df.columns:
        df = weather.add_degree_days(df)
        df = weather.add_weather_lags(df)

    df = lags.add_target_lags(
        df, target=target, lags=tuple(config.get("target_lags", (1, 48, 336)))
    )
    df = lags.add_rolling_features(
        df, target=target, windows=tuple(config.get("rolling_windows", (48, 336)))
    )

    logger.info("Built feature table: %d rows, %d columns", len(df), df.shape[1])
    return df


def run(config: dict[str, Any]) -> Path:
    """Read processed data, build features, write ``features.parquet``."""
    src = PROCESSED_DIR / config.get("input_filename", "demand.parquet")
    df = pd.read_parquet(src)
    out = build_features(df, config)
    out_path = PROCESSED_DIR / config.get("output_filename", "features.parquet")
    out.to_parquet(out_path, index=False)
    logger.info("Wrote %s", out_path)
    return out_path
