"""raw -> interim -> processed pipeline.

Invoked by ``make data`` / ``sense-energy build-dataset``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import PROCESSED_DIR, ensure_dirs
from ..logging_utils import get_logger
from . import cleaning, loaders, validation

logger = get_logger(__name__)


def build_dataset(config: dict[str, Any]) -> Path:
    """Run the full raw-to-processed pipeline and write a parquet file.

    Returns the path of the written file.
    """
    ensure_dirs()
    freq = config.get("frequency", "30min")

    logger.info("Loading raw sources")
    meter = loaders.read_hh_meter_data(config.get("meter_path"))
    validation.validate_schema(meter)

    logger.info("Cleaning")
    meter = cleaning.drop_duplicate_readings(meter)
    meter = cleaning.reindex_to_grid(meter, freq=freq)
    meter = cleaning.flag_outliers(meter, z_threshold=config.get("outlier_z", 5.0))
    meter = cleaning.interpolate_short_gaps(meter, max_gap=config.get("max_gap", 4))

    logger.info("Joining site metadata and weather")
    sites = loaders.read_site_metadata(config.get("site_metadata_path"))
    weather = loaders.read_weather(config.get("weather_path"))
    df = meter.merge(sites, on="site_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")

    out_path = PROCESSED_DIR / config.get("output_filename", "demand.parquet")
    df.to_parquet(out_path, index=False)
    logger.info("Wrote %s (%d rows, %d columns)", out_path, len(df), df.shape[1])
    return out_path


def load_processed(filename: str = "demand.parquet") -> pd.DataFrame:
    """Read the processed dataset back."""
    return pd.read_parquet(PROCESSED_DIR / filename)
