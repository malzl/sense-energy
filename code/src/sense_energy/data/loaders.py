"""Readers for the raw source files.

One function per source. Each returns a tidy long-format DataFrame with, at
minimum: ``site_id``, ``timestamp`` (tz-aware UTC), ``value``, ``unit``.

TODO: fill these in once the real files land in ``code/data/raw/``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import RAW_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)


def read_hh_meter_data(path: str | Path | None = None) -> pd.DataFrame:
    """Read half-hourly meter readings.

    Expected tidy output columns:
        site_id : str    - NHS site / MPAN identifier
        timestamp: datetime64[ns, UTC] - period *start*
        value   : float  - consumption in the period
        unit    : str    - e.g. "kWh"
    """
    path = Path(path) if path else RAW_DIR / "hh_meter_data.csv"
    logger.info("Reading half-hourly meter data from %s", path)
    raise NotImplementedError("Implement once the raw meter export format is known.")


def read_site_metadata(path: str | Path | None = None) -> pd.DataFrame:
    """Read the site register: floor area, trust, site type, lat/lon, bed count."""
    path = Path(path) if path else RAW_DIR / "site_metadata.csv"
    logger.info("Reading site metadata from %s", path)
    raise NotImplementedError("Implement once the site register is available.")


def read_weather(path: str | Path | None = None) -> pd.DataFrame:
    """Read weather observations/forecasts joined on ``site_id`` and ``timestamp``."""
    path = Path(path) if path else RAW_DIR / "weather.csv"
    logger.info("Reading weather data from %s", path)
    raise NotImplementedError("Implement once the weather source is chosen.")
