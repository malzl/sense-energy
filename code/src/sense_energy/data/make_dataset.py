"""raw -> interim pipeline.

Produces two files in ``code/data/interim/``:

``consumption_halfhourly.parquet``
    The cleaned demand series at **meter** grain — one row per
    (``mpxn``, ``energy_type``, ``datetime``) on a complete half-hourly grid,
    with quality flags. Meter grain is preserved deliberately so the series can
    be aggregated up a hierarchy later; see
    :func:`sense_energy.data.cleaning.aggregate_to_level`.

``sites.parquet``
    One row per site: name, organisation, region, floor area, use type. These
    attributes are denormalised onto all 21M source rows and are extracted once.

Invoked by ``make data`` / ``sense-energy build-interim``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..config import INTERIM_DIR, ensure_dirs
from ..logging_utils import get_logger
from . import cleaning, loaders, validation

logger = get_logger(__name__)

CONSUMPTION_FILENAME = "consumption_halfhourly.parquet"
SITES_FILENAME = "sites.parquet"

#: Column order of the interim consumption table.
INTERIM_COLUMNS = [
    "mpxn",
    "site_code",
    "energy_type",
    "datetime",
    "consumption_kwh",
    "is_estimated",
    "is_interpolated",
    "is_outlier",
]


def build_interim(config: dict[str, Any]) -> dict[str, Path]:
    """Clean the raw extract and write the interim tables.

    Returns a mapping of logical name to written path.
    """
    ensure_dirs()
    freq = config.get("frequency", "30min")

    # --- site dimension ----------------------------------------------------
    sites = loaders.read_site_dimension(config.get("source_path"))
    sites_path = INTERIM_DIR / config.get("sites_filename", SITES_FILENAME)
    sites.to_parquet(sites_path, index=False)
    logger.info("Wrote %s (%d sites)", sites_path.name, len(sites))

    # --- readings ----------------------------------------------------------
    df = loaders.read_readings(
        config.get("source_path"),
        reading_types=tuple(config.get("reading_types", loaders.ACTIVE_HH_READING_TYPES)),
        energy_types=tuple(config["energy_types"]) if config.get("energy_types") else None,
    )
    validation.validate_schema(df)

    df = cleaning.drop_duplicate_readings(df)
    df["is_estimated"] = df["reading_type"].isin(loaders.ESTIMATED_READING_TYPES)

    if config.get("zeros_as_missing", True):
        df = cleaning.zeros_to_nan(df, column="consumption")

    df = df.rename(columns={"consumption": "consumption_kwh"})
    df = df.drop(columns=["reading_type"])

    df = cleaning.reindex_to_grid(df, freq=freq)
    df = cleaning.flag_outliers(df, z_threshold=config.get("outlier_z", 10.0))
    df = cleaning.interpolate_short_gaps(df, max_gap=config.get("max_gap", 4))

    df["is_estimated"] = df["is_estimated"].fillna(False).astype(bool)
    df = df.sort_values(["mpxn", "energy_type", "datetime"]).reset_index(drop=True)
    df = df[INTERIM_COLUMNS]

    out_path = INTERIM_DIR / config.get("consumption_filename", CONSUMPTION_FILENAME)
    df.to_parquet(out_path, index=False, compression="zstd")
    logger.info("Wrote %s (%s rows, %d columns)", out_path.name, f"{len(df):,}", df.shape[1])

    return {"consumption": out_path, "sites": sites_path}


def load_interim(filename: str = CONSUMPTION_FILENAME) -> pd.DataFrame:
    """Read the interim consumption table back."""
    return pd.read_parquet(INTERIM_DIR / filename)


def load_sites(filename: str = SITES_FILENAME) -> pd.DataFrame:
    """Read the site dimension table back."""
    return pd.read_parquet(INTERIM_DIR / filename)
