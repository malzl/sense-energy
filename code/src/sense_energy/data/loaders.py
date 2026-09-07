"""Readers for the raw NHS consumption extract.

The source is a single wide parquet from Energy Systems Catapult
(``nhs_trust_consumption_data.parquet``) holding ~21M rows at **meter**
granularity, with site attributes denormalised onto every row.

Two things about this file matter more than anything else:

1. ``consumption`` mixes units. Roughly 9.6M rows are reactive power in kVArh
   (``reading_type`` 8/9/10/11), not energy. They must be filtered out before
   the column means anything.
2. The grain is ``mpxn`` (the meter), not ``site_code``. A third of sites have
   several meters — Homerton has 20 — so site demand is the *sum* over meters
   at a timestamp, never a de-duplication.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import EXTERNAL_DIR, RAW_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

#: ``reading_type`` -> (label, quantity, is_estimated)
READING_TYPES: dict[str, tuple[str, str, bool]] = {
    "1": ("30 minute aggregated kWh", "active_energy", False),
    "2": ("Estimate 30 minute aggregated kWh", "active_energy", True),
    "3": ("Part actual 30 minute aggregate kWh", "active_energy", True),
    "4": ("Cumulative kWh", "cumulative", False),
    "5": ("Monthly aggregated kWh", "monthly", False),
    "6": ("Export 30 minute aggregated kWh", "export", False),
    "8": ("Reactive Export 30 minute aggregated kVArh", "reactive", False),
    "9": ("Reactive Export estimate 30 minute aggregated kVArh", "reactive", True),
    "10": ("Reactive Import 30 minute aggregated kVArh", "reactive", False),
    "11": ("Reactive Import estimate 30 minute aggregated kVArh", "reactive", True),
    "14": ("Time of use cumulative kWh", "cumulative", False),
}

#: Half-hourly *active energy* in kWh — the only rows that belong in a demand series.
ACTIVE_HH_READING_TYPES: tuple[str, ...] = ("1", "2", "3")

#: Of those, the ones that are estimated or only partly actual.
ESTIMATED_READING_TYPES: tuple[str, ...] = ("2", "3")

#: Columns that describe the site rather than the reading. Constant per site_code.
SITE_COLUMNS: tuple[str, ...] = (
    "site_code",
    "site_name",
    "postcode",
    "organisation_name",
    "organisation_type",
    "comissioning_region",  # [sic] - misspelled in the source
    "integrated_care_board",
    "local_authority",
    "occupied_floor_area",
    "site_gross_internal_area",
    "site_heated_volume",
    "site_construction_year_band",
    "site_use_type",
    "fossil_fuel_led_chp_units_operated_on_site",
)

#: Columns needed to build the half-hourly series.
READING_COLUMNS: tuple[str, ...] = (
    "datetime",
    "site_code",
    "mpxn",
    "energy_type",
    "reading_type",
    "consumption",
)

SOURCE_GLOB = "*nhs_trust_consumption_data.parquet"


def find_source_file(path: str | Path | None = None) -> Path:
    """Locate the NHS extract, preferring ``data/raw`` then ``data/external``."""
    if path is not None:
        resolved = Path(path)
        if not resolved.exists():
            raise FileNotFoundError(f"No such file: {resolved}")
        return resolved

    for directory in (RAW_DIR, EXTERNAL_DIR):
        matches = sorted(directory.glob(SOURCE_GLOB))
        if matches:
            if len(matches) > 1:
                logger.warning("Multiple extracts in %s; using %s", directory, matches[-1].name)
            return matches[-1]

    raise FileNotFoundError(
        f"No file matching {SOURCE_GLOB!r} in {RAW_DIR} or {EXTERNAL_DIR}. See code/data/README.md."
    )


def read_readings(
    path: str | Path | None = None,
    reading_types: tuple[str, ...] = ACTIVE_HH_READING_TYPES,
    energy_types: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Read meter-level readings, filtered to real half-hourly energy.

    Filtering happens in Arrow before materialising to pandas, so the reactive
    rows never occupy memory.

    Returns columns: ``datetime`` (UTC), ``site_code``, ``mpxn``,
    ``energy_type``, ``reading_type``, ``consumption`` (kWh).
    """
    source = find_source_file(path)
    logger.info("Reading %s", source.name)

    table = pq.read_table(source, columns=list(READING_COLUMNS))
    n_all = table.num_rows

    table = table.filter(pc.is_in(table.column("reading_type"), value_set=pa.array(reading_types)))
    if energy_types is not None:
        table = table.filter(
            pc.is_in(table.column("energy_type"), value_set=pa.array(energy_types))
        )

    logger.info(
        "Kept %s of %s rows (reading_type in %s%s)",
        f"{table.num_rows:,}",
        f"{n_all:,}",
        reading_types,
        f", energy_type in {energy_types}" if energy_types else "",
    )
    return table.to_pandas()


def read_site_dimension(path: str | Path | None = None) -> pd.DataFrame:
    """Read the one-row-per-site attribute table.

    These columns are denormalised onto all 21M reading rows in the source but
    are constant per ``site_code``, so they are extracted once here.
    """
    source = find_source_file(path)
    table = pq.read_table(source, columns=list(SITE_COLUMNS))
    sites = table.to_pandas().drop_duplicates(subset="site_code").reset_index(drop=True)

    inconsistent = sites["site_code"].duplicated().sum()
    if inconsistent:
        raise ValueError(
            f"{inconsistent} site_code(s) carry conflicting attributes; "
            "site metadata is no longer safe to treat as a dimension table."
        )

    sites = sites.rename(columns={"comissioning_region": "commissioning_region"})
    logger.info("Site dimension: %d sites", len(sites))
    return sites
