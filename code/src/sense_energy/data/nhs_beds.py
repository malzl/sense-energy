"""KH03 bed availability and occupancy (ESC extract): occupied beds per trust and quarter.

Four tables in ``external/nhs_beds/``: occupied beds overnight and day-only, each
by sector (general & acute, maternity, mental illness, learning disabilities)
and by specialty (78 codes), quarterly snapshots 2010 (2001 overnight by sector)
to June 2024. Values are the quarter's average number of occupied beds.

Outputs (``interim/``):
* ``kh03_occupied_by_sector.parquet``  - ods_code, quarter_end, sector, overnight, day_only
* ``kh03_occupied_by_specialty.parquet`` - ods_code, quarter_end, specialty, overnight, day_only
"""

from __future__ import annotations

import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

BEDS_DIR = EXTERNAL_DIR / "nhs_beds"
STEM = "energy_systems_catapult.nhs_bed_availability."
FILES = {
    ("sector", "overnight"): "kh03occupied_overnight_only.csv",
    ("sector", "day_only"): "kh03_occupied_day_only.csv",
    ("specialty", "overnight"): "kh03occupied_by_spec_overnight_only.parquet",
    ("specialty", "day_only"): "kh03_occupied_by_spec_day_only.parquet",
}


def _read(name: str) -> pd.DataFrame:
    p = BEDS_DIR / f"{STEM}{name}"
    f = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    f["quarter_end"] = pd.to_datetime(f["Effective_Snapshot_Date"], dayfirst=True, errors="coerce")
    f["beds"] = pd.to_numeric(f["Number_Of_Beds"], errors="coerce")
    f["ods_code"] = f["Organisation_Code"].astype(str).str.strip()
    return f.dropna(subset=["quarter_end"])


def load(kind: str) -> pd.DataFrame:
    """kind = 'sector' or 'specialty'; overnight and day-only side by side."""
    key = "Sector" if kind == "sector" else "Specialty"
    frames = []
    for when in ("overnight", "day_only"):
        f = _read(FILES[(kind, when)])
        if key == "Sector":
            f[key] = f[key].astype(str).str.strip().str.replace("Mental illness", "Mental Illness")
        f = f.groupby(["ods_code", "quarter_end", key], as_index=False)["beds"].sum()
        frames.append(f.rename(columns={"beds": when, key: kind}))
    return frames[0].merge(frames[1], on=["ods_code", "quarter_end", kind], how="outer")


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    sector, specialty = load("sector"), load("specialty")
    sector.to_parquet(INTERIM_DIR / "kh03_occupied_by_sector.parquet", index=False)
    specialty.to_parquet(INTERIM_DIR / "kh03_occupied_by_specialty.parquet", index=False)
    logger.info(
        "KH03 occupied beds: %d trust-quarter-sector rows (%s -> %s), %d trust-quarter-specialty rows",
        len(sector),
        sector["quarter_end"].min().date(),
        sector["quarter_end"].max().date(),
        len(specialty),
    )
    return sector, specialty
