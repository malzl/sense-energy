"""Energy Systems Catapult: aggregate mobile-phone visitation for one NHS trust.

Dataset *Aggregate mobile data visitation for UK campus* (provider: The Oxford
Partnership, via the ESC SENSE platform). Daily visit counts for calendar 2024
at three Royal Devon University Healthcare sites - RH801 Royal Devon & Exeter
(Wonford), RH802 Heavitree, RH8K6 Nightingale Exeter - and a university campus
comparator, plus static visitor profiles (age bands, household income
quintiles, MBI Consumer Styles segments, travel-distance bands), all in percent.

Licence (EULA ESC3888-3.0): research purposes only; no re-identification; keep
on access-controlled systems; delete on expiry; outputs must carry the
attribution in :data:`ATTRIBUTION`. Schedule 3's secure-environment option is
not ticked, so local copies are permitted.
"""

from __future__ import annotations

import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

RAW_DIR = EXTERNAL_DIR / "esc_mobility"

ATTRIBUTION = (
    "This product contains data provided by The Oxford Partnership. Other than providing certain "
    "data sets, The Oxford Partnership has not been involved in the development of this product and "
    "accepts no responsibility whatsoever for the accuracy, completeness or reliability of this product, "
    "nor for any advice contained within it or for the use or reliance any reader may give to it."
)

#: ids in the files -> site_code in the NHS register (the comparator has none)
SITE_IDS = {"RH801": "RH801", "RH802": "RH802", "RH8K6": "RH8K6", "university": None}

PROFILE_FILES = {
    "age_band": "campus_agebands.csv",
    "household_income_quintile": "campus_hh_income.csv",
    "consumer_style": "campus_styles.csv",
    "travel_distance_m": "campus_travel_distance.csv",
}


def load_visits() -> pd.DataFrame:
    """Daily visits per id. RH802 carries one extra day (2025-01-01), kept as delivered."""
    v = pd.read_csv(RAW_DIR / "campus_visitation_2024.csv")
    v["date"] = pd.to_datetime(v["date"]).dt.date
    v["site_code"] = v["id"].map(SITE_IDS)
    v = v.sort_values(["id", "date"]).reset_index(drop=True)
    return v[["id", "site_code", "date", "visits"]]


def load_profiles() -> pd.DataFrame:
    """Static visitor profiles, long: id, profile, category, pct.

    Two of the four files (styles, travel distance) list ``RH801`` twice and
    ``university`` not at all, while the other two list ``university`` in the
    same fourth position. The duplicate is treated as the university row and
    flagged ``id_corrected``.
    """
    frames = []
    for profile, name in PROFILE_FILES.items():
        f = pd.read_csv(RAW_DIR / name)
        f["id_corrected"] = False
        dup = f["id"].duplicated(keep="first")
        if dup.any() and "university" not in set(f["id"]):
            f.loc[dup, "id"] = "university"
            f.loc[dup, "id_corrected"] = True
        long = f.melt(id_vars=["id", "id_corrected"], var_name="category", value_name="pct")
        long["profile"] = profile
        frames.append(long)
    out = pd.concat(frames, ignore_index=True)
    out["site_code"] = out["id"].map(SITE_IDS)
    return out[["id", "site_code", "profile", "category", "pct", "id_corrected"]]


def build() -> dict[str, object]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    visits, profiles = load_visits(), load_profiles()
    p1, p2 = (
        INTERIM_DIR / "mobility_visits_daily.parquet",
        INTERIM_DIR / "mobility_visitor_profiles.parquet",
    )
    visits.to_parquet(p1, index=False)
    profiles.to_parquet(p2, index=False)
    logger.info("Wrote %s (%d rows) and %s (%d rows)", p1.name, len(visits), p2.name, len(profiles))
    return {"visits": p1, "profiles": p2}
