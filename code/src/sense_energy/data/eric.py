"""Estates Returns Information Collection (ERIC) 2023/24, site and trust tables.

Source: the ESC-provided extract in ``external/eric/`` (NHS England Digital,
financial year 1 April 2023 - 31 March 2024). The loader keeps the fields the
project uses and makes them numeric:

* **electricity**: total consumed = green tariff + trust-owned solar + third-party
  solar + other renewables + other. The headline "Electrical energy consumption"
  field is filled for only ~5 % of sites, so the components are summed instead.
* gas, oil, steam and hot water consumed; solar generated; maximum electrical
  demand (kW) and available electrical capacity (kVA);
* building: gross internal floor area, heated volume, a floor-area-weighted mean
  construction year from the age-profile bands, LED coverage, backlog cost;
* technology: heat pumps, fossil-fuel CHP units, gas boilers older than 10 years,
  EV charge points; site type, tenure, single bedrooms;
* overheating: the number of occasions on which an occupied ward or clinical area
  exceeded 26 degC and triggered a risk assessment (field S16_07), the one estates
  field that speaks to cooling; trust-owned and third-party solar consumed on site.

Output: ``interim/eric_2023_24_site.parquet`` and ``interim/eric_2023_24_trust.parquet``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

ERIC_DIR = EXTERNAL_DIR / "eric"
STEM = "energy_systems_catapult.estates_returns_information_collection_eric_2023_24.eric_2023_24_"
FY_START, FY_END = "2023-04-01", "2024-04-01"  # half-open, local calendar
#: The collections held: the ESC extract of 2023/24 and the NHS England files of 2024/25
#: (published 16 October 2025, latin-1 encoded, with new CHP, export and floor-area fields).
YEARS = {
    "2023_24": {
        "dir": ERIC_DIR,
        "files": {k: f"{STEM}{k}_data.csv" for k in ("site", "trust", "pfi")},
        "encoding": "utf-8",
        "fy": ("2023-04-01", "2024-04-01"),
    },
    "2024_25": {
        "dir": EXTERNAL_DIR / "eric_2024_25",
        "files": {
            "site": "ERIC_-_2024_25_-_Site_data.csv",
            "trust": "ERIC_-_2024_25_-_Trust_data.csv",
            "pfi": "ERIC_-_2024_25_-_PFI_data.csv",
        },
        "encoding": "latin-1",
        "fy": ("2024-04-01", "2025-04-01"),
    },
    # Published 15 October 2026. The files sit behind NHS England Digital's publication page,
    # which refuses scripted requests: download the three CSVs in a browser into
    # external/eric_2025_26/. The names follow the 2024/25 pattern and have not been checked
    # against the files; adjust them here if the publication changed them.
    "2025_26": {
        "dir": EXTERNAL_DIR / "eric_2025_26",
        "files": {
            "site": "ERIC_-_2025_26_-_Site_data.csv",
            "trust": "ERIC_-_2025_26_-_Trust_data.csv",
            "pfi": "ERIC_-_2025_26_-_PFI_data.csv",
        },
        "encoding": "latin-1",
        "fy": ("2025-04-01", "2026-04-01"),
    },
}

ELEC_COMPONENTS = [
    "Electricity - green electricity consumed (kWh)",
    "Electricity - trust owned solar consumed (kWh)",
    "Electricity - third party owned solar consumed (kWh)",
    "Electricity - other renewables consumed (kWh)",
    "Electricity - other consumed (kWh)",
]
NUMERIC = {
    "gas_kwh": "Gas consumed (kWh)",
    "oil_kwh": "Oil consumed (kWh)",
    "steam_kwh": "Steam consumed (kWh)",
    "hot_water_kwh": "Hot water consumed (kWh)",
    "renewable_heat_kwh": "Non-fossil fuel - renewable consumed (kWh)",
    "solar_generated_kwh": "Solar electricity generated (kWh)",
    "max_demand_kw": "Maximum electrical demand (kW)",
    "capacity_kva": "Available electrical capacity (kVA)",
    "gia_m2": "Gross internal floor area (m²)",
    "heated_volume_m3": "Site heated volume (m³)",
    "led_pct": "LED lighting coverage (%)",
    "heat_pumps": "Heat pumps installed on site (No.)",
    "chp_units": "Fossil-fuel led CHP units operated on site (No.)",
    "old_boilers_large": "Number of primary heating gas boilers older than 10 years (100kW and above) (No.)",
    "old_boilers_small": "Number of primary heating gas boilers older than 10 years (less than 100kW) (No.)",
    "ev_charge_points": "Electric vehicle charging points (No.)",
    "solar_consumed_kwh": "Electricity - trust owned solar consumed (kWh)",
    "third_party_solar_kwh": "Electricity - third party owned solar consumed (kWh)",
    "overheating_occurrences": "Overheating occurrences triggering a risk assessment (No.)",
    "single_rooms_ensuite": "Single bedrooms for patients with en-suite facilities (No.)",
    "single_rooms_other": "Single bedrooms for patients without en-suite facilities (No.)",
    "backlog_high_risk_gbp": "Cost to eradicate high risk backlog (£)",
    "backlog_significant_risk_gbp": "Cost to eradicate significant risk backlog (£)",
    # 2024/25 additions (missing in 2023/24)
    "occupied_floor_area_m2": "Occupied floor area (m²)",
    "chp_elec_output_kwh": "Electrical energy output of CHP system/s (kWh)",
    "chp_thermal_output_kwh": "Thermal energy output of CHP system/s (kWh)",
    "chp_gas_input_kwh": "Fossil energy (Gas) input to CHP system/s (kWh)",
    "chp_other_input_kwh": "Fossil energy (Other) input to CHP system/s (kWh)",
    "chp_elec_purchased_kwh": "Electricity purchased from CHP consumed (kWh)",
    "chp_thermal_purchased_kwh": "Thermal energy purchased from CHP consumed (kWh)",
    "chp_thermal_wasted_kwh": "Thermal energy discharged to waste (kWh)",
    "exported_elec_kwh": "Exported electricity (kWh)",
    "exported_thermal_kwh": "Exported thermal energy (kWh)",
    "chp_elec_kw": "CHP unit/s size - electrical (kW)",
    "chp_thermal_kw": "CHP unit/s size - thermal (kW)",
}
AGE_BANDS = {  # band -> representative construction year
    "Age profile - 2025 to 2034 (%)": 2029.5,  # 2024/25 only
    "Age profile - 2015 to 2024 (%)": 2019.5,
    "Age profile - 2005 to 2014 (%)": 2009.5,
    "Age profile - 1995 to 2004 (%)": 1999.5,
    "Age profile - 1985 to 1994 (%)": 1989.5,
    "Age profile - 1975 to 1984 (%)": 1979.5,
    "Age profile - 1965 to 1974 (%)": 1969.5,
    "Age profile - 1955 to 1964 (%)": 1959.5,
    "Age profile - 1948 to 1954 (%)": 1951.0,
    "Age profile - pre 1948 (%)": 1935.0,
}


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.strip(),
        errors="coerce",
    )


def normalise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Column names as in the 2023/24 extract: the 2024/25 files are latin-1 and carry the pound
    and squared signs as replacement characters, and whitespace varies."""
    import re

    def fix(c: str) -> str:
        c = c.replace("\ufffd", "£").replace("Â", "").replace("Ã‚", "")
        return re.sub(r"\s+", " ", c).strip()

    return frame.rename(columns={c: fix(c) for c in frame.columns})


def mean_build_year(frame: pd.DataFrame) -> pd.Series:
    """Floor-area-weighted mean construction year from the ERIC age-profile bands."""
    bands = {c: y for c, y in AGE_BANDS.items() if c in frame.columns}
    w = pd.concat([_num(frame[c]).rename(c) for c in bands], axis=1)
    total = w.sum(axis=1, min_count=1)
    years = (w * pd.Series(bands)).sum(axis=1, min_count=1)
    return (years / total).where(total > 0)


def _path(kind: str, year: str):
    y = YEARS[year]
    return y["dir"] / y["files"][kind]


def load_site(path=None, year: str = "2023_24") -> pd.DataFrame:
    raw = pd.read_csv(
        path or _path("site", year), low_memory=False, encoding=YEARS[year]["encoding"]
    )
    raw = normalise_columns(raw)
    out = pd.DataFrame(
        {
            "site_code": raw["Site Code"].astype(str).str.strip(),
            "site_name": raw["Site Name"],
            "trust_code": raw["Trust Code"].astype(str).str.strip(),
            "trust_name": raw["Trust Name"],
            "trust_type": raw["Trust Type"],
            "commissioning_region": raw["Commissioning Region"],
            "site_type": raw["Site Type"],
            "tenure": raw["Tenure"],
            "postcode": raw["Post Code"],
        }
    )
    parts = pd.concat([_num(raw[c]) for c in ELEC_COMPONENTS], axis=1)
    out["elec_kwh"] = parts.sum(axis=1, min_count=1)
    headline = "Electrical energy consumption (KWh)"  # dropped from the 2024/25 collection
    out["elec_headline_kwh"] = _num(raw[headline]) if headline in raw else np.nan
    for name, col in NUMERIC.items():
        out[name] = _num(raw[col]) if col in raw else np.nan
    out["collection"] = year.replace("_", "/")
    out["mean_build_year"] = mean_build_year(raw)
    out["fossil_heat_kwh"] = out[["gas_kwh", "oil_kwh"]].sum(axis=1, min_count=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["elec_kwh_per_m2"] = (out["elec_kwh"] / out["gia_m2"]).where(out["gia_m2"] > 0)
        out["gas_kwh_per_m2"] = (out["gas_kwh"] / out["gia_m2"]).where(out["gia_m2"] > 0)
        out["gas_to_elec"] = (out["gas_kwh"] / out["elec_kwh"]).where(out["elec_kwh"] > 0)
    return out


def load_trust(path=None, year: str = "2023_24") -> pd.DataFrame:
    raw = pd.read_csv(
        path or _path("trust", year), low_memory=False, encoding=YEARS[year]["encoding"]
    )
    raw = normalise_columns(raw)
    out = raw.rename(columns={"Trust Code": "trust_code", "Trust Name": "trust_name"})
    out["trust_code"] = out["trust_code"].astype(str).str.strip()
    return out


def locate(site: pd.DataFrame) -> pd.DataFrame:
    """Postcode centroids (postcodes.io), cached in ``interim/eric_2023_24_postcodes.parquet``."""
    from .geo import geocode_postcodes

    cache = INTERIM_DIR / "eric_2023_24_postcodes.parquet"
    keys = site["postcode"].astype(str).str.strip().str.upper()
    known = pd.read_parquet(cache) if cache.exists() else pd.DataFrame(columns=["postcode"])
    missing = sorted(set(keys) - set(known["postcode"]) - {"NAN", ""})
    if missing:
        known = pd.concat([known, geocode_postcodes(missing)], ignore_index=True)
        known = known.drop_duplicates("postcode", keep="last")
        known.to_parquet(cache, index=False)
    ll = known.set_index("postcode")[["latitude", "longitude"]]
    return site.assign(
        latitude=keys.map(ll["latitude"]).astype(float),
        longitude=keys.map(ll["longitude"]).astype(float),
    )


def build(geocode: bool = True, year: str = "2023_24") -> tuple[pd.DataFrame, pd.DataFrame]:
    site, trust = load_site(year=year), load_trust(year=year)
    if geocode:
        site = locate(site)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    site.to_parquet(INTERIM_DIR / f"eric_{year}_site.parquet", index=False)
    trust.to_parquet(INTERIM_DIR / f"eric_{year}_trust.parquet", index=False)
    logger.info(
        "ERIC %s: %d sites (%d with electricity, %d with max demand and capacity), %d trusts",
        year.replace("_", "/"),
        len(site),
        int((site["elec_kwh"] > 0).sum()),
        int(((site["max_demand_kw"] > 0) & (site["capacity_kva"] > 0)).sum()),
        len(trust),
    )
    return site, trust
