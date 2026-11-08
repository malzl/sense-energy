"""Site context for the weather work: urban form, population density and planned solar.

Three open sources, one row per site in ``interim/sites.parquet``:

* **postcodes.io** (ONS open data): the 2021 census geographies of the site postcode (output
  area, LSOA, built-up area) and the rural-urban classification (RUC 2011 and 2021).
* **Census 2021 population density** (ONS table TS006, usual residents per km2) of the site's
  LSOA, pulled from the Nomis API into ``external/ons/census2021_ts006_density_lsoa21.csv``.
* **Renewable Energy Planning Database** (DESNZ quarterly extract, schemes of 150 kW and
  above): solar PV and battery schemes at the site postcode with capacity, planning status and
  operational date, ``external/repd/REPD_Publication_*.csv``.

Output: ``interim/site_context.parquet``.
"""

from __future__ import annotations

import time
import urllib.error
from pathlib import Path

import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger
from .geo import BULK_CHUNK, POSTCODES_IO_BULK, _post_json

logger = get_logger(__name__)

DENSITY_FILE = EXTERNAL_DIR / "ons" / "census2021_ts006_density_lsoa21.csv"
REPD_DIR = EXTERNAL_DIR / "repd"
SAMPLE_END = pd.Timestamp("2026-05-08")  # last day of the electricity extract

#: ONS rural-urban classification 2011 (output areas), collapsed to four classes.
RUC11_CLASS = {
    "A1": "major conurbation",
    "B1": "minor conurbation",
    "C1": "city and town",
    "C2": "city and town",
    "D1": "rural",
    "D2": "rural",
    "E1": "rural",
    "E2": "rural",
    "F1": "rural",
    "F2": "rural",
}
URBAN_ORDER = ("major conurbation", "minor conurbation", "city and town", "rural")
REPD_SOLAR = "Solar Photovoltaics"
REPD_BATTERY = "Battery"
#: REPD development statuses, best first.
REPD_STATUS_RANK = (
    "Operational",
    "Under Construction",
    "Awaiting Construction",
    "Application Submitted",
    "Revised",
    "Appeal Lodged",
    "Application Refused",
    "Application Withdrawn",
    "Abandoned",
)


def normalise_postcode(s: pd.Series) -> pd.Series:
    """Upper case, no spaces; anything that is not 5-7 characters becomes missing."""
    out = s.astype("string").str.upper().str.replace(r"\s+", "", regex=True)
    return out.where(out.str.len().between(5, 7), pd.NA)


def urban_class(ruc11: pd.Series) -> pd.Series:
    return ruc11.map(RUC11_CLASS)


def parse_result(query: str | None, result: dict | None) -> dict:
    codes = (result or {}).get("codes") or {}
    return {
        "postcode_norm": query,
        "oa21": codes.get("oa21"),
        "lsoa21": codes.get("lsoa21"),
        "lsoa_name": (result or {}).get("lsoa"),
        "bua": codes.get("bua"),
        "ruc11": codes.get("ruc11"),
        "ruc21": codes.get("ruc21"),
    }


def lookup_postcodes(postcodes: list[str], pause: float = 0.3) -> pd.DataFrame:
    """Census geographies and rural-urban class from postcodes.io, in chunks of 100.
    Terminated postcodes come back empty: the register no longer carries their geographies."""
    unique = sorted({p for p in postcodes if isinstance(p, str) and p})
    rows = []
    for start in range(0, len(unique), BULK_CHUNK):
        chunk = unique[start : start + BULK_CHUNK]
        try:
            response = _post_json(POSTCODES_IO_BULK, {"postcodes": chunk})
        except urllib.error.URLError as exc:
            raise RuntimeError(f"postcodes.io request failed: {exc}") from exc
        for item in response.get("result", []):
            rows.append(parse_result(item.get("query"), item.get("result")))
        if start + BULK_CHUNK < len(unique):
            time.sleep(pause)
    out = pd.DataFrame(rows)
    logger.info(
        "postcodes.io: %d of %d postcodes resolved", int(out["lsoa21"].notna().sum()), len(out)
    )
    return out


def density_table(path: Path = DENSITY_FILE) -> pd.Series:
    """Usual residents per km2 by 2021 LSOA (Census 2021 TS006)."""
    d = pd.read_csv(path)
    d = d[d["CELL_NAME"].str.contains("per square kilometre", na=False)]
    return d.set_index("GEOGRAPHY_CODE")["OBS_VALUE"].astype(float).rename("density_km2")


def repd_table(path: Path | None = None) -> pd.DataFrame:
    """The REPD extract, England, with a normalised postcode and a parsed operational date."""
    if path is None:
        files = sorted(REPD_DIR.glob("REPD_Publication_*.csv"))
        if not files:
            raise FileNotFoundError(f"no REPD extract in {REPD_DIR}")
        path = files[-1]
    raw = pd.read_csv(path, encoding="latin-1", low_memory=False)
    out = pd.DataFrame(
        {
            "ref": raw["Ref ID"],
            "scheme": raw["Site Name"],
            "technology": raw["Technology Type"],
            "capacity_mw": pd.to_numeric(raw["Installed Capacity (MWelec)"], errors="coerce"),
            "status": raw["Development Status (short)"],
            "operational": pd.to_datetime(raw["Operational"], dayfirst=True, errors="coerce"),
            "postcode_norm": normalise_postcode(raw["Post Code"]),
            "country": raw["Country"],
        }
    )
    return out[out["postcode_norm"].notna() & out["country"].eq("England")].reset_index(drop=True)


def _best_status(statuses: pd.Series) -> str | None:
    rank = {s: i for i, s in enumerate(REPD_STATUS_RANK)}
    present = [s for s in statuses.dropna().unique()]
    if not present:
        return None
    return min(present, key=lambda s: rank.get(s, len(rank)))


def repd_at_sites(
    postcodes: pd.Series, repd: pd.DataFrame, sample_end: pd.Timestamp = SAMPLE_END
) -> pd.DataFrame:
    """Per site (index of ``postcodes``, values normalised postcodes): solar and battery schemes
    at the site postcode. ``repd_solar_operational_mw`` counts capacity operational by
    ``sample_end``; the rest is planned or under construction."""
    sites = pd.DataFrame({"site_code": postcodes.index, "postcode_norm": postcodes.to_numpy()})
    m = sites.dropna(subset=["postcode_norm"]).merge(repd, on="postcode_norm", how="inner")
    solar = m[m["technology"] == REPD_SOLAR]
    battery = m[m["technology"] == REPD_BATTERY]
    live = solar[solar["operational"].notna() & (solar["operational"] <= sample_end)]
    out = pd.DataFrame(index=pd.Index(postcodes.index, name="site_code"))
    g = solar.groupby("site_code")
    out["repd_solar_schemes"] = g.size().reindex(out.index).fillna(0).astype(int)
    out["repd_solar_mw"] = g["capacity_mw"].sum(min_count=1).reindex(out.index)
    out["repd_solar_status"] = g["status"].agg(_best_status).reindex(out.index)
    out["repd_solar_operational_mw"] = (
        live.groupby("site_code")["capacity_mw"].sum(min_count=1).reindex(out.index)
    )
    out["repd_solar_first_operational"] = (
        live.groupby("site_code")["operational"].min().reindex(out.index)
    )
    out["repd_battery_mw"] = (
        battery.groupby("site_code")["capacity_mw"].sum(min_count=1).reindex(out.index)
    )
    return out


def build(sample_end: pd.Timestamp = SAMPLE_END) -> pd.DataFrame:
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet")[["site_code", "postcode"]]
    sites = sites.drop_duplicates("site_code").reset_index(drop=True)
    sites["postcode_norm"] = normalise_postcode(sites["postcode"])
    geo = lookup_postcodes(sites["postcode_norm"].dropna().tolist())
    out = sites.merge(geo, on="postcode_norm", how="left")
    out["urban"] = urban_class(out["ruc11"])
    out["density_km2"] = out["lsoa21"].map(density_table())
    repd = repd_at_sites(out.set_index("site_code")["postcode_norm"], repd_table(), sample_end)
    out = out.merge(repd, left_on="site_code", right_index=True, how="left")
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INTERIM_DIR / "site_context.parquet", index=False)
    logger.info(
        "site context: %d sites, %d with a rural-urban class, %d with density, "
        "%d with a REPD solar scheme at the postcode (%d operational by %s)",
        len(out),
        int(out["urban"].notna().sum()),
        int(out["density_km2"].notna().sum()),
        int((out["repd_solar_schemes"] > 0).sum()),
        int(out["repd_solar_operational_mw"].notna().sum()),
        sample_end.date(),
    )
    return out
