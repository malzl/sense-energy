"""Display Energy Certificates (DECs) for the site postcodes, from the MHCLG energy performance
of buildings data API (bearer token ``EPB_API_KEY`` in ``.env``).

Every occupied public building over 250 m2 displays a DEC, renewed every year above 1,000 m2.
Each certificate records, per building: floor area, main heating fuel, the building environment
(air conditioning / mixed mode / heating with natural or mechanical ventilation), whether air
conditioning is present and its rated output, the metered annual electricity and fuel of the
assessment year, the operational rating and the nominated date. It is the one public source on
cooling plant, building by building.

Fetching: ``GET /api/display/search?postcode=`` for every site postcode (every certificate ever
registered at the postcode), then ``GET /api/certificate?certificate_number=`` for each; both
cached as JSON under ``external/dec/`` so a rerun costs nothing. The service allows 6,000
requests per five minutes; the fetch paces itself at five per second and backs off on 429.

Each DEC has an **advisory report** (its ``related_certificate_number`` on the newer schemas):
the assessor's recommendations by payback class (short / medium / long / other) with a
recommendation code and a CO2-impact grade. These are the "opportunities" of the estate,
building by building; they are fetched for the latest certificate of each building and grouped
into themes (cooling, heating, hot water, lighting, controls, fabric, ventilation, renewables).

Outputs: ``interim/dec_certificates.parquet`` (one row per certificate, flattened),
``interim/dec_recommendations.parquet`` (one row per recommendation) and
``interim/dec_sites.parquet`` (one row per site: the latest certificate of each building at the
site postcode, aggregated by floor area, with recommendation counts by theme). Open Government
Licence v3.0.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR, SECRETS
from ..logging_utils import get_logger
from .site_context import normalise_postcode

logger = get_logger(__name__)

API = "https://api.get-energy-performance-data.communities.gov.uk"
DEC_DIR = EXTERNAL_DIR / "dec"
SEARCH_DIR = DEC_DIR / "search"
CERT_DIR = DEC_DIR / "certificates"
PAUSE = 0.2  # seconds between requests: 300 a minute against a limit of 1,200
BACKOFF = 60.0
USER_AGENT = "sense-energy research (NHS site energy study)"

#: Building environment as the DEC records it, collapsed.
ENVIRONMENT_CLASS = {
    "Air Conditioning": "air conditioning",
    "Mixed-mode with Natural Ventilation": "mixed mode",
    "Mixed-mode with Mechanical Ventilation": "mixed mode",
    "Heating and Mechanical Ventilation": "heating, mechanical ventilation",
    "Heating and Natural Ventilation": "heating, natural ventilation",
    "Natural Ventilation Only": "heating, natural ventilation",
    "Unconditioned": "unconditioned",
}
#: A building whose latest certificate is older than this is treated as no longer part of the
#: estate (DECs are renewed yearly above 1,000 m2; the register keeps every certificate since 2008).
SINCE = pd.Timestamp("2019-01-01")
ENVIRONMENT_ORDER = (
    "air conditioning",
    "mixed mode",
    "heating, mechanical ventilation",
    "heating, natural ventilation",
    "unconditioned",
)
ELECTRIC_FUELS = ("Grid Supplied Electricity", "Grid Displaced Electricity", "Electricity")
ADVISORY_DIR = DEC_DIR / "advisory"
EPC_SEARCH_DIR = DEC_DIR / "epc_search"
EPC_CERT_DIR = DEC_DIR / "epc_certificates"
#: Non-domestic EPCs are valid for ten years; a building whose latest is older is dropped.
EPC_SINCE = pd.Timestamp("2016-01-01")
PAYBACKS = ("short", "medium", "long", "other")
#: Recommendation themes by keyword in the recommendation text (a recommendation can carry
#: several themes).
RECOMMENDATION_THEMES = {
    "cooling": ("air conditioning", "cooling", "chiller", "refrigerat", "free cooling"),
    "heating": ("boiler", "heating", "radiator", "heat pump", "chp", "combined heat"),
    "hot water": ("hot water", "dhw"),
    "lighting": ("lighting", "daylight", "lamp", "luminaire"),
    "controls": (
        "control",
        "bms",
        "thermostat",
        "timer",
        "sensor",
        "zoning",
        "power save",
        "metering",
    ),
    "fabric": ("insulation", "glazing", "fabric", "draught", "air tightness", "sealing", "roof"),
    "ventilation": ("ventilation", "fan", "air handling", "ahu"),
    "renewables": ("photovoltaic", "solar", "wind turbine", "biomass", "renewable"),
}


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #


def _request(path: str, params: dict[str, Any] | None = None) -> dict:
    if not SECRETS.epb_api_key:
        raise RuntimeError("EPB_API_KEY is not set (.env)")
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {SECRETS.epb_api_key}",
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=120) as response:  # noqa: S310
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 3:
                logger.warning("rate limited; sleeping %.0f s", BACKOFF)
                time.sleep(BACKOFF)
                continue
            if exc.code == 404:
                return {}
            raise
    raise RuntimeError(f"gave up on {path}")


def search_postcode(postcode: str, refresh: bool = False, kind: str = "display") -> list[dict]:
    """Every certificate of ``kind`` (``display`` for DECs, ``non-domestic`` for EPCs) registered
    at a postcode (cached)."""
    directory = SEARCH_DIR if kind == "display" else EPC_SEARCH_DIR
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{postcode}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    rows, page = [], 1
    while True:
        d = _request(f"/api/{kind}/search", {"postcode": postcode, "current_page": page})
        rows.extend(d.get("data") or [])
        pagination = d.get("pagination") or {}
        if not pagination.get("nextPage"):
            break
        page = pagination["nextPage"]
        time.sleep(PAUSE)
    cache.write_text(json.dumps(rows))
    time.sleep(PAUSE)
    return rows


def fetch_certificate(number: str, refresh: bool = False, directory=None) -> dict:
    """The full record of one certificate or report (cached)."""
    directory = CERT_DIR if directory is None else directory
    directory.mkdir(parents=True, exist_ok=True)
    cache = directory / f"{number}.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    d = _request("/api/certificate", {"certificate_number": number}).get("data") or {}
    cache.write_text(json.dumps(d))
    time.sleep(PAUSE)
    return d


def site_postcodes() -> pd.DataFrame:
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet")[["site_code", "postcode"]]
    sites = sites.drop_duplicates("site_code")
    sites["postcode_norm"] = normalise_postcode(sites["postcode"])
    return sites.dropna(subset=["postcode_norm"]).reset_index(drop=True)


def fetch_all(refresh: bool = False, kind: str = "display") -> pd.DataFrame:
    """Search every site postcode and fetch every certificate found; returns the search rows."""
    directory = CERT_DIR if kind == "display" else EPC_CERT_DIR
    sites = site_postcodes()
    rows = []
    for i, pc in enumerate(sorted(sites["postcode_norm"].unique())):
        for r in search_postcode(pc, refresh, kind):
            rows.append({**r, "postcode_norm": pc})
        if i % 25 == 0:
            logger.info("%s: searched %d postcodes, %d certificates so far", kind, i + 1, len(rows))
    found = pd.DataFrame(rows)
    if found.empty:
        return found
    logger.info(
        "%s: %d certificates at %d postcodes", kind, len(found), found["postcode_norm"].nunique()
    )
    numbers = found["certificateNumber"].unique()
    for i, number in enumerate(numbers):
        fetch_certificate(number, refresh, directory)
        if i % 200 == 0:
            logger.info("%s: fetched %d of %d certificates", kind, i + 1, len(numbers))
    return found


def fetch_advisory_reports(latest: pd.DataFrame, refresh: bool = False) -> int:
    """The advisory report of every certificate in ``latest`` that links one."""
    numbers = latest["related_certificate_number"].dropna().unique()
    for i, number in enumerate(numbers):
        fetch_certificate(number, refresh, ADVISORY_DIR)
        if i % 200 == 0:
            logger.info("fetched %d of %d advisory reports", i + 1, len(numbers))
    return len(numbers)


# --------------------------------------------------------------------------- #
# Flatten
# --------------------------------------------------------------------------- #


def _get(d: Any, *path: str, default: Any = None) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return default
        d = d.get(key)
    return default if d is None else d


def _num(v: Any) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def flatten(d: dict) -> dict:
    """The fields the project uses, from a certificate record of any CEPC schema version."""
    ti = d.get("technical_information") or {}
    ac = d.get("ac_questionnaire") or {}
    rated = ac.get("ac_rated_output") or {}
    cons = d.get("or_energy_consumption") or {}
    summary = d.get("dec_annual_energy_summary") or {}
    this = d.get("this_assessment") or {}
    ac_kw = np.nan
    if isinstance(rated, dict) and not rated.get("ac_rating_unknown_flag"):
        ac_kw = _num(rated.get("ac_kw_rating", rated.get("value")))
    elif not isinstance(rated, dict):
        ac_kw = _num(rated)
    fuels = {k: _num(_get(v, "consumption")) for k, v in cons.items() if isinstance(v, dict)}
    electricity_kwh = fuels.pop("electricity", np.nan)
    thermal_kwh = float(np.nansum(list(fuels.values()))) if fuels else np.nan
    if fuels and all(np.isnan(v) for v in fuels.values()):
        thermal_kwh = np.nan
    return {
        "certificate_number": d.get("certificate_number") or d.get("certificateNumber"),
        "uprn": str(d.get("uprn")) if d.get("uprn") not in (None, "", 0) else None,
        "postcode": d.get("postcode"),
        "address_line_1": d.get("address_line_1"),
        "address_line_2": d.get("address_line_2"),
        "post_town": d.get("post_town"),
        "property_type": d.get("property_type"),
        "building_category": d.get("building_category"),
        "schema_type": d.get("schema_type"),
        "registration_date": d.get("registration_date"),
        "inspection_date": d.get("inspection_date"),
        "nominated_date": this.get("nominated_date") or d.get("nominated_date"),
        "valid_until": d.get("valid_until"),
        "assessment_start": d.get("or_assessment_start_date"),
        "assessment_end": d.get("or_assessment_end_date"),
        "floor_area_m2": _num(ti.get("floor_area", d.get("floor_area"))),
        "main_heating_fuel": ti.get("main_heating_fuel", d.get("main_heating_fuel")),
        "building_environment": ti.get("building_environment", d.get("building_environment")),
        "separately_metered_electric_heating": _num(ti.get("separately_metered_electric_heating")),
        "hvac_system": _get(d, "or_building_data", "hvac_system"),
        "ac_present": ac.get("ac_present", d.get("ac_present")),
        "ac_rated_kw": ac_kw,
        "ac_estimated_output": ac.get("ac_estimated_output"),
        "ac_inspection_commissioned": ac.get("ac_inspection_commissioned"),
        "operational_rating": _num(this.get("energy_rating", d.get("operational_rating"))),
        "asset_rating": _num(d.get("asset_rating")),
        "assessment_type": d.get("assessment_type"),
        "electricity_co2": _num(this.get("electricity_co2")),
        "heating_co2": _num(this.get("heating_co2")),
        "renewables_co2": _num(this.get("renewables_co2")),
        "electricity_kwh": electricity_kwh,
        "electricity_estimated": _num(_get(cons, "electricity", "estimate")),
        "thermal_kwh": thermal_kwh,
        "thermal_fuels": ";".join(sorted(fuels)) if fuels else None,
        "elec_kwh_m2": _num(summary.get("annual_energy_use_electrical")),
        "thermal_kwh_m2": _num(summary.get("annual_energy_use_fuel_thermal")),
        "typical_elec_kwh_m2": _num(summary.get("typical_electrical_use")),
        "typical_thermal_kwh_m2": _num(summary.get("typical_thermal_use")),
        "renewables_electrical": _num(summary.get("renewables_electrical")),
        "renewables_thermal": _num(summary.get("renewables_fuel_thermal")),
        "main_benchmark": _get(d, "or_benchmark_data", "main_benchmark"),
        "related_certificate_number": d.get("related_certificate_number"),
    }


def certificates_table(directory=None) -> pd.DataFrame:
    directory = CERT_DIR if directory is None else directory
    rows = []
    for path in sorted(directory.glob("*.json")) if directory.exists() else []:
        d = json.loads(path.read_text())
        if not d:
            continue
        r = flatten(d)
        r["certificate_number"] = r["certificate_number"] or path.stem
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in (
        "registration_date",
        "inspection_date",
        "nominated_date",
        "valid_until",
        "assessment_start",
        "assessment_end",
    ):
        out[c] = pd.to_datetime(out[c], errors="coerce")
    out["postcode_norm"] = normalise_postcode(out["postcode"])
    out["environment"] = out["building_environment"].map(ENVIRONMENT_CLASS)
    out["electric_heating"] = out["main_heating_fuel"].isin(ELECTRIC_FUELS)
    out["ac_yes"] = out["ac_present"].astype(str).str.lower().eq("yes")
    return out


def themes_of(text: str) -> list[str]:
    t = (text or "").lower()
    return [theme for theme, words in RECOMMENDATION_THEMES.items() if any(w in t for w in words)]


def recommendations_table() -> pd.DataFrame:
    """One row per recommendation of every cached advisory report."""
    rows = []
    for path in sorted(ADVISORY_DIR.glob("*.json")) if ADVISORY_DIR.exists() else []:
        d = json.loads(path.read_text())
        if not d:
            continue
        ti = d.get("technical_information") or {}
        for payback in PAYBACKS:
            for r in d.get(f"{payback}_payback") or []:
                if not isinstance(r, dict):
                    continue
                text = r.get("recommendation")
                rows.append(
                    {
                        "advisory_number": path.stem,
                        "dec_number": d.get("related_certificate_number"),
                        "postcode_norm": normalise_postcode(pd.Series([d.get("postcode")])).iloc[0],
                        "inspection_date": d.get("inspection_date"),
                        "floor_area_m2": _num(ti.get("floor_area")),
                        "building_environment": ti.get("building_environment"),
                        "payback": payback,
                        "code": r.get("recommendation_code"),
                        "recommendation": text,
                        "co2_impact": r.get("co2_impact"),
                        "themes": ";".join(themes_of(text)),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["inspection_date"] = pd.to_datetime(out["inspection_date"], errors="coerce")
    return out


def aggregate_recommendations(recs: pd.DataFrame, latest: pd.DataFrame) -> pd.DataFrame:
    """Per postcode: buildings with an advisory report among the latest certificates, the number
    of recommendations, and the number (and share of reporting buildings) per theme."""
    if recs.empty:
        return pd.DataFrame(columns=["postcode_norm"])
    link = latest.dropna(subset=["related_certificate_number"])[
        ["postcode_norm", "building", "related_certificate_number"]
    ].rename(columns={"related_certificate_number": "advisory_number"})
    r = recs.drop(columns=["postcode_norm"]).merge(link, on="advisory_number", how="inner")
    out = r.groupby("postcode_norm").agg(
        rec_buildings=("building", "nunique"), rec_total=("recommendation", "size")
    )
    out["rec_short_payback"] = r[r["payback"] == "short"].groupby("postcode_norm").size()
    for theme in RECOMMENDATION_THEMES:
        hit = r["themes"].str.split(";").apply(lambda t, th=theme: th in t)
        out[f"rec_{theme.replace(' ', '_')}"] = r[hit].groupby("postcode_norm").size()
        out[f"rec_{theme.replace(' ', '_')}_building_share"] = (
            r[hit].groupby("postcode_norm")["building"].nunique() / out["rec_buildings"]
        )
    return out.fillna({c: 0 for c in out.columns if not c.endswith("_share")}).reset_index()


# --------------------------------------------------------------------------- #
# Per site
# --------------------------------------------------------------------------- #


def latest_per_building(
    certs: pd.DataFrame, until: pd.Timestamp | None = None, since: pd.Timestamp | None = None
) -> pd.DataFrame:
    """One certificate per building at each postcode: the latest nominated date (registration
    date as fallback), optionally not after ``until``; with ``since``, buildings whose latest
    certificate is older are dropped. Buildings are keyed by UPRN, else by the first address
    line."""
    c = certs.copy()
    c["when"] = c["nominated_date"].fillna(c["registration_date"])
    if until is not None:
        c = c[c["when"] <= until]
    key_addr = (
        c["address_line_1"].astype(str).str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    )
    c["building"] = c["uprn"].where(c["uprn"].notna(), "ADDR:" + key_addr)
    c = c.sort_values("when")
    latest = c.groupby(["postcode_norm", "building"], as_index=False).tail(1)
    if since is not None:
        latest = latest[latest["when"] >= since]
    return latest


def _wshare(mask: pd.Series, w: pd.Series) -> float:
    return float((w * mask).sum() / w.sum()) if w.sum() > 0 else np.nan


def aggregate_sites(latest: pd.DataFrame, sites: pd.DataFrame, prefix: str = "dec") -> pd.DataFrame:
    """Floor-area-weighted site summaries of the latest certificate per building."""
    rows = []
    for _, s in sites.iterrows():
        b = latest[latest["postcode_norm"] == s["postcode_norm"]]
        w = b["floor_area_m2"].fillna(0)
        row = {
            "site_code": s["site_code"],
            "postcode_norm": s["postcode_norm"],
            "dec_buildings": len(b),
        }
        if b.empty:
            rows.append(row)
            continue
        known_kw = b["ac_rated_kw"].notna() & b["ac_yes"]
        row.update(
            {
                "dec_floor_area_m2": float(w.sum()),
                "dec_ac_share": _wshare(b["ac_yes"], w),
                "dec_ac_buildings": int(b["ac_yes"].sum()),
                "dec_ac_kw": float(b.loc[known_kw, "ac_rated_kw"].sum())
                if known_kw.any()
                else np.nan,
                "dec_ac_kw_m2": (
                    float(b.loc[known_kw, "ac_rated_kw"].sum() / w[known_kw].sum())
                    if known_kw.any() and w[known_kw].sum() > 0
                    else np.nan
                ),
                "dec_ac_kw_known_share": _wshare(known_kw, w[b["ac_yes"]])
                if b["ac_yes"].any()
                else np.nan,
                "dec_cooled_share": _wshare(
                    b["environment"].isin(["air conditioning", "mixed mode"]), w
                ),
                "dec_mech_vent_share": _wshare(
                    b["environment"].eq("heating, mechanical ventilation"), w
                ),
                "dec_electric_heating_share": _wshare(b["electric_heating"], w),
                "dec_elec_kwh_m2": _wmean(b["elec_kwh_m2"], w),
                "dec_thermal_kwh_m2": _wmean(b["thermal_kwh_m2"], w),
                "dec_operational_rating": _wmean(b["operational_rating"], w),
                "dec_renewables_electrical": _wmean(b["renewables_electrical"], w),
                "dec_electricity_kwh": float(b["electricity_kwh"].sum(min_count=1)),
                "dec_latest_nominated": b["when"].max(),
                "dec_oldest_nominated": b["when"].min(),
                "dec_main_environment": (
                    w.groupby(b["environment"]).sum().idxmax()
                    if b["environment"].notna().any()
                    else None
                ),
                "dec_property_types": ";".join(
                    sorted(b["property_type"].dropna().astype(str).unique())
                ),
            }
        )
        for env in ENVIRONMENT_ORDER:
            row[f"dec_env_{env.split(',')[0].replace(' ', '_')}_share"] = _wshare(
                b["environment"].eq(env), w
            )
        rows.append(row)
    out = pd.DataFrame(rows)
    if prefix != "dec":
        out = out.rename(columns={c: prefix + c[3:] for c in out.columns if c.startswith("dec_")})
    return out


def _wmean(x: pd.Series, w: pd.Series) -> float:
    m = x.notna() & (w > 0)
    return float((x[m] * w[m]).sum() / w[m].sum()) if m.any() else np.nan


def build(
    fetch: bool = True, until: pd.Timestamp | None = None, since: pd.Timestamp | None = SINCE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if fetch:
        fetch_all()
    certs = certificates_table()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    certs.to_parquet(INTERIM_DIR / "dec_certificates.parquet", index=False)
    latest = latest_per_building(certs, until, since)
    sites = site_postcodes()
    out = aggregate_sites(latest, sites)
    ever = latest_per_building(certs, until).groupby("postcode_norm").size()
    out["dec_buildings_ever"] = out["postcode_norm"].map(ever).fillna(0).astype(int)
    if fetch:
        fetch_advisory_reports(latest)
    recs = recommendations_table()
    recs.to_parquet(INTERIM_DIR / "dec_recommendations.parquet", index=False)
    out = out.merge(aggregate_recommendations(recs, latest), on="postcode_norm", how="left")
    out.to_parquet(INTERIM_DIR / "dec_sites.parquet", index=False)
    logger.info(
        "DEC: %d certificates, %d buildings (latest per building), %d of %d sites with a DEC at "
        "the postcode, %d with air conditioning somewhere on site; %d recommendations from %d "
        "advisory reports",
        len(certs),
        len(latest),
        int((out["dec_buildings"] > 0).sum()),
        len(out),
        int((out["dec_ac_buildings"].fillna(0) > 0).sum()),
        len(recs),
        int(recs["advisory_number"].nunique()) if not recs.empty else 0,
    )
    return certs, out


def build_epc(
    fetch: bool = True, since: pd.Timestamp | None = EPC_SINCE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Non-domestic EPCs at the site postcodes: the asset-rated buildings DECs do not cover
    (leased premises, new builds), with the same HVAC fields. Outputs
    ``interim/epc_certificates.parquet`` and ``interim/epc_sites.parquet`` (``epc_*`` columns)."""
    if fetch:
        fetch_all(kind="non-domestic")
    certs = certificates_table(EPC_CERT_DIR)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    certs.to_parquet(INTERIM_DIR / "epc_certificates.parquet", index=False)
    latest = latest_per_building(certs, since=since) if not certs.empty else certs
    out = aggregate_sites(latest, site_postcodes(), prefix="epc")
    out.to_parquet(INTERIM_DIR / "epc_sites.parquet", index=False)
    logger.info(
        "EPC: %d certificates, %d current buildings, %d of %d sites with an EPC at the postcode",
        len(certs),
        len(latest),
        int((out["epc_buildings"] > 0).sum()),
        len(out),
    )
    return certs, out
