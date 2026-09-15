"""Public NHS activity statistics as trust-level covariates.

Pulled from the NHS England statistics site (direct file downloads; NHS
Digital publications block scripted access and are listed in
``doc/external_activity_data.md`` for manual retrieval):

* **A&E attendances and emergency admissions** - monthly, per provider (ODS
  code): attendances by department type, 4-hour breaches, emergency
  admissions. Acute and some community trusts.
* **AmbSYS** - monthly ambulance systems indicators per ambulance service:
  calls, incidents, response times.
* **KH03** - quarterly overnight bed availability and occupancy per trust
  (open time-series CSVs; later quarters as spreadsheets on the same page).

Everything lands in ``interim/nhs_activity_<source>.parquet`` keyed by ODS
code and period, plus ``interim/nhs_activity_trusts.parquet`` mapping the
demand panel's trusts to ODS codes. Monthly and quarterly series are slow
covariates: level shifts, kWh-per-attendance normalisation and structure, not
day-ahead dynamics.
"""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd
import requests

from ..config import INTERIM_DIR, PROCESSED_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

BASE = "https://www.england.nhs.uk/statistics/statistical-work-areas"
AE_PAGES = {
    "2023-24": f"{BASE}/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2023-24/",
    "2024-25": f"{BASE}/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2024-25/",
    "2025-26": f"{BASE}/ae-waiting-times-and-activity/ae-attendances-and-emergency-admissions-2025-26/",
}
AMBSYS_PAGE = f"{BASE}/ambulance-quality-indicators/"
KH03_CSVS = {
    "available": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2024/09/KH03-Available-Overnight-only.csv",
    "occupied": "https://www.england.nhs.uk/statistics/wp-content/uploads/sites/2/2024/09/KH03-Occupied-Overnight-only.csv",
}
UA = {"User-Agent": "sense-energy research (nicolas.malz05@googlemail.com)"}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"


def _get(url: str) -> bytes:
    r = requests.get(url, headers=UA, timeout=120)
    r.raise_for_status()
    return r.content


def _csv_links(page_html: str) -> list[str]:
    return sorted(
        set(
            re.findall(
                r'href="(https://www\.england\.nhs\.uk/statistics/wp-content/uploads/[^"]+\.csv)"',
                page_html,
            )
        )
    )


def _norm(name: str) -> str:
    return re.sub(r"[^A-Z0-9 ]", "", str(name).upper()).replace("  ", " ").strip()


def fetch_ae_monthly() -> pd.DataFrame:
    """Every monthly provider-level A&E CSV linked from the financial-year pages."""
    frames = []
    for fy, page in AE_PAGES.items():
        html = _get(page).decode("utf-8", "ignore")
        links = [u for u in _csv_links(html) if re.search(r"(Monthly-AE|CSV)", u, re.I)]
        for url in links:
            try:
                f = pd.read_csv(io.BytesIO(_get(url)))
            except Exception as exc:  # noqa: BLE001
                logger.warning("skip %s: %s", url, exc)
                continue
            if "Org Code" not in f.columns or "Period" not in f.columns:
                continue
            f["source_file"] = url.rsplit("/", 1)[-1]
            f["financial_year"] = fy
            frames.append(f)
        logger.info("A&E %s: %d files", fy, len(links))
    out = pd.concat(frames, ignore_index=True)
    label = out["Period"].astype(str).str.replace("MSitAE-", "", regex=False).str.title()
    out["period"] = pd.to_datetime(label, format="%B-%Y", errors="coerce")
    out = out[out["period"].notna()]  # drops the TOTAL rows
    out = out.drop(columns=["Period"]).rename(
        columns={"Org Code": "ods_code", "Org name": "org_name", "Parent Org": "parent_org"}
    )
    out = out[
        [
            c
            for c in out.columns
            if not str(c).lower().startswith("unnamed") and str(c).strip() not in ("", "a")
        ]
    ]
    out.columns = [
        re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_")
        if c
        not in ("ods_code", "org_name", "parent_org", "period", "source_file", "financial_year")
        else c
        for c in out.columns
    ]
    out = out.drop_duplicates(subset=["ods_code", "period"], keep="last")
    out = out[out["ods_code"].astype(str).str.len().between(3, 5)]
    out.to_parquet(INTERIM_DIR / "nhs_activity_ae_monthly.parquet", index=False)
    logger.info(
        "A&E monthly: %d provider-months, %s -> %s",
        len(out),
        out["period"].min(),
        out["period"].max(),
    )
    return out


def fetch_ambsys() -> pd.DataFrame:
    html = _get(AMBSYS_PAGE).decode("utf-8", "ignore")
    links = [u for u in _csv_links(html) if "AmbSYS-to-" in u]
    if not links:
        raise RuntimeError("AmbSYS CSV link not found")
    f = pd.read_csv(io.BytesIO(_get(links[-1])))
    f = f.rename(columns={"Org Code": "ods_code", "Org Name": "org_name"})
    f["period"] = pd.to_datetime(
        f["Year"].astype(str) + "-" + f["Month"].astype(str).str.zfill(2) + "-01"
    )
    f.to_parquet(INTERIM_DIR / "nhs_activity_ambsys_monthly.parquet", index=False)
    logger.info(
        "AmbSYS: %d rows, %d services, %s -> %s",
        len(f),
        f["ods_code"].nunique(),
        f["period"].min(),
        f["period"].max(),
    )
    return f


def fetch_kh03() -> pd.DataFrame:
    parts = []
    for kind, url in KH03_CSVS.items():
        f = pd.read_csv(io.BytesIO(_get(url)), encoding="utf-8-sig")
        f = f.rename(
            columns={
                "Organisation_Code": "ods_code",
                "Number_Of_Beds": f"beds_{kind}",
                "Sector": "sector",
                "Effective_Snapshot_Date": "snapshot",
            }
        )
        parts.append(f)
    out = parts[0].merge(parts[1], on=["ods_code", "sector", "snapshot"], how="outer")
    out["snapshot"] = pd.to_datetime(out["snapshot"], dayfirst=True)
    for c in ("beds_available", "beds_occupied"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["occupancy"] = out["beds_occupied"] / out["beds_available"]
    out.to_parquet(INTERIM_DIR / "nhs_activity_kh03_quarterly.parquet", index=False)
    logger.info(
        "KH03: %d rows, %d orgs, %s -> %s",
        len(out),
        out["ods_code"].nunique(),
        out["snapshot"].min(),
        out["snapshot"].max(),
    )
    return out


def trust_mapping(ae: pd.DataFrame, amb: pd.DataFrame, kh: pd.DataFrame) -> pd.DataFrame:
    """The demand panel's trusts -> ODS codes, by site-code prefix or by normalised name."""
    site = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "series_index.parquet")
    names = pd.concat(
        [ae[["ods_code", "org_name"]], amb[["ods_code", "org_name"]]]
    ).drop_duplicates()
    names["key"] = names["org_name"].map(_norm)
    by_name = names.drop_duplicates("key").set_index("key")["ods_code"]
    rows = []
    for trust, g in site.groupby("organisation_name"):
        prefixes = g["series_id"].str[:3].unique()
        known = {*ae["ods_code"], *amb["ods_code"], *kh["ods_code"]}
        code = next((p for p in prefixes if p in known), None)
        if code is None:
            code = by_name.get(_norm(trust))
        rows.append(
            {
                "organisation_name": trust,
                "ods_code": code,
                "n_sites": len(g),
                "site_prefixes": ",".join(prefixes),
            }
        )
    out = pd.DataFrame(rows)
    # ambulance services report under their own AmbSYS code; match on the first three words
    amb_names = (
        amb.drop_duplicates("ods_code")
        .set_index("ods_code")["org_name"]
        .map(lambda n: " ".join(_norm(n).split()[:3]))
    )
    out["ambsys_code"] = [
        next((c for c, n in amb_names.items() if n == " ".join(_norm(t).split()[:3])), None)
        if "AMBULANCE" in str(t).upper()
        else None
        for t in out["organisation_name"]
    ]
    for src, frame in (("ae", ae), ("kh03", kh)):
        out[f"in_{src}"] = out["ods_code"].isin(set(frame["ods_code"]))
    out["in_ambsys"] = out["ambsys_code"].notna()
    out.to_parquet(INTERIM_DIR / "nhs_activity_trusts.parquet", index=False)
    logger.info(
        "trust mapping: %d trusts, %d with an ODS code; in A&E %d, AmbSYS %d, KH03 %d",
        len(out),
        out["ods_code"].notna().sum(),
        out["in_ae"].sum(),
        out["in_ambsys"].sum(),
        out["in_kh03"].sum(),
    )
    return out


def fetch_all(config: dict[str, Any] | None = None) -> pd.DataFrame:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    ae, amb, kh = fetch_ae_monthly(), fetch_ambsys(), fetch_kh03()
    return trust_mapping(ae, amb, kh)
