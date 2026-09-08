"""GB electricity price signals: Octopus Agile (day-ahead-known) and Elexon MID.

Why not ENTSO-E: GB left EU day-ahead market coupling in January 2021 and its
day-ahead prices stopped appearing on the Transparency Platform; the REST host
also answers 404 for every bidding zone at the time of writing.

What is "known beforehand" matters for a forecasting feature. The N2EX
day-ahead auction clears mid-morning on D-1 and Octopus publishes the derived
Agile rates that afternoon, so for a forecast issued on the evening of D-1 the
rates for D are known. For a forecast issued earlier in the day they are not -
check the issue time before using them as a feature.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from typing import Any

import pandas as pd

from ..config import INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

OCTOPUS_ROOT = "https://api.octopus.energy/v1"
ELEXON_ROOT = "https://data.elexon.co.uk/bmrs/api/v1"


def _get_json(url: str, timeout: int = 60) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "sense-energy/0.1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


# --------------------------------------------------------------------------- #
# Octopus Agile
# --------------------------------------------------------------------------- #


def octopus_product(code: str) -> dict[str, Any]:
    """Product metadata: on-sale window and the regional tariff codes."""
    return _get_json(f"{OCTOPUS_ROOT}/products/{code}/")


def fetch_agile_rates(
    product: str, region: str, start: str, end: str, page_size: int = 1500, pause: float = 0.15
) -> pd.DataFrame:
    """All half-hourly unit rates for one product/region in [start, end)."""
    tariff = f"E-1R-{product}-{region}"
    url = (
        f"{OCTOPUS_ROOT}/products/{product}/electricity-tariffs/{tariff}/standard-unit-rates/?"
        + urllib.parse.urlencode(
            {"period_from": f"{start}T00:00Z", "period_to": f"{end}T00:00Z", "page_size": page_size}
        )
    )
    rows: list[dict[str, Any]] = []
    while url:
        payload = _get_json(url)
        rows.extend(payload.get("results", []))
        url = payload.get("next")
        if url:
            time.sleep(pause)

    if not rows:
        return pd.DataFrame(
            columns=["datetime", "valid_to", "product", "region", "rate_exc_vat", "rate_inc_vat"]
        )
    frame = pd.DataFrame(rows)
    frame = frame.rename(
        columns={
            "valid_from": "datetime",
            "value_exc_vat": "rate_exc_vat",
            "value_inc_vat": "rate_inc_vat",
        }
    )
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)
    frame["valid_to"] = pd.to_datetime(frame["valid_to"], utc=True)
    frame["product"] = product
    frame["region"] = region
    return frame[["datetime", "valid_to", "product", "region", "rate_exc_vat", "rate_inc_vat"]]


def fetch_all_agile(config: dict[str, Any]) -> pd.DataFrame:
    """Every product x region over the window, in one long table."""
    oc = config["octopus"]
    frames = []
    for product in oc["products"]:
        for region in oc["regions"]:
            frame = fetch_agile_rates(
                product,
                region,
                config["start"],
                config["end"],
                oc.get("page_size", 1500),
                oc.get("pause_seconds", 0.15),
            )
            logger.info("Agile %s region %s: %s rates", product, region, f"{len(frame):,}")
            frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out = (
        out.drop_duplicates(["product", "region", "datetime"])
        .sort_values(["product", "region", "datetime"])
        .reset_index(drop=True)
    )
    return out


def on_sale_windows(products: list[str]) -> pd.DataFrame:
    """When each product was on sale - used to pick a canonical product per day."""
    rows = []
    for code in products:
        meta = octopus_product(code)
        rows.append(
            {
                "product": code,
                "available_from": pd.to_datetime(meta.get("available_from"), utc=True),
                "available_to": pd.to_datetime(meta.get("available_to"), utc=True),
            }
        )
    return pd.DataFrame(rows)


def canonical_agile(
    rates: pd.DataFrame, windows: pd.DataFrame, fallback: str = "AGILE-18-02-21"
) -> pd.DataFrame:
    """One rate per (region, datetime): the product on sale that day, else the fallback.

    Products differ in caps and multipliers, so mixing them within a series is
    a step change. The chosen product is kept as a column so that can be seen.
    """
    rates = rates.reset_index(drop=True)  # boolean .loc below needs unique labels
    order = {p: i for i, p in enumerate(windows.sort_values("available_from")["product"])}
    rates["_is_on_sale"] = False
    for _, w in windows.iterrows():
        mask = (rates["product"] == w["product"]) & (rates["datetime"] >= w["available_from"])
        if pd.notna(w["available_to"]):
            mask &= rates["datetime"] < w["available_to"]
        rates.loc[mask, "_is_on_sale"] = True
    rates["_rank"] = rates["product"].map(order).fillna(len(order))
    rates["_pref"] = (~rates["_is_on_sale"]).astype(int) * 10 + (
        rates["product"] != fallback
    ).astype(int)
    chosen = rates.sort_values(["region", "datetime", "_pref", "_rank"]).drop_duplicates(
        ["region", "datetime"], keep="first"
    )
    return chosen.drop(columns=["_is_on_sale", "_rank", "_pref"]).reset_index(drop=True)


def gsp_group_for_postcode(postcode: str) -> str | None:
    """Octopus' postcode -> Grid Supply Point group lookup ('_C' style)."""
    compact = "".join(postcode.split()).upper()
    payload = _get_json(
        f"{OCTOPUS_ROOT}/industry/grid-supply-points/?postcode={urllib.parse.quote(compact)}"
    )
    results = payload.get("results") or []
    return results[0]["group_id"].lstrip("_") if results else None


def map_sites_to_gsp(sites: pd.DataFrame, pause: float = 0.1) -> pd.DataFrame:
    """Add ``gsp_group`` (A-P) to a site table via each site's postcode."""
    sites = sites.copy()
    postcodes = sites["postcode"].dropna().str.strip().str.upper()
    lookup: dict[str, str | None] = {}
    for pc in sorted(set(postcodes)):
        try:
            lookup[pc] = gsp_group_for_postcode(pc)
        except Exception as exc:  # noqa: BLE001 - one bad postcode must not stop the rest
            logger.warning("GSP lookup failed for %s: %s", pc, exc)
            lookup[pc] = None
        time.sleep(pause)
    sites["gsp_group"] = sites["postcode"].str.strip().str.upper().map(lookup)
    logger.info(
        "GSP groups resolved for %d of %d sites", int(sites["gsp_group"].notna().sum()), len(sites)
    )
    return sites


# --------------------------------------------------------------------------- #
# Elexon Market Index Data
# --------------------------------------------------------------------------- #


def fetch_elexon_mid(
    start: str, end: str, window_days: int = 7, pause: float = 0.2
) -> pd.DataFrame:
    """APX and N2EX market index price/volume per settlement period, ex-post."""
    frames = []
    for chunk_start in pd.date_range(start, end, freq=f"{window_days}D"):
        chunk_end = min(chunk_start + pd.Timedelta(days=window_days), pd.Timestamp(end))
        url = f"{ELEXON_ROOT}/datasets/MID?from={chunk_start.date()}&to={chunk_end.date()}&format=json"
        data = _get_json(url).get("data", [])
        if data:
            frames.append(pd.DataFrame(data))
        time.sleep(pause)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out.rename(
        columns={
            "startTime": "datetime",
            "dataProvider": "provider",
            "settlementDate": "settlement_date",
            "settlementPeriod": "settlement_period",
            "price": "price_gbp_mwh",
            "volume": "volume_mwh",
        }
    )
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out = out[
        [
            "datetime",
            "settlement_date",
            "settlement_period",
            "provider",
            "price_gbp_mwh",
            "volume_mwh",
        ]
    ]
    return (
        out.drop_duplicates(["datetime", "provider"])
        .sort_values(["provider", "datetime"])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build_prices(config: dict[str, Any]) -> dict[str, Any]:
    """Pull both sources and write interim parquet files."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    written = {}

    rates = fetch_all_agile(config)
    raw_path = INTERIM_DIR / "prices_agile_all_products.parquet"
    rates.to_parquet(raw_path, index=False, compression="zstd")
    written["agile_all"] = raw_path

    windows = on_sale_windows(config["octopus"]["products"])
    canon = canonical_agile(rates, windows)
    canon_path = INTERIM_DIR / "prices_agile.parquet"
    canon.to_parquet(canon_path, index=False, compression="zstd")
    written["agile"] = canon_path
    logger.info(
        "Agile canonical: %s rows, %d regions, %s -> %s",
        f"{len(canon):,}",
        canon["region"].nunique(),
        canon["datetime"].min(),
        canon["datetime"].max(),
    )

    el = config.get("elexon", {})
    mid = fetch_elexon_mid(config["start"], config["end"], el.get("window_days", 7))
    mid_path = INTERIM_DIR / "prices_elexon_mid.parquet"
    mid.to_parquet(mid_path, index=False, compression="zstd")
    written["elexon_mid"] = mid_path
    logger.info("Elexon MID: %s rows", f"{len(mid):,}")
    return written
