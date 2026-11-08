"""NOAA Integrated Surface Database (ISD): hourly observations from the Met Office synoptic
and airfield stations, as an independent thermometer for the sites.

Files: one CSV per station and year from NOAA's open-data bucket
(``https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{USAF}{WBAN}.csv``, no key), and the
station history ``isd-history.csv``. Stations are those filed under country UK or GB that are
among the ``K_NEAREST`` stations within ``MAX_KM`` of some located site and whose record runs
into 2025 (the history snapshot's last update).

Parsing: ``TMP`` is ``+0123,1`` - tenths of degC and a quality code; ``+9999`` is missing; quality
codes 0, 1, 4, 5 and 9 pass, the rest (suspect, erroneous) are dropped. Daily and monthly summary
report types are skipped. Observations are averaged within the clock hour.

Site series: for each site and hour, the nearest station among its candidates that reported
that hour. Days with fewer than ``MIN_HOURS`` hours are dropped.

Outputs: ``external/noaa_isd/raw/{year}/{id}.csv`` (as received), ``interim/isd_hourly.parquet``
(station_id, datetime UTC, t_c, td_c), ``interim/isd_site_hourly.parquet`` (site_code, datetime,
t_c, station_id, distance_km) and ``interim/weather_daily_isd.parquet`` (site_code, date,
t_mean, t_min, t_max, t_eve, n_hours, distance_km).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, GEO_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

ISD_DIR = EXTERNAL_DIR / "noaa_isd"
BUCKET = "https://noaa-global-hourly-pds.s3.amazonaws.com"
HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
COUNTRIES = {"UK", "GB"}
BOX = {"lat": (49.8, 55.9), "lon": (-6.5, 2.0)}
K_NEAREST, MAX_KM = 3, 60.0
MIN_HOURS = 18
GOOD_QUALITY = {"0", "1", "4", "5", "9"}
SKIP_REPORT_TYPES = {"SOD", "SOM"}
TZ = "Europe/London"
EVENING_FROM = 18


def haversine_km(lat1, lon1, lat2, lon2) -> np.ndarray:
    la1, lo1, la2, lo2 = (np.radians(np.asarray(x, float)) for x in (lat1, lon1, lat2, lon2))
    a = (
        np.sin((la1[:, None] - la2[None, :]) / 2) ** 2
        + np.cos(la1)[:, None]
        * np.cos(la2)[None, :]
        * np.sin((lo1[:, None] - lo2[None, :]) / 2) ** 2
    )
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def station_table(path: Path | None = None, active_from: str = "2025-01-01") -> pd.DataFrame:
    """UK stations in the England box with records running past ``active_from``."""
    path = path or ISD_DIR / "isd-history.csv"
    h = pd.read_csv(path, dtype=str)
    h.columns = [c.strip().lower().replace(" ", "_") for c in h.columns]
    h = h[h["ctry"].isin(COUNTRIES) & h["lat"].notna() & h["lon"].notna()].copy()
    h["lat"], h["lon"] = h["lat"].astype(float), h["lon"].astype(float)
    h["begin"] = pd.to_datetime(h["begin"], format="%Y%m%d", errors="coerce")
    h["end"] = pd.to_datetime(h["end"], format="%Y%m%d", errors="coerce")
    h = h[
        h["lat"].between(*BOX["lat"])
        & h["lon"].between(*BOX["lon"])
        & (h["end"] >= active_from)
        & (h["begin"] <= "2022-12-01")
    ]
    h["station_id"] = h["usaf"] + h["wban"]
    return h[["station_id", "station_name", "lat", "lon", "begin", "end"]].reset_index(drop=True)


def candidates(sites: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """The K_NEAREST stations within MAX_KM of each site, ranked by distance."""
    km = haversine_km(sites["latitude"], sites["longitude"], stations["lat"], stations["lon"])
    rows = []
    for i, site in enumerate(sites["site_code"]):
        order = np.argsort(km[i])[:K_NEAREST]
        for rank, j in enumerate(order):
            if km[i, j] <= MAX_KM:
                rows.append(
                    {
                        "site_code": site,
                        "station_id": stations["station_id"].iloc[j],
                        "rank": rank,
                        "distance_km": float(km[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def _download(url: str, dest: Path) -> bool:
    import requests

    r = requests.get(url, timeout=120)
    if r.status_code == 404:
        return False
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def fetch(years: list[int], station_ids: list[str], max_workers: int = 6) -> dict[str, int]:
    """Download every station-year file not already present. Missing files (a station without
    data that year) are recorded as empty markers so they are not retried."""
    jobs = [(y, s) for y in years for s in station_ids]
    pending = [
        (y, s)
        for y, s in jobs
        if not (ISD_DIR / "raw" / str(y) / f"{s}.csv").exists()
        and not (ISD_DIR / "raw" / str(y) / f"{s}.missing").exists()
    ]
    logger.info("ISD: %d station-years, %d to fetch", len(jobs), len(pending))
    got, missing = 0, 0
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _download, f"{BUCKET}/{y}/{s}.csv", ISD_DIR / "raw" / str(y) / f"{s}.csv"
            ): (y, s)
            for y, s in pending
        }
        for f in as_completed(futures):
            y, s = futures[f]
            if f.result():
                got += 1
            else:
                missing += 1
                (ISD_DIR / "raw" / str(y) / f"{s}.missing").touch()
    logger.info("ISD: fetched %d files, %d station-years have no file", got, missing)
    return {"fetched": got, "missing": missing, "present": len(jobs) - len(pending)}


def _field(series: pd.Series) -> pd.Series:
    """``+0123,1`` -> 12.3 degC where the quality code passes, else NaN."""
    parts = series.fillna("+9999,9").str.split(",", n=1, expand=True)
    value = pd.to_numeric(parts[0], errors="coerce") / 10.0
    ok = parts[1].isin(GOOD_QUALITY) & (parts[0] != "+9999")
    return value.where(ok)


def parse_file(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, usecols=["STATION", "DATE", "REPORT_TYPE", "TMP", "DEW"], dtype=str)
    raw = raw[~raw["REPORT_TYPE"].isin(SKIP_REPORT_TYPES)]
    out = pd.DataFrame(
        {
            "station_id": raw["STATION"].str.zfill(11),
            "datetime": pd.to_datetime(raw["DATE"], utc=True, errors="coerce").dt.floor("h"),
            "t_c": _field(raw["TMP"]),
            "td_c": _field(raw["DEW"]),
        }
    ).dropna(subset=["datetime", "t_c"])
    return out.groupby(["station_id", "datetime"], as_index=False)[["t_c", "td_c"]].mean()


def build_hourly() -> pd.DataFrame:
    paths = sorted((ISD_DIR / "raw").glob("*/*.csv"))
    frames = [parse_file(p) for p in paths]
    hourly = pd.concat(frames, ignore_index=True).sort_values(["station_id", "datetime"])
    hourly.to_parquet(INTERIM_DIR / "isd_hourly.parquet", index=False, compression="zstd")
    logger.info(
        "ISD hourly: %s rows, %d stations", f"{len(hourly):,}", hourly["station_id"].nunique()
    )
    return hourly


def site_hourly(hourly: pd.DataFrame, cand: pd.DataFrame) -> pd.DataFrame:
    """Each site's hourly temperature from the nearest candidate station that reported."""
    wide = hourly.pivot(index="datetime", columns="station_id", values="t_c")
    frames = []
    for site, c in cand.sort_values(["site_code", "rank"]).groupby("site_code", sort=False):
        t = pd.Series(np.nan, index=wide.index, dtype=float)
        src = pd.Series("", index=wide.index, dtype=object)
        dist = pd.Series(np.nan, index=wide.index, dtype=float)
        for r in c.itertuples():
            if r.station_id not in wide:
                continue
            fill = t.isna() & wide[r.station_id].notna()
            t[fill] = wide.loc[fill, r.station_id]
            src[fill] = r.station_id
            dist[fill] = r.distance_km
        frames.append(
            pd.DataFrame(
                {
                    "site_code": site,
                    "datetime": wide.index,
                    "t_c": t.to_numpy(),
                    "station_id": src.to_numpy(),
                    "distance_km": dist.to_numpy(),
                }
            ).dropna(subset=["t_c"])
        )
    return pd.concat(frames, ignore_index=True)


def daily(site_h: pd.DataFrame) -> pd.DataFrame:
    local = site_h["datetime"].dt.tz_convert(TZ)
    x = site_h.assign(date=local.dt.normalize().dt.tz_localize(None), hour=local.dt.hour)
    x["t_eve"] = x["t_c"].where(x["hour"] >= EVENING_FROM)
    g = x.groupby(["site_code", "date"])
    out = pd.concat(
        [
            g["t_c"].agg(t_mean="mean", t_min="min", t_max="max", n_hours="size"),
            g["t_eve"].mean().rename("t_eve"),
            g["distance_km"].mean().rename("distance_km"),
        ],
        axis=1,
    ).reset_index()
    return out[out["n_hours"] >= MIN_HOURS].reset_index(drop=True)


def build() -> pd.DataFrame:
    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet").dropna(subset=["latitude", "longitude"])
    stations = station_table()
    cand = candidates(sites, stations)
    cand.to_parquet(INTERIM_DIR / "isd_site_stations.parquet", index=False)
    hourly = build_hourly()
    sh = site_hourly(hourly, cand)
    sh.to_parquet(INTERIM_DIR / "isd_site_hourly.parquet", index=False, compression="zstd")
    out = daily(sh)
    out.to_parquet(INTERIM_DIR / "weather_daily_isd.parquet", index=False, compression="zstd")
    logger.info(
        "ISD daily: %s site-days, %d sites, median station distance %.1f km",
        f"{len(out):,}",
        out["site_code"].nunique(),
        float(out["distance_km"].median()),
    )
    return out
