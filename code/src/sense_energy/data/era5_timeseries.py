"""Decades of hourly ERA5 2 m temperature at the panel's grid cells, for the design-winter work.

The gridded ERA5 pull (``data/weather.py``) covers the demand record only (December 2022 on).
A design winter needs the long record. The Climate Data Store serves ERA5 as point time series
(dataset ``reanalysis-era5-single-levels-timeseries``): one request returns every hour since
1940 at the grid point nearest to a location, in minutes, with no field downloads. One request
per 0.25 degree cell that holds a panel site; cells are the same nearest grid points the gridded
extraction used, so the two records splice (checked on their overlap in ``splice_check``).

``fetch_all`` is resumable: a cell whose file exists is skipped. ``build`` writes

* ``interim/era5_long_hourly.parquet``: (cell, datetime UTC, t2m degC), and
* ``interim/weather_daily_long.parquet``: (site_code, date, t_mean, t_min, t_max) per
  ``Europe/London`` day, the long record followed by the gridded record from its first day.

Contains modified Copernicus Climate Change Service information (ERA5, CC BY 4.0).
"""

from __future__ import annotations

import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, GEO_DIR, INTERIM_DIR, PROCESSED_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

DATASET = "reanalysis-era5-single-levels-timeseries"
OUT_DIR = EXTERNAL_DIR / "weather" / "era5_timeseries"
START, END = "1960-01-01", "2022-12-31"
TZ = "Europe/London"
GRID = 0.25


def cell_id(lat: float, lon: float) -> str:
    return f"{lat:+07.2f}_{lon:+07.2f}"


def panel_cells() -> pd.DataFrame:
    """Every panel site (electricity sites and the sites of the gas meters) with the 0.25 degree
    grid point nearest to its postcode centroid."""
    g = pd.read_parquet(GEO_DIR / "sites_geo.parquet").dropna(subset=["latitude", "longitude"])
    elec = set(pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet").columns)
    gas = set(
        pd.read_parquet(PROCESSED_DIR / "gas" / "meter" / "series_index.parquet")["site_code"]
        .astype(str)
        .unique()
    )
    g = g[g["site_code"].isin(elec | gas)][["site_code", "latitude", "longitude"]].copy()
    g["cell_lat"] = (g["latitude"] / GRID).round() * GRID
    g["cell_lon"] = (g["longitude"] / GRID).round() * GRID
    g["cell"] = [cell_id(a, b) for a, b in zip(g["cell_lat"], g["cell_lon"], strict=True)]
    return g.reset_index(drop=True)


def fetch_cell(lat: float, lon: float, start: str = START, end: str = END) -> Path:
    from .weather import make_client

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{cell_id(lat, lon)}.zip"
    if path.exists() and path.stat().st_size > 0:
        return path
    tmp = path.with_suffix(".part")
    make_client().retrieve(
        DATASET,
        {
            "variable": ["2m_temperature"],
            "location": {"longitude": float(lon), "latitude": float(lat)},
            "date": [f"{start}/{end}"],
            "data_format": "csv",
        },
        str(tmp),
    )
    tmp.rename(path)
    logger.info("era5 time series: %s (%.1f MB)", path.name, path.stat().st_size / 1e6)
    return path


def fetch_all(start: str = START, end: str = END, workers: int = 4) -> int:
    """Fetch every panel cell not yet on disk; returns the number of cells that failed."""
    cells = panel_cells().drop_duplicates("cell")[["cell_lat", "cell_lon"]].to_numpy()
    failed = 0

    def one(c):
        try:
            fetch_cell(c[0], c[1], start, end)
            return 0
        except Exception as exc:  # one cell must not stop the rest
            logger.error("era5 time series: cell %s failed: %s", cell_id(c[0], c[1]), exc)
            return 1

    with ThreadPoolExecutor(max_workers=workers) as ex:
        failed = sum(ex.map(one, cells))
    logger.info("era5 time series: %d cells, %d failed", len(cells), failed)
    return failed


def read_cell(path: Path) -> pd.DataFrame:
    """Hourly (datetime UTC, t2m degC) of one cell file."""
    try:
        with zipfile.ZipFile(path) as z:
            d = pd.read_csv(z.open(z.namelist()[0]), usecols=["valid_time", "t2m"])
    except zipfile.BadZipFile:
        d = pd.read_csv(path, usecols=["valid_time", "t2m"])
    return pd.DataFrame(
        {
            "datetime": pd.to_datetime(d["valid_time"], utc=True),
            "t2m": (d["t2m"] - 273.15).astype("float32"),
        }
    )


def daily_stats(hourly: pd.DataFrame) -> pd.DataFrame:
    """Mean, minimum and maximum per ``Europe/London`` day of an hourly (datetime, t2m) table."""
    day = hourly["datetime"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    g = hourly.groupby(day)["t2m"]
    out = pd.DataFrame({"t_mean": g.mean(), "t_min": g.min(), "t_max": g.max(), "n": g.size()})
    out = out[out["n"] >= 23].drop(columns="n")  # whole days only (23 or 25 on clock changes)
    out.index.name = "date"
    return out.reset_index()


def splice_check(long_daily: pd.DataFrame) -> dict:
    """Agreement of the long record with the gridded record on the days both cover."""
    g = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_mean"]
    )
    m = long_daily.merge(g, on=["site_code", "date"], suffixes=("_long", "_grid"))
    if m.empty:
        return {"overlap_site_days": 0}
    d = m["t_mean_long"] - m["t_mean_grid"]
    return {
        "overlap_site_days": int(len(m)),
        "mean_diff_c": round(float(d.mean()), 4),
        "max_abs_diff_c": round(float(d.abs().max()), 4),
    }


def build() -> dict:
    """Cell files on disk -> the hourly cell table and the long daily table per site."""
    sites = panel_cells()
    hourly, daily = [], []
    for cell, grp in sites.groupby("cell"):
        path = OUT_DIR / f"{cell}.zip"
        if not path.exists():
            logger.warning("era5 time series: no file for cell %s (%d sites)", cell, len(grp))
            continue
        h = read_cell(path)
        hourly.append(h.assign(cell=cell))
        d = daily_stats(h)
        daily.extend(d.assign(site_code=s) for s in grp["site_code"])
    if not hourly:
        raise RuntimeError("no ERA5 time-series files on disk: run fetch-era5-timeseries first")
    hourly = pd.concat(hourly, ignore_index=True)
    hourly["cell"] = hourly["cell"].astype("category")
    hourly.to_parquet(INTERIM_DIR / "era5_long_hourly.parquet", index=False)
    long_daily = pd.concat(daily, ignore_index=True)[
        ["site_code", "date", "t_mean", "t_min", "t_max"]
    ]
    check = splice_check(long_daily)
    grid = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet",
        columns=["site_code", "date", "t_mean", "t_min", "t_max"],
    )
    grid = grid[grid["site_code"].isin(long_daily["site_code"].unique())]
    first = grid["date"].min()
    out = pd.concat([long_daily[long_daily["date"] < first], grid], ignore_index=True)
    out = out.sort_values(["site_code", "date"]).reset_index(drop=True)
    out.to_parquet(INTERIM_DIR / "weather_daily_long.parquet", index=False)
    sites[["site_code", "cell"]].to_parquet(INTERIM_DIR / "era5_long_site_cells.parquet")
    summary = {
        "cells": int(hourly["cell"].nunique()),
        "sites": int(out["site_code"].nunique()),
        "first_day": str(out["date"].min().date()),
        "last_day": str(out["date"].max().date()),
        "splice": check,
    }
    logger.info("era5 long record: %s", summary)
    return summary


def hourly_for_sites(sites: list[str]) -> pd.DataFrame:
    """Wide hourly temperature (degC, UTC index) per site from the long cell table, extended by
    the gridded record from December 2022 on."""
    cells = pd.read_parquet(INTERIM_DIR / "era5_long_site_cells.parquet").set_index("site_code")
    cells = cells.loc[[s for s in sites if s in cells.index], "cell"]
    h = pd.read_parquet(INTERIM_DIR / "era5_long_hourly.parquet")
    h = h[h["cell"].isin(cells.unique())]
    wide = h.pivot(index="datetime", columns="cell", values="t2m")
    long_part = pd.DataFrame({s: wide[c] for s, c in cells.items()})
    g = pd.read_parquet(
        INTERIM_DIR / "weather_reanalysis.parquet", columns=["site_code", "datetime", "t2m"]
    )
    g = g[g["site_code"].isin(long_part.columns)]
    grid = g.pivot(index="datetime", columns="site_code", values="t2m") - 273.15
    grid.index = pd.to_datetime(grid.index, utc=True)
    out = pd.concat([long_part[long_part.index < grid.index.min()], grid[long_part.columns]])
    return out.astype(np.float32)
