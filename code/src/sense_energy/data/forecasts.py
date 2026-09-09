"""ECMWF AIFS-ENS forecasts from ECMWF open data.

ECMWF publishes its operational forecasts under CC BY 4.0 with a rolling
archive of about four days. That shapes everything here: this is a *harvester*
that runs daily and keeps whatever it can reach, not a historical pull. The
51 members (control + 50 perturbed) of every kept run are cropped to the site
bounding box on arrival and stored as one small NetCDF per run.

Global GRIB fields for 50 members are large (~30 MB per parameter per step);
the open-data client fetches only the requested parameters by byte range, and
the crop happens in memory before anything is written.

Attribution: "Contains ECMWF open data, CC BY 4.0".
"""

from __future__ import annotations

import datetime as dt
import re
import shutil
import urllib.request
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

OPEN_DATA_ROOT = "https://data.ecmwf.int/forecasts"

#: Scalar GRIB coordinates that differ between parameter groups and block merging.
_LEVEL_COORDS = (
    "heightAboveGround",
    "surface",
    "entireAtmosphere",
    "meanSea",
    "depthBelowLandLayer",
)


def output_dir(config: dict[str, Any]) -> Path:
    return EXTERNAL_DIR / config.get("output_subdir", "weather/aifs_ens")


def run_path(date: str, time: str, config: dict[str, Any]) -> Path:
    return output_dir(config) / f"{date}T{time}z.nc"


def list_available_runs(config: dict[str, Any]) -> list[tuple[str, str]]:
    """(YYYYMMDD, HH) pairs currently in the open-data archive for this model."""
    with urllib.request.urlopen(f"{OPEN_DATA_ROOT}/", timeout=60) as r:  # noqa: S310
        dates = sorted(set(re.findall(r"(20\d{6})/", r.read().decode())))
    runs = []
    for date in dates:
        with urllib.request.urlopen(f"{OPEN_DATA_ROOT}/{date}/", timeout=60) as r:  # noqa: S310
            times = sorted(set(re.findall(r"(\d{2})z/", r.read().decode())))
        for time in times:
            if time not in config["run_times"]:
                continue
            with urllib.request.urlopen(f"{OPEN_DATA_ROOT}/{date}/{time}z/", timeout=60) as r:  # noqa: S310
                if config["model"] in r.read().decode():
                    runs.append((date, time))
    return runs


def _open_grib_merged(path: Path):
    """Open a GRIB holding mixed surface parameters as one Dataset.

    cfgrib refuses to put 2 m, 10 m and surface fields in one dataset because
    their level coordinates disagree; open the compatible groups and merge
    after dropping those scalar coordinates.
    """
    import cfgrib
    import xarray as xr

    with warnings.catch_warnings():
        # cfgrib's internal merge trips xarray's compat-default FutureWarning
        # once per message group; it is not actionable here.
        warnings.simplefilter("ignore", FutureWarning)
        groups = cfgrib.open_datasets(str(path), backend_kwargs={"indexpath": ""})
    cleaned = []
    for ds in groups:
        ds = ds.drop_vars([c for c in _LEVEL_COORDS if c in ds.coords], errors="ignore")
        if "number" not in ds.dims:
            # The control run carries no member dimension; it is member 0.
            ds = ds.expand_dims(number=[0])
        cleaned.append(ds)
    # compat="override" would take each variable from the first group only -
    # the control run - and silently drop the 50 perturbed members. The groups
    # have disjoint member coordinates, so no_conflicts merges them by outer join.
    return xr.merge(cleaned, compat="no_conflicts", join="outer")


def _read_grib_cropped(path: Path, area: list[float]):
    """Decode a regular-grid GRIB with ecCodes directly and crop to ``area``.

    cfgrib decodes every global field into xarray before any crop; on 51
    members x 12 parameters that is minutes per step. Reading the messages
    with ecCodes and slicing the flat array to the box takes seconds.
    """
    import eccodes
    import xarray as xr

    north, west, south, east = area
    fields: dict[str, dict[int, np.ndarray]] = {}
    lat = lon = None
    rows = cols = None
    with path.open("rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                if lat is None:
                    ni, nj = eccodes.codes_get(gid, "Ni"), eccodes.codes_get(gid, "Nj")
                    lat0, dlat = (
                        eccodes.codes_get(gid, "latitudeOfFirstGridPointInDegrees"),
                        eccodes.codes_get(gid, "jDirectionIncrementInDegrees"),
                    )
                    lon0, dlon = (
                        eccodes.codes_get(gid, "longitudeOfFirstGridPointInDegrees"),
                        eccodes.codes_get(gid, "iDirectionIncrementInDegrees"),
                    )
                    scan_neg_j = eccodes.codes_get(gid, "jScansPositively") == 0
                    lats = (
                        lat0 - np.arange(nj) * dlat if scan_neg_j else lat0 + np.arange(nj) * dlat
                    )
                    lons = ((lon0 + np.arange(ni) * dlon) + 180.0) % 360.0 - 180.0
                    rows = np.flatnonzero((lats <= north) & (lats >= south))
                    cols = np.flatnonzero((lons >= west) & (lons <= east))
                    lat, lon = lats[rows], lons[cols]
                name = eccodes.codes_get(gid, "shortName")
                name = {"2t": "t2m", "2d": "d2m", "10u": "u10", "10v": "v10"}.get(name, name)
                number = (
                    int(eccodes.codes_get(gid, "perturbationNumber"))
                    if eccodes.codes_get(gid, "dataType") == "pf"
                    else 0
                )
                values = eccodes.codes_get_values(gid).reshape(nj, ni)
                fields.setdefault(name, {})[number] = values[np.ix_(rows, cols)].astype("float32")
            finally:
                eccodes.codes_release(gid)
    members = sorted({m for v in fields.values() for m in v})
    data = {
        name: (("number", "latitude", "longitude"), np.stack([grid[m] for m in members]))
        for name, grid in fields.items()
        if set(grid) == set(members)
    }
    return xr.Dataset(data, coords={"number": members, "latitude": lat, "longitude": lon})


def _crop(ds, area: list[float]):
    north, west, south, east = area
    lon = ds["longitude"]
    if float(lon.max()) > 180:
        ds = ds.assign_coords(longitude=(((lon + 180) % 360) - 180)).sortby("longitude")
    lat_desc = bool(ds["latitude"][0] > ds["latitude"][-1])
    return ds.sel(
        latitude=slice(north, south) if lat_desc else slice(south, north),
        longitude=slice(west, east),
    )


def harvest_run(date: str, time: str, config: dict[str, Any]) -> Path:
    """Fetch one run, all members, crop to the area, write a NetCDF. Idempotent."""
    import xarray as xr
    from ecmwf.opendata import Client

    out = run_path(date, time, config)
    if out.exists() and out.stat().st_size > 0:
        logger.info("%s already harvested", out.name)
        return out

    client = Client(source="ecmwf", model=config["model"], resol=config["resolution"])
    work = out.with_suffix(".work")
    work.mkdir(parents=True, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    def fetch(step: int) -> Path:
        grib = work / f"{step:03d}h.grib2"
        client.retrieve(
            date=date,
            time=int(time),
            stream=config["stream"],
            type=list(config["types"]),
            step=step,
            param=list(config["params"]),
            target=str(grib),
        )
        return grib

    per_step = []
    try:
        steps = list(config["steps"])
        logger.info(
            "AIFS-ENS %sT%sz: fetching %d steps x %d params x %s, %d in parallel",
            date,
            time,
            len(steps),
            len(config["params"]),
            config["types"],
            int(config.get("download_workers", 4)),
        )
        with ThreadPoolExecutor(max_workers=int(config.get("download_workers", 4))) as pool:
            gribs = list(pool.map(fetch, steps))
        for step, grib in zip(steps, gribs, strict=True):
            try:
                ds = _read_grib_cropped(grib, config["area"])
            except Exception as exc:  # noqa: BLE001 - fall back to cfgrib on anything unexpected
                logger.warning("ecCodes read failed for step %d (%s); using cfgrib", step, exc)
                ds = _crop(_open_grib_merged(grib), config["area"]).load()
                ds = ds.drop_vars(
                    [c for c in ("time", "valid_time") if c in ds.coords], errors="ignore"
                )
            per_step.append(
                ds.expand_dims(step=[pd.Timedelta(hours=step)]) if "step" not in ds.dims else ds
            )
            grib.unlink(missing_ok=True)
        run = xr.concat(per_step, dim="step").sortby("number")
        # Any scalar level coordinate that survived the per-group drop is noise.
        run = run.drop_vars([c for c in run.coords if c not in run.dims], errors="ignore")
        run = run.assign_coords(
            run_time=pd.Timestamp(f"{date}T{time}:00", tz="UTC").tz_convert(None)
        )
        run.attrs.update(
            source="ECMWF open data, AIFS-ENS",
            licence="CC BY 4.0",
            attribution="Contains ECMWF open data",
        )
        encoding = {v: {"zlib": True, "complevel": 4} for v in run.data_vars}
        run.to_netcdf(out, encoding=encoding)
        logger.info(
            "Wrote %s: %d members x %d steps x %d vars (%.1f MB)",
            out.name,
            run.sizes["number"],
            run.sizes["step"],
            len(run.data_vars),
            out.stat().st_size / 1e6,
        )
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return out


def harvest_available(config: dict[str, Any]) -> list[Path]:
    """Harvest every run in the archive not yet on disk. Safe to run on a schedule.

    A lock file makes a second concurrent invocation (cron firing while a
    manual run is still going) exit at once instead of racing on the same
    work directory.
    """
    import fcntl

    out_dir = output_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    lock_path = out_dir / ".harvest.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            logger.warning("Another harvest holds %s; exiting", lock_path)
            return []

        runs = list_available_runs(config)
        logger.info(
            "%d %s runs in the open-data archive at %s",
            len(runs),
            config["model"],
            config["run_times"],
        )
        written = []
        for date, time in runs:
            try:
                written.append(harvest_run(date, time, config))
            except Exception as exc:  # noqa: BLE001 - one bad run must not stop the rest
                logger.error("Run %sT%sz failed: %s", date, time, exc)
        return written


def extract_sites(nc_path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-point values per site, long over (member, step)."""
    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        picked = ds.sel(
            latitude=xr.DataArray(sites["latitude"].to_numpy(), dims="site"),
            longitude=xr.DataArray(sites["longitude"].to_numpy(), dims="site"),
            method="nearest",
        ).assign_coords(site=("site", sites["site_code"].to_numpy()))
        run_time = pd.Timestamp(picked["run_time"].values)
        frame = picked.to_dataframe().reset_index()

    frame = frame.rename(columns={"site": "site_code", "number": "member"})
    frame["run_time"] = run_time.tz_localize("UTC")
    frame["valid_time"] = frame["run_time"] + frame["step"]
    frame["step_hours"] = (frame["step"] / pd.Timedelta(hours=1)).astype(int)
    drop = [c for c in ("latitude", "longitude", "step") if c in frame.columns]
    frame = frame.drop(columns=drop)
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member"]
    return frame[keys + [c for c in frame.columns if c not in keys]]


def build_forecast_sites(config: dict[str, Any], sites: pd.DataFrame) -> Path:
    """Extract every harvested run into one parquet under interim/."""
    located = sites[sites["latitude"].notna() & sites["longitude"].notna()]
    paths = sorted(output_dir(config).glob("*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"No harvested runs under {output_dir(config)}. Run harvest-forecasts first."
        )
    frames = [extract_sites(p, located) for p in paths]
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["site_code", "run_time", "step_hours", "member"]
    )
    out_path = INTERIM_DIR / "forecast_aifs_ens.parquet"
    out.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "Wrote %s (%s rows, %d runs)", out_path.name, f"{len(out):,}", out["run_time"].nunique()
    )
    return out_path


def today_utc() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%d")
