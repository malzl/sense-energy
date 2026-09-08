"""ECMWF IFS ENS historical forecasts from TIGGE, via the ECMWF Web API.

TIGGE archives every operational ensemble run since 2006 at 0.5 deg, which
makes it the source for *what the forecast said* over the training window -
the quantity a day-ahead demand model would really have had, as opposed to the
reanalysis in :mod:`weather` (what the weather turned out to be).

One MARS request per (type, month); each lands as ``<type>/<YYYY-MM>.nc`` and
is skipped on re-run if it opens cleanly. MARS queues are slow and opaque:
minutes to an hour per request is normal, so this runs in the background.

Credentials: ``ECMWF_API_URL`` / ``ECMWF_API_KEY`` / ``ECMWF_API_EMAIL`` in
``.env``. The account must additionally have accepted the TIGGE licence.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR
from ..logging_utils import get_logger
from .weather import month_range

logger = get_logger(__name__)

TIGGE_LICENCE_URL = "http://apps.ecmwf.int/datasets/licences/tigge"


def output_dir(config: dict[str, Any]) -> Path:
    return EXTERNAL_DIR / config.get("output_subdir", "weather/tigge_ens")


def target_path(product_type: str, year: int, month: int, config: dict[str, Any]) -> Path:
    suffix = ".nc" if config.get("format", "netcdf") == "netcdf" else ".grib"
    return output_dir(config) / product_type / f"{year}-{month:02d}{suffix}"


def build_request(
    product_type: str, year: int, month: int, config: dict[str, Any], target: Path
) -> dict[str, str]:
    """The MARS request for one type/month. Lists are joined with '/' as MARS expects."""
    if product_type not in ("cf", "pf"):
        raise KeyError(f"product_type must be 'cf' or 'pf', got '{product_type}'")
    last_day = pd.Period(f"{year}-{month:02d}").days_in_month
    request = {
        "class": "ti",
        "dataset": config.get("dataset", "tigge"),
        "expver": config.get("expver", "prod"),
        "origin": config.get("origin", "ecmf"),
        "levtype": "sfc",
        "type": product_type,
        "param": "/".join(str(p) for p in config["params"]),
        "date": f"{year}-{month:02d}-01/to/{year}-{month:02d}-{last_day:02d}",
        "time": "/".join(config["times"]),
        "step": "/".join(str(s) for s in config["steps"]),
        "grid": config.get("grid", "0.5/0.5"),
        "area": config["area"],
        "format": config.get("format", "netcdf"),
        "target": str(target),
    }
    if product_type == "pf":
        request["number"] = config.get("members", "1/to/50")
    return request


def _opens_cleanly(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        import xarray as xr

        engine = "cfgrib" if path.suffix == ".grib" else None
        with xr.open_dataset(path, engine=engine) as ds:
            return len(ds.data_vars) > 0
    except Exception:  # noqa: BLE001 - any failure means re-download
        logger.warning("%s does not open cleanly; will re-download", path.name)
        return False


def make_server():
    from ecmwfapi import ECMWFDataServer

    url = os.getenv("ECMWF_API_URL", "https://api.ecmwf.int/v1")
    key, email = os.getenv("ECMWF_API_KEY", ""), os.getenv("ECMWF_API_EMAIL", "")
    if not key or not email:
        raise RuntimeError("ECMWF_API_KEY / ECMWF_API_EMAIL are not set in .env")
    return ECMWFDataServer(url=url, key=key, email=email, verbose=False)


def fetch_month(
    product_type: str, year: int, month: int, config: dict[str, Any], server=None
) -> Path:
    """Retrieve one type/month if not present. Returns the file path."""
    path = target_path(product_type, year, month, config)
    if _opens_cleanly(path):
        logger.info("%s/%s already present", product_type, path.name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    server = server or make_server()
    logger.info("MARS request: tigge %s %d-%02d", product_type, year, month)
    try:
        server.retrieve(build_request(product_type, year, month, config, tmp))
    except Exception as exc:
        message = str(exc)
        if "not access to datasets/tigge" in message or "accept the terms" in message:
            raise RuntimeError(
                f"The ECMWF account has not accepted the TIGGE licence. Accept it at {TIGGE_LICENCE_URL} and re-run."
            ) from exc
        raise
    if not _opens_cleanly(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"TIGGE download for {product_type} {year}-{month:02d} is unreadable")
    tmp.replace(path)
    logger.info("Wrote %s/%s (%.1f MB)", product_type, path.name, path.stat().st_size / 1e6)
    return path


def fetch_all(config: dict[str, Any], product_types: list[str] | None = None) -> list[Path]:
    """Every (type, month) in the config, a couple of MARS requests at a time. Resumable."""
    types = product_types or list(config["types"])
    jobs = [(t, y, m) for t in types for (y, m) in month_range(config["start"], config["end"])]
    pending = [j for j in jobs if not _opens_cleanly(target_path(*j, config))]
    logger.info(
        "%d of %d type-months present; requesting %d",
        len(jobs) - len(pending),
        len(jobs),
        len(pending),
    )

    done = [target_path(*j, config) for j in jobs if j not in pending]
    failures = []
    server = make_server()
    with ThreadPoolExecutor(max_workers=int(config.get("max_workers", 2))) as pool:
        futures = {
            pool.submit(fetch_month, t, y, m, config, server): (t, y, m) for (t, y, m) in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                failures.append(job)
                logger.error("Failed tigge %s %d-%02d: %s", *job, exc)
    if failures:
        logger.error("%d type-months failed: %s", len(failures), failures)
    return done


# --------------------------------------------------------------------------- #
# Extraction to sites
# --------------------------------------------------------------------------- #

_DROP = ("latitude", "longitude", "surface", "heightAboveGround", "expver")


def extract_sites(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-point values per site, long over (run_time, step, member).

    MARS NetCDF carries the run in ``time`` and the lead in ``step``; ``number``
    is present for perturbed members and absent for the control, which becomes
    member 0.
    """
    import xarray as xr

    engine = "cfgrib" if path.suffix == ".grib" else None
    with xr.open_dataset(path, engine=engine) as ds:
        if "number" not in ds.dims:
            ds = ds.expand_dims(number=[0])
        lon = ds["longitude"]
        site_lon = sites["longitude"].to_numpy()
        if float(lon.max()) > 180:
            site_lon = (site_lon + 360) % 360
        picked = ds.sel(
            latitude=xr.DataArray(sites["latitude"].to_numpy(), dims="site"),
            longitude=xr.DataArray(site_lon, dims="site"),
            method="nearest",
        ).assign_coords(site=("site", sites["site_code"].to_numpy()))
        frame = picked.to_dataframe().reset_index()

    frame = frame.rename(columns={"site": "site_code", "number": "member", "time": "run_time"})
    frame["run_time"] = pd.to_datetime(frame["run_time"], utc=True)
    if "valid_time" in frame.columns:
        frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    else:
        frame["valid_time"] = frame["run_time"] + pd.to_timedelta(frame["step"])
    frame["step_hours"] = (pd.to_timedelta(frame["step"]) / pd.Timedelta(hours=1)).astype(int)
    frame = frame.drop(columns=[c for c in (*_DROP, "step") if c in frame.columns])
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member"]
    return frame[keys + [c for c in frame.columns if c not in keys]]


def build_forecast_sites(config: dict[str, Any], sites: pd.DataFrame) -> Path:
    """Extract every downloaded month (both types) into one parquet under interim/."""
    located = sites[sites["latitude"].notna() & sites["longitude"].notna()]
    paths = sorted(
        p
        for t in config["types"]
        for p in (output_dir(config) / t).glob("*.*")
        if not p.name.endswith(".part")
    )
    if not paths:
        raise FileNotFoundError(
            f"No TIGGE files under {output_dir(config)}. Run fetch-tigge first."
        )
    frames = []
    for path in paths:
        logger.info("Extracting %s/%s", path.parent.name, path.name)
        frames.append(extract_sites(path, located))
    out = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["site_code", "run_time", "step_hours", "member"]
    )
    out = out.sort_values(["site_code", "run_time", "member", "step_hours"]).reset_index(drop=True)
    out_path = INTERIM_DIR / "forecast_tigge_ens.parquet"
    out.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "Wrote %s (%s rows, %d runs, %d members)",
        out_path.name,
        f"{len(out):,}",
        out["run_time"].nunique(),
        out["member"].nunique(),
    )
    return out_path
