"""ERA5 weather from the Copernicus Climate Data Store.

Two products are pulled for the same window and area:

``reanalysis``
    Deterministic best estimate, 0.25 deg, hourly.
``ensemble_members``
    ERA5's 10-member Ensemble of Data Assimilations, 0.5 deg, 3-hourly. Not a
    forecast ensemble, but the spread is real analysis uncertainty and is the
    only ensemble product CDS serves for this period.

Requests are one month per product; each lands as a NetCDF under
``code/data/external/weather/era5/<product>/<YYYY-MM>.nc`` and is skipped on
re-run if it already opens cleanly, so an interrupted pull resumes. Site
extraction takes the nearest grid point to each site's postcode centroid.

Credentials come from ``CDSAPI_URL``/``CDSAPI_KEY`` in ``.env`` - never from
code, never logged.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR, SECRETS
from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Hour-of-day strings each product accepts. EDA is archived 3-hourly.
TIMES: dict[str, list[str]] = {
    "reanalysis": [f"{h:02d}:00" for h in range(24)],
    "ensemble_members": [f"{h:02d}:00" for h in range(0, 24, 3)],
    "ensemble_mean": [f"{h:02d}:00" for h in range(0, 24, 3)],
    "ensemble_spread": [f"{h:02d}:00" for h in range(0, 24, 3)],
}

#: CDS variable name -> NetCDF short name, for the extracted site table.
SHORT_NAMES: dict[str, str] = {
    "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m",
    "total_cloud_cover": "tcc",
    "low_cloud_cover": "lcc",
    "total_precipitation": "tp",
    "snowfall": "sf",
    "10m_u_component_of_wind": "u10",
    "10m_v_component_of_wind": "v10",
    "instantaneous_10m_wind_gust": "i10fg",
    "surface_solar_radiation_downwards": "ssrd",
    "surface_thermal_radiation_downwards": "strd",
    "surface_pressure": "sp",
}


def month_range(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    periods = pd.period_range(start, end, freq="M")
    return [(p.year, p.month) for p in periods]


def build_request(
    product_type: str, year: int, month: int, config: dict[str, Any]
) -> dict[str, Any]:
    """The CDS request body for one product/month."""
    if product_type not in TIMES:
        raise KeyError(f"Unknown product_type '{product_type}'. Known: {sorted(TIMES)}")
    days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
    return {
        "product_type": [product_type],
        "variable": list(config["variables"]),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
        "time": TIMES[product_type],
        "area": list(config["area"]),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def output_dir(config: dict[str, Any]) -> Path:
    return EXTERNAL_DIR / config.get("output_subdir", "weather/era5")


def target_path(product_type: str, year: int, month: int, config: dict[str, Any]) -> Path:
    return output_dir(config) / product_type / f"{year}-{month:02d}.nc"


def open_download(path: Path):
    """Open a CDS download as one Dataset.

    When a request mixes instantaneous (t2m, tcc ...) and accumulated (tp, ssrd
    ...) variables, the new CDS delivers a zip holding one NetCDF per step type
    even with ``download_format: unarchived``. Merge them on the time axis.
    """
    import zipfile

    import xarray as xr

    if not zipfile.is_zipfile(path):
        return xr.open_dataset(path)

    # Members are extracted beside the archive; the netCDF4 engine reads from
    # paths, not buffers, and a month of hourly fields is too big to hold twice.
    extract_dir = path.with_suffix(".parts")
    extract_dir.mkdir(exist_ok=True)
    parts = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith(".nc"):
                parts.append(xr.open_dataset(zf.extract(name, extract_dir)))
    if not parts:
        raise ValueError(f"{path.name} is a zip with no NetCDF members")
    return xr.merge(parts, compat="override", join="outer")


def _opens_cleanly(path: Path) -> bool:
    """A partially written file is worse than a missing one; check before skipping."""
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with open_download(path) as ds:
            return len(ds.data_vars) > 0
    except Exception:  # noqa: BLE001 - any failure means re-download
        logger.warning("%s does not open cleanly; will re-download", path.name)
        return False


def make_client():
    """A cdsapi client bound to the credentials in .env."""
    import cdsapi

    if not SECRETS.cdsapi_key:
        raise RuntimeError("CDSAPI_KEY is not set. Copy .env.example to .env and fill it in.")
    return cdsapi.Client(url=SECRETS.cdsapi_url, key=SECRETS.cdsapi_key, quiet=True, progress=False)


def fetch_month(
    product_type: str, year: int, month: int, config: dict[str, Any], client=None
) -> Path:
    """Download one product/month if not already present. Returns the file path."""
    path = target_path(product_type, year, month, config)
    if _opens_cleanly(path):
        logger.info("%s/%s already present", product_type, path.name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".nc.part")
    client = client or make_client()
    logger.info("Requesting %s %d-%02d", product_type, year, month)
    client.retrieve(config["dataset"], build_request(product_type, year, month, config), str(tmp))

    try:
        with open_download(tmp) as ds:
            if not ds.data_vars:
                raise ValueError("no variables")
            encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
            ds.load().to_netcdf(path, encoding=encoding)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Download for {product_type} {year}-{month:02d} is unreadable: {exc}"
        ) from exc
    tmp.unlink(missing_ok=True)
    shutil.rmtree(tmp.with_suffix(".parts"), ignore_errors=True)
    logger.info("Wrote %s/%s (%.1f MB)", product_type, path.name, path.stat().st_size / 1e6)
    return path


def fetch_all(config: dict[str, Any], product_types: list[str] | None = None) -> list[Path]:
    """Pull every product/month in the config, a few CDS jobs at a time.

    Failures are logged and skipped so one bad month does not sink the run;
    re-running picks up whatever is missing.
    """
    products = product_types or list(config["product_types"])
    jobs = [(p, y, m) for p in products for (y, m) in month_range(config["start"], config["end"])]
    pending = [j for j in jobs if not _opens_cleanly(target_path(*j, config))]
    logger.info(
        "%d of %d product-months already present; fetching %d",
        len(jobs) - len(pending),
        len(jobs),
        len(pending),
    )

    done: list[Path] = [target_path(*j, config) for j in jobs if j not in pending]
    failures: list[tuple[str, int, int]] = []
    client = make_client()
    with ThreadPoolExecutor(max_workers=int(config.get("max_workers", 3))) as pool:
        futures = {
            pool.submit(fetch_month, p, y, m, config, client): (p, y, m) for (p, y, m) in pending
        }
        for future in as_completed(futures):
            job = futures[future]
            try:
                done.append(future.result())
            except Exception as exc:  # noqa: BLE001 - keep going, report at the end
                failures.append(job)
                logger.error("Failed %s %d-%02d: %s", *job, exc)

    if failures:
        logger.error("%d product-months failed: %s", len(failures), failures)
    return done


# --------------------------------------------------------------------------- #
# Extraction to sites
# --------------------------------------------------------------------------- #


def extract_sites(nc_path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-point values for each site, as a long table.

    Output columns: ``site_code``, ``datetime`` (UTC), ``member`` (0 for the
    deterministic product), plus one column per variable short name.
    """
    import xarray as xr

    with open_download(nc_path) as ds:
        time_dim = "valid_time" if "valid_time" in ds.dims else "time"
        lon = ds["longitude"]
        # CDS serves area subsets as -180..180; guard the 0..360 case anyway.
        site_lon = sites["longitude"].to_numpy()
        if float(lon.max()) > 180:
            site_lon = np.where(site_lon < 0, site_lon + 360, site_lon)

        picked = ds.sel(
            latitude=xr.DataArray(sites["latitude"].to_numpy(), dims="site"),
            longitude=xr.DataArray(site_lon, dims="site"),
            method="nearest",
        )
        picked = picked.assign_coords(site=("site", sites["site_code"].to_numpy()))
        frame = picked.to_dataframe().reset_index()

    frame = frame.rename(columns={time_dim: "datetime", "site": "site_code", "number": "member"})
    if "member" not in frame.columns:
        frame["member"] = 0
    frame["datetime"] = pd.to_datetime(frame["datetime"], utc=True)

    drop = [c for c in ("latitude", "longitude", "expver", "surface", "time") if c in frame.columns]
    frame = frame.drop(columns=drop)
    keep = ["site_code", "datetime", "member"] + [
        c for c in frame.columns if c not in ("site_code", "datetime", "member")
    ]
    return frame[keep]


def build_weather_sites(
    config: dict[str, Any], sites: pd.DataFrame, product_type: str = "reanalysis"
) -> Path:
    """Extract every downloaded month for one product into a single parquet."""
    located = sites[sites["latitude"].notna() & sites["longitude"].notna()]
    paths = sorted((output_dir(config) / product_type).glob("*.nc"))
    if not paths:
        raise FileNotFoundError(
            f"No NetCDF files under {output_dir(config) / product_type}. Run fetch-weather first."
        )

    frames = []
    for path in paths:
        logger.info("Extracting %s", path.name)
        frames.append(extract_sites(path, located))
    out = pd.concat(frames, ignore_index=True).sort_values(["site_code", "member", "datetime"])
    out = out.drop_duplicates(["site_code", "member", "datetime"], keep="last").reset_index(
        drop=True
    )

    out_path = INTERIM_DIR / f"weather_{product_type}.parquet"
    out.to_parquet(out_path, index=False, compression="zstd")
    logger.info(
        "Wrote %s (%s rows, %d sites, %d members)",
        out_path.name,
        f"{len(out):,}",
        out["site_code"].nunique(),
        out["member"].nunique(),
    )
    return out_path
