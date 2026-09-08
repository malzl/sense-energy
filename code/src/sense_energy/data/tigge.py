"""ECMWF IFS ENS historical forecasts from TIGGE, via the ECMWF Data Store.

TIGGE archives every operational ensemble run since 2006 at 0.5 deg, which
makes it the source for *what the forecast said* over the training window -
the quantity a day-ahead demand model would really have had, as opposed to the
reanalysis in :mod:`weather` (what the weather turned out to be).

Access is through ECDS (``ecds.ecmwf.int``), which runs the same data-store
software as the Copernicus CDS and speaks to the same ``cdsapi`` client. The
former Public Datasets Web API was decommissioned on 2026-05-27.

One request per (forecast type, month); each lands as ``<type>/<YYYY-MM>.grib``
and is skipped on re-run if it opens cleanly. Data-store queues are slow and
opaque, so this runs in the background.

Credentials: ``ECDS_API_URL`` / ``ECDS_API_KEY`` in ``.env``. The account must
also have accepted the TIGGE licence on the ECDS dataset page.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, INTERIM_DIR, SECRETS
from ..logging_utils import get_logger
from .forecasts import _open_grib_merged
from .weather import month_range

logger = get_logger(__name__)

TIGGE_DATASET_URL = "https://ecds.ecmwf.int/datasets/tigge-forecasts"

#: forecast_type -> short label used for directories and member numbering.
TYPE_LABELS = {"control_forecast": "cf", "perturbed_forecast": "pf"}


def output_dir(config: dict[str, Any]) -> Path:
    return EXTERNAL_DIR / config.get("output_subdir", "weather/tigge_ens")


def target_path(forecast_type: str, year: int, month: int, config: dict[str, Any]) -> Path:
    label = TYPE_LABELS.get(forecast_type, forecast_type)
    return output_dir(config) / label / f"{year}-{month:02d}.grib"


def build_request(
    forecast_type: str, year: int, month: int, config: dict[str, Any]
) -> dict[str, Any]:
    """The ECDS request body for one forecast type and month."""
    if forecast_type not in TYPE_LABELS:
        raise KeyError(f"forecast_type must be one of {sorted(TYPE_LABELS)}, got '{forecast_type}'")
    days_in_month = pd.Period(f"{year}-{month:02d}").days_in_month
    return {
        "origin": config.get("origin", "ecmwf"),
        "level_type": config.get("level_type", "single_level"),
        "forecast_type": forecast_type,
        "variable": list(config["variables"]),
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, days_in_month + 1)],
        "time": list(config["times"]),
        "leadtime_hour": [str(h) for h in config["leadtime_hours"]],
        "area": list(config["area"]),
        "data_format": config.get("data_format", "grib"),
    }


def _opens_cleanly(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with _open_grib_merged(path) as ds:
            return len(ds.data_vars) > 0
    except Exception:  # noqa: BLE001 - any failure means re-download
        logger.warning("%s does not open cleanly; will re-download", path.name)
        return False


def make_client():
    """A cdsapi client bound to ECDS."""
    import cdsapi

    if not SECRETS.ecds_api_key:
        raise RuntimeError(
            "ECDS_API_KEY is not set. Get a token from your profile at https://ecds.ecmwf.int and add it to .env."
        )
    return cdsapi.Client(
        url=SECRETS.ecds_api_url, key=SECRETS.ecds_api_key, quiet=True, progress=False
    )


def sites_parquet_path(product_type: str, year: int, month: int, config: dict[str, Any]) -> Path:
    return output_dir(config) / f"{product_type}_sites" / f"{year}-{month:02d}.parquet"


def month_present(product_type: str, year: int, month: int, config: dict[str, Any]) -> bool:
    """A month counts as done only once its site extract exists.

    A GRIB on disk without an extract is not done: fetch_month will skip the
    download and just extract it.
    """
    return sites_parquet_path(product_type, year, month, config).exists()


def _located_sites() -> pd.DataFrame:
    from ..config import GEO_DIR

    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    return sites[sites["latitude"].notna() & sites["longitude"].notna()][
        ["site_code", "latitude", "longitude"]
    ].reset_index(drop=True)


def fetch_month(
    product_type: str, year: int, month: int, config: dict[str, Any], client=None
) -> Path:
    """Retrieve one type/month, extract the sites, and drop the GRIB unless configured to keep it.

    Perturbed-member GRIBs are ~1 GB a month; the site extract of the same
    month is a small parquet. Keeping only the extract is what makes the full
    archive fit on disk. Returns the site-extract path.
    """
    out_parquet = sites_parquet_path(product_type, year, month, config)
    if out_parquet.exists():
        logger.info("%s/%s already extracted", product_type, out_parquet.name)
        return out_parquet

    path = target_path(product_type, year, month, config)
    if not _opens_cleanly(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        client = client or make_client()
        logger.info("ECDS request: tigge %s %d-%02d", product_type, year, month)
        client.retrieve(
            config["dataset"], build_request(product_type, year, month, config), str(tmp)
        )
        if not _opens_cleanly(tmp):
            tmp.unlink(missing_ok=True)
            raise RuntimeError(
                f"TIGGE download for {product_type} {year}-{month:02d} is unreadable"
            )
        tmp.replace(path)
        logger.info("Wrote %s/%s (%.1f MB)", product_type, path.name, path.stat().st_size / 1e6)

    frame = extract_sites_fast(path, _located_sites())
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_parquet, index=False, compression="zstd")
    logger.info(
        "Extracted %s/%s (%s rows, %.1f MB)",
        product_type,
        out_parquet.name,
        f"{len(frame):,}",
        out_parquet.stat().st_size / 1e6,
    )
    if product_type not in config.get("keep_grib_for", ["cf"]):
        path.unlink(missing_ok=True)
        logger.info("Deleted %s/%s to save disk", product_type, path.name)
    return out_parquet


def fetch_all(config: dict[str, Any], forecast_types: list[str] | None = None) -> list[Path]:
    """Every (type, month) in the config, a couple of data-store jobs at a time. Resumable."""
    types = forecast_types or list(config["forecast_types"])
    jobs = [(t, y, m) for t in types for (y, m) in month_range(config["start"], config["end"])]
    pending = [j for j in jobs if not month_present(*j, config)]
    logger.info(
        "%d of %d type-months present; requesting %d",
        len(jobs) - len(pending),
        len(jobs),
        len(pending),
    )

    done = [sites_parquet_path(*j, config) for j in jobs if j not in pending]
    failures = []
    client = make_client()
    with ThreadPoolExecutor(max_workers=int(config.get("max_workers", 2))) as pool:
        futures = {
            pool.submit(fetch_month, t, y, m, config, client): (t, y, m) for (t, y, m) in pending
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

_DROP = ("latitude", "longitude", "surface", "heightAboveGround", "expver", "step")


def nearest_grid_indices(
    lat_grid: np.ndarray, lon_grid: np.ndarray, lats: np.ndarray, lons: np.ndarray
) -> np.ndarray:
    """Index of the nearest grid point for each site on a 1-D (unstructured) grid.

    ECDS serves TIGGE on the model's native reduced grid rather than a regular
    lat/lon box, so cfgrib exposes one ``values`` dimension with latitude and
    longitude as coordinates along it. Distances use an equirectangular
    approximation, fine at this scale.
    """
    lon_grid = ((np.asarray(lon_grid) + 180.0) % 360.0) - 180.0
    lat_grid = np.asarray(lat_grid)
    out = np.empty(len(lats), dtype=int)
    for i, (la, lo) in enumerate(zip(lats, lons, strict=True)):
        dlat = np.deg2rad(lat_grid - la)
        dlon = np.deg2rad(lon_grid - lo) * np.cos(np.deg2rad(la))
        out[i] = int(np.argmin(dlat**2 + dlon**2))
    return out


def extract_sites(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-point values per site, long over (run_time, step, member).

    Handles both a regular lat/lon grid and ECDS's reduced grid. The control
    run carries a single member 0; perturbed members are 1-50.
    """
    import xarray as xr

    with _open_grib_merged(path) as ds:
        if "values" in ds.dims:
            idx = nearest_grid_indices(
                ds["latitude"].values,
                ds["longitude"].values,
                sites["latitude"].to_numpy(),
                sites["longitude"].to_numpy(),
            )
            picked = ds.isel(values=xr.DataArray(idx, dims="site"))
            picked = picked.drop_vars([c for c in ("latitude", "longitude") if c in picked.coords])
        else:
            lon = ds["longitude"]
            site_lon = sites["longitude"].to_numpy()
            if float(lon.max()) > 180:
                site_lon = (site_lon + 360) % 360
            picked = ds.sel(
                latitude=xr.DataArray(sites["latitude"].to_numpy(), dims="site"),
                longitude=xr.DataArray(site_lon, dims="site"),
                method="nearest",
            )
            picked = picked.drop_vars([c for c in ("latitude", "longitude") if c in picked.coords])
        picked = picked.assign_coords(site=("site", sites["site_code"].to_numpy()))
        frame = picked.to_dataframe().reset_index()

    frame = frame.rename(columns={"site": "site_code", "number": "member", "time": "run_time"})
    frame["run_time"] = pd.to_datetime(frame["run_time"], utc=True)
    step = pd.to_timedelta(frame["step"])
    frame["valid_time"] = frame["run_time"] + step
    frame["step_hours"] = (step / pd.Timedelta(hours=1)).astype(int)
    if "member" not in frame.columns:
        frame["member"] = 0
    frame = frame.drop(columns=[c for c in (*_DROP, "step") if c in frame.columns])
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member"]
    values = [c for c in frame.columns if c not in keys]
    frame[values] = frame[values].astype("float32")
    frame["member"] = frame["member"].astype("int16")
    frame["step_hours"] = frame["step_hours"].astype("int16")
    return frame[keys + values]


#: ecCodes short names -> the cfgrib-style names used everywhere else.
_SHORTNAME_ALIASES = {"2t": "t2m", "2d": "d2m", "10u": "u10", "10v": "v10"}


def extract_sites_fast(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Same output as :func:`extract_sites`, decoded directly with ecCodes.

    cfgrib costs tens of milliseconds per message; a perturbed month holds
    ~480k messages of 739 points each. Reading the messages in a loop and
    keeping only the site indices is two orders of magnitude faster.
    """
    import eccodes

    lats = sites["latitude"].to_numpy()
    lons = sites["longitude"].to_numpy()
    site_codes = sites["site_code"].to_numpy()
    idx = None
    records: dict[tuple, dict[str, np.ndarray]] = {}

    with path.open("rb") as fh:
        while True:
            gid = eccodes.codes_grib_new_from_file(fh)
            if gid is None:
                break
            try:
                if idx is None:
                    idx = nearest_grid_indices(
                        eccodes.codes_get_array(gid, "latitudes"),
                        eccodes.codes_get_array(gid, "longitudes"),
                        lats,
                        lons,
                    )
                key = (
                    int(eccodes.codes_get(gid, "dataDate")),
                    int(eccodes.codes_get(gid, "dataTime")),
                    int(eccodes.codes_get(gid, "endStep")),
                    int(eccodes.codes_get(gid, "perturbationNumber"))
                    if eccodes.codes_is_defined(gid, "perturbationNumber")
                    else 0,
                )
                name = eccodes.codes_get(gid, "shortName")
                name = _SHORTNAME_ALIASES.get(name, name)
                values = eccodes.codes_get_values(gid)[idx].astype("float32")
                missing = eccodes.codes_get(gid, "missingValue")
                values[values == missing] = np.nan
                records.setdefault(key, {})[name] = values
            finally:
                eccodes.codes_release(gid)

    rows = []
    for (date, time, step, member), fields in records.items():
        run_time = pd.Timestamp(f"{date:08d}T{time:04d}", tz="UTC")
        frame = pd.DataFrame(fields, index=site_codes)
        frame.insert(0, "site_code", site_codes)
        frame.insert(1, "run_time", run_time)
        frame.insert(2, "valid_time", run_time + pd.Timedelta(hours=step))
        frame.insert(3, "step_hours", np.int16(step))
        frame.insert(4, "member", np.int16(member))
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True)
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member"]
    return out[keys + sorted(c for c in out.columns if c not in keys)]


def build_forecast_sites(config: dict[str, Any], sites: pd.DataFrame) -> Path:
    """Concatenate every month's site extract (both types) into one parquet under interim/."""
    paths = sorted(
        p for t in config["types"] for p in (output_dir(config) / f"{t}_sites").glob("*.parquet")
    )
    if not paths:
        raise FileNotFoundError(
            f"No site extracts under {output_dir(config)}. Run fetch-tigge first."
        )
    out = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = (
        out.drop_duplicates(["site_code", "run_time", "step_hours", "member"])
        .sort_values(["site_code", "run_time", "member", "step_hours"])
        .reset_index(drop=True)
    )
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
