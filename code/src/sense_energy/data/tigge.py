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


def fetch_month(
    forecast_type: str, year: int, month: int, config: dict[str, Any], client=None
) -> Path:
    """Retrieve one type/month if not present. Returns the file path."""
    path = target_path(forecast_type, year, month, config)
    if _opens_cleanly(path):
        logger.info("%s/%s already present", path.parent.name, path.name)
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    client = client or make_client()
    logger.info("ECDS request: tigge %s %d-%02d", TYPE_LABELS[forecast_type], year, month)
    try:
        client.retrieve(
            config["dataset"], build_request(forecast_type, year, month, config), str(tmp)
        )
    except Exception as exc:
        message = str(exc).lower()
        if "licence" in message or "license" in message or "403" in message:
            raise RuntimeError(
                f"ECDS refused the request - is the TIGGE licence accepted at {TIGGE_DATASET_URL}? ({exc})"
            ) from exc
        raise
    if not _opens_cleanly(tmp):
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"TIGGE download for {forecast_type} {year}-{month:02d} is unreadable")
    tmp.replace(path)
    logger.info("Wrote %s/%s (%.1f MB)", path.parent.name, path.name, path.stat().st_size / 1e6)
    return path


def fetch_all(config: dict[str, Any], forecast_types: list[str] | None = None) -> list[Path]:
    """Every (type, month) in the config, a couple of data-store jobs at a time. Resumable."""
    types = forecast_types or list(config["forecast_types"])
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


def extract_sites(path: Path, sites: pd.DataFrame) -> pd.DataFrame:
    """Nearest-grid-point values per site, long over (run_time, step, member).

    cfgrib carries the run in ``time`` and the lead in ``step``; ``number`` is
    present for perturbed members and becomes 0 for the control.
    """
    import xarray as xr

    with _open_grib_merged(path) as ds:
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
    frame = frame.drop(columns=[c for c in _DROP if c in frame.columns])
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member"]
    return frame[keys + [c for c in frame.columns if c not in keys]]


def build_forecast_sites(config: dict[str, Any], sites: pd.DataFrame) -> Path:
    """Extract every downloaded month (both types) into one parquet under interim/."""
    located = sites[sites["latitude"].notna() & sites["longitude"].notna()]
    paths = sorted(
        p
        for t in config["forecast_types"]
        for p in (output_dir(config) / TYPE_LABELS[t]).glob("*.grib")
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
