"""ECMWF IFS ENS (51 members) from the AWS mirror of ECMWF open data.

ECMWF publishes its operational forecasts as open data (CC BY 4.0) and the
AWS mirror keeps them from 18 January 2023. Each ensemble step is one GRIB
holding every parameter for all 51 members, with a JSON-lines ``.index`` that
gives the byte offset and length of every message. So instead of the whole
4.5 GB file we fetch the ~400 messages we want by HTTP range - eight surface
parameters x 51 members - decode them with ecCodes, crop to the site box and
extract the nearest grid point for every site. One parquet per run.

This is the same model family as TIGGE and the exact product a day-ahead
system would have consumed. The mirror moved from 0.4 to 0.25 degrees in
February 2024; ``grid`` in each extract says which.

Attribution: "Contains ECMWF open data".
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, GEO_DIR, INTERIM_DIR
from ..logging_utils import get_logger
from .forecasts import _read_grib_cropped

logger = get_logger(__name__)

#: Candidate key prefixes per run, newest layout first.
LAYOUTS = (
    "{date}/{time}z/ifs/0p25/enfo/",
    "{date}/{time}z/ifs/0p4-beta/enfo/",
    "{date}/{time}z/0p4-beta/enfo/",
)


def output_dir(config: dict[str, Any]) -> Path:
    return EXTERNAL_DIR / config.get("output_subdir", "weather/ifs_ens")


def run_extract_path(date: str, time: str, config: dict[str, Any]) -> Path:
    return output_dir(config) / "sites" / f"{date}T{time}z.parquet"


def all_runs(config: dict[str, Any]) -> list[tuple[str, str]]:
    days = pd.date_range(config["start"], config["end"], freq="D")
    return [(d.strftime("%Y%m%d"), t) for d in days for t in config["run_times"]]


def missing_runs(config: dict[str, Any]) -> list[tuple[str, str]]:
    return [r for r in all_runs(config) if not run_extract_path(*r, config).exists()]


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #


def _session():
    import requests

    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=3)
    s.mount("https://", adapter)
    return s


def index_key(grib_key: str) -> str:
    """The .index sits beside the GRIB with the extension replaced, not appended."""
    return (
        grib_key[: -len(".grib2")] + ".index"
        if grib_key.endswith(".grib2")
        else grib_key + ".index"
    )


def resolve_layout(session, bucket: str, date: str, time: str, step: int) -> tuple[str, str] | None:
    """The key of the step's GRIB under whichever layout the mirror used that day, and its grid label."""
    for layout in LAYOUTS:
        prefix = layout.format(date=date, time=time)
        key = f"{prefix}{date}{time}0000-{step}h-enfo-ef.grib2"
        r = session.head(f"{bucket}/{index_key(key)}", timeout=60)
        if r.status_code == 200:
            return key, ("0p25" if "0p25" in prefix else "0p4")
    return None


def select_messages(index_text: str, params: list[str]) -> list[tuple[int, int, str, int]]:
    """(offset, length, param, member) for every wanted message in an index file."""
    wanted = set(params)
    out = []
    for line in index_text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("param") not in wanted:
            continue
        number = int(rec.get("number") or 0)  # the control run carries no number
        out.append((int(rec["_offset"]), int(rec["_length"]), rec["param"], number))
    return out


def fetch_step(
    session,
    bucket: str,
    key: str,
    messages: list[tuple[int, int, str, int]],
    target: Path,
    workers: int,
) -> Path:
    """Fetch the selected messages by byte range, in parallel, into one GRIB file."""
    url = f"{bucket}/{key}"

    def get(msg):
        offset, length, _, _ = msg
        r = session.get(
            url, headers={"Range": f"bytes={offset}-{offset + length - 1}"}, timeout=120
        )
        r.raise_for_status()
        if len(r.content) != length:
            raise OSError(f"short read {len(r.content)} != {length} for {key}@{offset}")
        return r.content

    with ThreadPoolExecutor(max_workers=workers) as pool:
        chunks = list(pool.map(get, messages))
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as fh:
        for c in chunks:
            fh.write(c)
    return target


def _decode(args: tuple[str, list[float], int]) -> tuple[int, dict[str, Any]]:
    """Decode one step's GRIB to a cropped dict (runs in a worker process)."""
    path, area, step = args
    ds = _read_grib_cropped(Path(path), area)
    payload = {
        "number": ds["number"].to_numpy(),
        "latitude": ds["latitude"].to_numpy(),
        "longitude": ds["longitude"].to_numpy(),
        "vars": {v: ds[v].to_numpy() for v in ds.data_vars},
    }
    return step, payload


def _located_sites() -> pd.DataFrame:
    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    return sites[sites["latitude"].notna() & sites["longitude"].notna()][
        ["site_code", "latitude", "longitude"]
    ].reset_index(drop=True)


def _extract(
    payloads: dict[int, dict[str, Any]], sites: pd.DataFrame, run_time: pd.Timestamp, grid: str
) -> pd.DataFrame:
    """Nearest grid point per site, long over (step, member)."""
    frames = []
    for step, p in sorted(payloads.items()):
        lat, lon = p["latitude"], p["longitude"]
        i = np.abs(lat[None, :] - sites["latitude"].to_numpy()[:, None]).argmin(axis=1)
        j = np.abs(lon[None, :] - sites["longitude"].to_numpy()[:, None]).argmin(axis=1)
        members = p["number"]
        n_site, n_mem = len(sites), len(members)
        base = pd.DataFrame(
            {
                "site_code": np.repeat(sites["site_code"].to_numpy(), n_mem),
                "member": np.tile(members.astype("int16"), n_site),
            }
        )
        for v, arr in p["vars"].items():  # arr: (number, lat, lon)
            base[v] = arr[:, i, j].T.reshape(-1).astype("float32")
        base["step_hours"] = np.int16(step)
        frames.append(base)
    out = pd.concat(frames, ignore_index=True)
    out["run_time"] = run_time
    out["valid_time"] = run_time + pd.to_timedelta(out["step_hours"].astype(int), unit="h")
    out["grid"] = grid
    keys = ["site_code", "run_time", "valid_time", "step_hours", "member", "grid"]
    return out[keys + [c for c in out.columns if c not in keys]]


def harvest_run(date: str, time: str, config: dict[str, Any], session=None) -> Path | None:
    """Fetch, decode and extract one run. Returns the extract path, or None if the mirror lacks it."""
    out = run_extract_path(date, time, config)
    if out.exists():
        return out
    session = session or _session()
    bucket = config["bucket"]
    work = output_dir(config) / "work" / f"{date}T{time}z"
    work.mkdir(parents=True, exist_ok=True)
    try:
        gribs: dict[int, Path] = {}
        grid = None
        for step in config["steps"]:
            found = resolve_layout(session, bucket, date, time, step)
            if found is None:
                logger.warning("%sT%sz step %d not on the mirror; skipping run", date, time, step)
                return None
            key, grid = found
            index_text = session.get(f"{bucket}/{index_key(key)}", timeout=120).text
            messages = select_messages(index_text, list(config["params"]))
            if not messages:
                logger.warning(
                    "%sT%sz step %d: none of %s in the index", date, time, step, config["params"]
                )
                return None
            gribs[step] = fetch_step(
                session,
                bucket,
                key,
                messages,
                work / f"{step:03d}h.grib2",
                int(config.get("download_workers", 32)),
            )
        with ProcessPoolExecutor(max_workers=int(config.get("decode_workers", 6))) as pool:
            payloads = dict(
                pool.map(_decode, [(str(p), list(config["area"]), s) for s, p in gribs.items()])
            )
        run_time = pd.Timestamp(f"{date}T{time}:00", tz="UTC")
        frame = _extract(payloads, _located_sites(), run_time, grid or "unknown")
        out.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(out, index=False, compression="zstd")
        logger.info(
            "%sT%sz: %d members x %d steps x %d vars, grid %s -> %s (%.1f MB)",
            date,
            time,
            frame["member"].nunique(),
            len(gribs),
            len(config["params"]),
            grid,
            out.name,
            out.stat().st_size / 1e6,
        )
        return out
    finally:
        shutil.rmtree(work, ignore_errors=True)


def harvest(
    config: dict[str, Any],
    until_complete: bool = False,
    pause_seconds: int = 1800,
    max_passes: int = 50,
) -> int:
    """Harvest every missing run; optionally keep re-passing for runs that failed."""
    import time as _time

    session = _session()
    for n in range(1, (max_passes if until_complete else 1) + 1):
        pending = missing_runs(config)
        if not pending:
            logger.info("All %d runs present", len(all_runs(config)))
            return 0
        logger.info("Pass %d: %d of %d runs missing", n, len(pending), len(all_runs(config)))
        skipped = 0
        for date, time in pending:
            try:
                if harvest_run(date, time, config, session) is None:
                    skipped += 1
            except Exception as exc:  # noqa: BLE001 - keep going; the next pass retries
                skipped += 1
                logger.error("%sT%sz failed: %s", date, time, exc)
        if not until_complete or not missing_runs(config):
            return skipped
        logger.info("Pass %d ended with %d runs missing; pausing %d s", n, skipped, pause_seconds)
        _time.sleep(pause_seconds)
    return len(missing_runs(config))


def build_forecast_sites(config: dict[str, Any]) -> list[Path]:
    """Concatenate run extracts into monthly parquet files under interim/forecast_ifs_ens/."""
    paths = sorted((output_dir(config) / "sites").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError("no IFS ENS run extracts yet")
    out_dir = INTERIM_DIR / "forecast_ifs_ens"
    out_dir.mkdir(parents=True, exist_ok=True)
    by_month: dict[str, list[Path]] = {}
    for p in paths:
        by_month.setdefault(p.name[:6], []).append(p)
    written = []
    for ym, files in by_month.items():
        frame = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        out = out_dir / f"{ym[:4]}-{ym[4:]}.parquet"
        frame.to_parquet(out, index=False, compression="zstd")
        written.append(out)
        logger.info("Wrote %s (%s rows, %d runs)", out.name, f"{len(frame):,}", len(files))
    return written
