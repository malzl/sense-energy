"""Two extensions of the day-ahead forecast, after Spoek et al. (2026, zero-shot forecasts of
district heat load), on the whole electricity record.

1. **A correction within the day** (``run_residual``). The day-ahead forecast of the whole
   record (``forecast_probe.run_record`` with the weather forecast) leaves an error that is
   known up to the present if the meter is read in real time. Every two hours a second
   zero-shot Chronos-2 forecasts the next four half-hours of that error from the last week of
   errors, and the day-ahead forecast is corrected by it. Comparators on the same half-hours:
   the day-ahead forecast alone, the last known error carried forward, the last metered value
   carried forward, and Chronos-2 run again on the demand itself (six weeks of context, no
   covariates). The sites with on-site generation switch their plant within the day, which no
   day-ahead input has predicted; this is the test of whether the error itself carries it.

2. **Measured and forecast weather as separate inputs** (``run_split_weather``). The
   covariate run gives each weather variable one channel: measured values (ERA5) over the past
   and the forecast of the day before (IFS ENS mean) over the future. The model then sees a
   change of source at the issue time. Two alternatives on every third day of the record:
   ``split`` gives the measured value as a past-only input and the day-ahead forecast as a
   second input over past and future; ``fc_only`` gives only the day-ahead forecast, over past
   and future. The history of forecasts is the 00z run of the day before each day, as issued.

Both use a GPU if one is visible. Outputs: ``code/reports/forecast_record/``.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import poc
from .fm_covariates import WEATHER_TO_IFS, CovariateWindows, clean
from .forecast_probe import OUT_DIR as PROBE_DIR
from .forecast_probe import RECORD_END, RECORD_START, predict, setup
from .zero_shot import _assemble, contexts_for

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_record"
RESID_CONTEXT = 336  # one week of errors
RESID_HORIZON = 4  # two hours
RESID_STRIDE = 3  # every third day (all weekdays in turn)
MIN_FINITE = 0.9  # share of the error context that must be known
BLOCKS_PER_CHUNK = 360
SPLIT_VARIANTS = ("split", "fc_only")
SPLIT_STRIDE = 3


# --------------------------------------------------------------------------- #
# 1. A correction within the day
# --------------------------------------------------------------------------- #


def base_grid(panel: poc.Panel) -> np.ndarray:
    """The day-ahead forecast of the whole record on the panel grid (T x S, kWh per period)."""
    f = pd.read_parquet(
        PROBE_DIR / "forecasts_record_ifs.parquet", columns=["site_code", "target", "point"]
    ).drop_duplicates(["site_code", "target"], keep="last")
    wide = f.pivot(index="target", columns="site_code", values="point")
    return wide.reindex(index=panel.index, columns=panel.sites).to_numpy("float32")


def block_starts(tod: np.ndarray, local_date: np.ndarray, days: np.ndarray) -> np.ndarray:
    """Positions of the first half-hour of each two-hour block of the local day on ``days``."""
    on = np.isin(local_date, days) & (tod % RESID_HORIZON == 0)
    pos = np.flatnonzero(on)
    return pos[(pos >= RESID_CONTEXT) & (pos + RESID_HORIZON <= len(tod))]


def fill_rows(x: np.ndarray) -> np.ndarray:
    """Fill gaps in each row with the last value, then with the row's median (in place)."""
    for i in np.flatnonzero(~np.isfinite(x).all(axis=1)):
        x[i] = poc._fill(x[i])
    return x


def windows_before(a: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    """For a (T x S) array: the ``length`` rows before each start, as (len(starts) * S, length)
    in the order start by start, site by site."""
    idx = starts[:, None] - length + np.arange(length)[None, :]
    return a[idx].transpose(0, 2, 1).reshape(-1, length)


def windows_from(a: np.ndarray, starts: np.ndarray, length: int) -> np.ndarray:
    idx = starts[:, None] + np.arange(length)[None, :]
    return a[idx].transpose(0, 2, 1).reshape(-1, length)


def run_residual(device: str | None = None, stride_days: int = RESID_STRIDE) -> pd.DataFrame:
    from ..compute import autocast, select_device
    from ..models.foundation import load_chronos2

    config, panel = setup()
    device = device or select_device()
    pipe = load_chronos2(device)
    quantiles = list(config["quantiles"])
    L = int(config["context_periods"])
    S = len(panel.sites)
    Y = panel.y_imp[panel.sites].to_numpy("float32")
    Yraw = panel.y_raw[panel.sites].to_numpy("float32")
    B = base_grid(panel)
    R = Y - B
    first = pd.Timestamp(RECORD_START) + pd.Timedelta(days=RESID_CONTEXT // 48)
    days = (
        pd.date_range(first, RECORD_END, freq=f"{stride_days}D").to_numpy().astype("datetime64[D]")
    )
    starts = block_starts(panel.tod, panel.local_date, days)
    starts = starts[starts >= L]
    frames, t0 = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a in range(0, len(starts), BLOCKS_PER_CHUNK):
        ps = starts[a : a + BLOCKS_PER_CHUNK]
        Xr = windows_before(R, ps, RESID_CONTEXT)
        base = windows_from(B, ps, RESID_HORIZON)
        ok = (np.isfinite(Xr).mean(axis=1) >= MIN_FINITE) & np.isfinite(base).all(axis=1)
        if not ok.any():
            continue
        last_error = np.nan_to_num(Xr[ok, -1])
        Xr = fill_rows(Xr[ok].copy())
        Xd = fill_rows(windows_before(Y, ps, L)[ok].copy())
        with autocast(device):
            _, Pr = predict(pipe, Xr, RESID_HORIZON, quantiles, None, 1024)
            _, Pd = predict(pipe, Xd, RESID_HORIZON, quantiles, None, 256)
        base = base[ok]
        pos = np.repeat(ps, S)[ok][:, None] + np.arange(RESID_HORIZON)[None, :]
        site = np.tile(np.arange(S), len(ps))[ok]
        out = pd.DataFrame(
            {
                "site_code": np.asarray(panel.sites)[np.repeat(site, RESID_HORIZON)],
                "target": panel.index[pos.ravel()],
                "step": np.tile(np.arange(1, RESID_HORIZON + 1), len(site)).astype("int8"),
                "y": windows_from(Yraw, ps, RESID_HORIZON)[ok].ravel(),
                "day_ahead": base.ravel(),
                "corrected": (base + Pr).ravel().astype("float32"),
                "error_carried": (base + last_error[:, None]).ravel().astype("float32"),
                "value_carried": np.repeat(Xd[:, -1], RESID_HORIZON),
                "direct": Pd.ravel().astype("float32"),
            }
        )
        frames.append(out)
        done = min(a + BLOCKS_PER_CHUNK, len(starts))
        print(f"residual: {done}/{len(starts)} issues, {time.time() - t0:.0f} s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(["site_code", "target"], keep="last")
    out.to_parquet(OUT_DIR / "forecasts_intraday_correction.parquet", index=False)
    return out


# --------------------------------------------------------------------------- #
# 2. Measured and forecast weather as separate inputs
# --------------------------------------------------------------------------- #


def forecast_history(
    cw: CovariateWindows, panel: poc.Panel, origins: list[poc.Origin]
) -> tuple[dict[str, np.ndarray], float]:
    """For each weather variable a (T x S) array with the day-ahead forecast of each day as it
    was issued (the run of the day before); where a run is missing, the measured value of the
    last day known at the issue (the fallback of ``CovariateWindows.window``). Also the share
    of site-days without a run."""
    T, S = len(panel.index), len(panel.sites)
    F = {v: np.full((T, S), np.nan, "float32") for v in cw.weather_vars}
    missing = 0
    for o in origins:
        tpos = o.target_pos
        targets = panel.index[tpos].asi8
        back = np.clip(tpos - 48 * np.ceil((tpos - o.origin_pos) / 48).astype(int), 0, T - 1)
        for j, s in enumerate(panel.sites):
            block = cw.cov.ifs_block(s, o, targets)
            for k, v in enumerate(cw.weather_vars):
                f = block[WEATHER_TO_IFS[v]].astype("float64")
                bad = ~np.isfinite(f)
                if bad.any():
                    f[bad] = cw.per_site[v][back, j][bad]
                    missing += int(k == 0 and bad.all())
                F[v][tpos, j] = f
    return F, missing / max(1, len(origins) * S)


def split_window(
    cw: CovariateWindows,
    F: dict[str, np.ndarray],
    o: poc.Origin,
    j: int,
    site: str,
    L: int,
    H: int,
    variant: str,
) -> tuple[dict, dict]:
    """The covariate window of the ``ifs`` run with the weather channels of one variant."""
    past, future = cw.window(o, j, site, L, H)
    T = len(cw.panel.index)
    ctx = np.clip(np.arange(o.origin_pos - L + 1, o.origin_pos + 1), 0, T - 1)
    for v in cw.weather_vars:
        history = clean(F[v][ctx, j])
        if variant == "split":
            past[f"{v}_obs"] = past.pop(v)
            past[f"{v}_fc"] = history
            future[f"{v}_fc"] = future.pop(v)
        elif variant == "fc_only":
            past[v] = history
        else:
            raise KeyError(variant)
    return past, future


def run_split_weather(
    device: str | None = None,
    stride_days: int = SPLIT_STRIDE,
    variants: tuple[str, ...] = SPLIT_VARIANTS,
    start: str = RECORD_START,
    end: str = RECORD_END,
) -> pd.DataFrame:
    from ..compute import autocast, select_device
    from ..models.foundation import load_chronos2

    config, panel = setup()
    device = device or select_device()
    pipe = load_chronos2(device)
    quantiles = list(config["quantiles"])
    L = int(config["context_periods"])
    cw = CovariateWindows(config, panel, "ifs")
    t0 = time.time()
    history_start = str((pd.Timestamp(start) - pd.Timedelta(days=L // 48 + 1)).date())
    F, missing = forecast_history(cw, panel, poc.make_origins(panel, config, history_start, end, 1))
    print(f"forecast history: {time.time() - t0:.0f} s, {100 * missing:.1f} % without a run")
    origins = poc.make_origins(panel, config, start, end, stride_days)
    index = {s: j for j, s in enumerate(panel.sites)}
    frames: list[pd.DataFrame] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a in range(0, len(origins), 20):
        chunk = origins[a : a + 20]
        keys, X, horizon = contexts_for(panel, chunk, config)
        for variant in variants:
            windows = [split_window(cw, F, o, index[s], s, L, horizon, variant) for o, s in keys]
            with autocast(device):
                Q, P = predict(pipe, X, horizon, quantiles, windows, 256)
            name = f"chronos2_ifs_{variant}"
            out = _assemble(name, keys, Q.astype("float32"), P.astype("float32"), panel, quantiles)
            frames.append(out[["model", "site_code", "target", "point"]])
        done = min(a + 20, len(origins))
        print(f"split weather: {done}/{len(origins)} days, {time.time() - t0:.0f} s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    tag = "" if tuple(variants) == SPLIT_VARIANTS else "_" + "_".join(variants)
    out.to_parquet(OUT_DIR / f"forecasts_split_weather{tag}.parquet", index=False)
    return out


def run(which: str, device: str | None = None) -> Any:
    return {"residual": run_residual, "split": run_split_weather}[which](device)
