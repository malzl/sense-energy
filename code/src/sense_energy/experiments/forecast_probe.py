"""CPU probes behind the forecast finding: more days, less history, a mobility signal.

Three runs of Chronos-2, zero-shot, on the CPU, with the proof of concept's set-up (issue at 16:30
on D-1, the 48 half-hours of D, six weeks of context unless stated):

* ``daily``: a forecast for **every** day of the test window at the 91 proof-of-concept sites
  (the proof of concept has one origin every five days). No covariates: the run is there to
  count errors by kind of day (holidays, strike days, level shifts), which needs all the days.
* ``context``: the proof-of-concept origins again with 1 day, 3 days, 1 week and 2 weeks of
  context instead of six weeks: how much history a site needs.
* ``mobility``: the three sites with daily footfall counts (Royal Devon, calendar 2024), every
  day from 13 February to 31 December 2024, with the proof of concept's known-ahead covariates and
  reanalysis weather, and the footfall given to the model in context in three ways:

  - ``prev_day``: the count of the last complete day as a past-only covariate (what a daily feed
    would give at the issue time);
  - ``oracle``: the count of every day, the target day included, as a known covariate (the upper
    bound: a perfect footfall forecast);
  - ``uni_oracle``: the same with the footfall of the university campus in the same city (a
    placebo: activity in the city that is not the hospital's).

  The base and the oracle variants run again with one week of context (a site with little
  history).
* ``intraday``: the proof-of-concept origins issued later, at 00:00 and at 06:00 on the target
  day itself, for the rest of that day: what a later issue time is worth where the error is a
  plant that switches.

Outputs: ``code/reports/forecast_anatomy/forecasts_<run>.parquet`` in the proof of concept's long
format. Mobility data: The Oxford Partnership (research licence, see ``doc/data_governance.md``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT, load_config
from ..logging_utils import get_logger
from . import poc
from .fm_covariates import CovariateWindows, clean
from .zero_shot import _assemble, _to_array, contexts_for

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_anatomy"
CONFIG = "code/configs/experiments/poc.yaml"
CONTEXTS = (48, 144, 336, 672)
MOBILITY_SITES = ("RH801", "RH802", "RH8K6")
MOBILITY_START, MOBILITY_END = "2024-02-13", "2024-12-31"
SHORT_CONTEXT = 336
MOBILITY_VARIANTS = ("plain", "base", "prev_day", "oracle", "uni_oracle")
ISSUE_HOURS = (0, 6)  # local hour on the target day for the intraday issues


def setup(sites: tuple[str, ...] | None = None) -> tuple[dict[str, Any], poc.Panel]:
    config = load_config(CONFIG)
    panel = poc.load_panel(config)
    if sites:
        panel.sites = [s for s in sites if s in panel.y_raw.columns]
        panel.members = {s: {s: 1.0} for s in panel.sites}
        panel.site_meta = None
    return config, panel


def _pipe(threads: int):
    import torch

    from ..models.foundation import load_chronos2

    pipe = load_chronos2("cpu")
    torch.set_num_threads(threads)
    return pipe


def predict(
    pipe,
    X: np.ndarray,
    horizon: int,
    quantiles: list[float],
    windows: list[tuple[dict, dict]] | None = None,
    batch_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles (N, H, Q) and means (N, H) for contexts ``X``, with or without covariates."""
    import torch

    qs, pts = [], []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            if windows is None:
                batch = torch.tensor(X[start : start + batch_size]).unsqueeze(1)
                q, mean = pipe.predict_quantiles(
                    batch, prediction_length=horizon, quantile_levels=quantiles
                )
            else:
                inputs = [
                    {"target": X[i], "past_covariates": past, "future_covariates": future}
                    for i, (past, future) in enumerate(
                        windows[start : start + batch_size], start=start
                    )
                ]
                q, mean = pipe.predict_quantiles(
                    inputs,
                    prediction_length=horizon,
                    quantile_levels=quantiles,
                    batch_size=batch_size,
                )
            qs.append(_to_array(q, torch))
            pts.append(_to_array(mean, torch))
    return np.concatenate(qs), np.concatenate(pts)


def _run(pipe, panel, origins, config, name: str, windows=None, batch_size: int = 32):
    t0 = time.time()
    keys, X, horizon = contexts_for(panel, origins, config)
    Q, P = predict(pipe, X, horizon, list(config["quantiles"]), windows, batch_size)
    out = _assemble(name, keys, Q, P, panel, list(config["quantiles"]))
    logger.info("%s: %d forecasts in %.0f s", name, len(keys), time.time() - t0)
    return out


def run_daily(threads: int = 12, limit_days: int | None = None) -> pd.DataFrame:
    """Chronos-2 without covariates for every day of the test window."""
    config, panel = setup()
    origins = poc.make_origins(panel, config, config["test_start"], config["test_end"], 1)
    if limit_days:
        origins = origins[:limit_days]
    pipe = _pipe(threads)
    out = _run(pipe, panel, origins, config, "chronos2_daily", batch_size=64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "forecasts_daily.parquet", index=False)
    return out


def run_context(threads: int = 12, limit_origins: int | None = None) -> pd.DataFrame:
    """The proof-of-concept origins with shorter contexts."""
    config, panel = setup()
    origins = poc.make_origins(
        panel, config, config["test_start"], config["test_end"], int(config["origin_stride_days"])
    )
    if limit_origins:
        origins = origins[:limit_origins]
    pipe = _pipe(threads)
    frames = [
        _run(pipe, panel, origins, {**config, "context_periods": L}, f"chronos2_ctx{L}", None, 64)
        for L in CONTEXTS
    ]
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "forecasts_context.parquet", index=False)
    return out


def visits_on_grid(panel: poc.Panel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Daily footfall on the panel's half-hour grid: the site's count of the day (T x S), the
    count of the day before (T x S) and the university campus's count of the day (T,)."""
    v = pd.read_parquet(INTERIM_DIR / "mobility_visits_daily.parquet")
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    wide = v.pivot_table(index="date", columns="id", values="visits")
    days = pd.DatetimeIndex(panel.local_date.astype("datetime64[ns]"))
    same = np.column_stack([wide[s].reindex(days).to_numpy(dtype="float64") for s in panel.sites])
    prev = np.column_stack(
        [
            wide[s].reindex(days - pd.Timedelta(days=1)).to_numpy(dtype="float64")
            for s in panel.sites
        ]
    )
    uni = wide["university"].reindex(days).to_numpy(dtype="float64")
    return same, prev, uni


def mobility_windows(
    panel: poc.Panel, origins: list[poc.Origin], config: dict[str, Any], variant: str
) -> list[tuple[dict, dict]] | None:
    """Covariate windows for one footfall variant, in the order of ``contexts_for``."""
    if variant == "plain":
        return None
    L = int(config["context_periods"])
    H = int(max(o.target_pos[-1] - o.origin_pos for o in origins))
    cw = CovariateWindows(config, panel, "era5")
    same, prev, uni = visits_on_grid(panel)
    T = len(panel.index)
    out = []
    for o in origins:
        ctx = np.clip(np.arange(o.origin_pos - L + 1, o.origin_pos + 1), 0, T - 1)
        fut = np.clip(np.arange(o.origin_pos + 1, o.origin_pos + 1 + H), 0, T - 1)
        for j, s in enumerate(panel.sites):
            past, future = cw.window(o, j, s, L, H)
            if variant == "prev_day":
                past["visits_prev_day"] = clean(prev[ctx, j])  # past only: not known ahead
            elif variant == "oracle":
                past["visits"], future["visits"] = clean(same[ctx, j]), clean(same[fut, j])
            elif variant == "uni_oracle":
                past["uni_visits"], future["uni_visits"] = clean(uni[ctx]), clean(uni[fut])
            elif variant != "base":
                raise KeyError(variant)
            out.append((past, future))
    return out


def run_mobility(threads: int = 12, limit_days: int | None = None) -> pd.DataFrame:
    """Chronos-2 at the three footfall sites with the footfall given in context."""
    config, panel = setup(MOBILITY_SITES)
    origins = poc.make_origins(panel, config, MOBILITY_START, MOBILITY_END, 1)
    if limit_days:
        origins = origins[:limit_days]
    pipe = _pipe(threads)
    short = {**config, "context_periods": SHORT_CONTEXT}
    runs = [(v, config, f"mob_{v}") for v in MOBILITY_VARIANTS]
    runs += [(v, short, f"mob_{v}_ctx{SHORT_CONTEXT}") for v in ("base", "oracle")]
    frames = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant, cfg, name in runs:
        windows = mobility_windows(panel, origins, cfg, variant)
        frames.append(_run(pipe, panel, origins, cfg, name, windows, 32))
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT_DIR / "forecasts_mobility.partial.parquet", index=False
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_DIR / "forecasts_mobility.parquet", index=False)
    (OUT_DIR / "forecasts_mobility.partial.parquet").unlink(missing_ok=True)
    return out


def intraday_origins(
    panel: poc.Panel, config: dict[str, Any], start: str, end: str, stride_days: int, hour: int
) -> list[poc.Origin]:
    """Origins issued at ``hour`` local time on the target day, for the periods of that day
    after the issue (same convention for the last known period as ``poc.make_origins``)."""
    tz = config["timezone"]
    out = []
    for d in pd.date_range(start, end, freq=f"{stride_days}D"):
        target_day = np.datetime64(d.date(), "D")
        origin_time = d.replace(hour=hour, minute=0).tz_localize(tz).tz_convert("UTC")
        pos = int(panel.index.searchsorted(origin_time, side="right") - 1)
        tpos = np.flatnonzero(panel.local_date == target_day)
        tpos = tpos[tpos > pos]
        if pos < 0 or len(tpos) == 0:
            continue
        lead = (panel.index[tpos] - panel.index[pos]) / pd.Timedelta(hours=1)
        out.append(poc.Origin(target_day, origin_time, pos, tpos, lead.to_numpy()))
    return out


def run_intraday(threads: int = 12, limit_origins: int | None = None) -> pd.DataFrame:
    """Chronos-2 without covariates, issued on the target day itself."""
    config, panel = setup()
    pipe = _pipe(threads)
    frames = []
    for hour in ISSUE_HOURS:
        origins = intraday_origins(
            panel,
            config,
            config["test_start"],
            config["test_end"],
            int(config["origin_stride_days"]),
            hour,
        )
        if limit_origins:
            origins = origins[:limit_origins]
        frames.append(_run(pipe, panel, origins, config, f"chronos2_issue{hour:02d}", None, 64))
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "forecasts_intraday.parquet", index=False)
    return out
