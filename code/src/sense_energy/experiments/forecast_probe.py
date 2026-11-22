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
import warnings
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
PLANT_VARIANTS = ("state_oracle", "state_past", "cross", "cross_flag")
STATE_STEP = 0.25  # import this share of the site's mean away from its usual profile
CROSS_BATCH = 91  # one batch = every site of one origin (the model sees all sites together)


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
    cross_learning: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Quantiles (N, H, Q) and means (N, H) for contexts ``X``, with or without covariates.
    ``cross_learning`` lets the model share information between the series of a batch."""
    import torch

    qs, pts = [], []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            if windows is None:
                batch = torch.tensor(X[start : start + batch_size]).unsqueeze(1)
                q, mean = pipe.predict_quantiles(
                    batch,
                    prediction_length=horizon,
                    quantile_levels=quantiles,
                    batch_size=batch_size,
                    cross_learning=cross_learning,
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
                    cross_learning=cross_learning,
                )
            qs.append(_to_array(q, torch))
            pts.append(_to_array(mean, torch))
    return np.concatenate(qs), np.concatenate(pts)


def _run(pipe, panel, origins, config, name: str, windows=None, batch_size: int = 32, cross=False):
    t0 = time.time()
    keys, X, horizon = contexts_for(panel, origins, config)
    Q, P = predict(pipe, X, horizon, list(config["quantiles"]), windows, batch_size, cross)
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


LONG_CONTEXTS = (4032, 8064)  # 12 and 24 weeks of half-hours


def run_context_long(device: str | None = None) -> pd.DataFrame:
    """The proof-of-concept origins with 12 and 24 weeks of context (``run_context`` stops at
    two weeks and the proof of concept uses six). Spoek et al. (2026) find 12 weeks best for
    district heat load; this is the same test on our data. Uses a GPU if one is visible."""
    from ..compute import autocast, select_device
    from ..models.foundation import load_chronos2

    config, panel = setup()
    origins = poc.make_origins(
        panel, config, config["test_start"], config["test_end"], int(config["origin_stride_days"])
    )
    device = device or select_device()
    pipe = load_chronos2(device)
    frames = []
    for periods in LONG_CONTEXTS:
        with autocast(device):
            cfg = {**config, "context_periods": periods}
            frames.append(_run(pipe, panel, origins, cfg, f"chronos2_ctx{periods}", None, 128))
    out = pd.concat(frames, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / "forecasts_context_long.parquet", index=False)
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


def plant_state(panel: poc.Panel, config: dict[str, Any]) -> np.ndarray:
    """A three-level stand-in for a plant schedule, from the import itself (T x S).

    +1 where the import is more than ``STATE_STEP`` of the site's mean above its usual profile
    (the plant is off when it usually runs), -1 where it is as far below, 0 otherwise. The usual
    profile is the median of the same half-hour and weekday in the last four weeks; the residual
    is smoothed over two hours. The signal is made from the quantity to forecast, so its value
    on the target day is an upper bound for what a schedule from the site could give."""
    y = panel.y_imp[panel.sites]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-missing slots stay missing
        usual = np.nanmedian(np.stack([y.shift(336 * k).to_numpy() for k in (1, 2, 3, 4)]), axis=0)
    r = (y - usual).rolling(5, center=True, min_periods=3).median()
    before = panel.index < pd.Timestamp(config["test_start"], tz="UTC")
    level = y[before].mean().to_numpy()
    state = np.sign(r.to_numpy()) * (np.abs(r.to_numpy()) > STATE_STEP * level)
    return np.nan_to_num(state).astype("float32")


def plant_windows(
    panel: poc.Panel, origins: list[poc.Origin], config: dict[str, Any], variant: str
) -> list[tuple[dict, dict]] | None:
    """Covariate windows for one plant variant, in the order of ``contexts_for``. No other
    covariate is given, so each variant compares with Chronos-2 without covariates."""
    if variant == "cross":
        return None
    L = int(config["context_periods"])
    H = int(max(o.target_pos[-1] - o.origin_pos for o in origins))
    T = len(panel.index)
    if variant == "cross_flag":
        chp = pd.read_csv(PROJECT_ROOT / "code/reports/chp_dispatch/classification.csv")
        share = chp.set_index("site_code")["chp_share"].reindex(panel.sites).fillna(0.0)
        out = []
        for _ in origins:
            for s in panel.sites:
                flag, sh = float(share[s] > 0), float(share[s])
                past = {
                    "has_chp": np.full(L, flag, "float32"),
                    "chp_share": np.full(L, sh, "float32"),
                }
                future = {
                    "has_chp": np.full(H, flag, "float32"),
                    "chp_share": np.full(H, sh, "float32"),
                }
                out.append((past, future))
        return out
    state = plant_state(panel, config)
    out = []
    for o in origins:
        ctx = np.clip(np.arange(o.origin_pos - L + 1, o.origin_pos + 1), 0, T - 1)
        fut = np.clip(np.arange(o.origin_pos + 1, o.origin_pos + 1 + H), 0, T - 1)
        for j, _ in enumerate(panel.sites):
            past = state[ctx, j].copy()
            past[-2:] = past[-3]  # the smoothing window looks one hour ahead: hold the last value
            if variant == "state_past":
                out.append(({"plant_state": past}, {}))
            elif variant == "state_oracle":
                out.append(({"plant_state": past}, {"plant_state": state[fut, j]}))
            else:
                raise KeyError(variant)
    return out


def run_plant(
    threads: int = 8,
    limit_origins: int | None = None,
    variants: tuple[str, ...] = PLANT_VARIANTS,
    tag: str = "",
) -> pd.DataFrame:
    """Chronos-2 with knowledge of the plant: a flag for on-site generation that the model can
    compare across the sites of a batch, and the state of the plant (past only, and with the
    target day as an upper bound)."""
    config, panel = setup()
    origins = poc.make_origins(
        panel, config, config["test_start"], config["test_end"], int(config["origin_stride_days"])
    )
    if limit_origins:
        origins = origins[:limit_origins]
    pipe = _pipe(threads)
    frames = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        cross = variant.startswith("cross")
        windows = plant_windows(panel, origins, config, variant)
        batch = CROSS_BATCH if cross else 32
        frames.append(_run(pipe, panel, origins, config, f"plant_{variant}", windows, batch, cross))
        pd.concat(frames, ignore_index=True).to_parquet(
            OUT_DIR / f"forecasts_plant{tag}.partial.parquet", index=False
        )
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_DIR / f"forecasts_plant{tag}.parquet", index=False)
    (OUT_DIR / f"forecasts_plant{tag}.partial.parquet").unlink(missing_ok=True)
    return out


RECORD_START, RECORD_END = "2023-01-18", "2026-05-08"
RECORD_CHUNK_DAYS = 30
RECORD_COLUMNS = ["model", "site_code", "target", "point", "q10", "q30", "q50", "q70", "q90"]


def run_record(
    weather: str = "none",
    device: str | None = None,
    start: str = RECORD_START,
    end: str = RECORD_END,
    batch_size: int = 256,
) -> pd.DataFrame:
    """Chronos-2, zero-shot, for every day of the electricity record at the proof-of-concept
    sites (from six weeks after the first meter reading). The model has no training period, so
    the whole record is a test: three summers and four winters instead of one test year.

    ``weather="none"`` gives no covariates; ``"ifs"`` gives the proof of concept's known-ahead
    covariates with the weather forecast of the day before (the 00z IFS ENS run). Written in
    chunks of days to ``forecasts_record_<weather>.parquet`` (point and five quantiles)."""
    from ..compute import autocast, select_device
    from ..models.foundation import load_chronos2

    config, panel = setup()
    origins = poc.make_origins(panel, config, start, end, 1)
    device = device or select_device()
    pipe = load_chronos2(device)
    quantiles = list(config["quantiles"])
    name = "chronos2_record" if weather == "none" else f"chronos2_{weather}_record"
    cw = None if weather == "none" else CovariateWindows(config, panel, weather)
    L = int(config["context_periods"])
    frames, t0 = [], time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for a in range(0, len(origins), RECORD_CHUNK_DAYS):
        chunk = origins[a : a + RECORD_CHUNK_DAYS]
        keys, X, horizon = contexts_for(panel, chunk, config)
        windows = None
        if cw is not None:
            index = {s: j for j, s in enumerate(panel.sites)}
            windows = [cw.window(o, index[s], s, L, horizon) for o, s in keys]
        with autocast(device):
            Q, P = predict(pipe, X, horizon, quantiles, windows, batch_size)
        out = _assemble(name, keys, Q.astype("float32"), P.astype("float32"), panel, quantiles)
        frames.append(out[RECORD_COLUMNS])
        done = min(a + RECORD_CHUNK_DAYS, len(origins))
        print(f"{name}: {done}/{len(origins)} days, {time.time() - t0:.0f} s", flush=True)
    out = pd.concat(frames, ignore_index=True)
    out.to_parquet(OUT_DIR / f"forecasts_record_{weather}.parquet", index=False)
    return out
