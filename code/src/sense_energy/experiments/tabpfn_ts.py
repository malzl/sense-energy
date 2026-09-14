"""TabPFN-TS for the PoC: one TabPFN regression per (site, origin), spread over the GPUs.

The published TabPFN-TS recipe is used unchanged: the context becomes a table
of the package's default temporal features (running index, calendar sine/cosine
terms, auto-detected seasonal terms) with demand as the target, the TabPFN v3
time-series checkpoint is fitted on it and queried at the future timestamps.
With ``weather`` set, the known-ahead covariates of :mod:`fm_covariates` are
extra columns of that table (known for the future rows too).

Each (site, origin) is an independent fit of a few seconds, so the items of a
chunk of origins are split across ``gpus x workers_per_gpu`` loky workers,
each holding one regressor on its device; the result frame is checkpointed
after every chunk so a long run can resume.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .fm_covariates import CovariateWindows
from .poc import Origin, Panel, _fill, forecast_frame, outputs_dir
from .zero_shot import model_name

logger = get_logger(__name__)


def _featurise(context_df: pd.DataFrame, future_df: pd.DataFrame, L: int):
    """The package's own preprocessing + feature generation on a long frame of items."""
    from tabpfn_time_series import TABPFN_TS_DEFAULT_FEATURES, TimeSeriesDataFrame
    from tabpfn_time_series.features import FeatureTransformer
    from tabpfn_time_series.pipeline import _featurize_context_future

    ctx = TimeSeriesDataFrame.from_data_frame(context_df)
    fut = TimeSeriesDataFrame.from_data_frame(future_df)
    train, test = _featurize_context_future(
        ctx, fut, FeatureTransformer(list(TABPFN_TS_DEFAULT_FEATURES)), L
    )
    cols = [c for c in train.columns if c != "target"]
    items = {}
    for item_id in train.item_ids:
        tr = train.loc[item_id]
        te = test.loc[item_id]
        items[item_id] = (
            tr[cols].to_numpy(dtype="float32"),
            tr["target"].to_numpy(dtype="float32"),
            te[cols].to_numpy(dtype="float32"),
        )
    return items, cols


def worker(device: str, ckpt: str, n_estimators: int, quantiles: list[float], items: list):
    """Fit-and-predict every item on one device with one regressor."""
    import warnings

    import torch
    from tabpfn import TabPFNRegressor
    from threadpoolctl import threadpool_limits

    warnings.simplefilter("ignore")
    torch.set_num_threads(2)  # the work is on the GPU; leave the cores to the CPU models
    reg = TabPFNRegressor(model_path=ckpt, device=device, n_estimators=int(n_estimators))
    with threadpool_limits(2):
        return _predict_items(reg, quantiles, items)


def _predict_items(reg, quantiles: list[float], items: list) -> list:
    out = []
    for key, X, y, Xf in items:
        reg.fit(X, y)
        pred = reg.predict(Xf, output_type="main", quantiles=list(quantiles))
        q = np.stack([np.asarray(v, dtype="float32") for v in pred["quantiles"]], axis=1)
        out.append((key, q, np.asarray(pred["median"], dtype="float32")))
    return out


def _devices(spec: dict) -> list[str]:
    import torch

    gpus = spec.get("gpus", "all")
    ids = list(range(torch.cuda.device_count())) if gpus == "all" else [int(g) for g in gpus]
    if not ids:
        raise RuntimeError("TabPFN-TS needs at least one CUDA device")
    return [f"cuda:{i}" for i in ids]


def run_tabpfn_ts(
    panel: Panel, origins: list[Origin], config: dict[str, Any], weather: str = "none"
) -> pd.DataFrame:
    from joblib import Parallel, delayed

    from ..models.foundation import tabpfn_ts_checkpoint

    name = model_name("tabpfn_ts", weather)
    spec = dict(config.get("tabpfn", {}))
    quantiles = [float(q) for q in config["quantiles"]]
    L = int(spec.get("context_periods", config["context_periods"]))
    horizon = int(max(o.target_pos[-1] - o.origin_pos for o in origins))
    devices = _devices(spec)
    workers = devices * int(spec.get("workers_per_gpu", 1))
    chunk_origins = int(spec.get("chunk_origins", 2))
    ckpt = tabpfn_ts_checkpoint()
    sites = panel.sites
    cw = CovariateWindows(config, panel, weather) if weather != "none" else None
    site_index = {s: j for j, s in enumerate(sites)}
    ts_utc = panel.index.tz_convert(None).to_numpy()  # regular naive UTC stamps
    T = len(panel.index)

    partial = Path(outputs_dir(config)) / f"forecasts_{name}.partial.parquet"
    frames, done = [], set()
    if partial.exists():
        prev = pd.read_parquet(partial)
        frames.append(prev)
        done = set(pd.to_datetime(prev["origin"], utc=True).unique())
        logger.info("%s: resuming, %d origins already on disk", name, len(done))
    todo = [o for o in origins if o.origin_time not in done]
    logger.info(
        "%s: %d origins x %d sites, context %d, horizon %d, workers %s",
        name,
        len(todo),
        len(sites),
        L,
        horizon,
        workers,
    )
    t0, n_done = time.time(), 0
    with Parallel(n_jobs=len(workers), backend="loky") as parallel:
        for c0 in range(0, len(todo), chunk_origins):
            chunk = todo[c0 : c0 + chunk_origins]
            ctx_rows, fut_rows, keys = [], [], []
            for o in chunk:
                ctx = np.clip(np.arange(o.origin_pos - L + 1, o.origin_pos + 1), 0, T - 1)
                fut = np.clip(np.arange(o.origin_pos + 1, o.origin_pos + 1 + horizon), 0, T - 1)
                for s in sites:
                    key = f"{s}|{o.origin_time.value}"
                    keys.append((key, o, s))
                    y = _fill(panel.y_imp[s].to_numpy()[ctx]).astype("float32")
                    c = {"item_id": key, "timestamp": ts_utc[ctx], "target": y}
                    f = {"item_id": key, "timestamp": ts_utc[fut]}
                    if cw is not None:
                        past, future = cw.window(o, site_index[s], s, L, horizon)
                        c |= past
                        f |= future
                    ctx_rows.append(pd.DataFrame(c))
                    fut_rows.append(pd.DataFrame(f))
            items, cols = _featurise(
                pd.concat(ctx_rows, ignore_index=True), pd.concat(fut_rows, ignore_index=True), L
            )
            parts = [[] for _ in workers]
            for i, (key, _, _) in enumerate(keys):
                X, y, Xf = items[key]
                parts[i % len(workers)].append((key, X, y, Xf))
            results = parallel(
                delayed(worker)(dev, ckpt, spec.get("n_estimators", 8), quantiles, part)
                for dev, part in zip(workers, parts, strict=True)
            )
            got = {key: (q, med) for part in results for key, q, med in part}
            for key, o, s in keys:
                q, med = got[key]
                steps = o.target_pos - o.origin_pos - 1
                frames.append(
                    forecast_frame(
                        name, s, o, panel, quantiles, np.sort(q[steps], axis=1), med[steps]
                    )
                )
            n_done += len(chunk)
            pd.concat(frames, ignore_index=True).to_parquet(partial, index=False)
            el = time.time() - t0
            logger.info(
                "%s: %d/%d origins (%d features), %.0f s elapsed, ~%.0f s left",
                name,
                n_done,
                len(todo),
                len(cols),
                el,
                el / n_done * (len(todo) - n_done),
            )
    out = pd.concat(frames, ignore_index=True)
    partial.unlink(missing_ok=True)
    logger.info("%s: %s forecast rows", name, f"{len(out):,}")
    return out
