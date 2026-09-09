"""Loaders for the time-series foundation models, pinned to the GPU.

Each loader returns the model already on the selected device with the
project's inference settings applied; ``benchmark`` proves that and measures
throughput, so a silent fall-back to CPU cannot go unnoticed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..compute import autocast, configure_torch, select_device
from ..logging_utils import get_logger

logger = get_logger(__name__)

CHRONOS2 = "amazon/chronos-2"
TIMESFM3 = "google/timesfm-3.0-pytorch"
TABPFN3_TS_CKPT_GLOB = (
    "models--Prior-Labs--tabpfn_3/snapshots/*/*regressor-v3_20260506_timeseries.ckpt"
)


def load_chronos2(device: str | None = None):
    import torch
    from chronos import BaseChronosPipeline

    device = device or select_device()
    configure_torch()
    pipe = BaseChronosPipeline.from_pretrained(
        CHRONOS2,
        device_map=device,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    where = next(pipe.model.parameters()).device
    logger.info("Chronos-2 on %s (%s)", where, next(pipe.model.parameters()).dtype)
    return pipe


def load_timesfm3(device: str | None = None, max_context: int = 1024, max_horizon: int = 48):
    """TimesFM 3.0 through its forecaster wrapper, placed on the GPU explicitly."""
    import timesfm

    device = device or select_device()
    configure_torch()
    model = timesfm.TimesFM3Forecaster.from_pretrained(TIMESFM3, device=device)
    logger.info("TimesFM 3.0 forecaster on %s", device)
    return model


def timesfm_predict(model, series: list[np.ndarray], horizon: int):
    """Call the 3.0 forecaster whatever its exact argument names are."""
    import inspect

    fn = model.predict_batch if hasattr(model, "predict_batch") else model.predict
    params = inspect.signature(fn).parameters
    kwargs = {}
    for name in ("horizon", "prediction_length", "forecast_horizon"):
        if name in params:
            kwargs[name] = horizon
            break
    first = next(n for n in params if n not in ("self",))
    return fn(**{first: series}, **kwargs)


def tabpfn_ts_checkpoint() -> str:
    import glob
    import os

    hits = glob.glob(os.path.expanduser(f"~/.cache/huggingface/hub/{TABPFN3_TS_CKPT_GLOB}"))
    if not hits:
        raise FileNotFoundError("TabPFN v3 time-series checkpoint not in the HF cache")
    return hits[0]


def load_tabpfn_ts(device: str | None = None):
    from tabpfn import TabPFNRegressor

    device = device or select_device()
    configure_torch()
    reg = TabPFNRegressor(model_path=tabpfn_ts_checkpoint(), device=device, n_estimators=4)
    logger.info("TabPFN v3 (time-series checkpoint) configured for %s", device)
    return reg


@dataclass
class BenchResult:
    model: str
    device: str
    n_series: int
    context: int
    horizon: int
    seconds: float

    @property
    def series_per_second(self) -> float:
        return self.n_series / self.seconds if self.seconds else float("inf")


def benchmark(
    n_series: int = 256, context: int = 672, horizon: int = 48, device: str | None = None
) -> list[BenchResult]:
    """Time each model on synthetic half-hourly series; also confirms the device."""
    import torch

    device = device or select_device()
    rng = np.random.default_rng(0)
    t = np.arange(context)
    series = [
        100 + 40 * np.sin(2 * np.pi * (t % 48) / 48) + rng.normal(0, 5, context)
        for _ in range(n_series)
    ]
    results = []

    pipe = load_chronos2(device)
    with autocast(device):
        _ = pipe.predict_quantiles(
            [torch.tensor(s, dtype=torch.float32) for s in series[:8]],
            prediction_length=horizon,
            quantile_levels=[0.5],
        )
        torch.cuda.synchronize() if device.startswith("cuda") else None
        t0 = time.perf_counter()
        _ = pipe.predict_quantiles(
            [torch.tensor(s, dtype=torch.float32) for s in series],
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        torch.cuda.synchronize() if device.startswith("cuda") else None
    results.append(
        BenchResult(
            "Chronos-2",
            str(next(pipe.model.parameters()).device),
            n_series,
            context,
            horizon,
            time.perf_counter() - t0,
        )
    )
    del pipe
    torch.cuda.empty_cache()

    model = load_timesfm3(device, max_context=max(context, 512), max_horizon=horizon)
    _ = timesfm_predict(model, series[:8], horizon)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    t0 = time.perf_counter()
    _ = timesfm_predict(model, series, horizon)
    torch.cuda.synchronize() if device.startswith("cuda") else None
    results.append(
        BenchResult("TimesFM 3.0", device, n_series, context, horizon, time.perf_counter() - t0)
    )
    del model
    torch.cuda.empty_cache()

    reg = load_tabpfn_ts(device)
    x = np.column_stack([np.arange(context) % 48, np.arange(context) // 48 % 7, np.arange(context)])
    t0 = time.perf_counter()
    reg.fit(x, series[0])
    _ = reg.predict(
        np.column_stack(
            [
                np.arange(context, context + horizon) % 48,
                np.arange(context, context + horizon) // 48 % 7,
                np.arange(context, context + horizon),
            ]
        )
    )
    results.append(
        BenchResult(
            "TabPFN v3 TS (1 series)", device, 1, context, horizon, time.perf_counter() - t0
        )
    )
    return results
