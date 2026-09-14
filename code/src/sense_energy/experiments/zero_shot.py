"""Zero-shot foundation-model forecasts for the PoC origins.

Each (site, origin) pair is a context of the last ``context_periods``
half-hours up to the issue time; the model predicts far enough ahead to cover
the target day and the target periods are picked out. Outputs the same long
quantile frame as the baselines. With ``weather`` set to ``era5`` or ``ifs``
the known-ahead covariates of :mod:`fm_covariates` are passed as well.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..compute import autocast, select_device
from ..logging_utils import get_logger
from .fm_covariates import covariate_inputs
from .poc import Origin, Panel, _fill, forecast_frame

logger = get_logger(__name__)


def model_name(base: str, weather: str) -> str:
    return base if weather == "none" else f"{base}_{weather}"


def contexts_for(
    panel: Panel, origins: list[Origin], config: dict[str, Any]
) -> tuple[list[tuple[Origin, str]], np.ndarray, int]:
    """All (origin, site) contexts as one array, plus the horizon needed."""
    L = int(config["context_periods"])
    keys, rows = [], []
    for o in origins:
        for s in panel.sites:
            ctx = panel.y_imp[s].to_numpy()[max(0, o.origin_pos - L + 1) : o.origin_pos + 1]
            if len(ctx) < L:
                ctx = np.concatenate([np.full(L - len(ctx), np.nan), ctx])
            rows.append(_fill(ctx).astype("float32"))
            keys.append((o, s))
    horizon = int(max(o.target_pos[-1] - o.origin_pos for o in origins))
    return keys, np.stack(rows), horizon


def _assemble(
    model: str,
    keys,
    quantile_grid: np.ndarray,
    point: np.ndarray,
    panel: Panel,
    quantiles: list[float],
) -> pd.DataFrame:
    """quantile_grid: (N, H, Q); point: (N, H). Pick each origin's target steps."""
    frames = []
    for i, (o, s) in enumerate(keys):
        steps = o.target_pos - o.origin_pos - 1
        q = np.sort(quantile_grid[i, steps, :], axis=1)  # enforce monotone quantiles
        frames.append(forecast_frame(model, s, o, panel, quantiles, q, point[i, steps]))
    return pd.concat(frames, ignore_index=True)


def _to_array(x, torch) -> np.ndarray:
    """Chronos returns a tensor for 2-D input and, for 3-D input or dict input, a list with
    one ``(n_variates, ...)`` tensor per series; the single variate axis is dropped."""
    if isinstance(x, list | tuple):
        x = torch.stack([t[0] for t in x])
    x = x.float().cpu().numpy()
    return x[:, 0] if x.ndim == 4 else x


def run_chronos2(
    panel: Panel,
    origins: list[Origin],
    config: dict[str, Any],
    weather: str = "none",
    batch_size: int = 256,
) -> pd.DataFrame:
    import torch

    from ..models.foundation import load_chronos2

    name = model_name("chronos2", weather)
    quantiles = list(config["quantiles"])
    device = select_device()
    pipe = load_chronos2(device)
    keys, X, horizon = contexts_for(panel, origins, config)
    logger.info(
        "%s: %d contexts x %d periods, horizon %d, device %s",
        name,
        len(keys),
        X.shape[1],
        horizon,
        device,
    )
    qs, pts = [], []
    if weather == "none":
        with autocast(device):
            for start in range(0, len(keys), batch_size):
                batch = torch.tensor(X[start : start + batch_size]).unsqueeze(
                    1
                )  # (n_series, 1 variate, history)
                q, mean = pipe.predict_quantiles(
                    batch, prediction_length=horizon, quantile_levels=quantiles
                )
                qs.append(_to_array(q, torch))
                pts.append(_to_array(mean, torch))
    else:
        windows, names = covariate_inputs(panel, keys, config, weather, X.shape[1], horizon)
        chunk = 512  # dicts per call; Chronos batches the series inside
        with autocast(device):
            for start in range(0, len(keys), chunk):
                inputs = [
                    {"target": X[i], "past_covariates": past, "future_covariates": future}
                    for i, (past, future) in enumerate(windows[start : start + chunk], start=start)
                ]
                q, mean = pipe.predict_quantiles(
                    inputs,
                    prediction_length=horizon,
                    quantile_levels=quantiles,
                    batch_size=batch_size,
                )
                qs.append(_to_array(q, torch))
                pts.append(_to_array(mean, torch))
    Q, P = np.concatenate(qs), np.concatenate(pts)
    assert Q.shape[:2] == P.shape[:2] == (len(keys), horizon), (Q.shape, P.shape)
    del pipe
    torch.cuda.empty_cache()
    return _assemble(name, keys, Q, P, panel, quantiles)


def run_timesfm3(
    panel: Panel,
    origins: list[Origin],
    config: dict[str, Any],
    weather: str = "none",
    batch_size: int = 128,
) -> pd.DataFrame:
    import torch

    from ..models.foundation import load_timesfm3, timesfm_predict

    name = model_name("timesfm3", weather)
    quantiles = list(config["quantiles"])
    device = select_device()
    model = load_timesfm3(device)
    keys, X, horizon = contexts_for(panel, origins, config)
    logger.info("%s: %d contexts, horizon %d, device %s", name, len(keys), horizon, device)
    pf = None
    if weather != "none":
        windows, names = covariate_inputs(
            panel, keys, config, weather, X.shape[1], horizon, standardised=True
        )
        # TimesFM 3.0 takes one (n_covariates, past + future) array per series
        pf = [
            np.stack([np.concatenate([past[n], future[n]]) for n in names]).astype("float32")
            for past, future in windows
        ]
    qs, pts = [], []
    for start in range(0, len(keys), batch_size):
        extra = {}
        if pf is not None:
            extra = {
                "past_future_covariates": pf[start : start + batch_size],
                "padding_mode": "edge",
            }
        outs = timesfm_predict(model, [x for x in X[start : start + batch_size]], horizon, **extra)
        for out in outs:
            point = np.asarray(out.forecast, dtype="float32").reshape(-1)[:horizon]
            qarr = np.asarray(out.quantiles, dtype="float32")
            if qarr.ndim == 2 and qarr.shape[0] != horizon and qarr.shape[1] == horizon:
                qarr = qarr.T
            qarr = qarr[:horizon]
            if qarr.shape[1] >= 10:  # [mean, q10..q90] layout: keep the nine quantiles
                qarr = qarr[:, -9:]
            if qarr.shape[1] != len(quantiles):
                raise ValueError(f"unexpected TimesFM quantile layout {qarr.shape}")
            qs.append(qarr)
            pts.append(point)
    Q, P = np.stack(qs), np.stack(pts)
    del model
    torch.cuda.empty_cache()
    return _assemble(name, keys, Q, P, panel, quantiles)
