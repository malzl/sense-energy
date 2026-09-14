"""Ensemble-driven demand distributions: one Chronos-2 forecast per IFS ENS member.

The ``chronos2_ifs`` run feeds the model the *ensemble mean* of the 00z IFS
ENS run of D-1. Here the same model, contexts and covariates are run once per
member (51 members: control + 50 perturbed), so each member's weather yields
its own demand quantiles. Three products come out of it:

* ``chronos2_ifs_members`` - the mixture of the 51 member distributions
  (member CDFs averaged, then inverted on the reporting quantile grid): the
  demand distribution that carries the weather uncertainty
* ``chronos2_ifs_ctrl``    - the control member alone (a deterministic-weather
  baseline)
* ``ensemble_members_<model>.parquet`` - each member's median per (site,
  origin, target), for spread-skill analysis

Members are processed one at a time and checkpointed, so a long run resumes.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..compute import autocast, select_device
from ..logging_utils import get_logger
from .fm_covariates import CovariateWindows
from .poc import Origin, Panel, outputs_dir
from .zero_shot import _assemble, _to_array, contexts_for

logger = get_logger(__name__)


def mixture_quantiles(Q: np.ndarray, levels: list[float], grid_points: int = 400) -> np.ndarray:
    """Quantiles of the equal-weight mixture of member distributions.

    ``Q``: (members, H, K) member quantiles at ``levels``. Each member's CDF is
    piecewise linear through its quantiles, extended linearly into the tails
    with the slope of the outer intervals; the mixture CDF is their mean and is
    inverted on a common grid per horizon step.
    """
    M, H, K = Q.shape
    tau = np.asarray(levels, dtype="float64")
    out = np.empty((H, K))
    for h in range(H):
        q = np.sort(Q[:, h, :], axis=1)  # M x K
        lo_slope = np.maximum(q[:, 1] - q[:, 0], 1e-9)
        hi_slope = np.maximum(q[:, -1] - q[:, -2], 1e-9)
        # extend to tau=0 and tau=1
        q0 = q[:, 0] - lo_slope * tau[0] / (tau[1] - tau[0])
        q1 = q[:, -1] + hi_slope * (1 - tau[-1]) / (tau[-1] - tau[-2])
        knots = np.column_stack([q0, q, q1])  # M x (K+2)
        levels_ext = np.concatenate([[0.0], tau, [1.0]])
        x = np.linspace(knots.min(), knots.max(), grid_points)
        F = np.mean([np.interp(x, knots[m], levels_ext) for m in range(M)], axis=0)
        F = np.maximum.accumulate(F)
        out[h] = np.interp(tau, F, x)
    return out


def _predict(pipe, torch, inputs, horizon, quantiles, batch_size=256, chunk=512):
    qs, pts = [], []
    for start in range(0, len(inputs), chunk):
        q, mean = pipe.predict_quantiles(
            inputs[start : start + chunk],
            prediction_length=horizon,
            quantile_levels=quantiles,
            batch_size=batch_size,
        )
        qs.append(_to_array(q, torch))
        pts.append(_to_array(mean, torch))
    return np.concatenate(qs), np.concatenate(pts)


def run_chronos2_members(
    panel: Panel, origins: list[Origin], config: dict[str, Any]
) -> pd.DataFrame:
    import torch

    from ..models.foundation import load_chronos2

    quantiles = list(config["quantiles"])
    spec = config.get("ensemble", {})
    n_members = int(spec.get("members", 51))
    device = select_device()
    pipe = load_chronos2(device)
    keys, X, horizon = contexts_for(panel, origins, config)
    L = X.shape[1]
    cw = CovariateWindows(config, panel, "ifs")
    site_index = {s: j for j, s in enumerate(panel.sites)}
    logger.info(
        "chronos2 x %d IFS members: %d contexts, horizon %d, device %s",
        n_members,
        len(keys),
        horizon,
        device,
    )
    # shared parts of every window: the past, and the future non-weather covariates
    t0 = time.time()
    past, future, member_w = [], [], []
    for o, s in keys:
        p, f = cw.window(o, site_index[s], s, L, horizon)
        past.append(p)
        future.append(f)
        member_w.append(cw.member_weather(o, site_index[s], s, horizon))
    n_avail = min(w[cw.weather_vars[0]].shape[0] for w in member_w)
    n_members = min(n_members, n_avail)
    logger.info(
        "windows built in %.0f s; %d members available everywhere", time.time() - t0, n_members
    )

    out_dir = outputs_dir(config)
    store = out_dir / "ensemble_members_chronos2.parquet"
    done = set()
    frames = []
    if store.exists():
        prev = pd.read_parquet(store)
        frames.append(prev)
        done = set(prev["member"].unique())
        logger.info("resuming: members %s on disk", sorted(done))
    Qm: dict[int, np.ndarray] = {}
    with autocast(device):
        for m in range(n_members):
            t1 = time.time()
            inputs = []
            for i in range(len(keys)):
                fut = dict(future[i])
                for v in cw.weather_vars:
                    fut[v] = member_w[i][v][m].astype("float32")
                inputs.append(
                    {"target": X[i], "past_covariates": past[i], "future_covariates": fut}
                )
            Q, P = _predict(pipe, torch, inputs, horizon, quantiles)
            Qm[m] = Q
            if m not in done:
                rows = []
                for i, (o, s) in enumerate(keys):
                    steps = o.target_pos - o.origin_pos - 1
                    rows.append(
                        pd.DataFrame(
                            {
                                "site_code": s,
                                "origin": o.origin_time,
                                "target": panel.index[o.target_pos],
                                "member": m,
                                "median": Q[i, steps, quantiles.index(0.5)].astype("float32"),
                                "mean": P[i, steps].astype("float32"),
                            }
                        )
                    )
                frames.append(pd.concat(rows, ignore_index=True))
                pd.concat(frames, ignore_index=True).to_parquet(store, index=False)
            if m == 0:
                ctrl = _assemble("chronos2_ifs_ctrl", keys, Q, P, panel, quantiles)
                ctrl.to_parquet(out_dir / "forecasts_chronos2_ifs_ctrl.parquet", index=False)
            logger.info("member %d/%d in %.0f s", m + 1, n_members, time.time() - t1)
    del pipe
    torch.cuda.empty_cache()
    # mixture over members
    allQ = np.stack([Qm[m] for m in range(n_members)])  # M x N x H x K
    mix = np.empty(allQ.shape[1:])
    for i in range(allQ.shape[1]):
        mix[i] = mixture_quantiles(allQ[:, i], quantiles)
    point = np.mean(np.stack([Qm[m][:, :, quantiles.index(0.5)] for m in range(n_members)]), axis=0)
    return _assemble("chronos2_ifs_members", keys, mix, point, panel, quantiles)
