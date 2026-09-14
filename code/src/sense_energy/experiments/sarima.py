"""SARIMA baseline for the PoC, fitted per site and origin across all cores.

Model (dynamic harmonic regression, Hyndman & Athanasopoulos ch. 12): on the
log scale, demand = weekly Fourier terms (K harmonics of the 336-period week)
+ SARIMA(p, d, q)(P, D, Q)_48 errors. The daily season enters through the
seasonal part with period 48, the weekly season through the Fourier
regressors: a seasonal period of 336 makes the state-space model hundreds of
times slower for no gain in accuracy.

Parameters are re-estimated at every origin on the same 6-week context the
zero-shot models see (``refit: origin``), warm-started from a per-site fit on
the last weeks of the training period; ``refit: site`` keeps the training
parameters and only filters the context. Gaps stay NaN - the Kalman filter
skips missing observations - so nothing is imputed for this model beyond the
panel's own short-gap imputation. Quantiles come from the Gaussian predictive
distribution on the fitting scale and are mapped back (the point forecast is
the median).

Every fit is one joblib task on a loky worker with BLAS pinned to one thread,
so 6,000 fits of ~15 s each finish in under an hour on 36 cores.
"""

from __future__ import annotations

import time
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm

from ..logging_utils import get_logger
from .poc import Origin, Panel, forecast_frame

logger = get_logger(__name__)

WEEK = 336


def fourier_week(hw: np.ndarray, k: int) -> np.ndarray:
    """Sine/cosine pairs of the first ``k`` harmonics of the week (``hw`` = half-hour of week)."""
    if k <= 0:
        return np.empty((len(hw), 0))
    t = 2 * np.pi * np.asarray(hw, dtype=float) / WEEK
    return np.column_stack([f(j * t) for j in range(1, k + 1) for f in (np.sin, np.cos)])


def half_hour_of_week(panel: Panel) -> np.ndarray:
    return panel.dow.astype(int) * 48 + panel.tod.astype(int)


def transform(y: np.ndarray, kind: str) -> np.ndarray:
    """Fitting scale: ``log`` (floored at 1% of the context median) or ``none``."""
    if kind == "none":
        return np.asarray(y, dtype=float)
    if kind != "log":
        raise ValueError(kind)
    med = np.nanmedian(y)
    floor = max(1e-3, 0.01 * med) if np.isfinite(med) and med > 0 else 1e-3
    return np.log(np.clip(np.asarray(y, dtype=float), floor, None))


def back_transform(x: np.ndarray, kind: str) -> np.ndarray:
    return np.exp(x) if kind == "log" else x


def _model(y, X, spec):
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    return SARIMAX(
        y,
        exog=X if X is not None and X.shape[1] else None,
        order=tuple(int(v) for v in spec["order"]),
        seasonal_order=tuple(int(v) for v in spec["seasonal_order"]),
        trend="n",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )


def fit_task(y: np.ndarray, X: np.ndarray, spec: dict, start_params, maxiter: int) -> dict:
    """Maximum-likelihood fit on one window; returns the parameters and convergence flag."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            res = _model(y, X, spec).fit(
                start_params=start_params, disp=False, maxiter=int(maxiter), method="lbfgs"
            )
        except Exception as exc:  # noqa: BLE001 - a failed fit is reported, not fatal
            return {"params": None, "converged": False, "error": repr(exc)}
    return {
        "params": np.asarray(res.params, dtype=float),
        "converged": bool(res.mle_retvals.get("converged", False)),
        "error": None,
    }


def forecast_task(
    y: np.ndarray,
    X: np.ndarray,
    X_future: np.ndarray,
    steps: int,
    spec: dict,
    params: np.ndarray | None,
    refit: bool,
    maxiter: int,
    z: np.ndarray,
) -> dict:
    """One (site, origin): refit (warm-started) or filter, then Gaussian quantiles ahead."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(1), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = _model(y, X, spec)
            if refit or params is None:
                res = model.fit(
                    start_params=params, disp=False, maxiter=int(maxiter), method="lbfgs"
                )
                converged = bool(res.mle_retvals.get("converged", False))
            else:
                res = model.filter(params)
                converged = True
            fc = res.get_forecast(int(steps), exog=X_future if X_future.shape[1] else None)
            mu = np.asarray(fc.predicted_mean, dtype=float)
            se = np.asarray(fc.se_mean, dtype=float)
            if not (np.isfinite(mu).all() and np.isfinite(se).all()):
                raise ValueError("non-finite forecast")
        except Exception as exc:  # noqa: BLE001
            return {"q": None, "mu": None, "converged": False, "error": repr(exc)}
    q = mu[:, None] + z[None, :] * se[:, None]
    return {"q": q, "mu": mu, "converged": converged, "error": None}


def _context(Y: np.ndarray, X_all: np.ndarray, j: int, o: Origin, L: int, kind: str):
    lo = max(0, o.origin_pos - L + 1)
    y = Y[lo : o.origin_pos + 1, j]
    X = X_all[lo : o.origin_pos + 1]
    steps = int(o.target_pos[-1] - o.origin_pos)
    X_future = X_all[o.origin_pos + 1 : o.origin_pos + 1 + steps]
    return transform(y, kind), X, X_future, steps


def run_sarima(panel: Panel, origins: list[Origin], config: dict[str, Any]) -> pd.DataFrame:
    from joblib import Parallel, delayed

    spec = dict(config["sarima"])
    quantiles = list(config["quantiles"])
    z = norm.ppf(quantiles)
    kind = str(spec.get("transform", "log"))
    L = int(config["context_periods"])
    refit = str(spec.get("refit", "origin")) == "origin"
    n_jobs = int(spec.get("n_jobs", 32))
    sites = panel.sites
    Y = panel.y_imp[sites].to_numpy(dtype=float)  # gaps stay NaN
    X_all = fourier_week(half_hour_of_week(panel), int(spec.get("weekly_fourier_k", 3)))
    logger.info(
        "SARIMA%s%s + %d weekly harmonics on %s scale, refit per %s, %d workers",
        tuple(spec["order"]),
        tuple(spec["seasonal_order"]),
        int(spec.get("weekly_fourier_k", 3)),
        kind,
        "origin" if refit else "site",
        n_jobs,
    )

    # 1. Per-site initial fits on the last weeks of the training period (warm starts).
    end = int(panel.index.searchsorted(pd.Timestamp(config["train_end"], tz="UTC"), "right")) - 1
    start = max(0, end - int(spec.get("init_weeks", 8)) * WEEK + 1)
    t0 = time.time()
    with Parallel(n_jobs=n_jobs, backend="loky") as parallel:
        inits = parallel(
            delayed(fit_task)(
                transform(Y[start : end + 1, j], kind),
                X_all[start : end + 1],
                spec,
                None,
                int(spec.get("maxiter", 50)),
            )
            for j in range(len(sites))
        )
        site_params = {s: r["params"] for s, r in zip(sites, inits, strict=True)}
        n_conv = sum(r["converged"] for r in inits)
        n_fail = sum(r["params"] is None for r in inits)
        logger.info(
            "initial fits: %d sites in %.0f s (%d converged, %d failed)",
            len(sites),
            time.time() - t0,
            n_conv,
            n_fail,
        )

        # 2. One task per (origin, site): warm-started refit (or filter) and forecast.
        keys = [(o, j) for o in origins for j in range(len(sites))]

        def tasks():
            for o, j in keys:
                y, X, X_future, steps = _context(Y, X_all, j, o, L, kind)
                yield delayed(forecast_task)(
                    y,
                    X,
                    X_future,
                    steps,
                    spec,
                    site_params[sites[j]],
                    refit,
                    int(spec.get("maxiter_refit", 30)),
                    z,
                )

        frames, n_conv, n_fail, t0 = [], 0, 0, time.time()
        results = Parallel(
            n_jobs=n_jobs, backend="loky", return_as="generator", pre_dispatch="2*n_jobs"
        )(tasks())
        for i, ((o, j), r) in enumerate(zip(keys, results, strict=True), start=1):
            n = len(o.target_pos)
            if r["q"] is None:
                n_fail += 1
                q = np.full((n, len(quantiles)), np.nan, dtype="float32")
                point = np.full(n, np.nan, dtype="float32")
            else:
                n_conv += int(r["converged"])
                steps = o.target_pos - o.origin_pos - 1
                q = back_transform(r["q"][steps], kind)
                point = back_transform(r["mu"][steps], kind)
            frames.append(forecast_frame("sarima", sites[j], o, panel, quantiles, q, point))
            if i % 250 == 0 or i == len(keys):
                el = time.time() - t0
                logger.info(
                    "sarima: %d/%d fits, %.0f s elapsed, ~%.0f s left (%d converged, %d failed)",
                    i,
                    len(keys),
                    el,
                    el / i * (len(keys) - i),
                    n_conv,
                    n_fail,
                )
    out = pd.concat(frames, ignore_index=True)
    logger.info("sarima: %s forecast rows", f"{len(out):,}")
    return out
