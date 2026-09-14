"""Hierarchical reconciliation of the PoC point forecasts.

The demand hierarchy is exact in the data: a site is the sum of its active
meters, a trust the sum of its sites, a region the sum of its trusts and the
national total the sum of the regions (over the 91 PoC sites). Every node has a
*base* forecast from the per-level runs; a reconciliation maps the base
forecasts of all nodes to a coherent set through ``S @ G`` (Hyndman et al.):

* ``base``        - the direct forecasts as they are (not coherent)
* ``bottom_up``   - sum the meters upwards
* ``top_down``    - split the national forecast by the meters' shares over the
                    last 28 days before the origin
* ``middle_out``  - trust forecasts split to meters by recent shares, summed above
* ``ols``         - least-squares projection (Hyndman et al. 2011)
* ``wls_struct``  - weights = number of meters under each node
* ``wls_var``     - weights = variance of each node's past base errors
* ``mint_shrink`` - full shrinkage covariance of past base errors (Wickramasuriya
                    et al. 2019, Schaefer-Strimmer shrinkage)

Past errors are those of earlier origins only (expanding window), so the
variance-based methods use nothing from the target day or later; every method
is scored on the same origins after a warm-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .hierarchy import LEVELS, group_of
from .poc import load_panel, outputs_dir

logger = get_logger(__name__)

METHODS = [
    "base",
    "bottom_up",
    "top_down",
    "middle_out",
    "ols",
    "wls_struct",
    "wls_var",
    "mint_shrink",
]


@dataclass
class Hierarchy:
    nodes: list[str]  # top to bottom
    level_of: dict[str, str]
    bottom: list[str]
    S: np.ndarray  # n x m summing matrix
    y: pd.DataFrame  # observed values, T x n (NaN where not strictly observed)
    y_imp: pd.DataFrame  # short-gap-imputed values, T x n
    parent_trust: dict[str, str]  # bottom node -> trust node


def build_hierarchy(config: dict[str, Any]) -> Hierarchy:
    panels = {lvl: load_panel({**config, "level": lvl}) for lvl in LEVELS}
    site = panels["site"]
    meter = panels["meter"]
    site_of = {m: next(iter(d)) for m, d in meter.members.items()}
    bottom = [m for m in meter.sites if site_of[m] in set(site.sites)]
    trust_of = group_of(site.site_meta, "trust").to_dict()
    region_of = group_of(site.site_meta, "region").to_dict()
    nodes = (
        panels["total"].sites + panels["region"].sites + panels["trust"].sites + site.sites + bottom
    )
    level_of = {n: lvl for lvl in LEVELS for n in panels[lvl].sites}
    ancestors = {}
    for m in bottom:
        s = site_of[m]
        ancestors[m] = {m, s, trust_of[s], region_of[s], "national"}
    S = np.zeros((len(nodes), len(bottom)))
    for j, m in enumerate(bottom):
        for i, n in enumerate(nodes):
            if n in ancestors[m]:
                S[i, j] = 1.0
    y = pd.concat([panels[lvl].y_raw[panels[lvl].sites] for lvl in LEVELS], axis=1)[nodes]
    y_imp = pd.concat([panels[lvl].y_imp[panels[lvl].sites] for lvl in LEVELS], axis=1)[nodes]
    y = y.loc[:, ~y.columns.duplicated()]
    y_imp = y_imp.loc[:, ~y_imp.columns.duplicated()]
    parent_trust = {m: trust_of[site_of[m]] for m in bottom}
    logger.info(
        "hierarchy: %d nodes (%s), %d bottom meters",
        len(nodes),
        {lvl: len(panels[lvl].sites) for lvl in LEVELS},
        len(bottom),
    )
    return Hierarchy(nodes, level_of, bottom, S, y[nodes], y_imp[nodes], parent_trust)


def base_forecasts(config: dict[str, Any], model: str, h: Hierarchy) -> pd.DataFrame:
    """Base point forecasts of ``model`` at every node: rows (origin, target), columns nodes."""
    frames = []
    for lvl in LEVELS:
        p = outputs_dir(config, lvl) / (
            "forecasts_baselines.parquet"
            if model in ("seasonal_naive", "profile_quantiles")
            else f"forecasts_{model}.parquet"
        )
        if not p.exists():
            raise FileNotFoundError(f"{model} has no {lvl}-level forecasts ({p})")
        f = pd.read_parquet(p, columns=["model", "site_code", "origin", "target", "point"])
        frames.append(f[f["model"] == model])
    f = pd.concat(frames, ignore_index=True)
    f["origin"] = pd.to_datetime(f["origin"], utc=True)
    f["target"] = pd.to_datetime(f["target"], utc=True)
    wide = f.pivot_table(index=["origin", "target"], columns="site_code", values="point")
    missing = [n for n in h.nodes if n not in wide.columns]
    if missing:
        raise KeyError(f"{model}: no base forecasts for nodes {missing[:5]}...")
    return wide[h.nodes].astype("float64")


def _shrink_cov(R: np.ndarray) -> np.ndarray:
    """Schaefer-Strimmer shrinkage of the sample covariance toward its diagonal."""
    n, p = R.shape
    Rc = R - R.mean(axis=0)
    cov = Rc.T @ Rc / n
    sd = np.sqrt(np.diag(cov))
    sd = np.where(sd > 0, sd, 1.0)
    X = Rc / sd
    corr = X.T @ X / n
    # variance of the correlation estimates (off-diagonal)
    var_corr = ((X[:, :, None] * X[:, None, :]) ** 2).sum(axis=0) / n**2 - corr**2 / n
    off = ~np.eye(p, dtype=bool)
    lam = (
        float(np.clip(var_corr[off].sum() / (corr[off] ** 2).sum(), 0.0, 1.0)) if off.any() else 0.0
    )
    return lam * np.diag(np.diag(cov)) + (1 - lam) * cov, lam


def projection(S: np.ndarray, W: np.ndarray | None = None) -> np.ndarray:
    """G such that reconciled = S @ G @ base; W = None gives OLS."""
    n, m = S.shape
    if W is None:
        G = np.linalg.solve(S.T @ S, S.T)
    else:
        Winv_S = np.linalg.solve(W, S)  # W^-1 S
        G = np.linalg.solve(S.T @ Winv_S, Winv_S.T)
    return G


def recent_shares(h: Hierarchy, origin_pos: int, days: int = 28) -> np.ndarray:
    """Each bottom meter's share of the national total over the last ``days`` before the origin."""
    lo = max(0, origin_pos - 48 * days + 1)
    block = h.y_imp[h.bottom].iloc[lo : origin_pos + 1]
    mean = block.mean().to_numpy(dtype="float64")
    mean = np.where(np.isfinite(mean) & (mean > 0), mean, 0.0)
    return mean / mean.sum() if mean.sum() > 0 else np.full(len(h.bottom), 1 / len(h.bottom))


def reconcile_origin(
    method: str,
    Yhat: np.ndarray,  # rows x n base forecasts of one origin
    h: Hierarchy,
    shares: np.ndarray,
    past_errors: np.ndarray | None,  # rows x n errors of earlier origins (complete rows)
) -> np.ndarray:
    n, m = h.S.shape
    bottom_idx = [h.nodes.index(b) for b in h.bottom]
    if method == "base":
        return Yhat
    if method == "bottom_up":
        return Yhat[:, bottom_idx] @ h.S.T
    if method == "top_down":
        top = Yhat[:, h.nodes.index("national")][:, None]
        return (top * shares[None, :]) @ h.S.T
    if method == "middle_out":
        trusts = sorted(set(h.parent_trust.values()))
        b = np.zeros((Yhat.shape[0], m))
        for t in trusts:
            js = [j for j, mtr in enumerate(h.bottom) if h.parent_trust[mtr] == t]
            w = shares[js]
            w = w / w.sum() if w.sum() > 0 else np.full(len(js), 1 / len(js))
            b[:, js] = Yhat[:, h.nodes.index(t)][:, None] * w[None, :]
        return b @ h.S.T
    if method == "ols":
        G = projection(h.S)
    elif method == "wls_struct":
        G = projection(h.S, np.diag(h.S.sum(axis=1)))
    elif method in ("wls_var", "mint_shrink"):
        if past_errors is None or len(past_errors) < 2 * n:
            return np.full_like(Yhat, np.nan)
        if method == "wls_var":
            var = past_errors.var(axis=0)
            W = np.diag(np.where(var > 0, var, var[var > 0].min() if (var > 0).any() else 1.0))
        else:
            W, _ = _shrink_cov(past_errors)
            W = W + 1e-8 * np.trace(W) / n * np.eye(n)
        G = projection(h.S, W)
    else:
        raise KeyError(method)
    return Yhat @ (h.S @ G).T


def run(
    config: dict[str, Any], models: list[str], methods: list[str] | None = None
) -> pd.DataFrame:
    methods = methods or METHODS
    spec = config.get("reconciliation", {})
    warmup = int(spec.get("warmup_origins", 8))
    share_days = int(spec.get("share_days", 28))
    h = build_hierarchy(config)
    n = len(h.nodes)
    rows = []
    for model in models:
        Yhat = base_forecasts(config, model, h)
        origins = sorted(Yhat.index.get_level_values("origin").unique())
        targets_all = Yhat.index.get_level_values("target")
        Y = h.y.reindex(targets_all).to_numpy(dtype="float64")  # rows x n
        errors_hist: list[np.ndarray] = []
        results = {mth: [] for mth in methods}
        scored_rows = []
        for k, o in enumerate(origins):
            mask = Yhat.index.get_level_values("origin") == o
            yh = Yhat.to_numpy(dtype="float64")[mask]
            y = Y[mask]
            opos = int(h.y.index.searchsorted(o, side="right") - 1)
            shares = recent_shares(h, opos, share_days)
            past = np.concatenate(errors_hist) if errors_hist else None
            if k >= warmup:
                scored_rows.append(mask)
                for mth in methods:
                    results[mth].append(reconcile_origin(mth, yh, h, shares, past))
            complete = np.isfinite(y).all(axis=1)
            if complete.any():
                errors_hist.append(yh[complete] - y[complete])
        scored = np.concatenate(scored_rows)
        y_s = Y[scored]
        y_mean = np.nanmean(y_s, axis=0)
        for mth in methods:
            R = np.concatenate(results[mth])
            ae = np.abs(R - y_s)
            mae = np.nanmean(ae, axis=0)  # per node over observed rows
            n_obs = np.isfinite(ae).sum(axis=0)
            for lvl in LEVELS:
                idx = [i for i, node in enumerate(h.nodes) if h.level_of[node] == lvl]
                nmae = mae[idx] / y_mean[idx]
                rows.append(
                    {
                        "model": model,
                        "method": mth,
                        "level": lvl,
                        "n_series": len(idx),
                        "n_origins": len(origins) - warmup,
                        "mae": float(np.nanmean(mae[idx])),
                        "nmae": float(np.nanmean(nmae)),
                        "n_scored": int(n_obs[idx].sum()),
                    }
                )
            coh = (
                np.nanmax(
                    np.abs(R @ np.eye(n)[:, [h.nodes.index(b) for b in h.bottom]] @ h.S.T - R)
                )
                if mth != "base"
                else np.nan
            )
            logger.info("%s / %s: coherence gap %.3g", model, mth, coh)
    out = pd.DataFrame(rows)
    path = outputs_dir(config, "hierarchy") / "reconciliation.csv"
    if path.exists():
        prev = pd.read_csv(path)
        prev = prev[~prev["model"].isin(models)]
        out = pd.concat([prev, out], ignore_index=True)
    out.to_csv(path, index=False)
    return out
