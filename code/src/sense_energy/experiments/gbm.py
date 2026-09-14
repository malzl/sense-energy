"""LightGBM direct multi-horizon quantile models for the PoC."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .features import CATEGORICAL, Covariates, _holiday_set, build_rows
from .poc import Origin, Panel, forecast_frame, make_origins

logger = get_logger(__name__)

NON_FEATURES = {"y", "origin", "target", "scale"}
VARIANT_WEATHER = {"lightgbm": "none", "lightgbm_era5": "era5", "lightgbm_ifs": "ifs"}


def _fit_quantile(
    X: pd.DataFrame,
    y: np.ndarray,
    Xv: pd.DataFrame,
    yv: np.ndarray,
    alpha: float,
    params: dict[str, Any],
):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        objective="quantile",
        alpha=alpha,
        n_estimators=int(params["n_estimators"]),
        learning_rate=float(params["learning_rate"]),
        num_leaves=int(params["num_leaves"]),
        min_child_samples=int(params["min_child_samples"]),
        subsample=float(params["subsample"]),
        subsample_freq=1,
        colsample_bytree=float(params["colsample_bytree"]),
        n_jobs=int(params.get("n_jobs", 32)),
        verbose=-1,
    )
    model.fit(
        X,
        y,
        eval_set=[(Xv, yv)],
        categorical_feature=[c for c in CATEGORICAL if c in X.columns],
        callbacks=[lgb.early_stopping(int(params["early_stopping_rounds"]), verbose=False)],
    )
    return model


def run_lightgbm(
    panel: Panel, test_origins: list[Origin], config: dict[str, Any], variant: str = "lightgbm"
) -> pd.DataFrame:
    params = config["lightgbm"]
    weather = VARIANT_WEATHER[variant]
    quantiles = list(config["quantiles"])
    fit_quantiles = [float(q) for q in params.get("quantiles", [0.1, 0.5, 0.9])]
    cov = Covariates(config, panel)
    hol = _holiday_set(range(2020, 2028))

    train_origins = make_origins(
        panel,
        config,
        config["train_start"],
        config["train_end"],
        int(params.get("train_origin_stride_days", 1)),
    )
    logger.info(
        "%s: building %d train origins x %d sites (weather=%s)",
        variant,
        len(train_origins),
        len(panel.sites),
        weather,
    )
    train = build_rows(panel, train_origins, cov, weather, hol).dropna(subset=["y"])
    cutoff = train["origin"].max() - pd.Timedelta(days=int(params.get("valid_last_days", 60)))
    fit, val = train[train["origin"] <= cutoff], train[train["origin"] > cutoff]
    features = [c for c in train.columns if c not in NON_FEATURES]
    logger.info(
        "%s: %s fit rows, %s validation rows, %d features",
        variant,
        f"{len(fit):,}",
        f"{len(val):,}",
        len(features),
    )

    models = {}
    for a in fit_quantiles:
        models[a] = _fit_quantile(
            fit[features], fit["y"].to_numpy(), val[features], val["y"].to_numpy(), a, params
        )
        logger.info("%s: q%.2f best iteration %d", variant, a, models[a].best_iteration_ or 0)

    test = build_rows(panel, test_origins, cov, weather, hol, with_target=False)
    preds = {a: models[a].predict(test[features]) * test["scale"].to_numpy() for a in fit_quantiles}
    # Interpolate the fitted quantiles onto the reporting grid (linear in tau)
    fitted = np.stack([preds[a] for a in fit_quantiles], axis=1)  # N x F
    fitted = np.sort(fitted, axis=1)
    grid = np.stack([np.interp(quantiles, fit_quantiles, row) for row in fitted])  # N x Q
    test = test.assign(**{f"_q{i}": grid[:, i] for i in range(len(quantiles))})
    frames = []
    by_key = {(o.origin_time, s): (o, s) for o in test_origins for s in panel.sites}
    for (origin_time, s), block in test.groupby(["origin", "site_code"], observed=True, sort=False):
        o, _ = by_key[(origin_time, s)]
        q = block[[f"_q{i}" for i in range(len(quantiles))]].to_numpy()
        frames.append(forecast_frame(variant, s, o, panel, quantiles, q))
    out = pd.concat(frames, ignore_index=True)
    imp = pd.Series(models[0.5].feature_importances_, index=features).sort_values(ascending=False)
    logger.info("%s top features: %s", variant, imp.head(10).to_dict())
    return out
