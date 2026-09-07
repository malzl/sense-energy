"""Forecast accuracy metrics.

MAPE is unstable when demand approaches zero; prefer MAE, RMSE and MASE, and
report MAPE only alongside them.
"""

from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Mean absolute percentage error, as a percentage."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon))) * 100)


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """Symmetric MAPE — bounded at 200%, safer near zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum((np.abs(y_true) + np.abs(y_pred)) / 2, epsilon)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)


def mase(
    y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray, season_length: int = 336
) -> float:
    """Mean absolute scaled error against an in-sample seasonal naive forecast.

    < 1 means the model beats seasonal naive on the training period.
    """
    y_train = np.asarray(y_train, dtype=float)
    scale = np.mean(np.abs(y_train[season_length:] - y_train[:-season_length]))
    if scale == 0:
        raise ValueError("Seasonal naive scale is zero; cannot compute MASE.")
    return mae(y_true, y_pred) / scale


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean error — positive means the model over-forecasts."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "bias": bias(y_true, y_pred),
    }
