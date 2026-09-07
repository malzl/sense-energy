"""Loading a trained model and generating forecasts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from ..config import MODELS_DIR


def load_model(name: str = "model") -> dict[str, Any]:
    """Load a persisted model bundle (model, feature list, target, config)."""
    path = MODELS_DIR / f"{name}.joblib" if not str(name).endswith(".joblib") else Path(name)
    return joblib.load(path)


def predict(bundle: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Predict on a feature table using a loaded bundle."""
    return bundle["model"].predict(df[bundle["features"]])


def forecast_horizon(bundle: dict[str, Any], df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Multi-step recursive forecast.

    TODO: implement once the operational horizon is fixed (day-ahead half-hourly
    vs. month-ahead). A direct multi-horizon model — one estimator per step — is
    often preferable to recursion, since it avoids compounding error.
    """
    raise NotImplementedError
