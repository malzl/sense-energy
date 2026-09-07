"""Model training entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

from ..config import MODELS_DIR, PROCESSED_DIR, ensure_dirs
from ..logging_utils import get_logger
from .baseline import ProfileMean, SeasonalNaive

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "seasonal_naive": SeasonalNaive,
    "profile_mean": ProfileMean,
    "ridge": Ridge,
    "lightgbm": LGBMRegressor,
}


def build_model(config: dict[str, Any]):
    """Instantiate a model from ``config['model']['name']`` and its params."""
    spec = config.get("model", {})
    name = spec.get("name", "lightgbm")
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**spec.get("params", {}))


def time_split(df: pd.DataFrame, cutoff: str, timestamp_col: str = "datetime"):
    """Split into train/test at a timestamp. Never split time series randomly."""
    cut = pd.Timestamp(cutoff, tz="UTC")
    return df[df[timestamp_col] < cut], df[df[timestamp_col] >= cut]


def train(config: dict[str, Any]) -> Path:
    """Fit a model on the feature table and persist it to ``code/models/``."""
    ensure_dirs()

    df = pd.read_parquet(PROCESSED_DIR / config.get("input_filename", "features.parquet"))
    target = config.get("target", "consumption_kwh")
    features = config["features"]

    train_df, _ = time_split(df, config["split"]["test_start"])
    train_df = train_df.dropna(subset=[*features, target])

    model = build_model(config)
    logger.info(
        "Fitting %s on %d rows, %d features", type(model).__name__, len(train_df), len(features)
    )
    model.fit(train_df[features], train_df[target])

    out_path = MODELS_DIR / f"{config.get('run_name', 'model')}.joblib"
    joblib.dump(
        {"model": model, "features": features, "target": target, "config": config}, out_path
    )
    logger.info("Saved model to %s", out_path)
    return out_path
