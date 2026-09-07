"""Forward-chaining (rolling-origin) backtesting.

The only valid validation scheme here: random k-fold leaks future information
into the training set and will flatter every model you try.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pandas as pd

from ..logging_utils import get_logger
from .metrics import all_metrics

logger = get_logger(__name__)


def rolling_origin_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    test_size: int = 336,
    gap: int = 0,
    timestamp_col: str = "datetime",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield expanding-window (train, test) folds ordered in time.

    ``gap`` leaves a blind period between train and test, matching the lead time
    at which the forecast would really be issued.
    """
    df = df.sort_values(timestamp_col)
    n = len(df)
    for i in reversed(range(n_splits)):
        test_end = n - i * test_size
        test_start = test_end - test_size
        train_end = test_start - gap
        if train_end <= 0:
            continue
        yield df.iloc[:train_end], df.iloc[test_start:test_end]


def backtest(
    model_factory,
    df: pd.DataFrame,
    features: list[str],
    target: str = "consumption_kwh",
    n_splits: int = 5,
    test_size: int = 336,
    gap: int = 0,
) -> pd.DataFrame:
    """Run a rolling-origin backtest and return one metrics row per fold."""
    results: list[dict[str, Any]] = []
    for fold, (train_df, test_df) in enumerate(
        rolling_origin_splits(df, n_splits=n_splits, test_size=test_size, gap=gap)
    ):
        train_df = train_df.dropna(subset=[*features, target])
        test_df = test_df.dropna(subset=[*features, target])
        if train_df.empty or test_df.empty:
            logger.warning("Fold %d empty after dropna; skipping", fold)
            continue

        model = model_factory()
        model.fit(train_df[features], train_df[target])
        preds = np.asarray(model.predict(test_df[features]))

        results.append(
            {
                "fold": fold,
                "train_end": train_df["datetime"].max(),
                "test_start": test_df["datetime"].min(),
                "test_end": test_df["datetime"].max(),
                "n_train": len(train_df),
                "n_test": len(test_df),
                **all_metrics(test_df[target].to_numpy(), preds),
            }
        )
    return pd.DataFrame(results)
