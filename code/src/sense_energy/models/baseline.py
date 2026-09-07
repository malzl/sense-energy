"""Naive baselines.

Any ML model must beat these before it is worth deploying. Seasonal naive
(same half-hour last week) is a genuinely strong baseline for hospital demand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Predict the value from ``season_length`` periods ago.

    Defaults to 336 periods = one week of half-hourly data.
    """

    def __init__(self, season_length: int = 336):
        self.season_length = season_length

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> SeasonalNaive:
        self.history_ = pd.Series(np.asarray(y)).tail(self.season_length).to_numpy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        reps = int(np.ceil(n / self.season_length))
        return np.tile(self.history_, reps)[:n]


class ProfileMean(BaseEstimator, RegressorMixin):
    """Predict the mean demand for the (day-of-week, time-of-day) cell.

    Requires ``day_of_week`` and ``minute_of_day`` columns from the calendar features.
    """

    def __init__(self, group_cols: tuple[str, ...] = ("day_of_week", "minute_of_day")):
        self.group_cols = group_cols

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> ProfileMean:
        frame = X[list(self.group_cols)].copy()
        frame["_y"] = np.asarray(y)
        self.profile_ = frame.groupby(list(self.group_cols))["_y"].mean()
        self.global_mean_ = float(np.mean(y))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[list(self.group_cols)])
        return self.profile_.reindex(idx).fillna(self.global_mean_).to_numpy()
