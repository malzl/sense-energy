"""Shared fixtures: synthetic half-hourly hospital demand."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_demand() -> pd.DataFrame:
    """Two sites, ~8 weeks of half-hourly demand with daily + weekly seasonality."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2025-01-01", periods=336 * 8, freq="30min", tz="UTC")

    frames = []
    for i, site_id in enumerate(["SITE_A", "SITE_B"]):
        tod = timestamps.hour + timestamps.minute / 60
        daily = 100 * (1 + 0.4 * np.sin(2 * np.pi * (tod - 6) / 24))
        weekly = np.where(timestamps.dayofweek >= 5, 0.75, 1.0)
        base = (1 + i * 0.5) * daily * weekly
        temperature = 10 + 6 * np.sin(2 * np.pi * (timestamps.dayofyear - 200) / 365.25)

        frames.append(
            pd.DataFrame(
                {
                    "site_id": site_id,
                    "timestamp": timestamps,
                    "value": base + rng.normal(0, 5, len(timestamps)),
                    "unit": "kWh",
                    "temperature_c": temperature + rng.normal(0, 1, len(timestamps)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
