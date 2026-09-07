"""Shared fixtures.

Two fixtures mirror the two stages of the pipeline: ``synthetic_readings`` looks
like the raw extract (meter grain, mixed reading types, dropout zeros), and
``synthetic_demand`` looks like the interim table after cleaning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

#: Site RXX01 has two meters; RXX02 has one. This mirrors the real extract,
#: where a third of sites are multi-meter.
METERS = [
    ("1000000000001", "RXX01"),
    ("1000000000002", "RXX01"),
    ("1000000000003", "RXX02"),
]


def _profile(timestamps: pd.DatetimeIndex, scale: float, rng) -> np.ndarray:
    tod = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
    daily = 1 + 0.4 * np.sin(2 * np.pi * (tod - 6) / 24)
    weekly = np.where(timestamps.dayofweek.to_numpy() >= 5, 0.75, 1.0)
    noise = rng.normal(0, scale * 0.05, len(timestamps))
    return np.asarray(scale * daily * weekly + noise, dtype=float)


@pytest.fixture
def synthetic_readings() -> pd.DataFrame:
    """Raw-shaped meter readings: 4 weeks half-hourly, with zeros and a duplicate."""
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2025-01-06", periods=336 * 4, freq="30min", tz="UTC")

    frames = []
    for i, (mpxn, site_code) in enumerate(METERS):
        values = _profile(timestamps, 100 * (1 + i), rng)
        values[10:20] = 0.0  # meter dropout recorded as zero
        frames.append(
            pd.DataFrame(
                {
                    "datetime": timestamps,
                    "mpxn": mpxn,
                    "site_code": site_code,
                    "energy_type": "elec",
                    "reading_type": "1",
                    "consumption": values,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    # One exact duplicate row, as the source contains 101 of them.
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def synthetic_demand() -> pd.DataFrame:
    """Interim-shaped cleaned demand: meter grain, complete grid, no zeros."""
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2025-01-06", periods=336 * 4, freq="30min", tz="UTC")

    frames = []
    for i, (mpxn, site_code) in enumerate(METERS):
        temperature = 10 + 6 * np.sin(2 * np.pi * (timestamps.dayofyear - 200) / 365.25)
        frames.append(
            pd.DataFrame(
                {
                    "datetime": timestamps,
                    "mpxn": mpxn,
                    "site_code": site_code,
                    "energy_type": "elec",
                    "consumption_kwh": _profile(timestamps, 100 * (1 + i), rng),
                    "temperature_c": temperature + rng.normal(0, 1, len(timestamps)),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)
