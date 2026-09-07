"""Cleaning steps: deduplication, gap handling, outlier flagging."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)


def drop_duplicate_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last reading per (site_id, timestamp) — later exports supersede earlier."""
    before = len(df)
    out = df.sort_values("timestamp").drop_duplicates(["site_id", "timestamp"], keep="last")
    logger.info("Dropped %d duplicate readings", before - len(out))
    return out


def reindex_to_grid(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Reindex each site onto a complete regular grid, leaving gaps as NaN."""
    frames = []
    for site_id, group in df.groupby("site_id"):
        g = group.set_index("timestamp").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq, tz="UTC")
        g = g.reindex(full)
        g["site_id"] = site_id
        g.index.name = "timestamp"
        frames.append(g.reset_index())
    return pd.concat(frames, ignore_index=True)


def flag_outliers(
    df: pd.DataFrame, column: str = "value", z_threshold: float = 5.0
) -> pd.DataFrame:
    """Add an ``is_outlier`` column using a per-site robust z-score (MAD-based).

    Flag, do not drop: a genuine spike (plant failure, a cold snap) is signal.
    """
    df = df.copy()

    def _robust_z(s: pd.Series) -> pd.Series:
        median = s.median()
        mad = (s - median).abs().median()
        if mad == 0 or np.isnan(mad):
            return pd.Series(0.0, index=s.index)
        return 0.6745 * (s - median) / mad

    z = df.groupby("site_id")[column].transform(_robust_z)
    df["is_outlier"] = z.abs() > z_threshold
    logger.info("Flagged %d outliers (|robust z| > %s)", int(df["is_outlier"].sum()), z_threshold)
    return df


def interpolate_short_gaps(
    df: pd.DataFrame, column: str = "value", max_gap: int = 4
) -> pd.DataFrame:
    """Linearly interpolate gaps of at most ``max_gap`` consecutive periods."""
    df = df.copy()
    df[column] = df.groupby("site_id")[column].transform(
        lambda s: s.interpolate(method="linear", limit=max_gap, limit_area="inside")
    )
    return df
