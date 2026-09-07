"""Schema and sanity checks applied at every stage boundary.

Fail loudly and early: a silently misaligned timestamp column is far more
expensive than a raised exception.
"""

from __future__ import annotations

import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = ("site_id", "timestamp", "value")


def validate_schema(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_COLUMNS) -> None:
    """Raise if required columns are missing or ``timestamp`` is not tz-aware UTC."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "timestamp" in df.columns:
        ts = df["timestamp"]
        if not pd.api.types.is_datetime64_any_dtype(ts):
            raise TypeError("'timestamp' must be a datetime dtype")
        if ts.dt.tz is None:
            raise TypeError("'timestamp' must be tz-aware (store everything in UTC)")


def check_regular_index(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Report gaps and duplicates per site on a regular ``freq`` grid."""
    rows = []
    for site_id, group in df.groupby("site_id"):
        ts = group["timestamp"].sort_values()
        expected = pd.date_range(ts.min(), ts.max(), freq=freq, tz="UTC")
        rows.append(
            {
                "site_id": site_id,
                "n_rows": len(group),
                "n_expected": len(expected),
                "n_missing": len(expected.difference(ts)),
                "n_duplicated": int(ts.duplicated().sum()),
                "start": ts.min(),
                "end": ts.max(),
            }
        )
    return pd.DataFrame(rows)


def summarise_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column null counts and share, for the data-quality report."""
    n = len(df)
    return pd.DataFrame(
        {
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().sum() / n * 100).round(2) if n else 0.0,
        }
    ).sort_values("n_missing", ascending=False)
