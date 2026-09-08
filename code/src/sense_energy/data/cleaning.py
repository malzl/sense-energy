"""Cleaning: de-duplication, gap handling, outlier flagging.

The interim table is kept at **meter** (``mpxn``) grain. Aggregating to site
level is lossy and cannot be undone, so it is deferred: hierarchical forecasting
needs the bottom level intact, and the upper levels are derived from it on
demand via :func:`aggregate_to_level`.

The natural hierarchy in this data is::

    mpxn -> site_code -> organisation_name -> integrated_care_board
                                           -> commissioning_region
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..logging_utils import get_logger

logger = get_logger(__name__)

#: Grain of the interim table: one row per meter, energy type and timestamp.
METER_GRAIN = ["mpxn", "energy_type", "datetime"]

#: Grain including reading_type, used only to spot true duplicate rows.
READING_GRAIN = ["mpxn", "energy_type", "reading_type", "datetime"]

#: Grouping key for per-series operations (lags, gaps, outliers).
SERIES_KEYS = ["mpxn", "energy_type"]


def drop_duplicate_readings(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the last reading per meter/timestamp — later rows supersede earlier.

    Note these are genuine duplicates. Rows sharing ``site_code`` and
    ``datetime`` are *not* duplicates: they are separate meters at one site.
    """
    before = len(df)
    out = df.drop_duplicates(subset=READING_GRAIN, keep="last")
    logger.info("Dropped %d duplicate meter readings", before - len(out))

    remaining = out.duplicated(subset=METER_GRAIN).sum()
    if remaining:
        logger.warning(
            "%d meter/timestamp pairs still carry multiple reading_types; "
            "keeping the most actual reading",
            remaining,
        )
        out = out.sort_values("reading_type").drop_duplicates(  # "1" (actual) sorts before "2"/"3"
            subset=METER_GRAIN, keep="first"
        )
    return out


def zeros_to_nan(df: pd.DataFrame, column: str = "consumption") -> pd.DataFrame:
    """Treat exact zeros as missing.

    A live meter never records exactly 0.000 kWh in a half hour: these are
    dropouts written as zero. They are concentrated in a handful of sites
    rather than spread evenly, which is what distinguishes them from genuine
    low readings.
    """
    df = df.copy()
    n_zero = int((df[column] == 0).sum())
    df[column] = df[column].replace(0.0, np.nan)
    logger.info("Converted %d exact zeros to NaN (%.2f%%)", n_zero, 100 * n_zero / max(len(df), 1))
    return df


def reindex_to_grid(df: pd.DataFrame, freq: str = "30min") -> pd.DataFrame:
    """Reindex each meter series onto a complete regular grid.

    Gaps become explicit NaN rows rather than absent timestamps, so lag features
    step over real elapsed time instead of silently skipping it.
    """
    static = [c for c in ("site_code",) if c in df.columns]
    frames = []

    for keys, group in df.groupby(SERIES_KEYS, observed=True, sort=False):
        g = group.set_index("datetime").sort_index()
        full = pd.date_range(g.index.min(), g.index.max(), freq=freq, tz="UTC")
        g = g.reindex(full)
        for key, value in zip(SERIES_KEYS, keys, strict=True):
            g[key] = value
        for col in static:
            g[col] = group[col].iloc[0]
        g.index.name = "datetime"
        frames.append(g.reset_index())

    out = pd.concat(frames, ignore_index=True)
    logger.info("Reindexed to %s grid: inserted %s timestamps", freq, f"{len(out) - len(df):,}")
    return out


def flag_outliers(
    df: pd.DataFrame,
    column: str = "consumption_kwh",
    z_threshold: float = 10.0,
    min_ratio_to_median: float = 5.0,
    ratio_to_p995: float = 3.0,
    mad_floor_share: float = 0.05,
    mad_floor_abs: float = 0.1,
) -> pd.DataFrame:
    """Flag implausible high readings per meter. Flag, never drop.

    A reading is an outlier only if it is extreme on a robust z-score AND at
    least ``min_ratio_to_median`` times the meter's median AND above
    ``ratio_to_p995`` times the meter's 99.5th percentile. The last condition
    is what keeps bimodal meters (idle most of the time, running at 20x idle
    for part of the day) from having a whole operating regime flagged: by
    construction it can only ever mark a sliver of any series. The MAD in the
    z-score is floored at ``mad_floor_share`` of the median (and an absolute
    ``mad_floor_abs`` kWh), because meters that read quantised small values
    (0.1, 0.2 kWh) have a near-zero MAD that turns ordinary readings into
    absurd z-scores. Only the high side is flagged: demand cannot be
    implausibly low except at zero, which is already treated as missing.
    """
    df = df.copy()

    def _score(s: pd.Series) -> pd.Series:
        median = s.median()
        if not np.isfinite(median) or median <= 0:
            return pd.Series(False, index=s.index)
        mad = max((s - median).abs().median(), mad_floor_share * median, mad_floor_abs)
        z = 0.6745 * (s - median) / mad
        p995 = s.quantile(0.995)
        return (z > z_threshold) & (s > min_ratio_to_median * median) & (s > ratio_to_p995 * p995)

    flagged = df.groupby(SERIES_KEYS, observed=True)[column].transform(_score)
    df["is_outlier"] = flagged.fillna(False).astype(bool)
    df.loc[df[column].isna(), "is_outlier"] = False
    logger.info(
        "Flagged %d outliers (z > %s, > %sx median, > %sx p99.5)",
        int(df["is_outlier"].sum()),
        z_threshold,
        min_ratio_to_median,
        ratio_to_p995,
    )
    return df


def interpolate_short_gaps(
    df: pd.DataFrame, column: str = "consumption_kwh", max_gap: int = 4
) -> pd.DataFrame:
    """Linearly fill gaps of at most ``max_gap`` periods, recording what was filled.

    Longer outages stay NaN: interpolating across a day of missing data invents
    a load profile rather than recovering one. ``is_interpolated`` lets these
    rows be excluded from evaluation.
    """
    df = df.sort_values([*SERIES_KEYS, "datetime"]).copy()
    was_missing = df[column].isna()
    df[column] = df.groupby(SERIES_KEYS, observed=True)[column].transform(
        lambda s: s.interpolate(method="linear", limit=max_gap, limit_area="inside")
    )
    df["is_interpolated"] = was_missing & df[column].notna()
    logger.info(
        "Interpolated %d readings across gaps <= %d periods",
        int(df["is_interpolated"].sum()),
        max_gap,
    )
    return df


def aggregate_to_level(
    df: pd.DataFrame,
    level: str = "site_code",
    column: str = "consumption_kwh",
    sites: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Sum the meter-level series up to a level of the hierarchy.

    Not applied when building the interim table — call it to construct the upper
    levels of a hierarchical forecast, or to reconcile bottom-level forecasts.

    ``level`` may be ``site_code`` (present on the interim table) or any column
    of the site dimension (``organisation_name``, ``integrated_care_board``,
    ``commissioning_region``), in which case pass ``sites``.

    ``n_reporting`` is carried through: a total assembled from fewer meters than
    usual is not comparable with a complete one, and should not be trusted blind.
    """
    if level not in df.columns:
        if sites is None:
            raise ValueError(f"'{level}' is not on the frame; pass sites= to join it on.")
        df = df.merge(sites[["site_code", level]], on="site_code", how="left")

    out = (
        df.groupby([level, "energy_type", "datetime"], observed=True)
        .agg(
            **{
                column: (column, lambda s: s.sum(min_count=1)),
                "n_reporting": (column, "count"),
                "n_expected": ("mpxn", "nunique"),
            }
        )
        .reset_index()
    )
    out["is_partial"] = out["n_reporting"] < out["n_expected"]
    logger.info(
        "Aggregated to %s: %s rows, %d partial totals",
        level,
        f"{len(out):,}",
        int(out["is_partial"].sum()),
    )
    return out
