"""Data preparation shared by the EDA figure scripts.

Each figure script does one thing: load a prepared frame from here, draw one
plot with the style helper, save. No figure script touches raw files.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..config import FIGURES_DIR, GEO_DIR, INTERIM_DIR, PROCESSED_DIR

OUT = FIGURES_DIR / "eda"
LOCAL_TZ = "Europe/London"

#: Gas is used at site level only where a series is reasonably observed.
GAS_MIN_COVERAGE = 0.5
GAS_MIN_DAYS = 180


def load_wide(energy: str, level: str, flavour: str = "imputed") -> pd.DataFrame:
    wide = pd.read_parquet(PROCESSED_DIR / energy / level / f"wide_{flavour}.parquet")
    wide.index = pd.to_datetime(wide.index, utc=True)
    return wide


def load_index(energy: str, level: str) -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / energy / level / "series_index.parquet")


def available(
    index: pd.DataFrame, min_coverage: float = GAS_MIN_COVERAGE, min_days: int = GAS_MIN_DAYS
) -> list[str]:
    ok = (index["coverage_in_span"] >= min_coverage) & (index["n_periods_span"] >= min_days * 48)
    return index.loc[ok, "series_id"].tolist()


def series_for(
    energy: str, level: str, flavour: str = "imputed"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide matrix and index, restricted to well-observed series for gas."""
    wide, index = load_wide(energy, level, flavour), load_index(energy, level)
    if energy == "gas":
        keep = [s for s in available(index) if s in wide.columns]
        wide, index = wide[keep], index[index["series_id"].isin(keep)]
    return wide, index


def normalise(wide: pd.DataFrame) -> pd.DataFrame:
    """Demand index: each series divided by its own mean (site mean = 1)."""
    return wide / wide.mean()


def local(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    out.index = out.index.tz_convert(LOCAL_TZ)
    return out


def daily_profile(wide_norm: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR across series of the mean profile by local half-hour and day type."""
    loc = local(wide_norm)
    slot = loc.index.hour * 2 + loc.index.minute // 30
    weekend = loc.index.dayofweek >= 5
    rows = []
    for day_type, mask in (("weekday", ~weekend), ("weekend", weekend)):
        per_series = loc[mask].groupby(slot[mask]).mean()  # slot x series
        q = per_series.quantile([0.25, 0.5, 0.75], axis=1).T
        q.columns = ["q25", "median", "q75"]
        q["hour"] = q.index / 2
        q["day_type"] = day_type
        rows.append(q.reset_index(drop=True))
    return pd.concat(rows, ignore_index=True)


def monthly_index(wide_norm: pd.DataFrame, min_share: float = 0.8) -> pd.DataFrame:
    """Monthly mean of the demand index per series; months < ``min_share`` observed are NaN."""
    monthly = wide_norm.resample("MS")
    mean = monthly.mean()
    share = monthly.count() / monthly.size().to_numpy()[:, None]
    return mean.where(share >= min_share)


def seasonal_profile(wide_norm: pd.DataFrame) -> pd.DataFrame:
    """Median and IQR across series of the calendar-month mean demand index."""
    m = monthly_index(wide_norm)
    by_month = m.groupby(m.index.month).mean()
    q = by_month.quantile([0.25, 0.5, 0.75], axis=1).T
    q.columns = ["q25", "median", "q75"]
    q["month"] = q.index
    return q.reset_index(drop=True)


def short_trust_name(name: str) -> str:
    s = re.sub(r"\bNHS\b|\bFoundation\b|\bTrust\b|\bHealthcare\b", "", str(name))
    return re.sub(r"\s+", " ", s).strip(" ,")


def site_points() -> pd.DataFrame:
    g = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    return g[g["latitude"].notna()][
        ["site_code", "site_name", "organisation_name", "latitude", "longitude"]
    ]


def england():
    import geopandas as gpd

    countries = gpd.read_file(GEO_DIR / "countries_uk.geojson")
    name_col = next(
        c for c in countries.columns if c.upper().startswith("CTRY") and c.upper().endswith("NM")
    )
    return countries[countries[name_col].eq("England")]


def daily_temperature_and_demand(energy: str) -> pd.DataFrame:
    """Site-days with daily mean 2 m temperature (ERA5 reanalysis) and demand index.

    Uses whatever ERA5 months have been extracted to ``interim/weather_reanalysis.parquet``.
    """
    w = pd.read_parquet(
        INTERIM_DIR / "weather_reanalysis.parquet",
        columns=["site_code", "datetime", "member", "t2m"],
    )
    w = w[w["member"] == 0]
    w["date"] = w["datetime"].dt.tz_convert(LOCAL_TZ).dt.date
    temp = w.groupby(["site_code", "date"])["t2m"].mean().sub(273.15).rename("t2m_c").reset_index()

    wide, _ = series_for(energy, "site")
    norm = local(normalise(wide))
    daily = norm.groupby(norm.index.date).mean()
    daily_count = norm.groupby(norm.index.date).count()
    daily = daily.where(daily_count >= 40)  # a day needs most of its half-hours
    demand = daily.stack(future_stack=True).rename("demand_index").reset_index()
    demand.columns = ["date", "site_code", "demand_index"]
    out = demand.merge(temp, on=["site_code", "date"], how="inner").dropna()
    return out


def bin_by_temperature(frame: pd.DataFrame, width: float = 1.0, min_n: int = 30) -> pd.DataFrame:
    frame = frame.copy()
    frame["t_bin"] = (np.floor(frame["t2m_c"] / width) * width) + width / 2
    g = frame.groupby("t_bin")["demand_index"]
    out = g.quantile([0.25, 0.5, 0.75]).unstack()
    out.columns = ["q25", "median", "q75"]
    out["n"] = g.size()
    return out[out["n"] >= min_n].reset_index()
