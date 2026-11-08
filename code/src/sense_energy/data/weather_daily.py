"""Daily weather per site and local day - cooling and heating need, humidity, solar,
wind, rain and snow - derived from hourly ERA5 at the site.

Nothing is fetched: the input is ``interim/weather_reanalysis.parquet`` (nearest
0.25 deg grid point to each site's postcode centroid, hourly, all 275 located
sites). Days are ``Europe/London`` calendar days so they line up with the demand
day; BST transition days have 23 or 25 hours.

Degree days use the **integration method** (CIBSE TM41): the mean over the day's
hours of ``max(T - base, 0)``, in degC day. It counts a hot afternoon on a day
whose mean stays below the base, which the daily-mean method misses. The
daily-mean form ``max(Tmean - base, 0)`` (Met Office / ASHRAE convention) is kept
alongside for comparison with published series, plus the Eurostat cooling
definition. Humidity terms cover the latent part of cooling need (theatres and
pharmacy stores dehumidify as well as cool). Wind, rain and snow feed the
extreme-event catalogue (``analysis/weather_events.py``).

Output: ``interim/weather_daily.parquet``, one row per (site_code, date).

ERA5 at 0.25 deg smooths urban heat islands and understates afternoon peaks
during heatwaves, so the hot tail of a city-centre site is biased low; the same
smoothing understates convective rain totals and peak gusts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

TZ = "Europe/London"
#: Cooling bases, degC: UK convention (CIBSE TM41), ASHRAE/US, and a high base
#: closer to the balance point of a lightly loaded building.
CDD_BASES = (15.5, 18.0, 22.0)
HDD_BASE = 15.5
#: Hours at or above these outdoor dry-bulb temperatures, degC.
HOT_HOURS = (25.0, 28.0)
#: Eurostat cooling degree days: (Tmean - 21) on days with Tmean > 24 degC.
EUROSTAT_BASE, EUROSTAT_THRESHOLD = 21.0, 24.0
#: Evening mean temperature covers local hours from this one to midnight: the part of the day
#: that a daily mean blurs and that demand responds to (the evening shift on hot days).
EVENING_FROM = 18


def _tag(base: float) -> str:
    return f"{base:g}".replace(".", "_")


def saturation_vapour_pressure(t_c):
    """hPa, Magnus form over water (Alduchov & Eskridge 1996 coefficients)."""
    return 6.1094 * np.exp(17.625 * t_c / (243.04 + t_c))


def relative_humidity(t_c, td_c):
    """Percent, clipped to [0, 100]."""
    return np.clip(100 * saturation_vapour_pressure(td_c) / saturation_vapour_pressure(t_c), 0, 100)


def wet_bulb_stull(t_c, rh):
    """Psychrometric wet-bulb temperature, degC (Stull 2011; RH 5-99 %, T -20-50 degC)."""
    return (
        t_c * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t_c + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )


def moist_air_enthalpy(t_c, td_c, p_pa):
    """kJ per kg dry air, from dry-bulb, dew point and surface pressure."""
    e = saturation_vapour_pressure(td_c)
    w = 0.622 * e / (p_pa / 100 - e)
    return 1.006 * t_c + w * (2501 + 1.86 * t_c)


def hourly_terms(weather: pd.DataFrame) -> pd.DataFrame:
    """Per-hour quantities that the daily table averages, sums or maxes."""
    t = weather["t2m"].to_numpy(np.float64) - 273.15
    td = weather["d2m"].to_numpy(np.float64) - 273.15
    rh = relative_humidity(t, td)
    local = weather["datetime"].dt.tz_convert(TZ)
    out = pd.DataFrame(
        {
            "site_code": weather["site_code"].to_numpy(),
            "date": local.dt.date.to_numpy(),
            "t": t,
            "td": td,
            "rh": rh,
            "tw": wet_bulb_stull(t, rh),
            "h": moist_air_enthalpy(t, td, weather["sp"].to_numpy(np.float64)),
            "hdd": np.maximum(HDD_BASE - t, 0),
            "t_eve": np.where(local.dt.hour >= EVENING_FROM, t, np.nan),
        }
    )
    for b in CDD_BASES:
        out[f"cdd_{_tag(b)}"] = np.maximum(t - b, 0)
    for b in HOT_HOURS:
        out[f"hours_ge_{_tag(b)}"] = (t >= b).astype(np.int8)
    if {"u10", "v10"} <= set(weather):
        out["wind"] = np.hypot(
            weather["u10"].to_numpy(np.float64), weather["v10"].to_numpy(np.float64)
        )
    if "i10fg" in weather:
        out["gust"] = weather["i10fg"].to_numpy(np.float64)
    return out


#: ERA5 hourly accumulations -> daily totals: (column, scale, output name).
ACCUMULATIONS = (
    ("ssrd", 1 / 3.6e6, "solar_kwh_m2"),  # J/m2 -> kWh/m2
    ("tp", 1000.0, "precip_mm"),  # m -> mm
    ("sf", 1000.0, "snow_mm_we"),  # m of water equivalent -> mm
)


def accumulated_daily(weather: pd.DataFrame) -> pd.DataFrame:
    """Daily totals of the accumulated fields present. ERA5 accumulates over the hour
    *ending* at the timestamp, so each value is dated by its hour's start."""
    start = (weather["datetime"] - pd.to_timedelta(1, "h")).dt.tz_convert(TZ).dt.date
    cols = {name: weather[c].astype(np.float64) * k for c, k, name in ACCUMULATIONS if c in weather}
    if not cols:  # a thermometer-only product (ERA5-Land) has no accumulated fields
        return pd.DataFrame(index=pd.MultiIndex.from_arrays([[], []], names=["site_code", "date"]))
    return pd.DataFrame(cols).groupby([weather["site_code"], start]).sum()


def daily(weather: pd.DataFrame) -> pd.DataFrame:
    """One row per (site_code, local date) from an hourly ERA5 site table."""
    h = hourly_terms(weather)
    g = h.groupby(["site_code", "date"])
    cdd = [f"cdd_{_tag(b)}" for b in CDD_BASES]
    hot = [f"hours_ge_{_tag(b)}" for b in HOT_HOURS]
    parts = [
        g.size().rename("n_hours"),
        g["t"].agg(t_mean="mean", t_min="min", t_max="max"),
        g["t_eve"].mean().rename("t_eve"),
        g[["hdd"] + cdd].mean().rename(columns={"hdd": f"hdd_{_tag(HDD_BASE)}"}),
        g[hot].sum(),
        g["td"].agg(td_mean="mean", td_max="max"),
        g["rh"].mean().rename("rh_mean"),
        g["tw"].max().rename("tw_max"),
        g["h"].agg(h_mean="mean", h_max="max"),
    ]
    if "wind" in h:
        parts.append(g["wind"].agg(wind_mean="mean", wind_max="max"))
    if "gust" in h:
        parts.append(g["gust"].max().rename("gust_max"))
    out = pd.concat(parts, axis=1)
    for b in CDD_BASES:
        out[f"cdd_dm_{_tag(b)}"] = np.maximum(out["t_mean"] - b, 0)
    out["cdd_eurostat"] = np.where(
        out["t_mean"] > EUROSTAT_THRESHOLD, out["t_mean"] - EUROSTAT_BASE, 0.0
    )
    out = out.join(accumulated_daily(weather).rename_axis(["site_code", "date"]))
    out = out.reset_index()
    out["date"] = pd.to_datetime(out["date"])
    return out


def build(product: str = "reanalysis", suffix: str = "") -> pd.DataFrame:
    """Write ``interim/weather_daily.parquet``. Partial first/last days (fewer
    than 23 hours, from the UTC edges of the ERA5 window) are dropped."""
    import pyarrow.parquet as pq

    source = INTERIM_DIR / f"weather_{product}.parquet"
    wanted = ["site_code", "datetime", "member", "t2m", "d2m", "sp", "ssrd", "tp", "sf"]
    wanted += ["u10", "v10", "i10fg"]
    have = set(pq.read_schema(source).names)
    weather = pd.read_parquet(source, columns=[c for c in wanted if c in have])
    weather = weather[weather["member"] == 0].drop(columns="member")
    out = daily(weather)
    out = out[out["n_hours"] >= 23].reset_index(drop=True)
    float_cols = out.select_dtypes("float64").columns
    out[float_cols] = out[float_cols].astype(np.float32)
    path = INTERIM_DIR / f"weather_daily{suffix}.parquet"
    out.to_parquet(path, index=False, compression="zstd")
    logger.info(
        "Wrote %s (%s rows, %d sites, %s -> %s)",
        path.name,
        f"{len(out):,}",
        out["site_code"].nunique(),
        out["date"].min().date(),
        out["date"].max().date(),
    )
    return out
