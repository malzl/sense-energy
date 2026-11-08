"""Nonlinear demand response to weather: binned temperature response with high-dimensional fixed effects.

Panel: site x local day, daily mean demand (kW, days with >= 44 observed half-hours) and the
daily maximum half-hour, joined to ``interim/weather_daily.parquet``.

Model (one per site group and outcome)::

    log y[s,d] = sum_b beta_b 1{T[s,d] in b} + controls + a[s, year-month] + c[s, weekday] + e

with 2 degC bins of daily mean temperature and 14-16 degC as the reference. The site x
year-month effects absorb the seasonal cycle, trends and level shifts, so each bin's
coefficient comes from day-to-day weather within a month: it answers "how much more does this
site use on a day this warm/cold than on a 15 degC day of the same month". The same model on
the daily maximum (bands to 32 degC and beyond, reference 18-20 degC) covers the 30+ degC days
that a daily mean never reaches in England. Controls: bank
holiday, Christmas week (24 Dec - 1 Jan), school summer holiday (21 Jul - 31 Aug). The fixed
effects are absorbed by alternating projections; standard errors are clustered by ISO week
(every site shares the weather of a day and weather persists for days), with the absorbed
fixed effects counted in the degrees of freedom. ``pct`` = 100 (exp(beta) - 1).

Extensions, same fixed effects:
* **peak** - the daily maximum half-hour instead of the daily mean.
* **hourly** - mean demand in each local hour of the day as the outcome (24 fits).
* **persistence** - consecutive-day terms on top of the bins: day 2 and day 3+ of a run of
  days with T_max at or above the site's Met Office heatwave threshold, and of days with
  T_mean <= 2 degC: the extra demand beyond what that day's own temperature implies.
* **other weather** - dew point on warm days (T_mean >= 16), wind on cold days
  (T_mean < 10), solar irradiation, on top of the bins.

Site groups: ``on-site CHP`` (fossil-fuel CHP; meters record grid import, see
doc/estates_analysis.md), and the other sites split into ``inpatient`` (sites with beds:
acute, community and mental-health hospitals) and ``outpatient`` (clinics, offices,
support buildings).

Outputs (``code/reports/weather_response/``): ``curve.csv``, ``hourly.csv``,
``persistence.csv``, ``other_weather.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..experiments.features import _holiday_set
from ..logging_utils import get_logger
from .cooling import MIN_PERIODS_PER_DAY, TZ

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "weather_response"
EDGES = [-np.inf, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, np.inf]
REF_BIN = 14.0  # lower edge of the reference bin [14, 16)
#: Daily-maximum bands: 4 degC up to 14, then 2 degC to 32 and beyond, so the 30-32 and
#: >= 32 degC days of 2023-2025 get bands of their own; reference 18-20 degC.
EDGES_MAX = [-np.inf, 2, 6, 10, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, np.inf]
REF_BIN_MAX = 18.0
#: Temperature variable -> (column, bin column, edges, reference lower edge, open-tail midpoints).
VARIABLES = {
    "daily mean": ("t_mean", "tbin", EDGES, REF_BIN, (-3.0, 25.0)),
    "daily maximum": ("t_max", "tbin_max", EDGES_MAX, REF_BIN_MAX, (0.0, 33.0)),
}
FE = ("fe_month", "fe_dow")
CLUSTER = "week"
#: ``site_use_type`` values of sites with inpatient beds.
INPATIENT_USES = {
    "General acute hospital",
    "Specialist hospital (acute only)",
    "Mixed service hospital",
    "Community hospital (with inpatient beds)",
    "Mental Health (including Specialist services)",
    "Mental Health and Learning Disabilities",
    "Other inpatient",
}
GROUPS = ("inpatient", "outpatient", "on-site CHP")
WARM_C, MILD_C, COLD_RUN_C = 16.0, 10.0, 2.0


def bin_label(lo: float, variable: str = "daily mean") -> str:
    edges = VARIABLES[variable][2]
    if not np.isfinite(lo):
        return f"<{edges[1]:g}"
    return f">={lo:g}" if lo >= edges[-2] else f"{lo:g}"


def site_groups() -> pd.Series:
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").set_index("site_code")
    chp = sites["fossil_fuel_led_chp_units_operated_on_site"].fillna(0) > 0
    inpatient = sites["site_use_type"].isin(INPATIENT_USES)
    return pd.Series(
        np.where(chp, "on-site CHP", np.where(inpatient, "inpatient", "outpatient")),
        index=sites.index,
        name="group",
    )


def _runs_day(flag: pd.Series) -> pd.Series:
    """1, 2, 3, ... within runs of True on consecutive dates (per site, sorted), else 0."""
    brk = (~flag).cumsum()
    return flag.groupby(brk).cumsum().where(flag, 0)


def panel() -> pd.DataFrame:
    """Daily site panel with outcomes, weather, calendar controls and fixed-effect ids."""
    from .weather_events import heat_thresholds

    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(TZ)
    day = local.normalize().tz_localize(None)
    n = w.notna().groupby(day).sum()
    ok = n >= MIN_PERIODS_PER_DAY
    mean = w.groupby(day).mean().where(ok)
    peak = w.groupby(day).max().where(ok)
    out = (
        pd.concat({"kw": mean.stack(), "peak_kw": peak.stack()}, axis=1)
        .rename_axis(["date", "site_code"])
        .reset_index()
    )
    out = out[(out["kw"] > 0) & (out["peak_kw"] > 0)]

    wx = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")
    keep = ["site_code", "date", "t_mean", "t_max", "td_mean", "wind_mean", "solar_kwh_m2"]
    out = out.merge(wx[keep], on=["site_code", "date"], how="inner")
    out = out.merge(heat_thresholds()[["site_code", "heat_threshold_c"]], on="site_code")
    out = out.merge(site_groups(), left_on="site_code", right_index=True)

    hol = pd.DatetimeIndex(sorted(_holiday_set(range(2022, 2027))))
    md = out["date"].dt.month * 100 + out["date"].dt.day
    out["holiday"] = out["date"].isin(hol).astype(float)
    out["christmas"] = ((md >= 1224) | (md <= 101)).astype(float)
    out["school_summer"] = ((md >= 721) & (md <= 831)).astype(float)
    for _, (col, bin_col, edges, _, _) in VARIABLES.items():
        out[bin_col] = pd.cut(out[col], edges, right=False).map(lambda iv: iv.left).astype(float)

    out = out.sort_values(["site_code", "date"]).reset_index(drop=True)
    # Consecutive-day counters need gap-free dates within a site.
    contiguous = out.groupby("site_code")["date"].diff().dt.days.eq(1)
    for name, flag in (
        ("hot", out["t_max"] >= out["heat_threshold_c"]),
        ("cold", out["t_mean"] <= COLD_RUN_C),
    ):
        f = flag & (contiguous | flag.groupby(out["site_code"]).cumcount().eq(0))
        k = f.groupby(out["site_code"], group_keys=False).apply(_runs_day)
        out[f"{name}_day2"] = (k == 2).astype(float)
        out[f"{name}_day3plus"] = (k >= 3).astype(float)

    out["fe_month"] = out["site_code"] + "|" + out["date"].dt.strftime("%Y-%m")
    out["fe_dow"] = out["site_code"] + "|" + out["date"].dt.dayofweek.astype(str)
    iso = out["date"].dt.isocalendar()
    out[CLUSTER] = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    return out


def fit(df: pd.DataFrame, y: str, terms: pd.DataFrame) -> pd.DataFrame:
    """pyfixest OLS of log y on terms, absorbing both fixed effects; CRV1 clustered by week."""
    import pyfixest as pf

    data = pd.concat([df[list(FE) + [CLUSTER]], np.log(df[y]).rename("log_y"), terms], axis=1)
    fml = f"log_y ~ {' + '.join(terms.columns)} | {' + '.join(FE)}"
    tidy = pf.feols(fml, data=data, vcov={"CRV1": CLUSTER}).tidy()
    return pd.DataFrame(
        {
            "term": tidy.index,
            "coef": tidy["Estimate"].to_numpy(),
            "se": tidy["Std. Error"].to_numpy(),
            "lo": tidy["2.5%"].to_numpy(),
            "hi": tidy["97.5%"].to_numpy(),
        }
    ).assign(n_obs=len(data))


def _pct(t: pd.DataFrame) -> pd.DataFrame:
    for c in ("coef", "lo", "hi"):
        t[f"{c}_pct"] = 100 * np.expm1(t[c])
    return t


def bin_term(lo: float, variable: str = "daily mean") -> str:
    """Formula-safe column name for the bin with lower edge ``lo`` (index into the edges)."""
    return f"tb{VARIABLES[variable][2].index(lo)}"


def bin_dummies(df: pd.DataFrame, variable: str = "daily mean") -> pd.DataFrame:
    _, bin_col, _, ref, _ = VARIABLES[variable]
    d = pd.get_dummies(df[bin_col], dtype=float)
    d.columns = [bin_term(lo, variable) for lo in d.columns]
    return d.drop(columns=bin_term(ref, variable))


def controls(df: pd.DataFrame) -> pd.DataFrame:
    return df[["holiday", "christmas", "school_summer"]]


def curve(df: pd.DataFrame, y: str, variable: str = "daily mean") -> pd.DataFrame:
    _, bin_col, edges, ref_bin, (tail_lo, tail_hi) = VARIABLES[variable]
    terms = pd.concat([bin_dummies(df, variable), controls(df)], axis=1)
    t = fit(df, y, terms)
    t = t[t["term"].str.startswith("tb")].copy()
    t["bin_lo"] = [edges[int(s[2:])] for s in t["term"]]
    ref = pd.DataFrame(
        {
            "term": [bin_term(ref_bin, variable)],
            "coef": [0.0],
            "se": [0.0],
            "lo": [0.0],
            "hi": [0.0],
            "bin_lo": [ref_bin],
        }
    )
    t = pd.concat([t, ref], ignore_index=True).sort_values("bin_lo")
    counts = df.groupby(bin_col).agg(site_days=("site_code", "size"), dates=("date", "nunique"))
    t = t.merge(counts, left_on="bin_lo", right_index=True, how="left")
    t["bin"] = [bin_label(lo, variable) for lo in t["bin_lo"]]
    idx = [edges.index(lo) for lo in t["bin_lo"]]
    t["bin_mid"] = [
        (edges[i] + edges[i + 1]) / 2
        if i > 0 and i < len(edges) - 2
        else (tail_lo if i == 0 else tail_hi)
        for i in idx
    ]
    t["variable"] = variable
    return _pct(t.drop(columns="n_obs"))


def hourly(df: pd.DataFrame, sites: list[str]) -> pd.DataFrame:
    """Bin response of the mean demand in each local hour, for the given sites."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(TZ)
    day, hour = local.normalize().tz_localize(None), local.hour
    rows = []
    sub = df[df["site_code"].isin(sites)]
    for h in range(24):
        m = hour == h
        hk = (
            w[m]
            .groupby(day[m])
            .mean()
            .stack()
            .rename("hour_kw")
            .rename_axis(["date", "site_code"])
            .reset_index()
        )
        x = sub.merge(hk[hk["hour_kw"] > 0], on=["date", "site_code"], how="inner")
        t = curve(x, "hour_kw")
        rows.append(t.assign(hour=h))
    return pd.concat(rows, ignore_index=True)


def extras(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = [bin_dummies(df), controls(df)]
    persist = fit(
        df,
        "kw",
        pd.concat(base + [df[["hot_day2", "hot_day3plus", "cold_day2", "cold_day3plus"]]], axis=1),
    )
    warm, cold = (df["t_mean"] >= WARM_C).astype(float), (df["t_mean"] < MILD_C).astype(float)
    other = pd.DataFrame(
        {
            "dew_point_warm_per_c": df["td_mean"] * warm,
            "wind_cold_per_m_s": df["wind_mean"] * cold,
            "solar_per_kwh_m2": df["solar_kwh_m2"],
        }
    )
    oth = fit(df, "kw", pd.concat(base + [other], axis=1))
    keep_p = persist["term"].str.contains("day")
    keep_o = oth["term"].isin(other.columns)
    return _pct(persist[keep_p].copy()), _pct(oth[keep_o].copy())


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = panel()
    curves, persist, other, hours = [], [], [], []
    with threadpool_limits(4):
        for g in GROUPS:
            sub = df[df["group"] == g]
            for y, outcome in (("kw", "daily mean"), ("peak_kw", "daily peak")):
                for variable in VARIABLES:
                    curves.append(
                        curve(sub, y, variable).assign(
                            group=g, outcome=outcome, sites=sub["site_code"].nunique()
                        )
                    )
            p, o = extras(sub)
            persist.append(p.assign(group=g))
            other.append(o.assign(group=g))
            hours.append(hourly(sub, sorted(sub["site_code"].unique())).assign(group=g))
    curves, persist, other, hours = (
        pd.concat(x, ignore_index=True) for x in (curves, persist, other, hours)
    )
    curves.to_csv(OUT_DIR / "curve.csv", index=False)
    persist.to_csv(OUT_DIR / "persistence.csv", index=False)
    other.to_csv(OUT_DIR / "other_weather.csv", index=False)
    hours.to_csv(OUT_DIR / "hourly.csv", index=False)

    def at(g: str, outcome: str, label: str, variable: str = "daily mean") -> list[float]:
        r = curves[
            (curves["group"] == g)
            & (curves["outcome"] == outcome)
            & (curves["variable"] == variable)
            & (curves["bin"] == label)
        ]
        return [round(float(r[c].iloc[0]), 1) for c in ("coef_pct", "lo_pct", "hi_pct")]

    summary = {
        "site_days": len(df),
        "sites": df.groupby("group")["site_code"].nunique().to_dict(),
        "window": [str(df["date"].min().date()), str(df["date"].max().date())],
        "reference_bin_c": [REF_BIN, REF_BIN + 2],
        "pct_ci": {
            f"{g} | {o} | {b}": at(g, o, b)
            for g in GROUPS
            for o in ("daily mean", "daily peak")
            for b in ("<-2", "0", "8", "20", "22", ">=24")
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Weather response: %d site-days, groups %s", len(df), summary["sites"])
    return {
        "curve": curves,
        "persistence": persist,
        "other": other,
        "hourly": hours,
        "summary": summary,
    }
