"""Electricity response to cooling need, per site (``interim/weather_daily.parquet``).

Daily mean demand (kW) on non-holiday weekdays is regressed on heating degree
days (15.5 degC) and cooling degree days (integration method) with a fixed
effect for every calendar month of the record. The slopes therefore come only
from day-to-day weather within a month - not from the seasonal cycle, the
August holiday dip or year-on-year level shifts. Standard errors are HAC
(Newey-West, 5 lags). The cooling slope is reported as % of the site's mean
demand per degC day; the base with the lowest AIC is recorded alongside.

At sites with fossil-fuel CHP the meter records grid import, which rises when a
heat-led unit turns down in hot weather, so their slope mixes cooling with CHP
operation; they are reported as a separate group.

Outputs (``code/reports/cooling/``): ``cooling_response_site.csv``,
``cooling_measure_r2.csv`` (which daily measure tracks summer demand best) and
``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..data.weather_daily import CDD_BASES, _tag
from ..experiments.features import _holiday_set
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "cooling"
TZ = "Europe/London"
PRIMARY = "cdd_15_5"
MIN_PERIODS_PER_DAY = 44
MIN_DAYS, MIN_SUMMER_DAYS = 300, 60
HAC_LAGS = 5
#: Daily measures compared on summer (June-September) weekdays.
MEASURES = (
    "t_mean",
    "t_max",
    "cdd_15_5",
    "cdd_18",
    "cdd_22",
    "cdd_dm_18",
    "h_mean",
    "tw_max",
    "td_mean",
    "solar_kwh_m2",
)


def daily_demand() -> pd.DataFrame:
    """Mean kW per site and local day; days with fewer than 44 observed half-hours are NaN
    and days with no site observed are dropped."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(TZ).date
    n = w.notna().groupby(local).sum()
    out = w.groupby(local).mean().where(n >= MIN_PERIODS_PER_DAY)
    out.index = pd.to_datetime(out.index)
    return out.dropna(how="all")


def fit_site(x: pd.DataFrame) -> dict:
    """Month-fixed-effect regression of daily kW on HDD and CDD, for every CDD base."""
    import statsmodels.api as sm

    fe = pd.get_dummies(x.index.to_period("M").astype(str), drop_first=True, dtype=float)
    fe.index = x.index
    fits = {}
    for b in CDD_BASES:
        col = f"cdd_{_tag(b)}"
        X = sm.add_constant(fe.assign(hdd=x["hdd_15_5"], cdd=x[col]))
        fits[col] = sm.OLS(x["kw"], X).fit(cov_type="HAC", cov_kwds={"maxlags": HAC_LAGS})
    f, mean = fits[PRIMARY], float(x["kw"].mean())
    return {
        "n_days": len(x),
        "mean_kw": mean,
        "cdd_pct": 100 * f.params["cdd"] / mean,
        "cdd_pct_lo": 100 * (f.params["cdd"] - 1.96 * f.bse["cdd"]) / mean,
        "cdd_pct_hi": 100 * (f.params["cdd"] + 1.96 * f.bse["cdd"]) / mean,
        "hdd_pct": 100 * f.params["hdd"] / mean,
        "best_base": min(fits, key=lambda c: fits[c].aic),
    }


def measure_r2(x: pd.DataFrame) -> dict:
    """Squared correlation of summer weekday demand with each measure, within each year."""
    s = x[x.index.month.isin([6, 7, 8, 9])]
    y = s["kw"] - s.groupby(s.index.year)["kw"].transform("mean")
    out = {}
    for m in MEASURES:
        v = s[m] - s.groupby(s.index.year)[m].transform("mean")
        out[m] = float(np.corrcoef(v, y)[0, 1] ** 2) if v.std() > 0 and y.std() > 0 else np.nan
    return out


def run() -> dict[str, pd.DataFrame]:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dd = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet").set_index(["site_code", "date"])
    kw = daily_demand()
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").set_index("site_code")
    chp = sites["fossil_fuel_led_chp_units_operated_on_site"].fillna(0) > 0
    hol = pd.DatetimeIndex(
        sorted(_holiday_set(range(kw.index.year.min(), kw.index.year.max() + 1)))
    )
    have = set(dd.index.get_level_values("site_code"))

    rows, r2 = [], []
    with threadpool_limits(4):
        for s in kw.columns:
            if s not in have:
                continue
            x = dd.loc[s].join(kw[s].rename("kw"), how="inner").dropna(subset=["kw"])
            x = x[(x.index.dayofweek < 5) & ~x.index.isin(hol)]
            if len(x) < MIN_DAYS or x.index.month.isin([6, 7, 8]).sum() < MIN_SUMMER_DAYS:
                continue
            rows.append({"site_code": s, "chp": bool(chp.get(s, False)), **fit_site(x)})
            r2.append({"site_code": s, **measure_r2(x)})

    resp = pd.DataFrame(rows).merge(
        sites[["site_name", "organisation_type", "site_use_type"]],
        left_on="site_code",
        right_index=True,
        how="left",
    )
    r2 = pd.DataFrame(r2)
    resp.to_csv(OUT_DIR / "cooling_response_site.csv", index=False)
    r2.to_csv(OUT_DIR / "cooling_measure_r2.csv", index=False)

    def group(g: pd.DataFrame) -> dict:
        return {
            "sites": len(g),
            "cdd_pct_q10_q50_q90": g["cdd_pct"].quantile([0.1, 0.5, 0.9]).round(2).tolist(),
            "demand_weighted_cdd_pct": round(
                float(np.average(g["cdd_pct"], weights=g["mean_kw"])), 2
            ),
            "significant_positive": int((g["cdd_pct_lo"] > 0).sum()),
            "significant_negative": int((g["cdd_pct_hi"] < 0).sum()),
        }

    summary = {
        "primary_measure": PRIMARY,
        "window": [str(dd.index.get_level_values("date").min().date()), str(kw.index.max().date())],
        "no_chp": group(resp[~resp["chp"]]),
        "chp": group(resp[resp["chp"]]),
        "best_base": resp["best_base"].value_counts().to_dict(),
        "summer_r2_median": r2.drop(columns="site_code").median().round(3).to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Cooling response: %d sites (%d with CHP)", len(resp), int(resp["chp"].sum()))
    return {"response": resp, "measure_r2": r2, "summary": summary}
