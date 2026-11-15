"""Where the day-ahead error sits and what could remove it.

The proof of concept ends with two hypotheses about the error that the best models leave:
on-site generation causes most of it, and occupancy or mobility signals would remove some of it
on atypical days. This module tests both with what the project holds, on the CPU.

1. **Anatomy** of the stored forecasts (91 sites, 69 origins) and of a forecast for every day of
   the test window (``experiments/forecast_probe.py``): the error by site group (on-site CHP,
   inpatient, outpatient), by kind of day (typical, bank holiday or Christmas, doctor strike, hot
   or cold, an unexplained shift of the day's level), and split into **level** and **shape**. The
   level part is what remains to be gained from knowing the day's mean demand exactly: the
   forecast is rescaled to the day's true mean, and the error that is left is shape. No daily
   signal (occupancy, footfall, plant state) can do better than that oracle.
2. **Decisions**: the same split for the peak cut of a battery that plans on the forecast
   (``flexibility.plan_threshold``), with the level oracle as a third plan.
3. **Strike days** as a shock to activity: national doctor strikes cancel elective work on dated
   days. Daily demand (whole day, working hours, night) on strike days against the same site,
   month and weekday, with weather and calendar controls; pseudo-strikes one week earlier and
   later as the placebo. If activity moved electricity, it would show here.
4. **Footfall** at the three sites that have it (2024): what the day's count explains of the
   day's demand beyond calendar and weather, and whether Chronos-2 forecasts better with the
   count given in context (``forecast_probe.run_mobility``).
5. **History and issue time**: the error of Chronos-2 by length of context, and by issue time
   (the day before, midnight, 06:00) with the peak cut of a battery that plans again on the day.
6. **Plant**: the largest step of each site-day that the last four weeks do not explain, the
   error by the size of that step, and the error of the sites with on-site generation against
   what sites without a plant have at the same total demand (import plus reported generation).
7. **Heat and cold waves** as Lu et al. (Nature Communications 17, 10041, 2026) define them:
   days on which the site's temperature is beyond the 5th or 95th percentile of the days before
   the test window.

Outputs in ``code/reports/forecast_anatomy/``.
"""

from __future__ import annotations

import json

import holidays
import numpy as np
import pandas as pd

from ..config import GEO_DIR, INTERIM_DIR, PROJECT_ROOT, load_config
from ..data.strikes import strike_days
from ..experiments import poc
from ..logging_utils import get_logger
from . import flexibility as fx
from . import weather_causal as wc
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_anatomy"
POC = PROJECT_ROOT / "code" / "reports" / "poc"
CONFIG = "code/configs/experiments/poc.yaml"
TZ = "Europe/London"
#: stored proof-of-concept forecasts: name -> (file, model filter)
STORED = {
    "seasonal_naive": ("forecasts_baselines.parquet", "seasonal_naive"),
    "lightgbm_ifs": ("forecasts_lightgbm_ifs.parquet", None),
    "tnp": ("forecasts_tnp.parquet", None),
    "pttnp": ("forecasts_pttnp.parquet", None),
    "gtnp": ("forecasts_gtnp.parquet", None),
    "tnp_ifs": ("forecasts_tnp_ifs.parquet", None),
    "tabpfn_ts_ifs": ("forecasts_tabpfn_ts_ifs.parquet", None),
    "timesfm3_ifs": ("forecasts_timesfm3_ifs.parquet", None),
    "chronos2": ("forecasts_chronos2.parquet", None),
    "chronos2_ifs": ("forecasts_chronos2_ifs.parquet", None),
}
BEST = "chronos2_ifs"
SURPRISE = 0.15  # a day's level differs from the same weekday of the last four weeks by this
HOT_TMAX, COLD_TMEAN = 28.0, 2.0
SWITCH_RATIO = 3.0  # a within-day step this many times the site's ordinary largest step
BIG_STEP = 1.0  # a within-day step larger than the site's mean demand
STEP_BINS = (0.0, 0.1, 0.25, 0.5, 1.0, np.inf)
STEP_LABELS = ("under 10 %", "10 to 25 %", "25 to 50 %", "50 to 100 %", "over 100 %")
WAVE_PCT = (5.0, 95.0)  # cold and heat wave days, as Lu et al. 2026 label them
WAVES = ("cold wave", "heat wave", "other days")
MIN_PERIODS = 40
KINDS = (
    "typical",
    "level shift",
    "holiday or Christmas",
    "strike day",
    "strike day near holidays",
    "hot or cold day",
)
SITE_TYPES = ("acute, no CHP", "acute, CHP import", "mental health", "community", "ambulance")
N_BOOT = 2000
BLOCK_DAYS = 7


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def poc_panel() -> tuple[dict, poc.Panel]:
    config = load_config(CONFIG)
    return config, poc.load_panel(config)


def actual_long(panel: poc.Panel, sites: list[str] | None = None) -> pd.DataFrame:
    """Observed demand (kWh per half-hour) with the local day of each period."""
    y = panel.y_raw[sites or panel.sites]
    day = pd.Series(panel.local_date.astype("datetime64[ns]"), index=y.index)
    out = y.stack().rename("y").rename_axis(["target", "site_code"]).reset_index()
    out["day"] = out["target"].map(day)
    return out


def load_stored(name: str) -> pd.DataFrame:
    file, model = STORED[name]
    f = pd.read_parquet(POC / file)
    if model is not None:
        f = f[f["model"] == model]
    return f.assign(model=name)


def day_table(
    panel: poc.Panel, sites: list[str] | None = None, test_start: str = "2025-06-01"
) -> pd.DataFrame:
    """Per site and local day: mean demand, the level expected from the same weekday of the last
    four weeks, calendar and weather flags, and one ``kind`` per day. ``wave`` marks the days
    whose temperature (mean of the day's highest and lowest) is beyond the 5th or 95th
    percentile of the site's days before ``test_start``."""
    sites = sites or panel.sites
    y = panel.y_raw[sites] * 2.0
    day = pd.DatetimeIndex(panel.local_date.astype("datetime64[ns]"))
    n = y.notna().groupby(day).sum()
    mean = y.groupby(day).mean().where(n >= MIN_PERIODS)
    lags = np.stack(
        [mean.shift(7 * k, freq="D").reindex(mean.index).to_numpy() for k in (1, 2, 3, 4)]
    )
    enough = np.isfinite(lags).sum(axis=0) >= 2
    with np.errstate(all="ignore"):
        ref = np.where(enough, np.nanmedian(lags, axis=0), np.nan)
    ref = pd.DataFrame(ref, index=mean.index, columns=mean.columns)
    out = (
        mean.stack(future_stack=True)
        .rename("mean_kw")
        .rename_axis(["day", "site_code"])
        .reset_index()
    )
    out["ref_kw"] = ref.stack(future_stack=True).to_numpy()
    out = out.dropna(subset=["mean_kw"])
    out["log_shift"] = np.log(out["mean_kw"] / out["ref_kw"])
    w = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet",
        columns=["site_code", "date", "t_mean", "t_min", "t_max"],
    ).rename(columns={"date": "day"})
    w["t_mid"] = (w["t_max"] + w["t_min"]) / 2.0
    before = w[w["day"] < pd.Timestamp(test_start)].groupby("site_code")["t_mid"]
    w["t_cold"] = w["site_code"].map(before.quantile(WAVE_PCT[0] / 100.0))
    w["t_hot"] = w["site_code"].map(before.quantile(WAVE_PCT[1] / 100.0))
    out = out.merge(w, on=["site_code", "day"], how="left")
    out["wave"] = np.select(
        [out["t_mid"] < out["t_cold"], out["t_mid"] > out["t_hot"]], WAVES[:2], default=WAVES[2]
    )
    years = range(out["day"].dt.year.min(), out["day"].dt.year.max() + 1)
    hol = pd.DatetimeIndex(sorted(holidays.UK(subdiv="ENG", years=years)))
    md = out["day"].dt.month * 100 + out["day"].dt.day
    out["christmas"] = (md >= 1224) | (md == 101)
    out["holiday"] = out["day"].isin(hol) | out["christmas"]
    out["strike"] = out["day"].isin(strike_days()["date"])
    out["near_holidays"] = near_holidays(out["day"])
    out["weather"] = (out["t_max"] >= HOT_TMAX) | (out["t_mean"] <= COLD_TMEAN)
    out["shift"] = out["log_shift"].abs() > np.log1p(SURPRISE)
    out["kind"] = np.select(
        [
            out["holiday"],
            out["strike"] & ~out["near_holidays"],
            out["strike"],
            out["weather"],
            out["shift"],
        ],
        [
            "holiday or Christmas",
            "strike day",
            "strike day near holidays",
            "hot or cold day",
            "level shift",
        ],
        default="typical",
    )
    out["group"] = out["site_code"].map(wr.site_groups())
    step = unusual_steps(y, day)
    out = out.merge(step, on=["site_code", "day"], how="left")
    out["switch"] = out["step_ratio"] > SWITCH_RATIO
    out["big_step"] = out["step_rel"] > BIG_STEP
    out["step_bin"] = pd.cut(out["step_rel"], STEP_BINS, labels=STEP_LABELS).astype(object)
    return out


def unusual_steps(y: pd.DataFrame, day: pd.DatetimeIndex) -> pd.DataFrame:
    """The largest sustained step of each site-day that is not in the site's usual profile.

    The usual profile of a half-hour is the median of the same half-hour and weekday in the last
    four weeks. A step is the mean of the residual over the next two hours minus its mean over
    the last two hours. ``step_ratio`` is the day's largest absolute step over the site's median
    daily largest step, so a value of 3 is a step three times the site's ordinary one: a plant
    that starts or stops, a supply that changes over. ``step_rel`` is the same step over the
    site's mean demand: 1 is a step as large as the site's mean import. A site whose plant
    starts and stops on most days has a large ordinary step and a small ratio, so the two
    measures answer different questions: unusual for the site, and large for the site."""
    usual = (
        pd.concat([y.shift(336 * k) for k in (1, 2, 3, 4)], keys=range(4))
        .groupby(level=1)
        .median()
        .reindex(y.index)
    )
    r = y - usual
    ahead = r.rolling(4, min_periods=4).mean().shift(-3)
    behind = r.rolling(4, min_periods=4).mean().shift(1)
    step = (ahead - behind).abs().groupby(day).max()
    ratio = (step / step.median()).stack().rename("step_ratio")
    rel = (step / y.mean()).stack().rename("step_rel")
    return pd.concat([ratio, rel], axis=1).rename_axis(["day", "site_code"]).reset_index()


# --------------------------------------------------------------------------- #
# 1. Error: level and shape
# --------------------------------------------------------------------------- #


def site_day_scores(
    f: pd.DataFrame, actual: pd.DataFrame, min_periods: int = MIN_PERIODS
) -> pd.DataFrame:
    """Per (model, site, day): summed absolute error of the forecast and of the forecast rescaled
    to the day's true mean (the level oracle), with the summed demand."""
    m = f[["model", "site_code", "target", "point"]].merge(
        actual, on=["target", "site_code"], how="inner"
    )
    key = ["model", "site_code", "day"]
    g = m.groupby(key, sort=False, observed=True)
    ratio = g["y"].transform("sum") / g["point"].transform("sum")
    m["ae"] = (m["y"] - m["point"]).abs()
    m["ae_shape"] = (m["y"] - m["point"] * ratio).abs()
    out = m.groupby(key, sort=False, observed=True).agg(
        n=("y", "size"),
        a_kwh=("y", "sum"),
        f_kwh=("point", "sum"),
        ae=("ae", "sum"),
        ae_shape=("ae_shape", "sum"),
    )
    out = out[out["n"] >= min_periods].reset_index()
    out["bias_pct"] = 100 * (out["f_kwh"] / out["a_kwh"] - 1)
    return out


def error_table(sd: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Error by ``by``: the site mean of each site's normalised error (as the proof of concept
    scores), the same after the level oracle, and the kWh of error and demand in the cell."""
    per = (
        sd.groupby(["model", *by, "site_code"], observed=True)
        .agg(
            ae=("ae", "sum"), ae_shape=("ae_shape", "sum"), a=("a_kwh", "sum"), days=("day", "size")
        )
        .reset_index()
    )
    per["nmae"] = per["ae"] / per["a"]
    per["nmae_shape"] = per["ae_shape"] / per["a"]
    out = (
        per.groupby(["model", *by], observed=True)
        .agg(
            sites=("site_code", "nunique"),
            site_days=("days", "sum"),
            nmae_pct=("nmae", lambda s: 100 * s.mean()),
            nmae_level_oracle_pct=("nmae_shape", lambda s: 100 * s.mean()),
            error_mwh=("ae", lambda s: s.sum() / 1000.0),
            shape_error_mwh=("ae_shape", lambda s: s.sum() / 1000.0),
            demand_mwh=("a", lambda s: s.sum() / 1000.0),
        )
        .reset_index()
    )
    out["level_share_pct"] = 100 * (1 - out["shape_error_mwh"] / out["error_mwh"])
    tot = out.groupby("model")[["error_mwh", "demand_mwh", "site_days"]].transform("sum")
    out["error_share_pct"] = 100 * out["error_mwh"] / tot["error_mwh"]
    out["demand_share_pct"] = 100 * out["demand_mwh"] / tot["demand_mwh"]
    out["day_share_pct"] = 100 * out["site_days"] / tot["site_days"]
    return out


def same_month(sd: pd.DataFrame, wave: str) -> pd.DataFrame:
    """The ``wave`` days and the other days of the site-months that have both."""
    x = sd.assign(month=sd["day"].dt.to_period("M").astype(str))
    x = x[x["wave"].isin([wave, WAVES[2]])]
    kinds = x.groupby(["site_code", "month"])["wave"].transform("nunique")
    return x[kinds == 2]


def wave_matched(sd: pd.DataFrame, wave: str) -> pd.DataFrame:
    """Error on ``wave`` days against the other days of the same site and calendar month, so
    that the season (a low summer level, plant that is off for maintenance) is not counted as
    an effect of the weather. Site-months without both kinds of day are left out. With a
    ``bias_pct`` column the median bias of the day's energy comes with it."""
    x = same_month(sd, wave)
    out = pd.concat(
        [
            error_table(x.assign(group="all sites"), ["group", "wave"]),
            error_table(x, ["group", "wave"]),
        ],
        ignore_index=True,
    )
    if "bias_pct" in x.columns:
        bias = pd.concat(
            [
                x.assign(group="all sites").groupby(["group", "wave"])["bias_pct"].median(),
                x.groupby(["group", "wave"])["bias_pct"].median(),
            ]
        ).rename("median_bias_pct")
        out = out.join(bias, on=["group", "wave"])
    return out.assign(matched_to=wave)


def wave_excess(sd: pd.DataFrame, wave: str) -> float:
    """The share of all error (kWh) that ``wave`` days have beyond the error rate of the other
    days of the same site and month: what bringing them to their neighbours would remove."""
    x = same_month(sd, wave)
    g = x.groupby(["site_code", "month", "wave"])[["ae", "a_kwh"]].sum().unstack("wave")
    rate = g[("ae", WAVES[2])] / g[("a_kwh", WAVES[2])]
    excess = g[("ae", wave)] - g[("a_kwh", wave)] * rate
    return float(100 * excess.sum() / sd["ae"].sum())


def wave_matched_value(v: pd.DataFrame, wave: str) -> pd.DataFrame:
    """The captured share of the peak cut on ``wave`` days against the other days of the same
    site and calendar month."""
    x = same_month(v, wave)
    out = pd.concat(
        [
            value_table(x.assign(group="all sites"), ["group", "wave"]),
            value_table(x, ["group", "wave"]),
        ],
        ignore_index=True,
    )
    return out.assign(matched_to=wave)


# --------------------------------------------------------------------------- #
# 2. Decisions
# --------------------------------------------------------------------------- #


def value_site_days(f: pd.DataFrame, actual: pd.DataFrame, mean_kw: pd.Series) -> pd.DataFrame:
    """Per (model, site, day) with 48 observed periods: the day's peak, the peak cut under
    perfect foresight, and the cut achieved with thresholds planned on the point forecast, on its
    70 % quantile and on the point forecast rescaled to the day's true mean."""
    rows = []
    for model, fm in f.groupby("model", observed=True):
        m = fm.merge(actual, on=["target", "site_code"], how="inner").sort_values(
            ["site_code", "day", "target"]
        )
        m["slot"] = m.groupby(["site_code", "day"]).cumcount()
        m = m[m.groupby(["site_code", "day"])["slot"].transform("size") == 48]
        key = m[["site_code", "day"]].drop_duplicates().reset_index(drop=True)
        key["row"] = np.arange(len(key))
        m = m.merge(key, on=["site_code", "day"])

        n_rows = len(key)

        def mat(col, m=m, n_rows=n_rows):
            a = np.full((n_rows, 48), np.nan)
            a[m["row"].to_numpy(), m["slot"].to_numpy()] = m[col].to_numpy() * 2.0  # kW
            return a

        act, point = mat("y"), mat("point")
        p_kw = fx.BATTERY_SHARE * key["site_code"].map(mean_kw).to_numpy()
        e_kwh = p_kw * fx.DURATION_H
        ok = np.isfinite(act).all(axis=1) & np.isfinite(point).all(axis=1) & (p_kw > 0)
        key, act, point, p_kw, e_kwh = key[ok], act[ok], point[ok], p_kw[ok], e_kwh[ok]
        raw = act.max(axis=1)
        out = key[["site_code", "day"]].assign(
            model=model,
            raw_peak_kw=raw,
            perfect_kw=raw - fx.plan_threshold(act, p_kw, e_kwh),
        )
        plans = {"point": point}
        if "q70" in m.columns and m["q70"].notna().all():
            q70 = mat("q70")[ok]
            if not np.allclose(q70, point):
                plans["q70"] = q70
        plans["level"] = point * (act.sum(axis=1) / point.sum(axis=1))[:, None]
        for name, fc in plans.items():
            theta = fx.plan_threshold(fc, p_kw, e_kwh)
            out[f"cut_{name}_kw"] = raw - fx.realised_peak(theta, act, p_kw, e_kwh)
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def value_table(v: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = [c for c in v.columns if c.startswith("cut_")]
    g = v.groupby(["model", *by], observed=True)
    out = g.agg(site_days=("day", "size"), perfect_kw=("perfect_kw", "sum")).reset_index()
    for c in cols:
        name = c[4:-3]
        got = g[c].sum(min_count=1).to_numpy()
        out[f"captured_{name}_pct"] = 100 * got / out["perfect_kw"].to_numpy()
    return out.drop(columns="perfect_kw")


# --------------------------------------------------------------------------- #
# 3. Strike days
# --------------------------------------------------------------------------- #


def near_holidays(dates: pd.Series) -> pd.Series:
    """Days in the run-up to Christmas, the first week of January, or the fortnight around
    Easter: weeks in which demand drifts with the school holidays, strike or not."""
    md = dates.dt.month * 100 + dates.dt.day
    out = (md >= 1217) | (md <= 108)
    years = range(int(dates.dt.year.min()), int(dates.dt.year.max()) + 1)
    hol = holidays.UK(subdiv="ENG", years=years)
    for d, name in hol.items():
        if "Easter Monday" in name:
            em = pd.Timestamp(d)
            out |= (dates >= em - pd.Timedelta(days=10)) & (dates <= em + pd.Timedelta(days=7))
    return out


def site_types() -> pd.Series:
    s = pd.read_parquet(INTERIM_DIR / "sites.parquet").drop_duplicates("site_code")
    s = s.set_index("site_code")
    org = s["organisation_type"].fillna("").str.upper()
    kind = np.select(
        [org.str.startswith("ACUTE"), org.str.contains("MENTAL"), org.str.contains("AMBULANCE")],
        ["acute", "mental health", "ambulance"],
        default="community",
    )
    chp = wr.site_groups().reindex(s.index).eq("on-site CHP")
    out = np.where(kind == "acute", np.where(chp, "acute, CHP import", "acute, no CHP"), kind)
    return pd.Series(out, index=s.index, name="site_type")


def window_means(hours: tuple[int, int], min_periods: int) -> pd.DataFrame:
    """Mean demand per site and local day over the local hours ``[hours[0], hours[1])``."""
    w = pd.read_parquet(PROJECT_ROOT / "code/data/processed/elec/site/wide_raw.parquet") * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(TZ)
    keep = (local.hour >= hours[0]) & (local.hour < hours[1])
    w, local = w[keep], local[keep]
    day = local.normalize().tz_localize(None).rename("date")
    n = w.notna().groupby(day).sum()
    m = w.groupby(day).mean().where(n >= min_periods)
    out = m.stack().rename("kw").rename_axis(["date", "site_code"]).reset_index()
    return out[out["kw"] > 0]


def strike_panel() -> pd.DataFrame:
    """The daily panel with strike flags, pseudo-strikes, site types and the working-hours and
    night outcomes."""
    df = wc.panel()
    st = strike_days()
    real = set(st["date"])
    df["site_type"] = df["site_code"].map(site_types()).fillna("community")
    weekday = df["date"].dt.dayofweek < 5
    quiet = ~df["holiday"].astype(bool) & ~df["christmas"].astype(bool)
    df["strike"] = df["date"].isin(real)
    near = near_holidays(df["date"])
    df["strike_wd"] = (df["strike"] & weekday & quiet).astype(float)
    df["strike_wd_clean"] = (df["strike"] & weekday & quiet & ~near).astype(float)
    df["strike_wd_near"] = (df["strike"] & weekday & quiet & near).astype(float)
    df["strike_we"] = (df["strike"] & ~weekday).astype(float)
    for name, shift in (("before", -7), ("after", 7)):
        pseudo = {d + pd.Timedelta(days=shift) for d in real} - real
        df[f"pseudo_{name}"] = (df["date"].isin(pseudo) & weekday & quiet & ~near).astype(float)
    work = window_means((8, 18), 16).rename(columns={"kw": "work_kw"})
    night = window_means((0, 6), 10).rename(columns={"kw": "night_kw"})
    df = df.merge(work, on=["date", "site_code"], how="left").merge(
        night, on=["date", "site_code"], how="left"
    )
    df["log_work"], df["log_night"] = np.log(df["work_kw"]), np.log(df["night_kw"])
    df["log_ratio"] = df["log_work"] - df["log_night"]  # activity is a working-hours matter
    for g in ("inpatient", "outpatient", "on-site CHP"):
        tag = g.replace(" ", "_").replace("-", "_")
        for v in ("hdd14", "cdd20", "solar_kwh_m2"):
            df[f"{v}__{tag}"] = df[v] * (df["group"] == g)
    return df


def strike_effect(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Demand on doctor strike weekdays against the same site, month and weekday, in per cent,
    for all sites and by site type; weekend strike days and pseudo-strikes a week before and
    after as placebos. Standard errors clustered by site and date."""
    import pyfixest as pf

    df = strike_panel() if df is None else df
    weather = [c for c in df.columns if "__" in c]
    controls = " + ".join([*weather, "holiday", "christmas", "school_summer"])
    rows = []

    def fit(y: str, terms: dict[str, pd.Series], label: str, site_type: str):
        d = df.dropna(subset=[y]).copy()
        for k, v in terms.items():
            d[k] = v.reindex(d.index)
        f = pf.feols(
            f"{y} ~ {' + '.join(terms)} + {controls} | fe_month + fe_dow",
            data=d,
            vcov={"CRV1": "site_code+date_id"},
        )
        t = f.tidy()
        for k in terms:
            b, se = t.loc[k, "Estimate"], t.loc[k, "Std. Error"]
            on = d[k] > 0
            rows.append(
                {
                    "outcome": y,
                    "term": label if len(terms) == 1 else k,
                    "site_type": site_type,
                    "sites": int(d.loc[on, "site_code"].nunique()),
                    "site_days": int(on.sum()),
                    "days": int(d.loc[on, "date"].nunique()),
                    "effect_pct": 100 * (np.exp(b) - 1),
                    "lo_pct": 100 * (np.exp(b - 1.96 * se) - 1),
                    "hi_pct": 100 * (np.exp(b + 1.96 * se) - 1),
                    "p": float(t.loc[k, "Pr(>|t|)"]),
                }
            )

    outcomes = {
        "log_kw": "whole day",
        "log_work": "08:00 to 18:00",
        "log_night": "00:00 to 06:00",
        "log_ratio": "08:00 to 18:00 against night",
    }
    for y in outcomes:
        fit(
            y,
            {
                "strike_wd_clean": df["strike_wd_clean"],
                "strike_wd_near": df["strike_wd_near"],
                "strike_we": df["strike_we"],
                "pseudo_before": df["pseudo_before"],
                "pseudo_after": df["pseudo_after"],
            },
            "",
            "all sites",
        )
    for st in SITE_TYPES:
        sub = df["site_type"] == st
        if df.loc[sub, "site_code"].nunique() < 3:
            continue
        for y in ("log_kw", "log_work", "log_ratio"):
            fit(y, {"strike_wd_clean": df["strike_wd_clean"] * sub}, "strike_wd_clean", st)
    out = pd.DataFrame(rows)
    out["outcome"] = out["outcome"].map(outcomes)
    out["term"] = out["term"].map(
        {
            "strike_wd_clean": "strike weekday, away from holidays",
            "strike_wd_near": "strike weekday, next to Christmas or Easter",
            "strike_we": "strike weekend day",
            "pseudo_before": "one week before (placebo)",
            "pseudo_after": "one week after (placebo)",
        }
    )
    return out


def hourly_means(min_periods: int = 2) -> pd.DataFrame:
    """Log of the mean demand per site, local day and local hour: one column per hour."""
    w = pd.read_parquet(PROJECT_ROOT / "code/data/processed/elec/site/wide_raw.parquet") * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(TZ)
    key = [local.normalize().tz_localize(None).rename("date"), local.hour.rename("hour")]
    n = w.notna().groupby(key).sum()
    m = w.groupby(key).mean().where(n >= min_periods)
    long = m.stack().rename("kw").rename_axis(["date", "hour", "site_code"]).reset_index()
    long = long[long["kw"] > 0]
    long["log_kw"] = np.log(long["kw"])
    wide = long.pivot(index=["date", "site_code"], columns="hour", values="log_kw")
    wide.columns = [f"log_h{h:02d}" for h in wide.columns]
    return wide.reset_index()


def strike_profile(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """The strike effect hour by hour: demand in each local hour of a strike weekday away from
    holidays against the same site, month and weekday, with the controls of ``strike_effect``.
    The pseudo-strikes one week before and after are in the same fit."""
    import pyfixest as pf

    df = strike_panel() if df is None else df
    df = df.merge(hourly_means(), on=["date", "site_code"], how="left")
    weather = [c for c in df.columns if "__" in c]
    controls = " + ".join([*weather, "holiday", "christmas", "school_summer"])
    terms = ["strike_wd_clean", "strike_wd_near", "strike_we", "pseudo_before", "pseudo_after"]
    rows = []
    for hour in range(24):
        y = f"log_h{hour:02d}"
        d = df.dropna(subset=[y])
        t = pf.feols(
            f"{y} ~ {' + '.join(terms)} + {controls} | fe_month + fe_dow",
            data=d,
            vcov={"CRV1": "site_code+date_id"},
        ).tidy()
        for k in ("strike_wd_clean", "pseudo_before", "pseudo_after"):
            b, se = t.loc[k, "Estimate"], t.loc[k, "Std. Error"]
            rows.append(
                {
                    "hour": hour,
                    "term": k,
                    "effect_pct": 100 * (np.exp(b) - 1),
                    "lo_pct": 100 * (np.exp(b - 1.96 * se) - 1),
                    "hi_pct": 100 * (np.exp(b + 1.96 * se) - 1),
                    "p": float(t.loc[k, "Pr(>|t|)"]),
                }
            )
    out = pd.DataFrame(rows)
    out["term"] = out["term"].map(
        {
            "strike_wd_clean": "strike weekday, away from holidays",
            "pseudo_before": "one week before (placebo)",
            "pseudo_after": "one week after (placebo)",
        }
    )
    return out


# --------------------------------------------------------------------------- #
# 4. Footfall
# --------------------------------------------------------------------------- #


def mobility_daily() -> pd.DataFrame:
    """The three footfall sites in 2024: daily demand, weather, calendar and counts."""
    from ..experiments.forecast_probe import MOBILITY_SITES

    df = wc.panel()
    df = df[df["site_code"].isin(MOBILITY_SITES) & (df["date"].dt.year == 2024)].copy()
    v = pd.read_parquet(INTERIM_DIR / "mobility_visits_daily.parquet")
    v["date"] = pd.to_datetime(v["date"]).dt.normalize()
    wide = v.pivot_table(index="date", columns="id", values="visits")
    own = wide[list(MOBILITY_SITES)].stack().rename("visits").rename_axis(["date", "site_code"])
    df = df.merge(own.reset_index(), on=["date", "site_code"], how="inner")
    df["uni_visits"] = df["date"].map(wide["university"])
    df["log_visits"], df["log_uni"] = np.log(df["visits"]), np.log(df["uni_visits"])
    df["month"], df["dow"] = df["date"].dt.month, df["date"].dt.dayofweek
    return df.sort_values(["site_code", "date"]).reset_index(drop=True)


def mobility_regression(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Elasticity of a site's daily demand to its daily footfall, within month and weekday, with
    weather and holidays held fixed; the university campus's footfall as the placebo regressor.
    Newey-West standard errors, seven lags."""
    import statsmodels.formula.api as smf

    df = mobility_daily() if df is None else df
    base = "hdd14 + cdd20 + solar_kwh_m2 + holiday + christmas + C(month) + C(dow)"
    rows = []
    for s, d in df.groupby("site_code"):
        d = d.assign(holiday=d["holiday"].astype(float), christmas=d["christmas"].astype(float))
        r0 = smf.ols(f"log_kw ~ {base}", d).fit()
        for label, extra, terms in (
            ("own footfall", "log_visits", ["log_visits"]),
            ("campus footfall (placebo)", "log_uni", ["log_uni"]),
            ("both", "log_visits + log_uni", ["log_visits", "log_uni"]),
        ):
            r = smf.ols(f"log_kw ~ {extra} + {base}", d).fit(
                cov_type="HAC", cov_kwds={"maxlags": 7}
            )
            for t in terms:
                rows.append(
                    {
                        "site_code": s,
                        "model": label,
                        "term": t,
                        "days": int(r.nobs),
                        "elasticity": r.params[t],
                        "lo": r.conf_int().loc[t, 0],
                        "hi": r.conf_int().loc[t, 1],
                        "p": r.pvalues[t],
                        "partial_r2": 1 - r.ssr / r0.ssr,
                        "raw_corr": float(np.corrcoef(d["log_kw"], d[t])[0, 1]),
                    }
                )
    return pd.DataFrame(rows)


def mobility_partial(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily demand and daily footfall of each site with month, weekday, weather and holidays
    taken out of both: the two residuals of the elasticity regression, in per cent."""
    import statsmodels.formula.api as smf

    df = mobility_daily() if df is None else df
    base = "hdd14 + cdd20 + solar_kwh_m2 + holiday + christmas + C(month) + C(dow)"
    rows = []
    for s, d in df.groupby("site_code"):
        d = d.assign(holiday=d["holiday"].astype(float), christmas=d["christmas"].astype(float))
        rows.append(
            pd.DataFrame(
                {
                    "site_code": s,
                    "date": d["date"].to_numpy(),
                    "demand_pct": 100 * smf.ols(f"log_kw ~ {base}", d).fit().resid.to_numpy(),
                    "footfall_pct": 100 * smf.ols(f"log_visits ~ {base}", d).fit().resid.to_numpy(),
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def mobility_day_kinds(df: pd.DataFrame) -> pd.DataFrame:
    """Kind of each footfall site-day: calendar, strike, an unusual count, or typical."""
    d = df[["site_code", "date", "visits", "holiday", "christmas"]].copy()
    ref = d.groupby("site_code")["visits"].transform(
        lambda s: pd.concat([s.shift(7 * k) for k in (1, 2, 3, 4)], axis=1).median(axis=1)
    )
    d["log_count_shift"] = np.log(d["visits"] / ref)
    d["kind"] = np.select(
        [
            d["holiday"].astype(bool) | d["christmas"].astype(bool),
            d["date"].isin(strike_days()["date"]),
            d["log_count_shift"].abs() > 0.2,
        ],
        ["holiday or Christmas", "strike day", "unusual footfall"],
        default="typical",
    )
    return d.rename(columns={"date": "day"})[["site_code", "day", "kind", "log_count_shift"]]


def quantile_scores(f: pd.DataFrame, actual: pd.DataFrame) -> pd.DataFrame:
    """Per (model, site, day): absolute error, pinball loss on the quantile grid, coverage."""
    m = f.merge(actual, on=["target", "site_code"], how="inner")
    qcols = [c for c in m.columns if c.startswith("q") and c[1:].isdigit()]
    y = m["y"].to_numpy()
    pb = np.zeros(len(m))
    for c in qcols:
        tau = int(c[1:]) / 100.0
        d = y - m[c].to_numpy()
        pb += np.maximum(tau * d, (tau - 1) * d)
    m["crps"] = 2 * pb / len(qcols)
    m["ae"] = (m["y"] - m["point"]).abs()
    m["in80"] = ((m["y"] >= m["q10"]) & (m["y"] <= m["q90"])).astype(float)
    return (
        m.groupby(["model", "site_code", "day"], observed=True)
        .agg(
            n=("y", "size"),
            a=("y", "sum"),
            ae=("ae", "sum"),
            crps=("crps", "sum"),
            in80=("in80", "mean"),
        )
        .reset_index()
    )


def _nmae(d: pd.DataFrame, col: str = "ae") -> float:
    per = d.groupby("site_code").agg(e=(col, "sum"), a=("a", "sum"))
    return float(100 * (per["e"] / per["a"]).mean())


def paired_bootstrap(sd: pd.DataFrame, a: str, b: str, col: str = "ae", seed: int = 0) -> dict:
    """Difference in the site-mean normalised error, model ``a`` minus model ``b``, with a
    95 % interval from resampling blocks of days. Both models get the same resampled days, so
    what they have in common cancels."""
    x = sd[sd["model"].isin([a, b])]
    days = np.sort(x["day"].unique())
    block = np.arange(len(days)) // BLOCK_DAYS
    n_blocks = int(block.max()) + 1
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_blocks, (N_BOOT, n_blocks))
    counts = np.zeros((N_BOOT, n_blocks))
    np.add.at(counts, (np.arange(N_BOOT)[:, None], pick), 1.0)
    weights = np.vstack([np.ones(len(days)), counts[:, block]])  # row 0: the sample itself

    def nmae(model: str) -> np.ndarray:
        m = x[x["model"] == model]
        err = m.pivot(index="day", columns="site_code", values=col).reindex(days).fillna(0.0)
        dem = m.pivot(index="day", columns="site_code", values="a").reindex(days).fillna(0.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = (weights @ err.to_numpy()) / (weights @ dem.to_numpy())
        return 100 * np.nanmean(rate, axis=1)

    diff = nmae(a) - nmae(b)
    lo, hi = np.nanquantile(diff[1:], [0.025, 0.975])
    return {"diff_pts": float(diff[0]), "lo": float(lo), "hi": float(hi)}


def mobility_forecasts_path():
    """The footfall forecasts: the finished file, or the variants written so far."""
    done = OUT_DIR / "forecasts_mobility.parquet"
    return done if done.exists() else OUT_DIR / "forecasts_mobility.partial.parquet"


def mobility_scores(panel: poc.Panel) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scores of the footfall variants: by site set and kind of day, the paired differences from
    the base variant, and the site-day table."""
    from ..experiments.forecast_probe import MOBILITY_SITES

    f = pd.read_parquet(mobility_forecasts_path())
    have = set(f["model"].unique())
    actual = actual_long(panel, list(MOBILITY_SITES))
    sd = quantile_scores(f, actual)
    sd = sd[sd["n"] >= MIN_PERIODS].merge(
        mobility_day_kinds(mobility_daily()), on=["site_code", "day"], how="left"
    )
    sets = {
        "all three sites": list(MOBILITY_SITES),
        "two sites without CHP": [s for s in MOBILITY_SITES if s != "RH801"],
        **{s: [s] for s in MOBILITY_SITES},
    }
    rows, diffs = [], []
    for set_name, members in sets.items():
        x = sd[sd["site_code"].isin(members)]
        for kind in ("all days", *sorted(x["kind"].dropna().unique())):
            xk = x if kind == "all days" else x[x["kind"] == kind]
            for model, g in xk.groupby("model"):
                rows.append(
                    {
                        "sites": set_name,
                        "kind": kind,
                        "model": model,
                        "site_days": len(g),
                        "nmae_pct": _nmae(g),
                        "crps_pct": _nmae(g, "crps"),
                        "coverage_80": float(g["in80"].mean()),
                    }
                )
        if len(members) > 1:
            for kind in ("all days", "not typical"):
                xk = x if kind == "all days" else x[x["kind"] != "typical"]
                for a, b in (
                    ("mob_prev_day", "mob_base"),
                    ("mob_oracle", "mob_base"),
                    ("mob_uni_oracle", "mob_base"),
                    ("mob_oracle", "mob_uni_oracle"),
                    ("mob_oracle_ctx336", "mob_base_ctx336"),
                    ("mob_base", "mob_plain"),
                ):
                    if a not in have or b not in have:
                        continue
                    for col in ("ae", "crps"):
                        diffs.append(
                            {
                                "sites": set_name,
                                "kind": kind,
                                "model": a,
                                "against": b,
                                "metric": "nMAE" if col == "ae" else "CRPS",
                                **paired_bootstrap(xk, a, b, col),
                            }
                        )
    return pd.DataFrame(rows), pd.DataFrame(diffs), sd


# --------------------------------------------------------------------------- #
# 5. Plant: switches, dispatch, issue time
# --------------------------------------------------------------------------- #


def dispatch_table(sd: pd.DataFrame) -> pd.DataFrame:
    """Error of Chronos-2 without and with the known-ahead covariates (calendar, price, weather)
    at the CHP sites, by how the plant is dispatched (``chp_dispatch``): plants that follow the
    price or avoid the red band act on signals the covariates carry."""
    path = PROJECT_ROOT / "code" / "reports" / "chp_dispatch" / "classification.csv"
    c = pd.read_csv(path).set_index("site_code")
    signal = (c["price_following"] | c["red_band_avoiding"]).rename("signal")
    x = sd[sd["model"].isin(["chronos2", "chronos2_ifs"]) & (sd["group"] == "on-site CHP")]
    x = x.merge(signal, left_on="site_code", right_index=True, how="inner")
    x["dispatch"] = np.where(x["signal"], "follows price or red band", "heat-led or no pattern")
    per = (
        x.groupby(["dispatch", "model", "site_code"])
        .agg(ae=("ae", "sum"), a=("a_kwh", "sum"))
        .reset_index()
    )
    per["nmae"] = 100 * per["ae"] / per["a"]
    out = per.pivot_table(index=["dispatch", "site_code"], columns="model", values="nmae")
    out["gain_pts"] = out["chronos2"] - out["chronos2_ifs"]
    return (
        out.groupby("dispatch")
        .agg(
            sites=("gain_pts", "size"),
            nmae_plain_pct=("chronos2", "mean"),
            nmae_covariates_pct=("chronos2_ifs", "mean"),
            gain_pts=("gain_pts", "mean"),
        )
        .reset_index()
    )


def site_errors(sd: pd.DataFrame, model: str = BEST) -> pd.DataFrame:
    """Per site: error and demand of one model, the site's place on the map, and for sites with
    on-site generation the share of the site's electricity that the plant supplies
    (``chp_dispatch``, from the estates returns)."""
    x = sd[sd["model"] == model]
    out = (
        x.groupby(["site_code", "group"], observed=True)
        .agg(
            days=("day", "size"),
            error_mwh=("ae", lambda v: v.sum() / 1000.0),
            demand_mwh=("a_kwh", lambda v: v.sum() / 1000.0),
            big_step_days=("big_step", "sum"),
        )
        .reset_index()
    )
    out["nmae_pct"] = 100 * out["error_mwh"] / out["demand_mwh"]
    out["mean_kw"] = out["demand_mwh"] * 1000.0 / (out["days"] * 24.0)
    out["error_share_pct"] = 100 * out["error_mwh"] / out["error_mwh"].sum()
    out["demand_share_pct"] = 100 * out["demand_mwh"] / out["demand_mwh"].sum()
    geo = pd.read_parquet(GEO_DIR / "sites_geo.parquet").drop_duplicates("site_code")
    out = out.merge(geo[["site_code", "latitude", "longitude"]], on="site_code", how="left")
    path = PROJECT_ROOT / "code" / "reports" / "chp_dispatch" / "classification.csv"
    chp = pd.read_csv(path)[["site_code", "chp_share", "capacity_factor", "plausible"]]
    out = out.merge(chp, on="site_code", how="left")
    known = out["plausible"].fillna(False).astype(bool) & out["chp_share"].between(0, 0.95)
    out["chp_share"] = out["chp_share"].where(known & (out["group"] == "on-site CHP"))
    return out.sort_values("error_mwh", ascending=False).reset_index(drop=True)


def size_adjusted(sites: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """What the error of the sites with on-site generation would be without the plant.

    The meter of such a site records import: demand minus generation. Its total demand is
    ``import / (1 - share)`` with the plant's reported share of the site's electricity. Sites
    without a plant give the error to expect at a given size: a line through log error against
    log mean demand. The expected error of a CHP site is that rate at its total demand, in kWh;
    the rest of its measured error is what the plant adds. Sites with no usable share count at
    the size of their import. The largest site without a plant has 1.5 MW, and the three largest
    CHP sites have 2.8 to 3.5 MW of total demand, so the rate is extrapolated."""
    plain = sites[sites["group"] != "on-site CHP"]
    slope, intercept = np.polyfit(np.log(plain["mean_kw"]), np.log(plain["nmae_pct"]), 1)
    chp = sites[sites["group"] == "on-site CHP"].copy()
    share = chp["chp_share"].fillna(0.0)
    chp["total_kw"] = chp["mean_kw"] / (1.0 - share)
    chp["total_demand_mwh"] = chp["demand_mwh"] / (1.0 - share)
    chp["expected_nmae_pct"] = np.exp(intercept + slope * np.log(chp["total_kw"]))
    chp["expected_error_mwh"] = chp["expected_nmae_pct"] / 100.0 * chp["total_demand_mwh"]
    chp["error_of_total_pct"] = 100 * chp["error_mwh"] / chp["total_demand_mwh"]
    chp["plant_added_mwh"] = chp["error_mwh"] - chp["expected_error_mwh"]
    total = float(sites["error_mwh"].sum())
    summary = {
        "sites_without_plant": int(len(plain)),
        "error_falls_with_size_slope": round(float(slope), 3),
        "chp_sites": int(len(chp)),
        "chp_sites_with_reported_share": int(chp["chp_share"].notna().sum()),
        "chp_error_share_pct": round(float(100 * chp["error_mwh"].sum() / total), 1),
        "chp_share_of_import_pct": round(
            float(100 * chp["demand_mwh"].sum() / sites["demand_mwh"].sum()), 1
        ),
        "chp_share_of_total_demand_pct": round(
            float(
                100
                * chp["total_demand_mwh"].sum()
                / (chp["total_demand_mwh"].sum() + plain["demand_mwh"].sum())
            ),
            1,
        ),
        "chp_error_of_import_pct": round(
            float(100 * chp["error_mwh"].sum() / chp["demand_mwh"].sum()), 1
        ),
        "chp_error_of_total_demand_pct": round(
            float(100 * chp["error_mwh"].sum() / chp["total_demand_mwh"].sum()), 1
        ),
        "expected_error_of_total_demand_pct": round(
            float(100 * chp["expected_error_mwh"].sum() / chp["total_demand_mwh"].sum()), 1
        ),
        "plant_added_share_of_all_error_pct": round(
            float(100 * chp["plant_added_mwh"].sum() / total), 1
        ),
        "plant_added_share_of_chp_error_pct": round(
            float(100 * chp["plant_added_mwh"].sum() / chp["error_mwh"].sum()), 1
        ),
        "top_three_sites_share_of_all_error_pct": round(
            float(sites["error_share_pct"].nlargest(3).sum()), 1
        ),
    }
    return chp, summary


def step_table(sd: pd.DataFrame, model: str = BEST) -> pd.DataFrame:
    """Error by the size of the day's largest unexplained step, per site group; the shares are
    of the whole panel's error, demand and site-days."""
    x = sd[(sd["model"] == model) & sd["step_bin"].notna()]
    out = (
        x.groupby(["group", "step_bin"], observed=True)
        .agg(
            site_days=("day", "size"),
            error_mwh=("ae", lambda v: v.sum() / 1000.0),
            demand_mwh=("a_kwh", lambda v: v.sum() / 1000.0),
        )
        .reset_index()
    )
    out["nmae_kwh_weighted_pct"] = 100 * out["error_mwh"] / out["demand_mwh"]
    out["error_share_pct"] = 100 * out["error_mwh"] / out["error_mwh"].sum()
    out["demand_share_pct"] = 100 * out["demand_mwh"] / out["demand_mwh"].sum()
    out["day_share_pct"] = 100 * out["site_days"] / out["site_days"].sum()
    out["day_share_of_group_pct"] = (
        100 * out["site_days"] / out.groupby("group")["site_days"].transform("sum")
    )
    order = {k: i for i, k in enumerate(STEP_LABELS)}
    return out.sort_values(["group", "step_bin"], key=lambda c: c.map(order).fillna(c)).reset_index(
        drop=True
    )


def example_days(f: pd.DataFrame, actual: pd.DataFrame, days: pd.DataFrame, n_sites: int = 3):
    """Observed demand and the forecast with its 80 % band, every forecast day of the sites
    that hold the most error (kW)."""
    sd = site_day_scores(f, actual)
    top = sd.groupby("site_code")["ae"].sum().nlargest(n_sites).index
    m = f[f["site_code"].isin(top)].merge(actual, on=["target", "site_code"], how="inner")
    m = m.merge(days[["site_code", "day", "step_rel", "big_step"]], on=["site_code", "day"])
    out = m[["site_code", "day", "target", "y", "point", "q10", "q90", "step_rel", "big_step"]]
    out = out.assign(**{c: out[c] * 2.0 for c in ("y", "point", "q10", "q90")})
    out = out.merge(
        sd[["site_code", "day", "ae", "a_kwh"]].assign(
            nmae_pct=lambda d: 100 * d["ae"] / d["a_kwh"]
        )[["site_code", "day", "nmae_pct"]],
        on=["site_code", "day"],
    )
    return out.sort_values(["site_code", "target"]).reset_index(drop=True)


def _slots(m: pd.DataFrame, col: str, key: pd.DataFrame) -> np.ndarray:
    a = np.full((len(key), 48), np.nan)
    a[m["row"].to_numpy(), m["slot"].to_numpy()] = m[col].to_numpy() * 2.0  # kW
    return a


def issue_value(actual: pd.DataFrame, mean_kw: pd.Series, kinds: pd.DataFrame) -> pd.DataFrame:
    """The peak cut of a battery whose threshold is planned the day before and planned again on
    the day, at 00:00 or at 06:00, with the forecast issued then for the rest of the day.

    Until the new issue the battery defends the day-ahead threshold on the actual demand. The
    new threshold is the lowest that the battery can hold on the new forecast from its state of
    charge at that time, and never lower than the peak that the day has had already."""
    late = pd.read_parquet(OUT_DIR / "forecasts_intraday.parquet")
    ahead = load_stored("chronos2")
    a = actual.sort_values(["site_code", "day", "target"]).copy()
    a["slot"] = a.groupby(["site_code", "day"]).cumcount()
    a = a[a.groupby(["site_code", "day"])["slot"].transform("size") == 48]
    days = ahead[["site_code", "target"]].merge(a, on=["site_code", "target"])[["site_code", "day"]]
    key = days.drop_duplicates().reset_index(drop=True)
    key["row"] = np.arange(len(key))
    a = a.merge(key, on=["site_code", "day"])
    act = _slots(a, "y", key)
    grid = a[["site_code", "target", "row", "slot"]]
    da = _slots(ahead.merge(grid, on=["site_code", "target"]), "point", key)
    p_kw = fx.BATTERY_SHARE * key["site_code"].map(mean_kw).to_numpy()
    plans = {"16:30 the day before": (0, da)}
    for model, label in (
        ("chronos2_issue00", "00:00 on the day"),
        ("chronos2_issue06", "06:00 on the day"),
    ):
        m = late[late["model"] == model].merge(grid, on=["site_code", "target"])
        fc = _slots(m, "point", key)
        first = int(m["slot"].min())
        plans[label] = (first, fc)
    ok = np.isfinite(act).all(axis=1) & np.isfinite(da).all(axis=1) & (p_kw > 0)
    for first, fc in plans.values():
        ok &= np.isfinite(fc[:, first:]).all(axis=1)
    key, act, da, p_kw = key[ok].reset_index(drop=True), act[ok], da[ok], p_kw[ok]
    e_kwh = p_kw * fx.DURATION_H
    raw = act.max(axis=1)
    out = key[["site_code", "day"]].assign(perfect_kw=raw - fx.plan_threshold(act, p_kw, e_kwh))
    theta1 = fx.plan_threshold(da, p_kw, e_kwh)
    for label, (first, fc) in plans.items():
        if first == 0:
            got = fx.realised_peak(theta1, act, p_kw, e_kwh)
        else:
            fc = fc[ok]
            peak1, soc = fx.realised_peak(theta1, act[:, :first], p_kw, e_kwh, return_soc=True)
            theta2 = np.maximum(peak1, fx.plan_threshold(fc[:, first:], p_kw, e_kwh, soc0=soc))
            got = np.maximum(peak1, fx.realised_peak(theta2, act[:, first:], p_kw, e_kwh, soc0=soc))
        out[label] = raw - got
    out = out.merge(kinds[["site_code", "day", "group"]], on=["site_code", "day"], how="left")
    rows = []
    for group, g in [("all sites", out), *out.groupby("group")]:
        for label in plans:
            rows.append(
                {
                    "group": group,
                    "issue": label,
                    "site_days": len(g),
                    "captured_pct": 100 * g[label].sum() / g["perfect_kw"].sum(),
                }
            )
    return pd.DataFrame(rows)


def issue_time_table(actual: pd.DataFrame, kinds: pd.DataFrame) -> pd.DataFrame:
    """Error of Chronos-2 by issue time (16:30 the day before, 00:00 and 06:00 on the day), on
    the periods that all three forecast: the target day from 06:30."""
    late = pd.read_parquet(OUT_DIR / "forecasts_intraday.parquet")
    common = late.loc[late["model"] == "chronos2_issue06", ["site_code", "target"]]
    f = pd.concat([late, load_stored("chronos2").assign(model="chronos2_issue_day_before")])
    f = f.merge(common, on=["site_code", "target"], how="inner")
    sd = site_day_scores(f, actual, 30).merge(kinds, on=["site_code", "day"], how="left")
    out = pd.concat(
        [error_table(sd.assign(group="all sites"), ["group"]), error_table(sd, ["group"])],
        ignore_index=True,
    )
    out["issue"] = out["model"].map(
        {
            "chronos2_issue_day_before": "16:30 the day before",
            "chronos2_issue00": "00:00 on the day",
            "chronos2_issue06": "06:00 on the day",
        }
    )
    return out


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, panel = poc_panel()
    actual = actual_long(panel)
    days = day_table(panel, test_start=str(config["test_start"]))
    mean_kw = (panel.y_raw[panel.sites] * 2.0).mean()
    kinds = days[
        ["site_code", "day", "kind", "group", "log_shift", "switch", "big_step", "step_bin", "wave"]
    ]
    days.to_csv(OUT_DIR / "day_table.csv", index=False)
    summary: dict = {"sites": len(panel.sites)}

    # stored forecasts: 69 origins, every model
    f = pd.concat([load_stored(n) for n in STORED], ignore_index=True)
    sd = site_day_scores(f, actual).merge(kinds, on=["site_code", "day"], how="left")
    by_group = error_table(sd, ["group"])
    by_kind = error_table(sd, ["group", "kind"])
    overall = error_table(sd.assign(all="all sites"), ["all"])
    by_group.to_csv(OUT_DIR / "error_by_group.csv", index=False)
    by_kind.to_csv(OUT_DIR / "error_by_group_kind.csv", index=False)
    overall.to_csv(OUT_DIR / "error_overall.csv", index=False)
    by_switch = error_table(sd.dropna(subset=["switch"]), ["group", "switch"])
    by_switch.to_csv(OUT_DIR / "error_by_group_switch.csv", index=False)
    dispatch = dispatch_table(sd)
    dispatch.to_csv(OUT_DIR / "chp_dispatch_gain.csv", index=False)
    by_step = step_table(sd)
    by_step.to_csv(OUT_DIR / "error_by_group_step.csv", index=False)
    by_wave = pd.concat(
        [
            error_table(sd.assign(group="all sites"), ["group", "wave"]),
            error_table(sd, ["group", "wave"]),
        ],
        ignore_index=True,
    )
    by_wave.to_csv(OUT_DIR / "error_by_group_wave.csv", index=False)
    sites = site_errors(sd)
    sites.to_csv(OUT_DIR / "error_by_site.csv", index=False)
    chp_sites, plant = size_adjusted(sites)
    chp_sites.to_csv(OUT_DIR / "chp_size_adjusted.csv", index=False)
    example_days(load_stored(BEST), actual, days).to_csv(OUT_DIR / "example_days.csv", index=False)
    v = value_site_days(f, actual, mean_kw).merge(kinds, on=["site_code", "day"], how="left")
    value_group = value_table(v, ["group"])
    value_all = value_table(v.assign(all="all sites"), ["all"])
    value_group.to_csv(OUT_DIR / "value_by_group.csv", index=False)
    value_all.to_csv(OUT_DIR / "value_overall.csv", index=False)
    # wave days against the same site and month, with and without weather inputs
    stored_wave = pd.concat(
        [
            wave_matched_value(v[v["model"] == m], w)
            for m in (BEST, "chronos2", "lightgbm_ifs")
            for w in WAVES[:2]
        ],
        ignore_index=True,
    )
    stored_wave.to_csv(OUT_DIR / "value_wave_matched_stored.csv", index=False)
    sw_all = stored_wave[stored_wave["group"] == "all sites"].set_index(
        ["model", "matched_to", "wave"]
    )
    summary["heat_wave_value_stored_forecasts"] = {
        m: {
            k: round(float(sw_all.loc[(m, WAVES[1], k), "captured_point_pct"]), 1)
            for k in (WAVES[1], WAVES[2])
        }
        for m in (BEST, "chronos2", "lightgbm_ifs")
    }
    # compact tables for the write-up
    keep = ["nmae_pct", "nmae_level_oracle_pct", "level_share_pct"]
    cut = ["captured_point_pct", "captured_q70_pct", "captured_level_pct"]
    overall.merge(value_all, on=["model", "all"])[["model", *keep, *cut]].sort_values(
        "nmae_pct"
    ).to_csv(OUT_DIR / "table_models.csv", index=False)
    by_group[by_group["model"] == BEST].merge(
        value_group[value_group["model"] == BEST], on=["model", "group"]
    )[["group", "sites", *keep, "error_share_pct", "demand_share_pct", *cut]].to_csv(
        OUT_DIR / "table_groups.csv", index=False
    )

    g = by_group[by_group["model"] == BEST].set_index("group")
    o = overall[overall["model"] == BEST].iloc[0]
    chp, inp = g.loc["on-site CHP"], g.loc["inpatient"]
    like_inpatient = chp["demand_mwh"] * inp["error_mwh"] / inp["demand_mwh"]
    summary["stored_best_model"] = {
        "model": BEST,
        "nmae_pct": round(float(o["nmae_pct"]), 2),
        "nmae_level_oracle_pct": round(float(o["nmae_level_oracle_pct"]), 2),
        "level_share_of_error_pct": round(float(o["level_share_pct"]), 1),
        "by_group": g[
            [
                "sites",
                "nmae_pct",
                "nmae_level_oracle_pct",
                "level_share_pct",
                "error_share_pct",
                "demand_share_pct",
            ]
        ]
        .round(1)
        .to_dict(orient="index"),
        "chp_error_share_pct": round(float(chp["error_share_pct"]), 1),
        "error_cut_if_chp_like_inpatient_pct": round(
            float(100 * (chp["error_mwh"] - like_inpatient) / g["error_mwh"].sum()), 1
        ),
        "value": value_all[value_all["model"] == BEST].round(1).to_dict(orient="records")[0],
        "value_by_group": value_group[value_group["model"] == BEST]
        .set_index("group")
        .round(1)
        .drop(columns="model")
        .to_dict(orient="index"),
    }
    sw = by_switch[by_switch["model"] == BEST].set_index(["group", "switch"])
    summary["plant_switch_days"] = {
        g_: {
            "share_of_days_pct": round(
                float(sw.loc[(g_, True), "site_days"] / sw.loc[g_, "site_days"].sum() * 100), 1
            ),
            "share_of_group_error_pct": round(
                float(sw.loc[(g_, True), "error_mwh"] / sw.loc[g_, "error_mwh"].sum() * 100), 1
            ),
            "nmae_switch_pct": round(float(sw.loc[(g_, True), "nmae_pct"]), 1),
            "nmae_other_pct": round(float(sw.loc[(g_, False), "nmae_pct"]), 1),
        }
        for g_ in ("on-site CHP", "inpatient", "outpatient")
    }
    summary["chp_dispatch_gain"] = dispatch.round(2).to_dict(orient="records")
    summary["plant"] = plant
    big = by_step[by_step["step_bin"] == STEP_LABELS[-1]].set_index("group")
    summary["steps_larger_than_mean_demand"] = (
        big[["site_days", "day_share_of_group_pct", "error_share_pct", "nmae_kwh_weighted_pct"]]
        .round(1)
        .to_dict(orient="index")
    )
    wv = by_wave[by_wave["model"] == BEST].set_index(["group", "wave"])
    summary["heat_and_cold_waves_stored"] = {
        g_: wv.loc[g_][["site_days", "nmae_pct", "nmae_level_oracle_pct", "error_share_pct"]]
        .round(1)
        .to_dict(orient="index")
        for g_ in ("all sites", "on-site CHP", "inpatient", "outpatient")
    }
    compact = overall[overall["model"].isin(["tnp", "pttnp", "gtnp", "tnp_ifs", "chronos2"])]
    summary["compact_models"] = (
        compact.merge(value_all, on=["model", "all"])
        .set_index("model")[["nmae_pct", "captured_point_pct", "captured_q70_pct"]]
        .round(1)
        .to_dict(orient="index")
    )

    # every day of the test window
    daily_path = OUT_DIR / "forecasts_daily.parquet"
    if daily_path.exists():
        fd = pd.read_parquet(daily_path)
        sdd = site_day_scores(fd, actual).merge(kinds, on=["site_code", "day"], how="left")
        sdd.to_parquet(OUT_DIR / "daily_site_days.parquet", index=False)
        dk = error_table(sdd, ["group", "kind"])
        dk_all = error_table(sdd, ["kind"])
        dg = error_table(sdd, ["group"])
        dk.to_csv(OUT_DIR / "daily_error_by_group_kind.csv", index=False)
        dk_all.to_csv(OUT_DIR / "daily_error_by_kind.csv", index=False)
        dg.to_csv(OUT_DIR / "daily_error_by_group.csv", index=False)
        bias = (
            sdd.groupby(["group", "kind"], observed=True)["bias_pct"]
            .median()
            .unstack("kind")
            .round(2)
        )
        bias.to_csv(OUT_DIR / "daily_bias_by_group_kind.csv")
        vd = value_site_days(fd, actual, mean_kw).merge(kinds, on=["site_code", "day"], how="left")
        vk = value_table(vd, ["group", "kind"])
        vk.to_csv(OUT_DIR / "daily_value_by_group_kind.csv", index=False)
        value_table(vd, ["kind"]).to_csv(OUT_DIR / "daily_value_by_kind.csv", index=False)
        dw = pd.concat(
            [
                error_table(sdd.assign(group="all sites"), ["group", "wave"]),
                error_table(sdd, ["group", "wave"]),
            ],
            ignore_index=True,
        )
        dw.to_csv(OUT_DIR / "daily_error_by_group_wave.csv", index=False)
        vw = pd.concat(
            [
                value_table(vd.assign(group="all sites"), ["group", "wave"]),
                value_table(vd, ["group", "wave"]),
            ],
            ignore_index=True,
        )
        vw.to_csv(OUT_DIR / "daily_value_by_group_wave.csv", index=False)
        step_table(sdd, "chronos2_daily").to_csv(
            OUT_DIR / "daily_error_by_group_step.csv", index=False
        )
        wm = pd.concat([wave_matched(sdd, w) for w in WAVES[:2]], ignore_index=True)
        wm.to_csv(OUT_DIR / "daily_error_wave_matched.csv", index=False)
        wmi = wm[wm["group"] == "all sites"].set_index(["matched_to", "wave"])
        wv = pd.concat([wave_matched_value(vd, w) for w in WAVES[:2]], ignore_index=True)
        wv.to_csv(OUT_DIR / "daily_value_wave_matched.csv", index=False)
        wvi = wv[wv["group"] == "all sites"].set_index(["matched_to", "wave"])
        summary["wave_excess_share_of_all_error_pct"] = {
            w: round(wave_excess(sdd, w), 2) for w in WAVES[:2]
        }
        summary["waves_against_same_site_and_month"] = {
            w: {
                k: {
                    **wmi.loc[(w, k)][
                        ["site_days", "nmae_pct", "nmae_level_oracle_pct", "median_bias_pct"]
                    ]
                    .astype(float)
                    .round(1)
                    .to_dict(),
                    **wvi.loc[(w, k)][
                        ["captured_point_pct", "captured_q70_pct", "captured_level_pct"]
                    ]
                    .astype(float)
                    .round(1)
                    .to_dict(),
                }
                for k in (w, WAVES[2])
            }
            for w in WAVES[:2]
            if w in wmi.index.get_level_values(0)
        }
        dwi = dw.set_index(["group", "wave"])
        summary["heat_and_cold_waves_every_day"] = {
            g_: dwi.loc[g_][
                [
                    "site_days",
                    "day_share_pct",
                    "nmae_pct",
                    "nmae_level_oracle_pct",
                    "error_share_pct",
                ]
            ]
            .round(1)
            .to_dict(orient="index")
            for g_ in ("all sites", "on-site CHP", "inpatient", "outpatient")
        }
        summary["daily"] = {
            "site_days": int(len(sdd)),
            "days": int(sdd["day"].nunique()),
            "by_kind": dk_all.set_index("kind")[
                [
                    "site_days",
                    "day_share_pct",
                    "nmae_pct",
                    "nmae_level_oracle_pct",
                    "error_share_pct",
                ]
            ]
            .round(1)
            .to_dict(orient="index"),
            "by_group": dg.set_index("group")[
                ["nmae_pct", "nmae_level_oracle_pct", "level_share_pct", "error_share_pct"]
            ]
            .round(1)
            .to_dict(orient="index"),
        }

    # history
    ctx_path = OUT_DIR / "forecasts_context.parquet"
    if ctx_path.exists():
        fc = pd.concat(
            [pd.read_parquet(ctx_path), load_stored("chronos2").assign(model="chronos2_ctx2016")],
            ignore_index=True,
        )
        sdc = site_day_scores(fc, actual).merge(kinds, on=["site_code", "day"], how="left")
        ct = pd.concat(
            [error_table(sdc.assign(group="all sites"), ["group"]), error_table(sdc, ["group"])],
            ignore_index=True,
        )
        ct["context_periods"] = ct["model"].str.extract(r"ctx(\d+)").astype(int)
        vc = value_site_days(fc, actual, mean_kw)
        cv = value_table(vc.assign(group="all sites"), ["group"])
        ct = ct.merge(
            cv[["model", "group", "captured_point_pct"]], on=["model", "group"], how="left"
        )
        ct.sort_values(["group", "context_periods"]).to_csv(OUT_DIR / "context.csv", index=False)
        ct[ct["group"] == "all sites"].assign(days=lambda d: d["context_periods"] / 48.0)[
            ["days", "nmae_pct", "nmae_level_oracle_pct", "captured_point_pct"]
        ].sort_values("days").to_csv(OUT_DIR / "table_context.csv", index=False)
        a = ct[ct["group"] == "all sites"].set_index("context_periods")
        summary["context"] = {
            int(k): {
                "nmae_pct": round(float(r["nmae_pct"]), 2),
                "captured_pct": round(float(r["captured_point_pct"]), 1),
            }
            for k, r in a.sort_index().iterrows()
        }

    # issue time
    if (OUT_DIR / "forecasts_intraday.parquet").exists():
        it = issue_time_table(actual, kinds)
        it.to_csv(OUT_DIR / "issue_time.csv", index=False)
        summary["issue_time"] = {
            g_: it[it["group"] == g_].set_index("issue")["nmae_pct"].round(2).to_dict()
            for g_ in ("all sites", "on-site CHP", "inpatient", "outpatient")
        }
        iv = issue_value(actual, mean_kw, kinds)
        iv.to_csv(OUT_DIR / "issue_value.csv", index=False)
        it[["group", "issue", "sites", "site_days", "nmae_pct", "nmae_level_oracle_pct"]].merge(
            iv[["group", "issue", "captured_pct"]], on=["group", "issue"], how="left"
        ).to_csv(OUT_DIR / "table_issue.csv", index=False)
        summary["issue_value_captured_pct"] = {
            g_: iv[iv["group"] == g_].set_index("issue")["captured_pct"].round(1).to_dict()
            for g_ in ("all sites", "on-site CHP", "inpatient", "outpatient")
        }

    # strike days
    strikes = strike_panel()
    se = strike_effect(strikes)
    se.to_csv(OUT_DIR / "strike_effect.csv", index=False)
    sp = strike_profile(strikes)
    sp.to_csv(OUT_DIR / "strike_profile.csv", index=False)
    spc = sp[sp["term"] == "strike weekday, away from holidays"]
    summary["strike_profile"] = {
        "largest_effect_pct": round(float(spc["effect_pct"].abs().max()), 2),
        "hours_significant_at_5_pct": int((spc["p"] < 0.05).sum()),
        "working_hours_8_to_18_mean_pct": round(
            float(spc[spc["hour"].between(8, 17)]["effect_pct"].mean()), 2
        ),
    }
    s_all = se[se["site_type"] == "all sites"]
    clean = s_all[s_all["term"] == "strike weekday, away from holidays"].set_index("outcome")
    summary["strike"] = {
        "strike_days_in_record": int(len(strike_days())),
        "away_from_holidays_by_outcome": clean[
            ["days", "site_days", "effect_pct", "lo_pct", "hi_pct", "p"]
        ]
        .round(3)
        .to_dict(orient="index"),
        "whole_day_all_terms": s_all[s_all["outcome"] == "whole day"]
        .set_index("term")[["days", "effect_pct", "lo_pct", "hi_pct", "p"]]
        .round(3)
        .to_dict(orient="index"),
        "by_site_type_working_hours_against_night": se[
            (se["site_type"] != "all sites") & (se["outcome"] == "08:00 to 18:00 against night")
        ]
        .set_index("site_type")[["sites", "effect_pct", "lo_pct", "hi_pct", "p"]]
        .round(3)
        .to_dict(orient="index"),
    }

    # footfall
    md = mobility_daily()
    reg = mobility_regression(md)
    reg.to_csv(OUT_DIR / "mobility_regression.csv", index=False)
    mobility_partial(md).to_csv(OUT_DIR / "mobility_partial.csv", index=False)
    own = reg[reg["model"] == "own footfall"].set_index("site_code")
    summary["mobility_regression"] = (
        own[["elasticity", "lo", "hi", "p", "partial_r2", "raw_corr"]]
        .round(3)
        .to_dict(orient="index")
    )
    if any(OUT_DIR.glob("forecasts_plant_*.parquet")):
        ps, pdiff = plant_scores(panel, config)
        ps.to_csv(OUT_DIR / "plant_scores.csv", index=False)
        pdiff.to_csv(OUT_DIR / "plant_differences.csv", index=False)
        summary["plant_knowledge"] = {
            "nmae_pct": ps.pivot(index="model", columns="sites", values="nmae_pct")
            .round(2)
            .to_dict(),
            "differences": pdiff.round(3).to_dict(orient="records"),
        }
    if mobility_forecasts_path().exists():
        ms, mdiff, msd = mobility_scores(panel)
        ms.to_csv(OUT_DIR / "mobility_scores.csv", index=False)
        ms[ms["sites"].isin(["two sites without CHP", "all three sites"])].to_csv(
            OUT_DIR / "table_mobility.csv", index=False
        )
        mdiff.to_csv(OUT_DIR / "mobility_differences.csv", index=False)
        msd.to_parquet(OUT_DIR / "mobility_site_days.parquet", index=False)
        top = ms[(ms["sites"] == "two sites without CHP") & (ms["kind"] == "all days")]
        summary["mobility_forecast"] = {
            "variants": sorted(ms["model"].unique()),
            "site_days_two_sites": int(top["site_days"].max()),
            "two_sites_without_chp_nmae_pct": top.set_index("model")["nmae_pct"].round(3).to_dict(),
            "differences": mdiff[
                (mdiff["sites"] == "two sites without CHP") & (mdiff["metric"] == "nMAE")
            ]
            .round(3)
            .to_dict(orient="records"),
        }

    text = json.dumps(summary, indent=2, default=float).replace("NaN", "null")
    (OUT_DIR / "summary.json").write_text(text + "\n")
    logger.info("forecast anatomy: %s", json.dumps(summary["stored_best_model"]["by_group"]))
    return {"summary": summary}


# --------------------------------------------------------------------------- #
# Knowledge of the plant: a flag, and the state
# --------------------------------------------------------------------------- #

PLANT_PAIRS = (
    ("plant_cross_flag", "plant_cross"),
    ("plant_cross", "chronos2"),
    ("plant_state_past", "chronos2"),
    ("plant_state_oracle", "chronos2"),
    ("lightgbm_ifs", "lightgbm_ifs_noid"),
    ("lightgbm_ifs_eric", "lightgbm_ifs"),
)


def plant_forecasts() -> pd.DataFrame:
    """The plant probe's forecasts with the stored runs they compare with."""
    files = sorted(OUT_DIR.glob("forecasts_plant_*.parquet"))
    files = [p for p in files if ".partial" not in p.name]
    frames = [pd.read_parquet(p) for p in files]
    frames.append(load_stored("chronos2"))
    for name in ("lightgbm_ifs", "lightgbm_ifs_noid", "lightgbm_ifs_eric"):
        frames.append(pd.read_parquet(POC / f"forecasts_{name}.parquet").assign(model=name))
    return pd.concat(frames, ignore_index=True)


def plant_scores(panel: poc.Panel, config: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Does knowledge of the plant help? Error and captured peak cut of each variant at the
    sites with on-site generation and at the others (and on the days with a step larger than the
    site's mean import), and the paired differences with their bootstrap intervals.

    * a trained model with and without the site's identity, and with the estates returns'
      descriptors (number of CHP units among them);
    * Chronos-2 that sees all sites of an origin together, with and without a flag and the
      plant's share of the site's electricity as covariates;
    * Chronos-2 with the plant state up to the issue time, and with the state of the target day
      (made from the import itself: an upper bound)."""
    f = plant_forecasts()
    actual = actual_long(panel)
    days = day_table(panel, test_start=str(config["test_start"]))
    chp = set(pd.read_csv(OUT_DIR.parent / "chp_dispatch" / "classification.csv")["site_code"])
    sd = quantile_scores(f, actual)
    sd = sd[sd["n"] >= MIN_PERIODS].merge(
        days[["site_code", "day", "big_step"]], on=["site_code", "day"], how="left"
    )
    sd["chp"] = sd["site_code"].isin(chp)
    mean_kw = (panel.y_raw[panel.sites] * 2.0).mean()
    v = value_site_days(f, actual, mean_kw)
    v["chp"] = v["site_code"].isin(chp)
    sets = {
        "all sites": lambda d: d,
        "sites with on-site generation": lambda d: d[d["chp"]],
        "other sites": lambda d: d[~d["chp"]],
        "on-site generation, large-step days": lambda d: d[d["chp"] & (d["big_step"] == True)],  # noqa: E712
    }
    rows, diffs = [], []
    for name, pick in sets.items():
        x = pick(sd)
        vx = pick(v.merge(days[["site_code", "day", "big_step"]], on=["site_code", "day"]))
        val = value_table(vx, []).set_index("model")
        for model, g in x.groupby("model"):
            rows.append(
                {
                    "sites": name,
                    "model": model,
                    "site_days": len(g),
                    "nmae_pct": _nmae(g),
                    "crps_pct": _nmae(g, "crps"),
                    "captured_point_pct": float(val.loc[model, "captured_point_pct"]),
                }
            )
        for a, b in PLANT_PAIRS:
            if {a, b} <= set(x["model"].unique()):
                diffs.append({"sites": name, "a": a, "b": b, **paired_bootstrap(x, a, b)})
    return pd.DataFrame(rows), pd.DataFrame(diffs)
