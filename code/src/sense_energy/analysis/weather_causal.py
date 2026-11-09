"""Quasi-experimental estimates of how extreme temperatures change NHS site electricity demand.

Identification rests on weather being assigned by nature: hospitals do not choose their sites'
weather, and conditional on the site and the calendar month, which days are hot or cold is as
good as random. What is *not* random is which sites are exposed to how much heat (southern and
urban sites run warmer, and those are more often acute hospitals); the designs below absorb that
with site-level fixed effects and differ in which time-varying confounders they rule out.

1. **Spatial difference-in-differences with a continuous dose** (two-way fixed effects)::

       log y[s,d] = b_c HDD14[s,d] + b_h CDD20[s,d] + a[s, year-month] + c[s, weekday] + g[d] + e

   HDD14 = max(14 - T, 0) and CDD20 = max(T - 20, 0) on daily mean temperature, the two arms of
   the hockey stick in ``weather_response``. The date effect g[d] absorbs everything common to all
   sites on a day (bank holidays, industrial action, school terms, national news, the grid), so
   the effect comes from how much warmer or colder one site was than the others on the same day,
   relative to its own month. Nested specifications: A (site x month, site x weekday, calendar
   controls), B (+ date), C (+ region x date: only sites in the same NHS region on the same day
   are compared), D (B plus the day's sunshine, wind and rain at the site, which co-move with
   temperature: sunshine lowers lighting demand on hot days). Standard errors are two-way
   clustered by site and date.
2. **Stacked event-study difference-in-differences** for heatwaves and cold spells. For every
   national event (``weather_events``): treated sites are inside a site-level event of that type;
   clean controls have no such event within +/- WINDOW days. Event time k counts days from the
   national onset; each stack has its own site and date effects, plus site x weekday effects; the
   reference period is k <= -5. Pre-event coefficients test parallel trends. The same regression
   with temperature as the outcome gives the dose, and demand ATT / temperature ATT the effect
   per degree of excess temperature. Stacks whose pre-period contains another national event of
   the same type are dropped. Stacking with clean controls avoids the bias of staggered two-way
   fixed effects when effects differ between events.
3. **Falsification**: (a) leads - tomorrow's and the day after's weather, given today's and the
   past week's, cannot cause today's demand (also with the same evening's temperature as a
   control, since a daily mean blurs the evening hours that demand responds to); (b)
   randomisation inference - whole daily weather
   series are reassigned between sites at random and design 1B is re-estimated N_PERM times; the
   observed effects should lie far outside the permutation distribution.
4. **Distributed lags** (0 - 7 days) in design 1B: the cumulative effect of a hot or cold day and
   any rebound in the following week.
5. **Effect modification** in design 1B: building age (ERIC mean construction year), site size and
   local climate (summer mean daily maximum). Descriptive: the moderators are not randomised.

6. **Extreme days**: indicators for a daily maximum of 28-30 and 30+ degC on top of the linear
   doses in A-D - the extra demand on such a day beyond what its mean implies.
7. **Space**: (a) the correlation between sites of day-to-day temperature anomalies and of
   specification-B residuals, by distance - how much of the panel is one weather system, and
   whether regional shocks survive the date effects; (b) a neighbour placebo - the mean dose of
   the sites 30-150 km away, given the site's own: it should not move the site's demand; (c)
   specification-B standard errors clustered by site and date (main), by region x week and by
   ICB, against cross-day regional correlation of the errors.

Outputs (``code/reports/weather_causal/``): ``twfe_specs.csv``, ``twfe_extreme.csv``,
``spatial_correlation.csv``, ``spatial_lag.csv``, ``inference.csv``, ``identifying_variation.csv``
(the dose's standard deviation left after each specification's fixed effects), ``stack_log.csv``
(every national event and why it was kept or dropped), ``event_study.csv``, ``event_att.csv``,
``event_per_degree.csv``, ``leads_lags.csv``, ``randomisation.csv``, ``heterogeneity.csv``,
``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import GEO_DIR, INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "weather_causal"
EVENTS_DIR = PROJECT_ROOT / "code" / "reports" / "weather_events"
COLD_BASE, HOT_BASE = 14.0, 20.0
DOSE = ("hdd14", "cdd20")
CALENDAR = ["holiday", "christmas", "school_summer"]
OTHER_WEATHER = ["solar_kwh_m2", "wind_mean", "precip_mm"]
SPECS = {
    "A": {
        "label": "site x month, site x weekday, calendar",
        "fe": ["fe_month", "fe_dow"],
        "controls": CALENDAR,
    },
    "B": {"label": "+ date", "fe": ["fe_month", "fe_dow", "date_id"], "controls": []},
    "C": {"label": "+ region x date", "fe": ["fe_month", "fe_dow", "region_date"], "controls": []},
    "D": {
        "label": "B + sunshine, wind, rain",
        "fe": ["fe_month", "fe_dow", "date_id"],
        "controls": OTHER_WEATHER,
    },
}
CLUSTER = "site_code+date_id"
WINDOW, REF_MAX, LEAD_MIN = 10, -5, -4
MIN_WINDOW_COVERAGE = 0.8
N_LAGS, N_LEADS = 7, 2
N_PERM, SEED = 200, 20261105
GROUPS = {
    "no CHP": ("inpatient", "outpatient"),
    "inpatient": ("inpatient",),
    "outpatient": ("outpatient",),
    "on-site CHP": ("on-site CHP",),
}


# --------------------------------------------------------------------------- #
# Panel
# --------------------------------------------------------------------------- #


#: Site-specific coefficients (dose x site interactions) take a Driscoll-Kraay variance: two-way
#: clustering by site and date understates them, because a regressor that belongs to one site
#: has a single site cluster (its score sums to zero there) while the date and intersection
#: terms cancel. Driscoll-Kraay sums the scores by date and applies a Newey-West kernel over
#: dates, robust to serial and cross-sectional correlation; in the site-slope regression it
#: matches a per-site Newey-West variance and is 1.7-3.2 times the two-way standard error.
HAC_LAG_DAYS = 14


def site_vcov(
    data: pd.DataFrame,
    date_col: str = "date",
    site_col: str = "site_code",
    lag: int = HAC_LAG_DAYS,
    time_index: np.ndarray | None = None,
):
    """Add numeric panel and time ids to ``data`` (in place) and return pyfixest's ``vcov`` and
    ``vcov_kwargs`` for a Driscoll-Kraay variance with ``lag`` periods."""
    data["_pid"] = pd.factorize(data[site_col])[0]
    if time_index is None:
        d = pd.to_datetime(data[date_col])
        time_index = (d - d.min()).dt.days.to_numpy()
    data["_tid"] = np.asarray(time_index)
    return "DK", {"time_id": "_tid", "panel_id": "_pid", "lag": lag}


def dose(t: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    return np.maximum(COLD_BASE - t, 0.0), np.maximum(t - HOT_BASE, 0.0)


def weather_doses(table: pd.DataFrame | None = None) -> pd.DataFrame:
    """HDD14/CDD20 per site and day with leads and lags, from ERA5's daily table or from any
    thermometer table (site_code, date, t_mean, t_eve). Shifts run on a complete date grid, so
    a missing day leaves its neighbours' lags empty rather than skipping to the day before."""
    if table is None:
        table = pd.read_parquet(
            INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_mean", "t_eve"]
        )
    t = table.pivot(index="date", columns="site_code", values="t_mean")
    t = t.reindex(pd.date_range(t.index.min(), t.index.max(), freq="D"))
    eve = table.pivot(index="date", columns="site_code", values="t_eve").reindex(t.index)
    hdd, cdd = dose(t.to_numpy())
    wide = {
        "hdd14": pd.DataFrame(hdd, index=t.index, columns=t.columns),
        "cdd20": pd.DataFrame(cdd, index=t.index, columns=t.columns),
    }
    parts = {"t_mean": t, "t_eve": eve, **wide}
    for v in DOSE:
        for k in range(1, N_LAGS + 1):
            parts[f"{v}_lag{k}"] = wide[v].shift(k)
        for k in range(1, N_LEADS + 1):
            parts[f"{v}_lead{k}"] = wide[v].shift(-k)
    out = pd.concat({k: v.stack(future_stack=True) for k, v in parts.items()}, axis=1).rename_axis(
        ["date", "site_code"]
    )
    out = out.dropna(subset=["t_mean"]).reset_index()
    out.loc[out["t_mean"].isna(), ["hdd14", "cdd20"]] = np.nan
    return out.drop(columns="t_mean")


def panel(thermometer: pd.DataFrame | None = None) -> pd.DataFrame:
    """The daily site panel of ``weather_response`` with doses, fixed-effect ids and clusters.
    ``thermometer`` (site_code, date, t_mean, t_max, t_eve) replaces ERA5's temperatures: the
    panel is then restricted to the site-days it covers and every dose, band and lag is
    recomputed from it; sunshine, wind and rain stay with ERA5."""
    df = wr.panel()
    if thermometer is not None:
        keep = ["site_code", "date", "t_mean", "t_max", "t_eve"]
        df = df.drop(columns=["t_mean", "t_max"]).merge(
            thermometer[keep], on=["site_code", "date"], how="inner"
        )
        for _, (col, bin_col, edges, _, _) in wr.VARIABLES.items():
            df[bin_col] = pd.cut(df[col], edges, right=False).map(lambda iv: iv.left).astype(float)
    geo = pd.read_parquet(GEO_DIR / "sites_geo.parquet")[["site_code", "region_ons"]]
    rain = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "precip_mm"]
    )
    doses = weather_doses(thermometer)
    if thermometer is not None:
        doses = doses.drop(columns="t_eve")
    df = (
        df.merge(geo, on="site_code", how="left")
        .merge(rain, on=["site_code", "date"], how="left")
        .merge(doses, on=["site_code", "date"], how="left")
    )
    df["date_id"] = df["date"].dt.strftime("%Y-%m-%d")
    df["region_date"] = df["region_ons"].fillna("unknown") + "|" + df["date_id"]
    df["log_kw"] = np.log(df["kw"])
    df["hdd14_eve"], df["cdd20_eve"] = dose(df["t_eve"])
    df["hot28"], df["hot30"] = extreme_flags(df["t_max"])
    iso = df["date"].dt.isocalendar()
    week = iso["year"].astype(str) + "-" + iso["week"].astype(str)
    df["region_week"] = df["region_ons"].fillna("unknown") + "|" + week
    icb = pd.read_parquet(INTERIM_DIR / "sites.parquet")[["site_code", "integrated_care_board"]]
    df = df.merge(icb, on="site_code", how="left")
    df["icb"] = df["integrated_care_board"].fillna("unknown")
    return df


def extreme_flags(t_max: pd.Series | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Indicators for a daily maximum in [28, 30) and at or above 30 degC."""
    t = np.asarray(t_max, dtype=float)
    return ((t >= 28) & (t < 30)).astype(float), (t >= 30).astype(float)


def _fit(data: pd.DataFrame, y: str, x: list[str], fe: list[str], cluster: str = CLUSTER):
    import pyfixest as pf

    return pf.feols(f"{y} ~ {' + '.join(x)} | {' + '.join(fe)}", data=data, vcov={"CRV1": cluster})


def _tidy(fit, keep: list[str] | None = None) -> pd.DataFrame:
    t = fit.tidy()
    t = t.loc[[k for k in t.index if keep is None or k in keep]]
    return pd.DataFrame(
        {
            "term": t.index,
            "coef": t["Estimate"].to_numpy(),
            "se": t["Std. Error"].to_numpy(),
            "lo": t["2.5%"].to_numpy(),
            "hi": t["97.5%"].to_numpy(),
            "p": t["Pr(>|t|)"].to_numpy(),
        }
    )


def _pct(t: pd.DataFrame) -> pd.DataFrame:
    """Log points -> % change for coefficients on log demand."""
    for c in ("coef", "lo", "hi"):
        t[f"{c}_pct"] = 100 * np.expm1(t[c])
    return t


# --------------------------------------------------------------------------- #
# 1. Two-way fixed effects with a continuous dose
# --------------------------------------------------------------------------- #


def twfe_specs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g, members in GROUPS.items():
        sub = df[df["group"].isin(members)]
        if sub.empty:
            continue
        for spec, s in SPECS.items():
            x = list(DOSE) + s["controls"]
            t = _pct(_tidy(_fit(sub, "log_kw", x, s["fe"]), list(DOSE)))
            rows.append(
                t.assign(
                    group=g,
                    spec=spec,
                    spec_label=s["label"],
                    sites=sub["site_code"].nunique(),
                    n_obs=len(sub),
                )
            )
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Stacked event study
# --------------------------------------------------------------------------- #


def identifying_variation(df: pd.DataFrame) -> pd.DataFrame:
    """Standard deviation of each dose after the fixed effects of each specification are
    partialled out: the variation that identifies the effect. Spec B keeps only the day's
    cross-site temperature gradient, so a 0.3-0.5 degC ERA5 representation error attenuates
    it far more than spec A."""
    import pyfixest as pf

    rows = []
    for g, members in GROUPS.items():
        sub = df[df["group"].isin(members)]
        if sub.empty:
            continue
        for spec, s in SPECS.items():
            for v in DOSE:
                r = pf.feols(f"{v} ~ 1 | {' + '.join(s['fe'])}", data=sub).resid()
                rows.append(
                    {
                        "group": g,
                        "spec": spec,
                        "spec_label": s["label"],
                        "dose": v,
                        "raw_sd": float(sub[v].std()),
                        "within_sd": float(np.std(r)),
                    }
                )
    out = pd.DataFrame(rows)
    out["share_of_variance"] = (out["within_sd"] / out["raw_sd"]) ** 2
    return out


MIN_UNITS = 5


def stacks(
    df: pd.DataFrame, kind: str, nat: pd.DataFrame | None = None, se: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stacked windows around each national event of ``kind``, and a log of every event."""
    if nat is None:
        nat = pd.read_csv(EVENTS_DIR / "national_events.csv", parse_dates=["start", "end"])
        nat = nat[nat["in_electricity_record"]]
    nat = nat[nat["type"] == kind].sort_values("start")
    if se is None:
        se = pd.read_csv(EVENTS_DIR / "site_events.csv", parse_dates=["start", "end"])
    se = se[se["type"] == kind]
    starts = nat["start"].tolist()
    out, log = [], []
    for e in nat.itertuples():
        s, f = e.start, e.end
        lo, hi = s - pd.to_timedelta(WINDOW, "D"), s + pd.to_timedelta(WINDOW, "D")
        entry = {"event_id": e.event_id, "type": kind, "start": s, "treated": 0, "controls": 0}
        if any(lo <= o < s for o in starts):  # another same-type onset in the pre-period
            log.append({**entry, "status": "dropped: onset in pre-period"})
            continue
        treated = set(se.loc[(se["start"] <= f) & (se["end"] >= s), "site_code"])
        exposed = set(se.loc[(se["start"] <= hi) & (se["end"] >= lo), "site_code"])
        w = df[df["date"].between(lo, hi)]
        n_obs = w.groupby("site_code").size()
        ok = set(n_obs.index[n_obs >= MIN_WINDOW_COVERAGE * (2 * WINDOW + 1)])
        units = (treated & ok) | (ok - exposed)
        w = w[w["site_code"].isin(units)].copy()
        w["treated"] = w["site_code"].isin(treated).astype(float)
        n = w.groupby("treated")["site_code"].nunique()
        entry.update(treated=int(n.get(1.0, 0)), controls=int(n.get(0.0, 0)))
        if min(entry["treated"], entry["controls"]) < MIN_UNITS:
            log.append(
                {
                    **entry,
                    "status": f"dropped: fewer than {MIN_UNITS} treated or clean-control sites",
                }
            )
            continue
        w["k"] = (w["date"] - s).dt.days
        w["duration"] = (f - s).days + 1
        w["stack"] = e.event_id
        out.append(w)
        log.append({**entry, "status": "kept"})
    log = pd.DataFrame(log)
    if not out:
        return pd.DataFrame(), log
    st = pd.concat(out, ignore_index=True)
    st["stack_site"] = st["stack"] + "|" + st["site_code"]
    st["stack_date"] = st["stack"] + "|" + st["date_id"]
    return st, log


def _k_name(k: int) -> str:
    return f"ev_m{-k}" if k < 0 else f"ev_p{k}"


def event_study(st: pd.DataFrame, y: str) -> tuple[pd.DataFrame, float, float]:
    """Dynamic coefficients D x 1{k} for k in [LEAD_MIN, WINDOW]; reference k <= REF_MAX.
    Returns the coefficients and the Wald p-value of the pre-event (lead) coefficients."""
    st = st.copy()
    ks = list(range(LEAD_MIN, WINDOW + 1))
    for k in ks:
        st[_k_name(k)] = st["treated"] * (st["k"] == k)
    fit = _fit(st, y, [_k_name(k) for k in ks], ["stack_site", "stack_date", "fe_dow"])
    t = _tidy(fit)
    t["k"] = [int(n[4:]) * (-1 if n.startswith("ev_m") else 1) for n in t["term"]]
    leads = [i for i, n in enumerate(fit._coefnames) if n.startswith("ev_m")]
    r = np.zeros((len(leads), len(fit._coefnames)))
    r[np.arange(len(leads)), leads] = 1.0
    wald = fit.wald_test(R=r)
    return t, float(wald["statistic"]), float(wald["pvalue"])


def event_att(st: pd.DataFrame, y: str) -> pd.DataFrame:
    st = st.copy()
    st["d_lead"] = st["treated"] * st["k"].between(LEAD_MIN, -1)
    st["d_event"] = st["treated"] * ((st["k"] >= 0) & (st["k"] < st["duration"]))
    st["d_after"] = st["treated"] * (st["k"] >= st["duration"])
    fit = _fit(st, y, ["d_lead", "d_event", "d_after"], ["stack_site", "stack_date", "fe_dow"])
    return _tidy(fit)


def event_designs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame]:
    dyn, att, meta, logs = [], [], [], []
    for kind in ("heatwave", "cold spell"):
        for g in ("no CHP", "on-site CHP"):
            st, log = stacks(df[df["group"].isin(GROUPS[g])], kind)
            logs.append(log.assign(group=g))
            if st.empty:
                continue
            for y, label in (("log_kw", "demand"), ("t_mean", "temperature")):
                t, w, p = event_study(st, y)
                dyn.append(t.assign(type=kind, group=g, outcome=label, wald_leads=w, p_leads=p))
                att.append(event_att(st, y).assign(type=kind, group=g, outcome=label))
            n = st.groupby("treated")["site_code"].nunique()
            meta.append(
                {
                    "type": kind,
                    "group": g,
                    "stacks": int(st["stack"].nunique()),
                    "treated_sites": int(n.get(1.0, 0)),
                    "control_sites": int(n.get(0.0, 0)),
                    "site_days": len(st),
                }
            )
    dyn = pd.concat(dyn, ignore_index=True)
    att = pd.concat(att, ignore_index=True)
    dem = dyn["outcome"] == "demand"
    dyn.loc[dem, ["coef_pct", "lo_pct", "hi_pct"]] = 100 * np.expm1(
        dyn.loc[dem, ["coef", "lo", "hi"]].to_numpy()
    )
    return dyn, att, meta, pd.concat(logs, ignore_index=True)


def per_degree(att: pd.DataFrame) -> pd.DataFrame:
    """Demand ATT over temperature ATT on event days: % demand per degC of excess temperature
    (delta-method SE ignoring the covariance between the two regressions)."""
    rows = []
    ev = att[att["term"] == "d_event"]
    for (kind, g), d in ev.groupby(["type", "group"]):
        dem, tmp = d[d["outcome"] == "demand"].iloc[0], d[d["outcome"] == "temperature"].iloc[0]
        ratio = 100 * np.expm1(dem["coef"]) / tmp["coef"]
        se = (
            100
            * abs(ratio / 100)
            * np.sqrt((dem["se"] / dem["coef"]) ** 2 + (tmp["se"] / tmp["coef"]) ** 2)
        )
        rows.append(
            {
                "type": kind,
                "group": g,
                "demand_att_pct": 100 * np.expm1(dem["coef"]),
                "demand_att_lo": 100 * np.expm1(dem["lo"]),
                "demand_att_hi": 100 * np.expm1(dem["hi"]),
                "temperature_att_c": tmp["coef"],
                "pct_per_c": ratio,
                "pct_per_c_se": se,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3-5. Leads and lags, randomisation inference, effect modification
# --------------------------------------------------------------------------- #


def leads_lags(df: pd.DataFrame, evening: bool = False) -> pd.DataFrame:
    """Specification B with leads and lags of both doses; ``evening`` adds the same day's
    evening doses as controls."""
    sub = df[df["group"].isin(GROUPS["no CHP"])]
    lags = [f"{v}_lag{k}" for v in DOSE for k in range(1, N_LAGS + 1)]
    leads = [f"{v}_lead{k}" for v in DOSE for k in range(1, N_LEADS + 1)]
    x = list(DOSE) + lags + leads + ([f"{v}_eve" for v in DOSE] if evening else [])
    sub = sub.dropna(subset=x)
    fit = _fit(sub, "log_kw", x, SPECS["B"]["fe"])
    t = _tidy(fit)
    t = t[~t["term"].str.endswith("_eve")].copy()
    t["dose"] = [n.split("_")[0] for n in t["term"]]
    t["lag"] = [
        0 if "_" not in n else (int(n.split("lag")[1]) if "lag" in n else -int(n.split("lead")[1]))
        for n in t["term"]
    ]
    out = []
    names = list(fit._coefnames)
    for v in DOSE:  # cumulative effect of a one-degree dose held for one day, lags 0..L
        for L in range(N_LAGS + 1):
            terms = [v] + [f"{v}_lag{k}" for k in range(1, L + 1)]
            w = np.zeros(len(names))
            w[[names.index(c) for c in terms]] = 1.0
            b = float(w @ fit.coef().to_numpy())
            se = float(np.sqrt(w @ fit._vcov @ w))
            out.append(
                {
                    "term": f"{v}_cum{L}",
                    "coef": b,
                    "se": se,
                    "lo": b - 1.96 * se,
                    "hi": b + 1.96 * se,
                    "p": np.nan,
                    "dose": v,
                    "lag": L,
                }
            )
    r = np.zeros((len(leads), len(names)))
    r[np.arange(len(leads)), [names.index(c) for c in leads]] = 1.0
    wald = fit.wald_test(R=r)
    t = pd.concat(
        [t.assign(kind="coefficient"), pd.DataFrame(out).assign(kind="cumulative")],
        ignore_index=True,
    )
    return _pct(t).assign(p_leads_joint=float(wald["pvalue"]))


def randomisation(df: pd.DataFrame, n_perm: int = N_PERM, seed: int = SEED) -> pd.DataFrame:
    """Design 1B re-estimated with weather series reassigned between sites at random."""
    sub = df[df["group"].isin(GROUPS["no CHP"])].copy()
    wx = weather_doses()[["site_code", "date"] + list(DOSE)]
    wide = {v: wx.pivot(index="date", columns="site_code", values=v) for v in DOSE}
    sites = np.array(wide["hdd14"].columns)
    di = wide["hdd14"].index.get_indexer(sub["date"])
    rng = np.random.default_rng(seed)
    obs = _tidy(_fit(sub, "log_kw", list(DOSE), SPECS["B"]["fe"]), list(DOSE)).set_index("term")[
        "coef"
    ]
    rows = [{"draw": -1, **obs.to_dict()}]
    panel_sites = set(sub["site_code"])
    for i in range(n_perm):
        while True:
            perm = dict(zip(sites, rng.permutation(sites), strict=True))
            if all(perm[s] != s for s in panel_sites if s in perm):
                break
        ci = wide["hdd14"].columns.get_indexer(sub["site_code"].map(perm))
        for v in DOSE:
            sub[v] = wide[v].to_numpy()[di, ci]
        t = _tidy(_fit(sub, "log_kw", list(DOSE), SPECS["B"]["fe"]), list(DOSE)).set_index("term")[
            "coef"
        ]
        rows.append({"draw": i, **t.to_dict()})
        if (i + 1) % 50 == 0:
            logger.info("randomisation draw %d / %d", i + 1, n_perm)
    return pd.DataFrame(rows)


def heterogeneity(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["group"].isin(GROUPS["no CHP"])].copy()
    eric = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").drop_duplicates("site_code")
    build = eric.set_index("site_code")["mean_build_year"]
    size = sub.groupby("site_code")["kw"].mean()
    summer = sub[sub["date"].dt.month.isin([6, 7, 8])].groupby("site_code")["t_max"].mean()
    moderators = {
        "older buildings": build.reindex(size.index) < build.reindex(size.index).median(),
        "larger sites": size > size.median(),
        "warmer climate": summer.reindex(size.index) > summer.reindex(size.index).median(),
    }
    rows = []
    for name, flag in moderators.items():
        known = flag[build.reindex(size.index).notna()] if name == "older buildings" else flag
        s = sub[sub["site_code"].isin(known.index)].copy()
        m = s["site_code"].map(known).astype(float)
        for v in DOSE:
            s[f"{v}_x"] = s[v] * m
        t = _tidy(_fit(s, "log_kw", list(DOSE) + [f"{v}_x" for v in DOSE], SPECS["B"]["fe"]))
        rows.append(t.assign(moderator=name, sites_flagged=int(known.sum()), sites=int(known.size)))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 6. Extreme days, spatial structure, inference
# --------------------------------------------------------------------------- #

EXTREME = ("hot28", "hot30")
DISTANCE_BANDS_KM = (0, 25, 50, 100, 200, 400)
NEIGHBOUR_KM = (30.0, 150.0)
CLUSTERS = {"site and date": "site_code+date_id", "region x week": "region_week", "ICB": "icb"}


def twfe_extreme(df: pd.DataFrame) -> pd.DataFrame:
    """Specifications A-D with indicators for days whose maximum reaches 28-30 and 30+ degC on
    top of the linear doses: the extra demand on such a day beyond what its mean implies."""
    rows = []
    for g, members in GROUPS.items():
        sub = df[df["group"].isin(members)]
        if sub.empty or sub["hot30"].sum() == 0:
            continue
        for spec, s in SPECS.items():
            x = list(DOSE) + list(EXTREME) + s["controls"]
            t = _pct(_tidy(_fit(sub, "log_kw", x, s["fe"]), list(DOSE) + list(EXTREME)))
            rows.append(
                t.assign(
                    group=g,
                    spec=spec,
                    spec_label=s["label"],
                    sites=sub["site_code"].nunique(),
                    site_days_30plus=int(sub["hot30"].sum()),
                )
            )
    return pd.concat(rows, ignore_index=True)


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Great-circle distances in km between all pairs of points (degrees in)."""
    la, lo = np.radians(np.asarray(lat, float)), np.radians(np.asarray(lon, float))
    dlat, dlon = la[:, None] - la[None, :], lo[:, None] - lo[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(la)[:, None] * np.cos(la)[None, :] * np.sin(dlon / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def site_distances(sites: list[str]) -> pd.DataFrame:
    geo = pd.read_parquet(GEO_DIR / "sites_geo.parquet").set_index("site_code")
    geo = geo.reindex(sites)[["latitude", "longitude"]].dropna()
    km = haversine_km(geo["latitude"].to_numpy(), geo["longitude"].to_numpy())
    return pd.DataFrame(km, index=geo.index, columns=geo.index)


def neighbour_weights(km: pd.DataFrame, band: tuple[float, float] = NEIGHBOUR_KM) -> pd.DataFrame:
    """Row-normalised weights over the sites in the distance band; NaN rows have no neighbour."""
    w = ((km >= band[0]) & (km < band[1])).astype(float)
    return w.div(w.sum(axis=1).replace(0.0, np.nan), axis=0)


def _band_means(c: pd.DataFrame, km: pd.DataFrame) -> list[dict]:
    cols = [s for s in c.columns if s in km.index]
    cm, dm = c.loc[cols, cols].to_numpy(), km.loc[cols, cols].to_numpy()
    upper = np.triu(np.ones_like(dm, dtype=bool), k=1)
    out = []
    for lo, hi in zip(DISTANCE_BANDS_KM[:-1], DISTANCE_BANDS_KM[1:], strict=True):
        m = upper & (dm >= lo) & (dm < hi) & np.isfinite(cm)
        out.append(
            {
                "band_km": f"{lo}-{hi}",
                "pairs": int(m.sum()),
                "mean_correlation": float(cm[m].mean()) if m.any() else np.nan,
            }
        )
    return out


def spatial_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Correlation between sites of day-to-day temperature anomalies (within site x month) and of
    specification-B demand residuals, by distance between the sites."""
    import pyfixest as pf

    sub = df[df["group"].isin(GROUPS["no CHP"])].dropna(subset=list(DOSE)).copy()
    fit = pf.feols(f"log_kw ~ {' + '.join(DOSE)} | {' + '.join(SPECS['B']['fe'])}", data=sub)
    sub["resid_b"] = np.asarray(fit.resid())
    sub["t_anom"] = sub["t_mean"] - sub.groupby("fe_month")["t_mean"].transform("mean")
    km = site_distances(sorted(sub["site_code"].unique()))
    rows = []
    for name, col in (("temperature anomaly", "t_anom"), ("demand residual, spec B", "resid_b")):
        wide = sub.pivot_table(index="date", columns="site_code", values=col)
        for r in _band_means(wide.corr(min_periods=200), km):
            rows.append({"series": name, **r})
    return pd.DataFrame(rows)


def spatial_lag(df: pd.DataFrame, doses_table: pd.DataFrame | None = None) -> pd.DataFrame:
    """Specifications A and B with the mean doses of the site's neighbours 30-150 km away (all
    located sites, not only the panel). Variants: the site's own doses alongside the neighbours'
    (a horse race - with regional shocks absent, the better thermometer wins), and the
    neighbours' doses alone (the effect per degree of regional temperature)."""
    sub = df[df["group"].isin(GROUPS["no CHP"])].copy()
    wx = weather_doses(doses_table)
    km = site_distances(sorted(wx["site_code"].unique()))
    w = neighbour_weights(km)
    nb = [f"{v}_nb" for v in DOSE]
    for v, col in zip(DOSE, nb, strict=True):
        wide = wx.pivot(index="date", columns="site_code", values=v)[w.index]
        lag = pd.DataFrame(wide.to_numpy() @ w.to_numpy().T, index=wide.index, columns=w.index)
        long = lag.stack().rename_axis(["date", "site_code"]).rename(col).reset_index()
        sub = sub.merge(long, on=["date", "site_code"], how="left")
    sub = sub.dropna(subset=nb)
    n_nb = w.gt(0).sum(axis=1).reindex(sub["site_code"].unique())
    rows = []
    for variant, terms in (("own and neighbours", list(DOSE) + nb), ("neighbours only", nb)):
        for spec in ("A", "B"):
            s = SPECS[spec]
            t = _pct(_tidy(_fit(sub, "log_kw", terms + s["controls"], s["fe"]), terms))
            rows.append(
                t.assign(
                    variant=variant,
                    spec=spec,
                    spec_label=s["label"],
                    sites=sub["site_code"].nunique(),
                    median_neighbours=float(n_nb.median()),
                )
            )
    return pd.concat(rows, ignore_index=True)


def inference_robustness(df: pd.DataFrame) -> pd.DataFrame:
    """Specification B under alternative clusterings of the errors."""
    sub = df[df["group"].isin(GROUPS["no CHP"])]
    rows = []
    for name, cl in CLUSTERS.items():
        t = _pct(_tidy(_fit(sub, "log_kw", list(DOSE), SPECS["B"]["fe"], cluster=cl), list(DOSE)))
        t["se_pct"] = 100 * t["se"]
        n = " x ".join(str(sub[c].nunique()) for c in cl.split("+"))
        rows.append(t.assign(cluster=name, n_clusters=n))
    return pd.concat(rows, ignore_index=True)


def response_curve(df: pd.DataFrame) -> pd.DataFrame:
    """The binned temperature response of ``weather_response`` re-estimated under specifications
    A, B and D at sites without CHP: the treatment-response curve with and without the national
    day shocks removed, on the daily mean and the daily maximum."""
    sub = df[df["group"].isin(GROUPS["no CHP"])]
    rows = []
    for variable, (_, _, edges, ref, (tail_lo, tail_hi)) in wr.VARIABLES.items():
        dummies = wr.bin_dummies(sub, variable)
        data = pd.concat([sub, dummies], axis=1)
        for spec in ("A", "B", "D"):
            s = SPECS[spec]
            x = list(dummies.columns) + s["controls"]
            t = _pct(_tidy(_fit(data, "log_kw", x, s["fe"]), list(dummies.columns)))
            t["bin_lo"] = [edges[int(n[2:])] for n in t["term"]]
            zero = {
                "term": wr.bin_term(ref, variable),
                "coef": 0.0,
                "se": 0.0,
                "lo": 0.0,
                "hi": 0.0,
                "p": 1.0,
            }
            zero.update(coef_pct=0.0, lo_pct=0.0, hi_pct=0.0, bin_lo=ref)
            t = pd.concat([t, pd.DataFrame([zero])], ignore_index=True).sort_values("bin_lo")
            idx = [edges.index(lo) for lo in t["bin_lo"]]
            t["bin_mid"] = [
                (edges[i] + edges[i + 1]) / 2
                if 0 < i < len(edges) - 2
                else (tail_lo if i == 0 else tail_hi)
                for i in idx
            ]
            t["bin"] = [wr.bin_label(lo, variable) for lo in t["bin_lo"]]
            rows.append(
                t.assign(
                    spec=spec,
                    spec_label=s["label"],
                    variable=variable,
                    sites=sub["site_code"].nunique(),
                )
            )
    return pd.concat(rows, ignore_index=True)


def run(n_perm: int = N_PERM) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = panel()
    specs = twfe_specs(df)
    specs.to_csv(OUT_DIR / "twfe_specs.csv", index=False)
    ident = identifying_variation(df)
    ident.to_csv(OUT_DIR / "identifying_variation.csv", index=False)
    extreme = twfe_extreme(df)
    extreme.to_csv(OUT_DIR / "twfe_extreme.csv", index=False)
    response_curve(df).to_csv(OUT_DIR / "response_curve.csv", index=False)
    spatial = spatial_correlation(df)
    spatial.to_csv(OUT_DIR / "spatial_correlation.csv", index=False)
    lag = spatial_lag(df)
    lag.to_csv(OUT_DIR / "spatial_lag.csv", index=False)
    infer = inference_robustness(df)
    infer.to_csv(OUT_DIR / "inference.csv", index=False)
    logger.info("TWFE specifications done")
    dyn, att, meta, stack_log = event_designs(df)
    stack_log.to_csv(OUT_DIR / "stack_log.csv", index=False)
    dyn.to_csv(OUT_DIR / "event_study.csv", index=False)
    pdeg = per_degree(att)
    att.to_csv(OUT_DIR / "event_att.csv", index=False)
    pdeg.to_csv(OUT_DIR / "event_per_degree.csv", index=False)
    logger.info("event studies done: %s", meta)
    ll = leads_lags(df)
    ll.to_csv(OUT_DIR / "leads_lags.csv", index=False)
    ll_eve = leads_lags(df, evening=True)
    ll_eve.to_csv(OUT_DIR / "leads_lags_evening.csv", index=False)
    het = heterogeneity(df)
    het.to_csv(OUT_DIR / "heterogeneity.csv", index=False)
    ri = randomisation(df, n_perm)
    ri.to_csv(OUT_DIR / "randomisation.csv", index=False)

    obs, draws = ri[ri["draw"] == -1].iloc[0], ri[ri["draw"] >= 0]
    summary = {
        "site_days": len(df),
        "twfe_pct_per_degree": {
            f"{r.group} | {r.spec} | {r.term}": [
                round(r.coef_pct, 2),
                round(r.lo_pct, 2),
                round(r.hi_pct, 2),
            ]
            for r in specs.itertuples()
        },
        "identifying_within_sd": {
            f"{r.group} | {r.spec} | {r.dose}": round(r.within_sd, 3)
            for r in ident.itertuples()
            if r.group == "no CHP"
        },
        "extreme_days_pct": {
            f"{r.group} | {r.spec} | {r.term}": [
                round(r.coef_pct, 2),
                round(r.lo_pct, 2),
                round(r.hi_pct, 2),
            ]
            for r in extreme.itertuples()
            if r.term in EXTREME
        },
        "spatial_correlation": spatial.round(3).to_dict("records"),
        "spatial_lag_pct": {
            f"{r.variant} | {r.spec} | {r.term}": [
                round(r.coef_pct, 2),
                round(r.lo_pct, 2),
                round(r.hi_pct, 2),
            ]
            for r in lag.itertuples()
        },
        "inference_se_pct": {
            f"{r.cluster} | {r.term}": round(r.se_pct, 3) for r in infer.itertuples()
        },
        "event_samples": meta,
        "event_stacks_dropped": stack_log.loc[
            stack_log["status"] != "kept", ["event_id", "group", "status"]
        ].to_dict("records"),
        "event_per_degree": pdeg.round(3).to_dict("records"),
        "event_pre_trend_p": {
            f"{t} | {g} | {o}": round(float(d["p_leads"].iloc[0]), 3)
            for (t, g, o), d in dyn.groupby(["type", "group", "outcome"])
        },
        "leads_joint_p": round(float(ll["p_leads_joint"].iloc[0]), 3),
        "leads_joint_p_evening_control": round(float(ll_eve["p_leads_joint"].iloc[0]), 3),
        "randomisation": {
            v: {
                "observed": round(float(obs[v]), 5),
                "perm_mean": round(float(draws[v].mean()), 5),
                "perm_sd": round(float(draws[v].std()), 5),
                "p_two_sided": round(float((np.abs(draws[v]) >= abs(obs[v])).mean()), 4),
                "draws": int(len(draws)),
            }
            for v in DOSE
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return {
        "specs": specs,
        "identifying_variation": ident,
        "event_study": dyn,
        "event_att": att,
        "per_degree": pdeg,
        "stack_log": stack_log,
        "leads_lags": ll,
        "heterogeneity": het,
        "randomisation": ri,
        "extreme": extreme,
        "spatial_correlation": spatial,
        "spatial_lag": lag,
        "inference": infer,
        "summary": summary,
    }
