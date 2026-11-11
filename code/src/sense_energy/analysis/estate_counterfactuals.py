"""The weather-normalised estate: a warmer climate and the 2025 heatwaves, half-hour by half-hour.

Both threads run the site-level counterfactual engine of ``hourly_response``: the demand of site
s in local hour h under a different daily mean temperature is the observed demand times
exp(beta_heat[s, h] x change in the heat dose + beta_cold[s, h] x change in the cold dose),
with the empirical-Bayes hourly posteriors, smoothed over neighbouring hours (weights 1/4, 1/2,
1/4, circular). Sites without CHP and without meter gaps.

**Warming stress test.** Every day of the record shifted by +1, +2 and +4 degC (uniform shifts
as sensitivity levels; summer warming by mid-century in UKCP18 lies in this range for southern
England, with the hottest days warming more than the mean). Per site: the change in annual
electricity, the summer (June-September) peak half-hour against the site's available capacity
(ERIC capacity x 0.95, the headroom analysis's basis, whole-site metered sites only), the
half-hours per summer above 90 % of capacity, and the same site's winter peak with heat pumps
from the headroom analysis, to see which binds first.

**The 2025 heatwaves.** For every site-day inside a site heatwave in 2025 (Met Office definition,
``weather_events``), the counterfactual temperature is the site's mean daily temperature over the
surrounding 29 days without heatwave days; the attributed demand is observed demand minus the
counterfactual demand, priced at the half-hour's market index price and costed in carbon at the
half-hour's grid intensity. Sites with CHP are added with the group's binned response of their
grid import (``weather_response``), reported separately because it is import, not demand. The
weather-blind JEPA's latent surprise (``experiments/jepa.py``) is compared between heatwave days
and other summer days, by group.

Outputs (``code/reports/estate_counterfactuals/``): ``warming_sites.csv``, ``warming_summary.csv``,
``heatwave_site_days.csv``, ``heatwave_events.csv``, ``heatwave_surprise.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "estate_counterfactuals"
HOURLY = PROJECT_ROOT / "code" / "reports" / "hourly_response"
SHIFTS = (1.0, 2.0, 4.0)
SUMMER = (6, 7, 8, 9)
HIGH_UTIL = 0.9
CF_HALF_WINDOW = 14
COLD_BASE, HOT_BASE = 14.0, 20.0


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #


def smoothed_profiles(path=HOURLY / "hourly_terms.csv") -> dict[str, pd.DataFrame]:
    """Site x hour posteriors of the heat and cold slopes, smoothed over neighbouring hours."""
    t = pd.read_csv(path)
    out = {}
    for term, name in (("cdd20", "heat"), ("hdd14", "cold")):
        p = (
            t[t["term"] == term]
            .pivot(index="site_code", columns="hour", values="posterior")
            .reindex(columns=range(24))
        )
        p = p.T.interpolate(limit_direction="both").T  # a missing hour takes its neighbours
        a = p.to_numpy()
        sm = 0.25 * np.roll(a, 1, axis=1) + 0.5 * a + 0.25 * np.roll(a, -1, axis=1)
        out[name] = pd.DataFrame(sm, index=p.index, columns=p.columns)
    return out


def doses(t: np.ndarray | pd.Series) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=float)
    return np.maximum(COLD_BASE - t, 0.0), np.maximum(t - HOT_BASE, 0.0)


def half_hourly(sites: list[str]) -> pd.DataFrame:
    """Long half-hourly kW with local date and hour, joined to the site's daily mean temperature."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    local = pd.to_datetime(w.index, utc=True).tz_convert(wr.TZ)
    w = w.assign(date=local.normalize().tz_localize(None), hour=local.hour, datetime=w.index)
    long = w.melt(
        id_vars=["datetime", "date", "hour"], var_name="site_code", value_name="kw"
    ).dropna(subset=["kw"])
    t = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_mean"]
    )
    return long.merge(t, on=["site_code", "date"], how="inner")


def multiplier(long: pd.DataFrame, prof: dict[str, pd.DataFrame], t_new: np.ndarray) -> np.ndarray:
    """exp(beta_heat x d(heat dose) + beta_cold x d(cold dose)) for every row of ``long``."""
    h0, c0 = doses(long["t_mean"])[1], doses(long["t_mean"])[0]
    c1, h1 = doses(t_new)
    sidx = prof["heat"].index.get_indexer(long["site_code"])
    hrs = long["hour"].to_numpy()
    bh = prof["heat"].to_numpy()[sidx, hrs]
    bc = prof["cold"].to_numpy()[sidx, hrs]
    return np.exp(bh * (h1 - h0) + bc * (c1 - c0))


# --------------------------------------------------------------------------- #
# Warming stress test
# --------------------------------------------------------------------------- #


def capacities() -> pd.DataFrame:
    h = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "estates" / "heat_pump_headroom_panel.csv")
    h = h[h["headroom_status"] == "included"].set_index("site_code")
    return h[["capacity_kw", "util_now", "util_hp", "site_type"]]


def warming(long: pd.DataFrame, prof: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cap = capacities()
    long = long.copy()
    long["year"] = long["date"].dt.year
    summer = long["date"].dt.month.isin(SUMMER).to_numpy()
    rows = []
    variants = {0.0: long["kw"].to_numpy()}
    for d in SHIFTS:
        variants[d] = long["kw"].to_numpy() * multiplier(long, prof, long["t_mean"].to_numpy() + d)
    base_e = long.groupby("site_code")["kw"].sum()
    for d, kw in variants.items():
        df = long[["site_code", "year"]].assign(kw=kw)
        energy = df.groupby("site_code")["kw"].sum()
        sm = df[summer]
        peak = sm.groupby(["site_code", "year"])["kw"].max().groupby("site_code").mean()
        for s in energy.index:
            row = {
                "site_code": s,
                "shift_c": d,
                "energy_change_pct": 100 * (energy[s] / base_e[s] - 1),
                "summer_peak_kw": peak.get(s, np.nan),
            }
            if s in cap.index:
                c = cap.loc[s, "capacity_kw"]
                ss = sm[sm["site_code"] == s]
                n_years = max(ss["year"].nunique(), 1)
                row.update(
                    {
                        "capacity_kw": c,
                        "summer_util": peak.get(s, np.nan) / c,
                        "halfhours_above_90pct_per_summer": float(
                            (ss["kw"] > HIGH_UTIL * c).sum() / n_years
                        ),
                        "winter_util_now": cap.loc[s, "util_now"],
                        "winter_util_heat_pumps": cap.loc[s, "util_hp"],
                    }
                )
            rows.append(row)
    sites = pd.DataFrame(rows)
    g = sites.groupby("shift_c")
    summ = pd.DataFrame(
        {
            "sites": g.size(),
            "median_energy_change_pct": g["energy_change_pct"].median(),
            "p90_energy_change_pct": g["energy_change_pct"].quantile(0.9),
            "sites_with_capacity": g["capacity_kw"].count(),
            "median_summer_util": g["summer_util"].median(),
            "max_summer_util": g["summer_util"].max(),
            "sites_summer_util_above_90pct": g["summer_util"].apply(
                lambda s: int((s > HIGH_UTIL).sum())
            ),
            "sites_summer_above_winter_hp": sites.assign(
                x=sites["summer_util"] > sites["winter_util_heat_pumps"]
            )
            .groupby("shift_c")["x"]
            .sum(),
        }
    ).reset_index()
    return sites, summ


# --------------------------------------------------------------------------- #
# The 2025 heatwaves
# --------------------------------------------------------------------------- #


def heatwave_days(year: int = 2025) -> pd.DataFrame:
    ev = pd.read_csv(
        PROJECT_ROOT / "code" / "reports" / "weather_events" / "site_events.csv",
        parse_dates=["start", "end"],
    )
    ev = ev[ev["type"] == "heatwave"]
    rows = []
    for r in ev.itertuples():
        for d in pd.date_range(r.start, r.end):
            rows.append({"site_code": r.site_code, "date": d})
    all_days = pd.DataFrame(rows).drop_duplicates()
    return all_days, all_days[all_days["date"].dt.year == year]


def counterfactual_temperature(hw: pd.DataFrame, all_hw: pd.DataFrame) -> pd.DataFrame:
    t = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_mean"]
    )
    t = t[t["site_code"].isin(hw["site_code"].unique())]
    flag = all_hw.assign(hw=True)
    t = t.merge(flag, on=["site_code", "date"], how="left")
    t["hw"] = t["hw"].astype("boolean").fillna(False).astype(bool)
    out = []
    for s, d in t.groupby("site_code"):
        d = d.set_index("date").sort_index()
        normal = (
            d["t_mean"]
            .where(~d["hw"])
            .rolling(f"{2 * CF_HALF_WINDOW + 1}D", center=True, min_periods=10)
            .mean()
        )
        out.append(
            pd.DataFrame(
                {
                    "site_code": s,
                    "date": d.index,
                    "t_cf": normal.to_numpy(),
                    "t_mean": d["t_mean"].to_numpy(),
                }
            )
        )
    cf = pd.concat(out, ignore_index=True)
    return hw.merge(cf, on=["site_code", "date"], how="left")


def prices_and_carbon() -> pd.DataFrame:
    from ..data.prices import load_mid

    p = load_mid()  # one row per half-hour: the traded provider only
    c = pd.read_parquet(
        INTERIM_DIR / "neso_covariates_halfhourly.parquet",
        columns=["datetime", "carbon_intensity_actual_gco2_kwh"],
    )
    c["datetime"] = pd.to_datetime(c["datetime"], utc=True)
    c = c.drop_duplicates("datetime")
    out = p.merge(c, on="datetime", how="outer")
    assert not out["datetime"].duplicated().any(), "prices and carbon must be one row per half-hour"
    return out


def heatwave_cost(
    long: pd.DataFrame, prof: dict[str, pd.DataFrame], hw_cf: pd.DataFrame
) -> pd.DataFrame:
    """Site-day attribution for sites with hourly profiles."""
    x = long.merge(hw_cf[["site_code", "date", "t_cf"]], on=["site_code", "date"], how="inner")
    x = x[x["site_code"].isin(prof["heat"].index)].dropna(subset=["t_cf"])
    m = multiplier(x, prof, x["t_cf"].to_numpy())  # demand ratio under the normal temperature
    x["kw_cf"] = x["kw"] * m
    x["kwh_attr"] = 0.5 * (x["kw"] - x["kw_cf"])
    pc = prices_and_carbon()
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True)
    x = x.merge(pc, on="datetime", how="left")
    x["cost_gbp"] = x["kwh_attr"] * x["price_gbp_mwh"] / 1000.0
    x["co2_kg"] = x["kwh_attr"] * x["carbon_intensity_actual_gco2_kwh"] / 1000.0
    g = x.groupby(["site_code", "date"]).agg(
        t_mean=("t_mean", "first"),
        t_cf=("t_cf", "first"),
        kwh=("kw", lambda s: 0.5 * s.sum()),
        kwh_attr=("kwh_attr", "sum"),
        cost_gbp=("cost_gbp", "sum"),
        co2_kg=("co2_kg", "sum"),
    )
    g["share_pct"] = 100 * g["kwh_attr"] / g["kwh"]
    g["group"] = "no CHP"
    return g.reset_index()


def heatwave_chp(hw_cf: pd.DataFrame) -> pd.DataFrame:
    """CHP sites: daily grid import attributed with the group's binned daily-mean response."""
    from . import weather_causal as wc

    c = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "weather_response" / "curve.csv")
    c = c[
        (c["group"] == "on-site CHP")
        & (c["variable"] == "daily mean")
        & (c["outcome"] == "daily mean")
    ]
    edges = np.array(wr.EDGES[1:-1], dtype=float)
    coef = dict(zip(c["bin_lo"].astype(float), c["coef"].astype(float), strict=True))

    def f(t):
        i = np.searchsorted(edges, t, side="right")
        lo = np.r_[-np.inf, edges][i]
        return np.array([coef.get(v, 0.0) for v in lo])

    df = wc.panel()
    chp = df[df["group"] == "on-site CHP"][["site_code", "date", "kw"]]
    x = hw_cf.merge(chp, on=["site_code", "date"], how="inner").dropna(subset=["t_cf"])
    x["kwh"] = 24 * x["kw"]
    x["kwh_attr"] = x["kwh"] * (1 - np.exp(f(x["t_cf"].to_numpy()) - f(x["t_mean"].to_numpy())))
    x["share_pct"] = 100 * x["kwh_attr"] / x["kwh"]
    pc = prices_and_carbon()
    local = pc["datetime"].dt.tz_convert(wr.TZ)
    daily = (
        pc.assign(date=local.dt.normalize().dt.tz_localize(None))
        .groupby("date")[["price_gbp_mwh", "carbon_intensity_actual_gco2_kwh"]]
        .mean()
    )
    x = x.merge(daily, left_on="date", right_index=True, how="left")
    x["cost_gbp"] = (
        x["kwh_attr"] * x["price_gbp_mwh"] / 1000.0
    )  # daily mean price: the profile is not known
    x["co2_kg"] = x["kwh_attr"] * x["carbon_intensity_actual_gco2_kwh"] / 1000.0
    x["group"] = "on-site CHP (import)"
    return x[
        [
            "site_code",
            "date",
            "t_mean",
            "t_cf",
            "kwh",
            "kwh_attr",
            "share_pct",
            "cost_gbp",
            "co2_kg",
            "group",
        ]
    ]


def event_table(sd: pd.DataFrame) -> pd.DataFrame:
    ne = pd.read_csv(
        PROJECT_ROOT / "code" / "reports" / "weather_events" / "national_events.csv",
        parse_dates=["start", "end"],
    )
    ne = ne[(ne["type"] == "heatwave") & (ne["start"].dt.year == 2025)]
    rows = []
    for r in ne.itertuples():
        d = sd[(sd["date"] >= r.start) & (sd["date"] <= r.end)]
        for grp, dd in d.groupby("group"):
            rows.append(
                {
                    "event_id": r.event_id,
                    "start": r.start.date(),
                    "end": r.end.date(),
                    "group": grp,
                    "sites": dd["site_code"].nunique(),
                    "site_days": len(dd),
                    "kwh_attr": dd["kwh_attr"].sum(),
                    "kwh": dd["kwh"].sum(),
                    "share_pct": 100 * dd["kwh_attr"].sum() / dd["kwh"].sum(),
                    "cost_gbp": dd.get("cost_gbp", pd.Series(dtype=float)).sum(),
                    "co2_t": dd.get("co2_kg", pd.Series(dtype=float)).sum() / 1000.0,
                }
            )
    return pd.DataFrame(rows)


def surprise_on_heatwaves(all_hw: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import mannwhitneyu

    from . import weather_causal as wc

    path = PROJECT_ROOT / "code" / "reports" / "jepa" / "surprise.parquet"
    if not path.exists():
        return pd.DataFrame()
    sp = pd.read_parquet(path)
    sp = sp[sp["date"].dt.month.isin(SUMMER)]
    sp = sp.merge(all_hw.assign(hw=True), on=["site_code", "date"], how="left")
    sp["hw"] = sp["hw"].astype("boolean").fillna(False).astype(bool)
    grp = wc.panel()[["site_code", "group"]].drop_duplicates().set_index("site_code")["group"]
    sp["group"] = np.where(
        grp.reindex(sp["site_code"]).to_numpy() == "on-site CHP", "on-site CHP", "no CHP"
    )
    rows = []
    for g, d in sp.groupby("group"):
        a, b = d.loc[d["hw"], "surprise"], d.loc[~d["hw"], "surprise"]
        rows.append(
            {
                "group": g,
                "heatwave_days": len(a),
                "other_summer_days": len(b),
                "median_heatwave": a.median(),
                "median_other": b.median(),
                "ratio": a.median() / b.median(),
                "p": mannwhitneyu(a, b, alternative="greater").pvalue
                if len(a) and len(b)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prof = smoothed_profiles()
    sites = sorted(prof["heat"].index)
    long = half_hourly(sites)
    ws, wsum = warming(long, prof)
    ws.to_csv(OUT_DIR / "warming_sites.csv", index=False)
    wsum.to_csv(OUT_DIR / "warming_summary.csv", index=False)
    logger.info("warming done")
    all_hw, hw = heatwave_days(2025)
    hw_cf = counterfactual_temperature(hw, all_hw)
    sd = heatwave_cost(long, prof, hw_cf)
    sd = pd.concat([sd, heatwave_chp(hw_cf)], ignore_index=True)
    sd.to_csv(OUT_DIR / "heatwave_site_days.csv", index=False)
    ev = event_table(sd)
    ev.to_csv(OUT_DIR / "heatwave_events.csv", index=False)
    sp = surprise_on_heatwaves(all_hw)
    sp.to_csv(OUT_DIR / "heatwave_surprise.csv", index=False)
    non = sd[sd["group"] == "no CHP"]
    chp = sd[sd["group"] != "no CHP"]
    summary = {
        "warming": wsum.round(3).to_dict("records"),
        "heatwaves_2025": {
            "no_chp": {
                "sites": int(non["site_code"].nunique()),
                "site_days": int(len(non)),
                "kwh_attr": round(float(non["kwh_attr"].sum())),
                "share_pct": round(float(100 * non["kwh_attr"].sum() / non["kwh"].sum()), 2),
                "median_site_day_share_pct": round(float(non["share_pct"].median()), 2),
                "cost_gbp_wholesale": round(float(non["cost_gbp"].sum())),
                "co2_t": round(float(non["co2_kg"].sum() / 1000), 1),
                "mean_t_excess_c": round(float((non["t_mean"] - non["t_cf"]).mean()), 2),
            },
            "chp_import": {
                "sites": int(chp["site_code"].nunique()),
                "site_days": int(len(chp)),
                "kwh_attr": round(float(chp["kwh_attr"].sum())),
                "share_pct": round(float(100 * chp["kwh_attr"].sum() / chp["kwh"].sum()), 2),
                "cost_gbp_wholesale": round(float(chp["cost_gbp"].sum())),
                "co2_t": round(float(chp["co2_kg"].sum() / 1000), 1),
            },
        },
        "surprise": sp.round(4).to_dict("records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    logger.info("estate counterfactuals: %s", json.dumps(summary, default=str)[:900])
    return {
        "warming_sites": ws,
        "warming": wsum,
        "heatwave_site_days": sd,
        "events": ev,
        "surprise": sp,
        "summary": summary,
    }
