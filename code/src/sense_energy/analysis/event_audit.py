"""Audit of the event findings: do they survive other choices, a placebo, and the next event?

``extreme_events`` measures what a site's electricity does in heat waves and cold spells with
one set of choices. This module asks how much the headline numbers owe to those choices and
whether they predict anything.

1. **Specification grid.** Every combination of the reference weeks (five sets), the event
   definition (the catalogue; heat thresholds one degree lower; the site's own 5th and 95th
   temperature percentiles), the site sample (complete meters; all sites; the forecast panel)
   and the treatment of holidays. The headline numbers are computed for each.
2. **Placebo events.** The catalogue's events moved by five to eight weeks and by one and two
   years (days that fall on a real event are left out, and real events cannot be reference
   days). A measure that reports a reaction where there is no event is not a measure.
3. **Does a site's past predict its next event?** Leave-one-event-out prediction of each
   site's reaction in each national event from: nothing; the other sites in the event; the
   other sites of the region; the site's own other events; both; and, in date order, the
   site's first k events.
4. **Is the build-up the temperature of the days before?** The daily reaction on the
   temperature differences of the day and of the three days before (daily maximum and daily
   minimum), with and without a term for the third day on of an event.
5. **What the plant does.** At the sites with on-site generation the meter shows import. With
   the plant's mean output from the estates return, the reaction of the import is split into
   what a demand reaction like that of the other sites would give with the plant unchanged,
   and the change in plant output that the rest implies.

Outputs: ``code/reports/event_audit/``.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import extreme_events as ee

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "event_audit"
PANEL_FILE = PROJECT_ROOT / "code" / "reports" / "forecast_anatomy" / "error_by_site.csv"
PLAIN, PLANT = "no on-site generation", "on-site generation"
WEEK_SETS = {
    "1 to 3 weeks": (-3, -2, -1, 1, 2, 3),
    "1 to 2 weeks": (-2, -1, 1, 2),
    "2 to 4 weeks": (-4, -3, -2, 2, 3, 4),
    "weeks before": (-3, -2, -1),
    "weeks after": (1, 2, 3),
}
EVENT_SETS = ("catalogue", "one degree lower", "site percentiles")
SAMPLES = ("complete meters", "all sites", "forecast panel")
PLACEBO_SHIFTS = (-56, -49, -42, -35, 35, 42, 49, 56, -728, -364, 364, 728)
PERCENTILES = (5.0, 95.0)
EVENING, MORNING = (16.0, 22.0), (8.0, 12.0)
LAGS = (0, 1, 2, 3)
WARM_MONTHS, COLD_MONTHS = (5, 6, 7, 8, 9), (11, 12, 1, 2, 3)
HOT_MAX, WARM_MIN = 24.0, 12.0  # degrees above which a day and a night count as heat
MIN_PLANT_SHARE = 0.1  # plant output as a share of the site's demand
ACUTE = "acute inpatient"


@dataclass(frozen=True)
class Spec:
    weeks: str = "1 to 3 weeks"
    events: str = "catalogue"
    sample: str = "complete meters"
    holidays_out: bool = True
    shift_days: int = 0

    @property
    def baseline(self) -> bool:
        return self == Spec()


@dataclass
class Inputs:
    kw: pd.DataFrame
    weather: pd.DataFrame
    cube: np.ndarray
    sites: pd.DataFrame
    panel_sites: set[str]

    @property
    def days(self) -> pd.DatetimeIndex:
        return self.kw.index

    @property
    def codes(self) -> list[str]:
        return list(self.kw.columns)


def load_inputs() -> Inputs:
    w = ee._half_hourly()
    kw = ee.daily_demand(w)
    weather = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")
    weather = weather[weather["site_code"].isin(kw.columns)]
    panel = set(pd.read_csv(PANEL_FILE)["site_code"]) if PANEL_FILE.exists() else set()
    return Inputs(kw, weather, ee.slot_cube(w, kw.index), ee.site_table(kw), panel)


# --------------------------------------------------------------------------- #
# Event definitions and samples
# --------------------------------------------------------------------------- #


def site_events(inp: Inputs, events: str) -> pd.DataFrame | None:
    """Site events under one definition (None: the catalogue as stored)."""
    from . import weather_events as we

    if events == "catalogue":
        return None
    if events == "one degree lower":
        ev = we.site_events(inp.weather, we.heat_thresholds(), heat_offset=1.0)
        return ev[ev["type"].isin(ee.TYPES)]
    if events != "site percentiles":
        raise KeyError(events)
    rows = []
    for site, x in inp.weather.groupby("site_code"):
        x = x.set_index("date").sort_index()
        cold = x["t_mean"] <= np.nanpercentile(x["t_mean"], PERCENTILES[0])
        hot = x["t_max"] >= np.nanpercentile(x["t_max"], PERCENTILES[1])
        for kind, flag, n in (("heatwave", hot, 3), ("cold spell", cold, 2)):
            rows += [
                {"site_code": site, "type": kind, "start": a, "end": b} for a, b in we.runs(flag, n)
            ]
    return pd.DataFrame(rows)


def sample_sites(inp: Inputs, sample: str) -> pd.DataFrame:
    """The site table with ``clean`` set to the sample of the specification."""
    s = inp.sites.copy()
    if sample == "all sites":
        s["clean"] = True
    elif sample == "forecast panel":
        s["clean"] = s["site_code"].isin(inp.panel_sites)
    elif sample != "complete meters":
        raise KeyError(sample)
    return s


def _shift(mask: pd.DataFrame, days: int) -> pd.DataFrame:
    return mask.shift(days, fill_value=False)


def prepare(inp: Inputs, spec: Spec):
    """Reactions, scored event days, masks and sites of one specification."""
    masks = ee.event_masks(inp.days, inp.codes, site_events(inp, spec.events))
    blocked = None
    if spec.shift_days:
        real = np.zeros(inp.kw.shape, dtype=bool)
        for m in masks.values():
            real |= m.to_numpy()
        masks = {k: _shift(m, spec.shift_days) & ~real for k, m in masks.items()}
        blocked = real
    sites = sample_sites(inp, spec.sample)
    weeks = WEEK_SETS[spec.weeks]
    r = ee.reactions(inp.kw, inp.weather, masks, weeks, spec.holidays_out, blocked)
    r["event_id"] = None
    ev = ee.event_days(r)
    ev = ev[ev["site_code"].isin(sites.loc[sites["clean"], "site_code"])]
    return r, ev, masks, sites, weeks, blocked


# --------------------------------------------------------------------------- #
# 1, 2. Headline numbers of one specification
# --------------------------------------------------------------------------- #


def _late_minus_first(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Reaction from the third day of an event on minus that of the first day, in points,
    paired within the sites without on-site generation."""
    keep = sites.loc[sites["clean"] & (sites["chp"] == PLAIN), "site_code"]
    x = ev[ev["site_code"].isin(keep)].copy()
    x["pos"] = x.groupby(["site_code", "type", "run"]).cumcount()
    x["part"] = np.select([x["pos"] == 0, x["pos"] >= 2], ["first", "late"], "")
    per = ee._ratios(x[x["part"] != ""], ["type", "site_code", "part"])
    return per[per["n"] >= 2]["pct"].unstack("part").dropna()


def headline(inp: Inputs, spec: Spec, hours: bool = True) -> list[dict]:
    """The headline numbers of the event findings under one specification, long format."""
    r, ev, masks, sites, weeks, blocked = prepare(inp, spec)
    sr = ee.site_reactions(ev, sites)
    rows = []

    def add(metric, kind, value, lo=np.nan, hi=np.nan, n=np.nan):
        rows.append({"metric": metric, "type": kind, "value": value, "lo": lo, "hi": hi, "n": n})

    late = _late_minus_first(ev, sites)
    rel = ee.reliability(ev[ev["site_code"].isin(sites.loc[sites["chp"] == PLAIN, "site_code"])])
    for kind in ee.TYPES:
        x = sr[sr["type"] == kind]
        plain, plant = x[x["chp"] == PLAIN], x[x["chp"] == PLANT]
        add("reaction, no plant", kind, *ee._mean_ci(plain["reaction_pct"]), len(plain))
        add("reaction, plant", kind, *ee._mean_ci(plant["reaction_pct"]), len(plant))
        add("degrees, no plant", kind, *ee._mean_ci(plain["degrees"]), len(plain))
        add("event days, no plant", kind, float(plain["event_days"].sum()), n=len(plain))
        if len(plain) > 2 and len(plant) > 2:
            d = plant["reaction_pct"].mean() - plain["reaction_pct"].mean()
            se = np.sqrt(
                plant["reaction_pct"].var() / len(plant) + plain["reaction_pct"].var() / len(plain)
            )
            add("plant minus no plant", kind, d, d - 1.96 * se, d + 1.96 * se, len(plant))
        if kind in late.index.get_level_values(0):
            w = late.loc[kind]
            add("third day on minus first day", kind, *ee._mean_ci(w["late"] - w["first"]), len(w))
        if len(plain) > 10:
            share = ee.region_share(
                plain.set_index("site_code")["reaction_pct"], plain.set_index("site_code")["region"]
            )
            add(
                "region share minus chance",
                kind,
                100 * (share["share"] - share["chance_share"]),
                n=share["n"],
            )
            add("region p", kind, share["p"], n=share["n"])
        got = rel[rel["type"] == kind]
        if len(got):
            add(
                "other events r2",
                kind,
                100 * float(got["other_events_r2"].iloc[0]),
                n=int(got["sites"].iloc[0]),
            )
    if not spec.shift_days:
        bu = ee.build_up(ev, sites)
        for b in bu.itertuples():
            add("build-up per degree", b.type, b.difference, b.lo, b.hi, b.sites)
    if hours:
        hp = ee.hour_profile(
            inp.cube, inp.days, inp.codes, masks, sites, weeks, spec.holidays_out, blocked
        )
        hp = hp[hp["group"] == PLAIN]
        for kind, h in hp.groupby("type"):
            eve = h[(h["hour"] >= EVENING[0]) & (h["hour"] < EVENING[1])]["reaction_pct"].mean()
            morn = h[(h["hour"] >= MORNING[0]) & (h["hour"] < MORNING[1])]["reaction_pct"].mean()
            add("evening minus morning", kind, eve - morn, n=int(h["sites"].max()))
    return [{**asdict(spec), **row} for row in rows]


def specification_grid(inp: Inputs) -> pd.DataFrame:
    rows = []
    grid = itertools.product(WEEK_SETS, EVENT_SETS, SAMPLES, (True, False))
    for weeks, events, sample, hol in grid:
        spec = Spec(weeks, events, sample, hol)
        rows += headline(inp, spec)
        logger.info("specification %s done", spec)
    out = pd.DataFrame(rows)
    out["baseline"] = (
        (out["weeks"] == Spec().weeks)
        & (out["events"] == Spec().events)
        & (out["sample"] == Spec().sample)
        & out["holidays_out"]
    )
    return out


def placebo(inp: Inputs) -> pd.DataFrame:
    rows = []
    for shift in PLACEBO_SHIFTS:
        rows += headline(inp, Spec(shift_days=shift))
    return pd.DataFrame(rows)


def grid_summary(grid: pd.DataFrame, plc: pd.DataFrame) -> pd.DataFrame:
    """Per headline number: the baseline, the range over the specifications, the share of the
    specifications with the baseline's sign and with an interval that excludes zero, and the
    range of the placebo events."""
    rows = []
    for (metric, kind), g in grid.groupby(["metric", "type"]):
        base = g[g["baseline"]].iloc[0]
        p = plc[(plc["metric"] == metric) & (plc["type"] == kind)]["value"]
        has_ci = g["lo"].notna().all()
        rows.append(
            {
                "metric": metric,
                "type": kind,
                "baseline": base["value"],
                "baseline_lo": base["lo"],
                "baseline_hi": base["hi"],
                "specifications": len(g),
                "min": g["value"].min(),
                "median": g["value"].median(),
                "max": g["value"].max(),
                "same_sign_share": float((np.sign(g["value"]) == np.sign(base["value"])).mean()),
                "interval_excludes_zero_share": float(((g["lo"] > 0) | (g["hi"] < 0)).mean())
                if has_ci
                else np.nan,
                "placebo_n": len(p),
                "placebo_min": p.min() if len(p) else np.nan,
                "placebo_mean": p.mean() if len(p) else np.nan,
                "placebo_max": p.max() if len(p) else np.nan,
                "placebo_as_large": int((p.abs() >= abs(base["value"])).sum())
                if len(p)
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Does a site's past predict its next event?
# --------------------------------------------------------------------------- #


def site_event_table(inp: Inputs) -> pd.DataFrame:
    """One row per site and national event of the baseline: reaction, size, date, region."""
    r, _, _, sites, _, _ = prepare(inp, Spec())
    r = ee.add_national(r, ee.national_events())
    ev = ee.event_days(r)
    ev = ev[ev["event_id"].notna() & ev["site_code"].isin(sites.loc[sites["clean"], "site_code"])]
    per = ee._ratios(ev, ["type", "event_id", "site_code"]).reset_index()
    per = per[per["n"] >= 2].rename(columns={"pct": "reaction_pct", "d": "degrees"})
    start = ev.groupby("event_id")["date"].min().rename("start")
    per = per.merge(start, on="event_id").merge(
        sites[["site_code", "chp", "region"]], on="site_code"
    )
    return per.replace([np.inf, -np.inf], np.nan).dropna(subset=["reaction_pct"])


def _loo_mean(values: pd.Series, groups: pd.Series) -> pd.Series:
    """Mean of the other members of the group (NaN for a group of one)."""
    g = values.groupby(groups)
    n, total = g.transform("size"), g.transform("sum")
    return ((total - values) / (n - 1)).where(n > 1)


def predictions(t: pd.DataFrame) -> pd.DataFrame:
    """Leave-one-out predictions of each site-event reaction (sites of one group, one type)."""
    t = t.copy()
    t["p_zero"] = 0.0
    t["p_event"] = _loo_mean(t["reaction_pct"], t["event_id"])
    region = _loo_mean(t["reaction_pct"], t["event_id"] + "|" + t["region"].astype(str))
    n_region = t.groupby(["event_id", "region"])["reaction_pct"].transform("size")
    t["p_region"] = region.where(n_region >= 4, t["p_event"])
    t["p_site"] = _loo_mean(t["reaction_pct"], t["site_code"])
    dev = t["reaction_pct"] - t["p_event"]
    t["p_event_site"] = t["p_event"] + _loo_mean(dev, t["site_code"])
    return t


def prediction_scores(inp: Inputs) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mean absolute error of each predictor over the site-events that every predictor can
    score, and, in date order, the error of 'other sites plus the site's first k events' by
    k. Sites without on-site generation; the plant sites separately."""
    t = site_event_table(inp)
    rows, curve = [], []
    names = {
        "p_zero": "no reaction",
        "p_event": "the other sites in the event",
        "p_region": "the other sites of the region",
        "p_site": "the site's other events",
        "p_event_site": "other sites and the site's other events",
    }
    for (kind, chp), g in t.groupby(["type", "chp"]):
        p = predictions(g).dropna(subset=list(names))
        if len(p) < 20:
            continue
        sst = float(((p["reaction_pct"] - p["p_event"]) ** 2).sum())
        for col, label in names.items():
            err = p["reaction_pct"] - p[col]
            rows.append(
                {
                    "type": kind,
                    "sites": chp,
                    "predictor": label,
                    "site_events": len(p),
                    "n_sites": p["site_code"].nunique(),
                    "mae_pts": float(err.abs().mean()),
                    "median_ae_pts": float(err.abs().median()),
                    "r2_against_event_mean": 1 - float((err**2).sum()) / sst,
                }
            )
        # in date order: the site's deviation from the other sites in its first k events
        g = predictions(g).sort_values("start")
        g["dev"] = g["reaction_pct"] - g["p_event"]
        g["k"] = g.groupby("site_code").cumcount()
        csum = g.groupby("site_code")["dev"].cumsum() - g["dev"]
        for k in range(0, 6):
            x = g[g["k"] >= k].copy()
            if k == 0:
                pred = x["p_event"]
            else:
                first_k = g[g["k"] < k].groupby("site_code")["dev"].mean()
                pred = x["p_event"] + x["site_code"].map(first_k)
            err = (x["reaction_pct"] - pred).dropna()
            x5 = x[x["k"] >= 5]
            curve.append(
                {
                    "type": kind,
                    "sites": chp,
                    "earlier_events_used": k,
                    "site_events": len(err),
                    "mae_pts": float(err.abs().mean()),
                    # the same later events for every k: the sixth event of a site and after
                    "mae_same_events_pts": float(
                        (x5["reaction_pct"] - pred.reindex(x5.index)).abs().mean()
                    )
                    if len(x5) >= 20
                    else np.nan,
                    "same_events": len(x5),
                }
            )
        del csum
    return pd.DataFrame(rows), pd.DataFrame(curve), t


# --------------------------------------------------------------------------- #
# 4. Is the build-up the temperature of the days before?
# --------------------------------------------------------------------------- #


def lag_panel(inp: Inputs) -> pd.DataFrame:
    """Site-days of the sites without on-site generation with the daily reaction (per cent)
    and the temperature differences from the reference days of the day and the three days
    before, plus the position of the day inside its event."""
    r, ev, _, sites, _, _ = prepare(inp, Spec())
    keep = sites.loc[sites["clean"] & (sites["chp"] == PLAIN), "site_code"]
    r = r[r["site_code"].isin(keep)].sort_values(["site_code", "date"]).copy()
    with np.errstate(all="ignore"):
        r["y"] = 100 * np.log(r["kw"] / r["ref_kw"])
    # heat is a dose above a threshold: degrees of daily maximum above 24 and of daily minimum
    # above 12, each as a difference from the same dose on the reference days
    r["d_hot"] = (r["t_max"] - HOT_MAX).clip(lower=0) - (r["t_max"] - r["d_t_max"] - HOT_MAX).clip(
        lower=0
    )
    r["d_warm_night"] = (r["t_min"] - WARM_MIN).clip(lower=0) - (
        r["t_min"] - r["d_t_min"] - WARM_MIN
    ).clip(lower=0)
    for col in ("d_hot", "d_warm_night", "d_t_min", "d_t_mean"):
        g = r.groupby("site_code")[col]
        for k in LAGS:
            r[f"{col}_{k}"] = g.shift(k)
    pos = ev.assign(pos=ev.groupby(["site_code", "type", "run"]).cumcount())
    r = r.merge(pos[["site_code", "date", "pos"]], on=["site_code", "date"], how="left")
    r["late"] = (r["pos"] >= 2).astype(float)
    r["in_event"] = r["pos"].notna().astype(float)
    return r.replace([np.inf, -np.inf], np.nan)


def lag_models(inp: Inputs) -> pd.DataFrame:
    """Daily reaction on the temperature of the day and of the three days before, as
    differences from the reference days. Cold season: degrees of daily mean (and of daily
    minimum). Warm season: degrees of daily maximum above 24 (and of daily minimum above 12),
    because heat acts above a threshold. Standard errors clustered by site and by date. Per
    model: the coefficients, R2, and the terms for a day inside an event and for the third
    day on of an event, which are zero if the temperatures explain the event."""
    import statsmodels.api as sm

    p = lag_panel(inp)
    sets = {
        "heat": ("heatwave", WARM_MONTHS, "d_hot", "d_warm_night"),
        "cold": ("cold spell", COLD_MONTHS, "d_t_mean", "d_t_min"),
    }
    rows = []
    for name, (kind, months, main, other) in sets.items():
        d = p[p["date"].dt.month.isin(months)].copy()
        d.loc[d["type"] != kind, ["late", "in_event"]] = 0.0
        cols_all = [f"{main}_{k}" for k in LAGS] + [f"{other}_{k}" for k in LAGS]
        d = d.dropna(subset=["y", *cols_all])
        d = d[d["y"].abs() < 150]  # meter changes
        models = {
            "the day's temperature": [f"{main}_0"],
            "the day and three days before": [f"{main}_{k}" for k in LAGS],
            "night temperature, day and three days before": [f"{other}_{k}" for k in LAGS],
            "both": cols_all,
        }
        groups = np.column_stack([pd.factorize(d["site_code"])[0], pd.factorize(d["date"])[0]])
        for label, cols in models.items():
            for with_late in (False, True):
                x = d[cols + (["in_event", "late"] if with_late else [])]
                fit = sm.OLS(d["y"], sm.add_constant(x)).fit(
                    cov_type="cluster", cov_kwds={"groups": groups}
                )
                ci = fit.conf_int()
                for term in x.columns:
                    rows.append(
                        {
                            "season": name,
                            "model": label,
                            "event_terms": with_late,
                            "term": term,
                            "coef": float(fit.params[term]),
                            "lo": float(ci.loc[term, 0]),
                            "hi": float(ci.loc[term, 1]),
                            "r2": float(fit.rsquared),
                            "site_days": int(fit.nobs),
                            "sites": d["site_code"].nunique(),
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 5. What the plant does
# --------------------------------------------------------------------------- #


def plant_decomposition(inp: Inputs, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the reaction of the import at the sites with on-site generation.

    Plant output is the plant's rated electrical power times its capacity factor from the
    estates return (a yearly mean; the meters do not show it); sites whose plant supplies less
    than a tenth of their demand are left out. Demand on the reference days is import plus
    that output. If the site's demand reacted like the sites without a plant and the plant did
    not change, the import would rise by the demand's rise in kilowatts. The rest of the
    observed change in import is a change in plant output. Two assumptions for the demand
    reaction: that of all sites without a plant, and that of the acute hospitals without a
    plant (the sites with a plant are acute hospitals, but only a few acute hospitals have no
    plant). Returns the sites and the pooled result, with intervals from resampling sites and
    the range when one site at a time is left out."""
    _, ev, _, sites, _, _ = prepare(inp, Spec())
    sr = ee.site_reactions(ev, sites)
    chp = pd.read_csv(ee.CHP_FILE)
    chp = chp[chp["plausible"] & chp["chp_elec_kw"].notna()]
    chp["output_kw"] = chp["chp_elec_kw"] * chp["capacity_factor"]
    rows = []
    for kind in ee.TYPES:
        x = sr[sr["type"] == kind]
        plain = x[x["chp"] == PLAIN]
        assumed = {
            "like all sites without a plant": float(plain["reaction_pct"].mean()),
            "like the acute sites without a plant": float(
                plain.loc[plain["use"] == ACUTE, "reaction_pct"].mean()
            ),
        }
        per = ee._ratios(ev[ev["type"] == kind], ["site_code"])
        p = x[x["chp"] == PLANT].merge(chp[["site_code", "output_kw", "heat_led"]], on="site_code")
        for s in p.itertuples():
            ref_import = per.loc[s.site_code, "ref"] / per.loc[s.site_code, "n"]
            d_import = per.loc[s.site_code, "kw"] / per.loc[s.site_code, "n"] - ref_import
            demand = ref_import + s.output_kw
            if s.output_kw / demand < MIN_PLANT_SHARE:
                continue
            for name, r_d in assumed.items():
                d_demand = demand * (np.exp(r_d / 100) - 1)
                rows.append(
                    {
                        "type": kind,
                        "assumption": name,
                        "demand_reaction_assumed_pct": r_d,
                        "site_code": s.site_code,
                        "heat_led": bool(s.heat_led),
                        "import_kw": ref_import,
                        "plant_output_kw": s.output_kw,
                        "demand_kw": demand,
                        "plant_share_of_demand": s.output_kw / demand,
                        "import_change_kw": d_import,
                        "import_reaction_pct": s.reaction_pct,
                        "demand_change_assumed_kw": d_demand,
                        "import_reaction_if_plant_unchanged_pct": 100
                        * np.log(1 + d_demand / ref_import),
                        "implied_output_change_kw": d_demand - d_import,
                        "implied_output_change_pct": 100 * (d_demand - d_import) / s.output_kw,
                    }
                )
    per_site = pd.DataFrame(rows)

    def stats(x):
        return (
            100 * x["implied_output_change_kw"].sum() / x["plant_output_kw"].sum(),
            100 * np.log(1 + x["import_change_kw"].sum() / x["import_kw"].sum()),
            100 * np.log(1 + x["demand_change_assumed_kw"].sum() / x["import_kw"].sum()),
        )

    rng = np.random.default_rng(seed)
    pooled = []
    for (kind, name), g0 in per_site.groupby(["type", "assumption"]):
        sets = [("all sites with a plant", g0)]
        sets += [
            ("plant follows heat" if led else "other plants", g)
            for led, g in g0.groupby("heat_led")
        ]
        for label, g in sets:
            obs = stats(g)
            boot = np.array([stats(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(2000)])
            loo = [stats(g.drop(index=i)) for i in g.index] if len(g) > 2 else [obs]
            pooled.append(
                {
                    "type": kind,
                    "assumption": name,
                    "demand_reaction_assumed_pct": float(g["demand_reaction_assumed_pct"].iloc[0]),
                    "sites": label,
                    "n": len(g),
                    "plant_share_of_demand": float(
                        g["plant_output_kw"].sum() / g["demand_kw"].sum()
                    ),
                    "import_reaction_summed_pct": obs[1],
                    "import_reaction_without_one_site_min_pct": min(v[1] for v in loo),
                    "import_reaction_without_one_site_max_pct": max(v[1] for v in loo),
                    "import_reaction_if_plant_unchanged_pct": obs[2],
                    "implied_output_change_pct": obs[0],
                    "lo": float(np.quantile(boot[:, 0], 0.025)),
                    "hi": float(np.quantile(boot[:, 0], 0.975)),
                    "without_one_site_min_pct": min(v[0] for v in loo),
                    "without_one_site_max_pct": max(v[0] for v in loo),
                    "sites_with_lower_output": int((g["implied_output_change_kw"] < 0).sum()),
                }
            )
    return per_site, pd.DataFrame(pooled)


# --------------------------------------------------------------------------- #
# 6. Which site effects are real? Placebo tests, false discovery rate, shrinkage
# --------------------------------------------------------------------------- #

NULL_DRAWS = 2000
SEASON_MONTHS = {"heatwave": (5, 6, 7, 8, 9), "cold spell": (11, 12, 1, 2, 3)}


def _block_null(kw: np.ndarray, ref: np.ndarray, lengths: list[int], rng) -> np.ndarray:
    """Reactions (per cent) of random sets of blocks of ordinary days with the lengths of the
    site's events: what the measure gives when there is no event."""
    n = len(kw)
    out = np.full(NULL_DRAWS, np.nan)
    if n < 3 * max(lengths) or n < 30:
        return out
    ckw, cref = np.r_[0.0, np.cumsum(kw)], np.r_[0.0, np.cumsum(ref)]
    num, den = np.zeros(NULL_DRAWS), np.zeros(NULL_DRAWS)
    for length in lengths:
        start = rng.integers(0, n - length + 1, NULL_DRAWS)
        num += ckw[start + length] - ckw[start]
        den += cref[start + length] - cref[start]
    return 100 * np.log(num / den)


def _bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjusted p-values."""
    order = p.sort_values().index
    m = len(order)
    q = (p.loc[order] * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1.0)
    return q.reindex(p.index)


def site_effect_tests(inp: Inputs, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Is the reaction of a site different from zero, and how much do the sites truly differ?

    For each site and event type the null distribution comes from the site itself: random
    blocks of its ordinary days in the event's season, with the lengths of its events, scored
    with the same measure. Its standard deviation is the standard error of the site's
    reaction, its centre the bias. From these: a two-sided placebo p-value, the
    Benjamini-Hochberg adjustment across sites, a random-effects decomposition (the true
    between-site standard deviation tau and the share of the observed variance that is real)
    and the empirical-Bayes estimate of each site, shrunk towards the mean of its group (with
    and without on-site generation)."""
    r, ev, _, sites, _, _ = prepare(inp, Spec())
    sr = ee.site_reactions(ev, sites)
    sr = sr[sr["clean"]]
    rng = np.random.default_rng(seed)
    ordinary = r[(r["type"] == "none") & r["kw"].notna() & r["ref_kw"].notna()]
    rows = []
    for kind in ee.TYPES:
        season = ordinary[ordinary["date"].dt.month.isin(SEASON_MONTHS[kind])]
        by_site = {s: g.sort_values("date") for s, g in season.groupby("site_code")}
        runs = ev[ev["type"] == kind].groupby(["site_code", "run"]).size()
        for x in sr[sr["type"] == kind].itertuples():
            g = by_site.get(x.site_code)
            if g is None:
                continue
            null = _block_null(
                g["kw"].to_numpy(), g["ref_kw"].to_numpy(), runs.loc[x.site_code].tolist(), rng
            )
            null = null[np.isfinite(null)]
            if len(null) < 100:
                continue
            centre, se = float(np.mean(null)), float(np.std(null, ddof=1))
            effect = x.reaction_pct - centre
            p = (1 + int((np.abs(null - centre) >= abs(effect)).sum())) / (len(null) + 1)
            rows.append(
                {
                    "type": kind,
                    "site_code": x.site_code,
                    "chp": x.chp,
                    "mean_kw": x.mean_kw,
                    "events": x.events,
                    "event_days": x.event_days,
                    "reaction_pct": x.reaction_pct,
                    "placebo_centre_pct": centre,
                    "effect_pct": effect,
                    "se_pct": se,
                    "lo": effect - 1.96 * se,
                    "hi": effect + 1.96 * se,
                    "p": p,
                }
            )
    t = pd.DataFrame(rows)
    t["q"] = t.groupby("type", group_keys=False)["p"].apply(_bh)
    parts, summary = [], []
    for (kind, chp), g in t.groupby(["type", "chp"]):
        w = 1 / g["se_pct"] ** 2
        mu = float((w * g["effect_pct"]).sum() / w.sum())
        qstat = float((w * (g["effect_pct"] - mu) ** 2).sum())
        c = float(w.sum() - (w**2).sum() / w.sum())
        tau2 = max(0.0, (qstat - (len(g) - 1)) / c)
        wr = 1 / (g["se_pct"] ** 2 + tau2)
        mean = float((wr * g["effect_pct"]).sum() / wr.sum())
        b = tau2 / (tau2 + g["se_pct"] ** 2)
        parts.append(
            g.assign(shrunk_pct=b * g["effect_pct"] + (1 - b) * mean, weight_on_own_data=b)
        )
        summary.append(
            {
                "type": kind,
                "sites_group": chp,
                "sites": len(g),
                "mean_effect_pct": mean,
                "mean_se_pct": float(np.sqrt(1 / wr.sum())),
                "true_sd_between_sites_pct": float(np.sqrt(tau2)),
                "share_of_variance_real": max(0.0, (qstat - (len(g) - 1)) / qstat)
                if qstat
                else 0.0,
                "median_site_se_pct": float(g["se_pct"].median()),
                "rise_p05": int(((g["p"] < 0.05) & (g["effect_pct"] > 0)).sum()),
                "fall_p05": int(((g["p"] < 0.05) & (g["effect_pct"] < 0)).sum()),
                "rise_q05": int(((g["q"] < 0.05) & (g["effect_pct"] > 0)).sum()),
                "fall_q05": int(((g["q"] < 0.05) & (g["effect_pct"] < 0)).sum()),
                "expected_false_at_p05": 0.05 * len(g),
            }
        )
    return pd.concat(parts, ignore_index=True), pd.DataFrame(summary)


# --------------------------------------------------------------------------- #
# 7. Road traffic near the site
# --------------------------------------------------------------------------- #

TRAFFIC_FILE = (
    PROJECT_ROOT
    / "code/data/external/sense_del/sense_del/traffic"
    / "energy_systems_catapult.road_traffic.dft_traffic_counts_aadf.parquet"
)
TRAFFIC_RADIUS_KM = 1.0


def site_traffic(sites: pd.DataFrame) -> pd.DataFrame:
    """Road traffic near each site from the Department for Transport count points: the
    annual average daily flow of motor vehicles (latest year of each count point from 2019
    on) at the busiest count point within one kilometre, and the number of count points
    there. The counts are yearly averages: they describe the place, not a day."""
    a = pd.read_parquet(
        TRAFFIC_FILE,
        columns=["count_point_id", "year", "latitude", "longitude", "all_motor_vehicles"],
    )
    a = a[a["year"] >= 2019].sort_values("year").drop_duplicates("count_point_id", keep="last")
    lat, lon = np.deg2rad(a["latitude"].to_numpy()), np.deg2rad(a["longitude"].to_numpy())
    flow = a["all_motor_vehicles"].to_numpy(dtype=float)
    rows = []
    for s in sites.dropna(subset=["latitude", "longitude"]).itertuples():
        la, lo = np.deg2rad(s.latitude), np.deg2rad(s.longitude)
        d = 6371.0 * np.hypot(lat - la, (lon - lo) * np.cos((lat + la) / 2))
        near = d <= TRAFFIC_RADIUS_KM
        rows.append(
            {
                "site_code": s.site_code,
                "count_points_1km": int(near.sum()),
                "busiest_flow_1km": float(flow[near].max()) if near.any() else np.nan,
                "nearest_count_point_km": float(d.min()),
            }
        )
    out = pd.DataFrame(rows)
    out["log_traffic"] = np.log(out["busiest_flow_1km"])
    return out


def traffic_moderator(inp: Inputs) -> pd.DataFrame:
    """Does road traffic near the site go with its reaction? The site reaction on the log of
    the traffic flow, with the size of the event held fixed (sites without on-site
    generation)."""
    if not TRAFFIC_FILE.exists():
        return pd.DataFrame()
    _, ev, _, sites, _, _ = prepare(inp, Spec())
    sr = ee.site_reactions(ev, sites).merge(site_traffic(sites), on="site_code", how="left")
    rows = []
    for kind, x in sr.groupby("type"):
        x = x[x["clean"] & (x["chp"] == PLAIN)].dropna(subset=["reaction_pct", "degrees"])
        got = ee._fit_descriptor(x, "log_traffic", "num")
        if got is None:
            continue
        d, fit, _, joint, gain = got
        ci = fit.conf_int().loc["z"]
        rows.append(
            {
                "type": kind,
                "sites": len(d),
                "sites_without_count_point": int(x["log_traffic"].isna().sum()),
                "median_flow": float(np.exp(d["log_traffic"]).median()),
                "points_per_sd": float(fit.params["z"]),
                "lo": float(ci[0]),
                "hi": float(ci[1]),
                "p": joint,
                "adj_r2_gain": gain,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run(grid: bool = True) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    summary: dict = {}
    if grid:
        g = specification_grid(inp)
        g.to_csv(OUT_DIR / "specification_grid.csv", index=False)
        plc = placebo(inp)
        plc.to_csv(OUT_DIR / "placebo.csv", index=False)
    else:
        g = pd.read_csv(OUT_DIR / "specification_grid.csv")
        plc = pd.read_csv(OUT_DIR / "placebo.csv")
    gs = grid_summary(g, plc)
    gs.to_csv(OUT_DIR / "grid_summary.csv", index=False)
    scores, curve, table = prediction_scores(inp)
    scores.to_csv(OUT_DIR / "prediction_scores.csv", index=False)
    curve.to_csv(OUT_DIR / "prediction_curve.csv", index=False)
    table.to_csv(OUT_DIR / "site_events.csv", index=False)
    lags = lag_models(inp)
    lags.to_csv(OUT_DIR / "lag_models.csv", index=False)
    per_site, pooled = plant_decomposition(inp)
    per_site.to_csv(OUT_DIR / "plant_decomposition_sites.csv", index=False)
    traffic = traffic_moderator(inp)
    if len(traffic):
        traffic.to_csv(OUT_DIR / "traffic_moderator.csv", index=False)
    tests, spread = site_effect_tests(inp)
    tests.to_csv(OUT_DIR / "site_effect_tests.csv", index=False)
    spread.to_csv(OUT_DIR / "site_effect_spread.csv", index=False)
    pooled.to_csv(OUT_DIR / "plant_decomposition.csv", index=False)
    summary = {
        "specifications": int(
            g[["weeks", "events", "sample", "holidays_out"]].drop_duplicates().shape[0]
        ),
        "placebo_runs": len(PLACEBO_SHIFTS),
        "grid": gs.round(3).to_dict(orient="records"),
        "prediction": scores.round(3).to_dict(orient="records"),
        "plant": pooled.round(2).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return {"summary": summary}
