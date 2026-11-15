"""Heat waves and cold spells at the sites: the reaction, where and when, and what explains it.

The event catalogue of ``weather_events`` (Met Office heat-wave definition, UKHSA cold-alert
criterion, per site) gives the days. This module measures what each site's electricity does on
them and sets it against what is known about the site.

* **Reaction.** Each site-day gets a reference: the median demand of the same weekday one to
  three weeks either side, leaving out days that are themselves in a heat wave or cold spell at
  the site, bank holidays and the Christmas fortnight. The reaction of a set of days is the
  demand summed over them against the summed reference, in per cent (log ratio x 100); sums,
  because the grid import of a site with a plant can be near zero on single days. The same
  difference for the temperature gives the size of the event at the site in degrees.
* **Time.** Per national event; by day from the onset at the site; by half-hour of the day; by
  season (the reaction per degree in each summer and winter of the record).
* **Space.** Per site; Moran's I of the site reactions and of the site temperature differences
  (nearest neighbours, permutation test); the share of the between-site variance that the
  region accounts for.
* **Site information.** On-site generation (CHP), the Display Energy Certificate (building
  environment, cooled floor share, electric heating share, operational rating), the estates
  return (use, gas-to-electricity ratio, heat pumps, build year, size): one regression per
  descriptor with the size of the event at the site held fixed, and all together.

For a site with CHP the meter records grid import, so its reaction is that of the import: the
building and the plant together.

Outputs: ``code/reports/extreme_events/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "extreme_events"
EVENTS_DIR = PROJECT_ROOT / "code" / "reports" / "weather_events"
CHP_FILE = PROJECT_ROOT / "code" / "reports" / "chp_dispatch" / "classification.csv"
TZ = "Europe/London"
TYPES = ("heatwave", "cold spell")
TEMP_COL = {"heatwave": "t_max", "cold spell": "t_mean"}
REF_WEEKS = (-3, -2, -1, 1, 2, 3)
MIN_REFS = 2
MIN_PERIODS = 44
EVENT_WINDOW = (-7, 14)  # days from the onset at the site
MIN_EVENT_DAYS = 3  # a site needs this many scored event days to get a reaction
BAD_COVERAGE = ("meter_gaps", "partial_supply", "aggregate_supply")
NEIGHBOURS = 8
N_PERM = 999
USE_GROUPS = {
    "acute inpatient": ("General acute hospital", "Specialist hospital (acute only)"),
    "community inpatient": ("Community hospital (with inpatient beds)",),
    "mental health": (
        "Mental Health (including Specialist services)",
        "Mental Health and Learning Disabilities",
        "Learning Disabilities",
    ),
}
CONTINUOUS = {
    "dec_cooled_share": "certificate: cooled floor share",
    "dec_ac_share": "certificate: floor share with any air conditioning",
    "dec_electric_heating_share": "certificate: floor share heated with electricity",
    "dec_operational_rating": "certificate: operational rating",
    "dec_elec_kwh_m2": "certificate: electricity per m2",
    "gas_to_elec": "estates return: gas over electricity",
    "mean_build_year": "estates return: mean build year",
    "log_mean_kw": "site size (log of mean demand)",
}
CATEGORICAL = {
    "chp": "on-site generation (CHP)",
    "use": "building use",
    "dec_environment": "certificate: building environment",
    "heat_pumps_any": "estates return: heat pumps",
}


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def _half_hourly() -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
    w.index = pd.to_datetime(w.index, utc=True).tz_convert(TZ)
    return w


def daily_demand(w: pd.DataFrame) -> pd.DataFrame:
    """Mean kW per site and local day; days with fewer than 44 observed half-hours are NaN."""
    day = w.index.tz_localize(None).normalize()
    n = w.notna().groupby(day).sum()
    return w.groupby(day).mean().where(n >= MIN_PERIODS)


def slot_cube(w: pd.DataFrame, days: pd.DatetimeIndex) -> np.ndarray:
    """kW by (day, half-hour of the local day, site); the repeated hour of the autumn clock
    change is averaged, the missing hour in spring stays missing."""
    day = w.index.tz_localize(None).normalize()
    slot = w.index.hour * 2 + w.index.minute // 30
    g = w.groupby([day, slot]).mean()
    full = pd.MultiIndex.from_product([days, range(48)])
    return g.reindex(full).to_numpy().reshape(len(days), 48, w.shape[1])


def excluded_days(days: pd.DatetimeIndex) -> np.ndarray:
    """Bank holidays in England and the Christmas fortnight (21 December to 3 January)."""
    import holidays

    hol = holidays.UK(subdiv="ENG", years=range(days.year.min(), days.year.max() + 1))
    md = days.month * 100 + days.day
    return np.array([d.date() in hol for d in days]) | (md >= 1221) | (md <= 103)


def event_masks(days: pd.DatetimeIndex, sites: list[str]) -> dict[str, pd.DataFrame]:
    """Per event type: True where the site-day is inside a site event (day x site)."""
    ev = pd.read_csv(EVENTS_DIR / "site_events.csv", parse_dates=["start", "end"])
    out = {}
    for kind in TYPES:
        m = pd.DataFrame(False, index=days, columns=sites)
        for r in ev[(ev["type"] == kind) & ev["site_code"].isin(sites)].itertuples():
            m.loc[r.start : r.end, r.site_code] = True
        out[kind] = m
    return out


def national_events() -> pd.DataFrame:
    nat = pd.read_csv(EVENTS_DIR / "national_events.csv", parse_dates=["start", "end"])
    nat = nat[nat["type"].isin(TYPES) & nat["in_electricity_record"]]
    return nat[["event_id", "type", "start", "end", "days", "sites", "peak_value"]].reset_index(
        drop=True
    )


def site_table(kw: pd.DataFrame) -> pd.DataFrame:
    """One row per site with electricity: place, plant, certificate and estates descriptors."""
    from ..eda import common as C

    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").drop_duplicates("site_code")
    sites = sites.set_index("site_code")
    eric = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").drop_duplicates("site_code")
    eric = eric.set_index("site_code")
    dec = pd.read_parquet(INTERIM_DIR / "dec_sites.parquet").set_index("site_code")
    flags = pd.read_parquet(INTERIM_DIR / "site_coverage_flags.parquet").set_index("site_code")
    chp = pd.read_csv(CHP_FILE).set_index("site_code")
    geo = C.site_points().drop_duplicates("site_code").set_index("site_code")

    out = pd.DataFrame(index=pd.Index(kw.columns, name="site_code"))
    out["mean_kw"] = kw.mean()
    out["log_mean_kw"] = np.log(out["mean_kw"])
    out["latitude"] = geo["latitude"].reindex(out.index)
    out["longitude"] = geo["longitude"].reindex(out.index)
    out["region"] = sites["commissioning_region"].reindex(out.index)
    out["trust"] = sites["organisation_name"].reindex(out.index)
    out["coverage_flag"] = flags["coverage_flag"].reindex(out.index)
    out["clean"] = ~out["coverage_flag"].isin(BAD_COVERAGE)
    out["chp"] = np.where(out.index.isin(chp.index), "on-site generation", "no on-site generation")
    out["chp_share"] = chp["chp_share"].reindex(out.index)
    out["chp_heat_led"] = chp["heat_led"].reindex(out.index)
    use = eric["site_type"].reindex(out.index)
    lookup = {name: group for group, names in USE_GROUPS.items() for name in names}
    out["use"] = use.map(lookup).where(use.notna()).fillna("non-inpatient").where(use.notna())
    out["gas_to_elec"] = eric["gas_to_elec"].reindex(out.index)
    out["mean_build_year"] = eric["mean_build_year"].reindex(out.index)
    hp = eric["heat_pumps"].reindex(out.index)
    out["heat_pumps_any"] = np.where(hp > 0, "heat pumps", "no heat pumps")
    out.loc[hp.isna(), "heat_pumps_any"] = None
    for c in (
        "dec_cooled_share",
        "dec_ac_share",
        "dec_electric_heating_share",
        "dec_operational_rating",
        "dec_elec_kwh_m2",
        "dec_buildings",
    ):
        out[c] = dec[c].reindex(out.index)
    env = dec["dec_main_environment"].reindex(out.index)
    out["dec_environment"] = env.replace(
        {
            "heating, natural ventilation": "natural ventilation",
            "heating, mechanical ventilation": "mechanical ventilation",
            "mixed mode": "mixed mode or air conditioning",
            "air conditioning": "mixed mode or air conditioning",
        }
    )
    return out.reset_index()


# --------------------------------------------------------------------------- #
# Reaction
# --------------------------------------------------------------------------- #


def reference(x: np.ndarray, usable: np.ndarray) -> np.ndarray:
    """Median over the reference days (same weekday, one to three weeks either side) of an
    array whose first axis is the day; ``usable`` (day x site) marks the days that can serve as
    a reference. Fewer than ``MIN_REFS`` references give NaN."""
    stack = []
    for k in REF_WEEKS:
        shifted = np.full(x.shape, np.nan)
        ok = np.zeros(usable.shape, dtype=bool)
        s = 7 * k
        if s > 0:
            shifted[s:], ok[s:] = x[:-s], usable[:-s]
        else:
            shifted[:s], ok[:s] = x[-s:], usable[-s:]
        okx = ok if x.ndim == 2 else ok[:, None, :]
        stack.append(np.where(okx, shifted, np.nan))
    stack = np.stack(stack)
    n = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(stack, axis=0)
    return np.where(n >= MIN_REFS, med, np.nan)


def reactions(
    kw: pd.DataFrame, weather: pd.DataFrame, masks: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """Per site-day: demand, its reference, the temperature differences from the reference
    days, and the event type the day is in (or none). Excluded days carry no reference."""
    days, sites = kw.index, list(kw.columns)
    skip = excluded_days(days)
    extreme = np.zeros(kw.shape, dtype=bool)
    for m in masks.values():
        extreme |= m.to_numpy()
    usable = ~extreme & ~skip[:, None]
    y = kw.to_numpy()
    ref = reference(y, usable & np.isfinite(y))
    ref[skip] = np.nan
    ref[~(ref > 0)] = np.nan
    out = pd.DataFrame(
        {
            "date": np.repeat(days, len(sites)),
            "site_code": np.tile(sites, len(days)),
            "kw": y.ravel(),
            "ref_kw": ref.ravel(),
        }
    )
    for col in sorted(set(TEMP_COL.values())):
        t = weather.pivot(index="date", columns="site_code", values=col).reindex(
            index=days, columns=sites
        )
        t = t.to_numpy()
        out[col] = t.ravel()
        out[f"d_{col}"] = (t - reference(t, usable & np.isfinite(t))).ravel()
    out["type"] = "none"
    for kind, m in masks.items():
        out.loc[m.to_numpy().ravel(), "type"] = kind
    return out


def add_national(r: pd.DataFrame, nat: pd.DataFrame) -> pd.DataFrame:
    """The national event a site-day inside a site event belongs to (or None)."""
    r = r.copy()
    r["event_id"] = None
    for e in nat.itertuples():
        hit = (r["type"] == e.type) & r["date"].between(e.start, e.end)
        r.loc[hit, "event_id"] = e.event_id
    return r


def event_days(r: pd.DataFrame) -> pd.DataFrame:
    """The scored site-days inside site events, with the size of the event in degrees
    (positive for the extreme: warmer daily maximum in a heat wave, colder daily mean in a
    cold spell) and the number of the site's event the day belongs to."""
    ev = r[(r["type"] != "none") & r["kw"].notna() & r["ref_kw"].notna()].copy()
    ev = ev.sort_values(["site_code", "type", "date"])
    ev["degrees"] = np.where(ev["type"] == "heatwave", ev["d_t_max"], -ev["d_t_mean"])
    new = ev.groupby(["site_code", "type"])["date"].diff().dt.days.ne(1)
    ev["run"] = new.groupby([ev["site_code"], ev["type"]]).cumsum()
    return ev


def ratio_pct(kw, ref) -> float:
    """Summed demand over summed reference, in per cent."""
    kw, ref = np.asarray(kw, dtype=float), np.asarray(ref, dtype=float)
    ok = np.isfinite(kw) & np.isfinite(ref)
    if not ok.any() or ref[ok].sum() <= 0 or kw[ok].sum() <= 0:
        return np.nan
    return float(100 * np.log(kw[ok].sum() / ref[ok].sum()))


def _ratios(x: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    g = x.groupby(by, observed=True)
    out = g.agg(kw=("kw", "sum"), ref=("ref_kw", "sum"), n=("kw", "size"), d=("degrees", "mean"))
    out["pct"] = 100 * np.log(out["kw"] / out["ref"])
    return out


def _mean_ci(x: pd.Series) -> tuple[float, float, float]:
    from scipy import stats

    x = pd.Series(x).replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) < 3:
        return (float(x.mean()) if len(x) else np.nan), np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean()), float(x.mean() - h), float(x.mean() + h)


def site_reactions(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per site and event type: the reaction over all its event days, the standard error from
    its events (each event is one observation) and the mean size of the events in degrees."""
    out = _ratios(ev, ["site_code", "type"]).rename(
        columns={"pct": "reaction_pct", "n": "event_days", "d": "degrees"}
    )
    per_event = _ratios(ev, ["site_code", "type", "run"])["pct"].groupby(level=[0, 1])
    out["events"] = per_event.size()
    out["se_pct"] = per_event.std(ddof=1) / np.sqrt(per_event.size())
    out["extra_kw"] = (out["kw"] - out["ref"]) / out["event_days"]
    out = out.drop(columns=["kw", "ref"]).reset_index()
    out = out[out["event_days"] >= MIN_EVENT_DAYS]
    return out.merge(sites, on="site_code", how="left")


def event_table(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Per national event and site group: sites and site-days, the mean over sites of the
    site's reaction (95 % interval across sites), the median, the reaction of the summed
    demand of the group (what the grid sees) and the size of the event in degrees."""
    x = ev[ev["event_id"].notna()].merge(sites[["site_code", "chp"]], on="site_code")
    rows = []
    for group, g in (("all sites", x), *x.groupby("chp")):
        per = _ratios(g, ["event_id", "type", "site_code"])
        for (event_id, kind), p in per.groupby(level=[0, 1]):
            mean, lo, hi = _mean_ci(p["pct"])
            rows.append(
                {
                    "event_id": event_id,
                    "type": kind,
                    "group": group,
                    "sites": len(p),
                    "site_days": int(p["n"].sum()),
                    "reaction_pct": mean,
                    "lo": lo,
                    "hi": hi,
                    "median_pct": float(p["pct"].median()),
                    "summed_pct": float(100 * np.log(p["kw"].sum() / p["ref"].sum())),
                    "extra_mw": float((p["kw"].sum() - p["ref"].sum()) / p["n"].sum() * len(p))
                    / 1000.0,
                    "degrees": float(p["d"].mean()),
                }
            )
    return pd.DataFrame(rows)


def compact_events(et: pd.DataFrame) -> pd.DataFrame:
    """One row per national event: the mean site reaction without and with on-site generation
    and the reaction of the summed demand of each group."""
    g = {name: x.set_index("event_id") for name, x in et.groupby("group")}
    a, n = g["all sites"], g["no on-site generation"]
    c = g["on-site generation"].reindex(a.index)
    return pd.DataFrame(
        {
            "type": a["type"],
            "start": a["start"],
            "days": a["days"],
            "degrees": a["degrees"],
            "sites_no_plant": n["sites"],
            "no_plant_pct": n["reaction_pct"],
            "no_plant_lo": n["lo"],
            "no_plant_hi": n["hi"],
            "sites_plant": c["sites"],
            "plant_pct": c["reaction_pct"],
            "summed_no_plant_pct": n["summed_pct"],
            "summed_plant_pct": c["summed_pct"],
        }
    ).reset_index()


def _wide(r: pd.DataFrame, col: str) -> pd.DataFrame:
    return r.pivot(index="date", columns="site_code", values=col)


def event_time(r: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Reaction and temperature difference by day from the onset of the event at the site (day
    0 is the first day of a site event inside a national event): per site the summed demand
    over the summed reference of that day across its events, then the mean over sites with a
    95 % interval, by site group. ``pct_per_degree`` is the mean reaction over the mean
    temperature difference."""
    kw, ref = _wide(r, "kw").to_numpy(), _wide(r, "ref_kw").to_numpy()
    kinds = _wide(r, "type")
    cols, days = kinds.columns, kinds.index
    national = _wide(r.assign(v=r["event_id"].notna()), "v").to_numpy().astype(bool)
    chp = sites.set_index("site_code")["chp"].reindex(cols)
    clean = sites.set_index("site_code")["clean"].reindex(cols).to_numpy()
    rows = []
    for kind in TYPES:
        sign = 1.0 if kind == "heatwave" else -1.0
        temp = sign * _wide(r, f"d_{TEMP_COL[kind]}").to_numpy()
        inside = kinds.eq(kind).to_numpy()
        onset = inside & ~np.vstack([np.zeros((1, len(cols)), bool), inside[:-1]]) & national
        for k in range(EVENT_WINDOW[0], EVENT_WINDOW[1] + 1):
            src = np.arange(len(days)) + k
            ok = (src >= 0) & (src < len(days))
            a, b, t = (np.full(kw.shape, np.nan) for _ in range(3))
            a[ok], b[ok], t[ok] = kw[src[ok]], ref[src[ok]], temp[src[ok]]
            use = onset & np.isfinite(a) & np.isfinite(b)
            num, den = np.where(use, a, 0).sum(axis=0), np.where(use, b, 0).sum(axis=0)
            with np.errstate(all="ignore"):
                pct = pd.Series(100 * np.log(num / den), index=cols)
                deg = pd.Series(
                    np.where(use, t, 0).sum(axis=0) / np.maximum(use.sum(axis=0), 1), index=cols
                )
            pct[(use.sum(axis=0) == 0) | ~clean] = np.nan
            for group in ("all sites", *sorted(chp.dropna().unique())):
                keep = cols if group == "all sites" else chp.index[chp == group]
                mean, lo, hi = _mean_ci(pct.reindex(keep))
                d = float(deg.reindex(keep)[pct.reindex(keep).notna()].mean())
                rows.append(
                    {
                        "type": kind,
                        "group": group,
                        "day": k,
                        "sites": int(pct.reindex(keep).notna().sum()),
                        "reaction_pct": mean,
                        "lo": lo,
                        "hi": hi,
                        "degrees": d,
                        "pct_per_degree": mean / d if abs(d) > 1 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def build_up(ev: pd.DataFrame, sites: pd.DataFrame) -> pd.DataFrame:
    """Does the reaction per degree grow inside an event? Per site the reaction per degree on
    the first day of its events and from the third day on; the paired difference over the
    sites that have both, with a 95 % interval (sites without on-site generation)."""
    keep = sites.loc[sites["clean"] & (sites["chp"] == "no on-site generation"), "site_code"]
    x = ev[ev["site_code"].isin(keep)].copy()
    x["pos"] = x.groupby(["site_code", "type", "run"]).cumcount()
    x["part"] = np.select([x["pos"] == 0, x["pos"] >= 2], ["first day", "third day on"], "")
    x = x[x["part"] != ""]
    per = _ratios(x, ["type", "site_code", "part"])
    per = per[(per["d"] > 2) & (per["n"] >= 2)]
    per["per_degree"] = per["pct"] / per["d"]
    wide = per["per_degree"].unstack("part").dropna()
    rows = []
    for kind, w in wide.groupby(level=0):
        first, later = _mean_ci(w["first day"]), _mean_ci(w["third day on"])
        diff = _mean_ci(w["third day on"] - w["first day"])
        rows.append(
            {
                "type": kind,
                "sites": len(w),
                "first_day_pct_per_degree": first[0],
                "third_day_on_pct_per_degree": later[0],
                "difference": diff[0],
                "lo": diff[1],
                "hi": diff[2],
            }
        )
    return pd.DataFrame(rows)


def hour_profile(
    cube: np.ndarray,
    days: pd.DatetimeIndex,
    site_codes: list[str],
    masks: dict[str, pd.DataFrame],
    sites: pd.DataFrame,
) -> pd.DataFrame:
    """Reaction by half-hour of the day: per site the demand of the half-hour summed over its
    event days against the same half-hour of the reference days; then the mean over sites with
    a 95 % interval and the reaction of the summed demand, by site group."""
    skip = excluded_days(days)
    extreme = np.zeros((len(days), len(site_codes)), dtype=bool)
    for m in masks.values():
        extreme |= m.to_numpy()
    usable = ~extreme & ~skip[:, None] & (np.isfinite(cube).sum(axis=1) >= MIN_PERIODS)
    ref = reference(cube, usable)
    info = sites.set_index("site_code").reindex(site_codes)
    chp, clean = info["chp"], info["clean"].to_numpy()
    rows = []
    for kind, m in masks.items():
        on = (m.to_numpy() & ~skip[:, None])[:, None, :] & np.isfinite(cube) & np.isfinite(ref)
        num, den = np.where(on, cube, 0).sum(axis=0), np.where(on, ref, 0).sum(axis=0)
        n = on.sum(axis=0)
        ok = (n >= MIN_EVENT_DAYS) & clean[None, :]
        with np.errstate(all="ignore"):
            prof = pd.DataFrame(np.where(ok, 100 * np.log(num / den), np.nan), columns=site_codes)
        num, den = (pd.DataFrame(np.where(ok, v, 0.0), columns=site_codes) for v in (num, den))
        for group in ("all sites", *sorted(chp.dropna().unique())):
            keep = site_codes if group == "all sites" else list(chp.index[chp == group])
            for slot in range(48):
                mean, lo, hi = _mean_ci(prof.loc[slot, keep])
                rows.append(
                    {
                        "type": kind,
                        "group": group,
                        "slot": slot,
                        "hour": slot / 2,
                        "sites": int(prof.loc[slot, keep].notna().sum()),
                        "reaction_pct": mean,
                        "lo": lo,
                        "hi": hi,
                        "median_pct": float(prof.loc[slot, keep].median()),
                        "summed_pct": float(
                            100 * np.log(num.loc[slot, keep].sum() / den.loc[slot, keep].sum())
                        ),
                    }
                )
    return pd.DataFrame(rows)


def seasons(ev: pd.DataFrame, sites: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Reaction per degree in each summer (heat waves) and winter (cold spells) of the record:
    the mean site reaction over the mean size of the events, with a 95 % interval from
    resampling sites (sites without on-site generation)."""
    keep = sites.loc[sites["clean"] & (sites["chp"] == "no on-site generation"), "site_code"]
    x = ev[ev["site_code"].isin(keep)].copy()
    year = x["date"].dt.year - (x["date"].dt.month <= 6) * (x["type"] == "cold spell")
    x["season"] = np.where(
        x["type"] == "heatwave",
        "summer " + year.astype(str),
        "winter " + year.astype(str) + "/" + ((year + 1) % 100).astype(str).str.zfill(2),
    )
    rng = np.random.default_rng(seed)
    rows = []
    for (kind, season), g in x.groupby(["type", "season"]):
        per = _ratios(g, ["site_code"])
        per = per[per["n"] >= MIN_EVENT_DAYS]
        if len(per) < 10:
            continue
        pct, deg = per["pct"].to_numpy(), per["d"].to_numpy()
        pick = rng.integers(0, len(per), (2000, len(per)))
        boot = pct[pick].mean(axis=1) / deg[pick].mean(axis=1)
        rows.append(
            {
                "type": kind,
                "season": season,
                "sites": len(per),
                "site_days": int(per["n"].sum()),
                "reaction_pct": float(pct.mean()),
                "degrees": float(deg.mean()),
                "pct_per_degree": float(pct.mean() / deg.mean()),
                "lo": float(np.quantile(boot, 0.025)),
                "hi": float(np.quantile(boot, 0.975)),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Space
# --------------------------------------------------------------------------- #


def morans_i(values: np.ndarray, lat: np.ndarray, lon: np.ndarray, seed: int = 0) -> dict:
    """Moran's I with row-standardised weights on the ``NEIGHBOURS`` nearest sites, and its
    permutation p-value (one-sided: more alike than chance)."""
    ok = np.isfinite(values) & np.isfinite(lat) & np.isfinite(lon)
    v, lat, lon = values[ok], np.deg2rad(lat[ok]), np.deg2rad(lon[ok])
    n = len(v)
    if n <= NEIGHBOURS + 2:
        return {"n": n, "morans_i": np.nan, "p": np.nan}
    dlat = lat[:, None] - lat[None, :]
    dlon = (lon[:, None] - lon[None, :]) * np.cos((lat[:, None] + lat[None, :]) / 2)
    dist = 6371.0 * np.hypot(dlat, dlon)
    np.fill_diagonal(dist, np.inf)
    w = np.zeros((n, n))
    nearest = np.argsort(dist, axis=1)[:, :NEIGHBOURS]
    w[np.arange(n)[:, None], nearest] = 1.0 / NEIGHBOURS

    def stat(z):
        z = z - z.mean()
        return float((z @ w @ z) / (z @ z))

    obs = stat(v)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(v)) for _ in range(N_PERM)])
    return {
        "n": n,
        "morans_i": obs,
        "p": float((1 + (perm >= obs).sum()) / (N_PERM + 1)),
        "neighbour_km": float(np.median(np.sort(dist, axis=1)[:, NEIGHBOURS - 1])),
    }


def region_share(values: pd.Series, region: pd.Series, seed: int = 0) -> dict:
    """Share of the between-site variance that the region means account for, with a
    permutation p-value."""
    ok = values.notna() & region.notna()
    v, g = values[ok].to_numpy(), region[ok].to_numpy()

    def eta(x):
        d = pd.DataFrame({"x": x, "g": g})
        between = (d.groupby("g")["x"].transform("mean") - x.mean()) ** 2
        return float(between.sum() / ((x - x.mean()) ** 2).sum())

    obs = eta(v)
    rng = np.random.default_rng(seed)
    perm = np.array([eta(rng.permutation(v)) for _ in range(N_PERM)])
    return {
        "n": len(v),
        "regions": len(set(g)),
        "share": obs,
        "chance_share": float(perm.mean()),
        "p": float((1 + (perm >= obs).sum()) / (N_PERM + 1)),
    }


def spatial(sr: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for kind, x in sr.groupby("type"):
        for label, sub in (
            ("all sites", x),
            ("no on-site generation", x[x["chp"] == "no on-site generation"]),
        ):
            for col, name in (("reaction_pct", "reaction"), ("degrees", "size of the event")):
                m = morans_i(
                    sub[col].to_numpy(), sub["latitude"].to_numpy(), sub["longitude"].to_numpy()
                )
                e = region_share(sub[col], sub["region"])
                trusts = sub["trust"].where(sub.groupby("trust")["trust"].transform("size") >= 2)
                t = region_share(sub[col], trusts)
                rows.append(
                    {
                        "type": kind,
                        "sites": label,
                        "quantity": name,
                        **m,
                        "region_share": e["share"],
                        "region_share_chance": e["chance_share"],
                        "region_p": e["p"],
                        "trust_sites": t["n"],
                        "trusts": t["regions"],
                        "trust_share": t["share"],
                        "trust_share_chance": t["chance_share"],
                        "trust_p": t["p"],
                    }
                )
    return pd.DataFrame(rows)


def reliability(ev: pd.DataFrame) -> pd.DataFrame:
    """How much of the between-site spread of the reaction belongs to the site: the correlation
    between a site's reaction in its odd and in its even events (in date order), stepped up to
    the full length (Spearman-Brown). ``other_events_r2`` is the share of the variance of the
    reaction in one half of the events that the other half explains (squared Pearson
    correlation): what a site's own earlier events say about its next ones."""
    x = ev.assign(half=ev["run"] % 2)
    rows = []
    for kind, g in x.groupby("type"):
        per = _ratios(g, ["site_code", "half"])
        per = per[per["n"] >= MIN_EVENT_DAYS]["pct"].unstack("half").dropna()
        rho = float(per[0].corr(per[1], method="spearman"))
        rows.append(
            {
                "type": kind,
                "sites": len(per),
                "split_half_r": rho,
                "reliability": 2 * rho / (1 + rho),
                "other_events_r2": float(per[0].corr(per[1]) ** 2),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Site information
# --------------------------------------------------------------------------- #


def _fit_descriptor(d: pd.DataFrame, col: str, typ: str):
    import statsmodels.formula.api as smf

    d = d.dropna(subset=[col]).copy()
    if typ == "cat":
        counts = d[col].value_counts()
        d = d[d[col].isin(counts[counts >= 3].index)]
        if d[col].nunique() < 2:
            return None
        rhs = f"C({col})"
    else:
        if len(d) < 15 or d[col].std() == 0:
            return None
        d["z"] = (d[col] - d[col].mean()) / d[col].std()
        rhs = "z"
    base = smf.ols("reaction_pct ~ degrees", d).fit()
    plain = smf.ols(f"reaction_pct ~ degrees + {rhs}", d).fit()
    fit = smf.ols(f"reaction_pct ~ degrees + {rhs}", d).fit(cov_type="HC3")
    names = [n for n in fit.params.index if n not in ("Intercept", "degrees")]
    joint = float(fit.f_test(", ".join(f"{n} = 0" for n in names)).pvalue)
    return d, fit, names, joint, float(plain.rsquared_adj - base.rsquared_adj)


def alignment(sr: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """What the descriptors of a site say about its reaction. One regression per descriptor of
    the site reaction on the descriptor, with the size of the event at the site held fixed
    (robust standard errors). On-site generation is tested on all sites; every other
    descriptor on the sites without on-site generation, so that the plant does not stand in
    for it. Categorical descriptors: the adjusted difference of each level from the first, and
    the joint p-value. Continuous: the change per standard deviation. Returns the terms, and
    per descriptor the sites, the joint p and the adjusted R2 gained over the size of the
    event alone."""
    terms, fits = [], []
    specs = [(c, "cat", lab) for c, lab in CATEGORICAL.items()]
    specs += [(c, "num", lab) for c, lab in CONTINUOUS.items()]
    for kind, x in sr.groupby("type"):
        x = x[x["clean"]].dropna(subset=["reaction_pct", "degrees"])
        plain = x[x["chp"] == "no on-site generation"]
        for col, typ, label in specs:
            sample = "all sites" if col == "chp" else "sites without on-site generation"
            got = _fit_descriptor(x if col == "chp" else plain, col, typ)
            if got is None:
                continue
            d, fit, names, joint, gain = got
            key = {"type": kind, "descriptor": label, "column": col, "sample": sample}
            fits.append({**key, "sites": len(d), "joint_p": joint, "adj_r2_gain": gain})
            ci = fit.conf_int()
            if typ == "cat":
                means = d.groupby(col)["reaction_pct"].agg(["mean", "size"])
                first = sorted(d[col].unique())[0]
                terms.append(
                    {
                        **key,
                        "level": first,
                        "sites": int(means.loc[first, "size"]),
                        "mean_reaction_pct": float(means.loc[first, "mean"]),
                        "difference_pts": 0.0,
                    }
                )
            for n in names:
                level = n.split("[T.")[-1].rstrip("]") if typ == "cat" else "per standard deviation"
                terms.append(
                    {
                        **key,
                        "level": level,
                        "sites": int(means.loc[level, "size"]) if typ == "cat" else len(d),
                        "mean_reaction_pct": float(means.loc[level, "mean"])
                        if typ == "cat"
                        else np.nan,
                        "difference_pts": float(fit.params[n]),
                        "lo": float(ci.loc[n, 0]),
                        "hi": float(ci.loc[n, 1]),
                        "p": float(fit.pvalues[n]),
                    }
                )
    return pd.DataFrame(terms), pd.DataFrame(fits)


def group_means(sr: pd.DataFrame) -> pd.DataFrame:
    """Mean site reaction with a 95 % interval by event type and level of each categorical
    descriptor (clean sites; every descriptor but on-site generation on the sites without
    it), and the reaction of the summed demand of the level."""
    rows = []
    x = sr[sr["clean"]]
    for kind, d in x.groupby("type"):
        for col, label in {**CATEGORICAL, "chp_heat_led": "plant follows heat demand"}.items():
            if col == "chp_heat_led":
                sub = d[d["chp"] == "on-site generation"]
            elif col == "chp":
                sub = d
            else:
                sub = d[d["chp"] == "no on-site generation"]
            for level, g in sub.dropna(subset=[col]).groupby(col):
                mean, lo, hi = _mean_ci(g["reaction_pct"])
                rows.append(
                    {
                        "type": kind,
                        "descriptor": label,
                        "column": col,
                        "level": str(level),
                        "sites": len(g),
                        "reaction_pct": mean,
                        "lo": lo,
                        "hi": hi,
                        "median_pct": float(g["reaction_pct"].median()),
                        "share_rising": float((g["reaction_pct"] > 0).mean()),
                        "degrees": float(g["degrees"].mean()),
                        "extra_kw_per_site": float(g["extra_kw"].mean()),
                        "mean_kw": float(g["mean_kw"].mean()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = _half_hourly()
    kw = daily_demand(w)
    days, site_codes = kw.index, list(kw.columns)
    weather = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")
    weather = weather[weather["site_code"].isin(site_codes)]
    masks = event_masks(days, site_codes)
    nat = national_events()
    sites = site_table(kw)
    sites.to_csv(OUT_DIR / "sites.csv", index=False)
    clean_codes = set(sites.loc[sites["clean"], "site_code"])

    r = add_national(reactions(kw, weather, masks), nat)
    ev_all = event_days(r)
    ev = ev_all[ev_all["site_code"].isin(clean_codes)]
    ev_all.to_csv(OUT_DIR / "event_site_days.csv", index=False)
    sr = site_reactions(ev_all, sites)
    sr.to_csv(OUT_DIR / "site_reactions.csv", index=False)
    et = event_table(ev, sites).merge(nat, on=["event_id", "type"], suffixes=("", "_affected"))
    et.to_csv(OUT_DIR / "events.csv", index=False)
    compact_events(et).to_csv(OUT_DIR / "table_events.csv", index=False)
    etime = event_time(r, sites)
    etime.to_csv(OUT_DIR / "event_time.csv", index=False)
    bu = build_up(ev, sites)
    bu.to_csv(OUT_DIR / "build_up.csv", index=False)
    hp = hour_profile(slot_cube(w, days), days, site_codes, masks, sites)
    hp.to_csv(OUT_DIR / "hour_profile.csv", index=False)
    ss = seasons(ev, sites)
    ss.to_csv(OUT_DIR / "seasons.csv", index=False)
    sp = spatial(sr[sr["clean"]])
    sp.to_csv(OUT_DIR / "spatial.csv", index=False)
    rel = reliability(ev)
    rel.to_csv(OUT_DIR / "reliability.csv", index=False)
    terms, fits = alignment(sr)
    terms.to_csv(OUT_DIR / "alignment_terms.csv", index=False)
    fits.to_csv(OUT_DIR / "alignment_fits.csv", index=False)
    gm = group_means(sr)
    gm.to_csv(OUT_DIR / "group_means.csv", index=False)

    clean = sr[sr["clean"]]
    summary = {
        "sites": len(site_codes),
        "clean_sites": len(clean_codes),
        "chp_sites_clean": int(((sites["chp"] == "on-site generation") & sites["clean"]).sum()),
        "clean_sites_with_certificate": int(
            (sites["dec_environment"].notna() & sites["clean"]).sum()
        ),
        "national_events": nat["type"].value_counts().to_dict(),
        "event_site_days": ev["type"].value_counts().to_dict(),
        "site_reaction": {
            kind: {
                "sites": len(x),
                "mean_pct": float(x["reaction_pct"].mean()),
                "median_pct": float(x["reaction_pct"].median()),
                "p10_pct": float(x["reaction_pct"].quantile(0.1)),
                "p90_pct": float(x["reaction_pct"].quantile(0.9)),
                "share_rising": float((x["reaction_pct"] > 0).mean()),
                "degrees": float(x["degrees"].mean()),
            }
            for kind, x in clean.groupby("type")
        },
        "build_up": bu.round(3).to_dict(orient="records"),
        "reliability": rel.set_index("type").round(3).to_dict(orient="index"),
        "spatial": sp.round(3).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("Extreme events: %s", summary["site_reaction"])
    return {"summary": summary}
