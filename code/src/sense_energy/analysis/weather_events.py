"""Extreme-weather events at the sites, from ``interim/weather_daily.parquet`` (ERA5).

Site-day criteria, then runs of consecutive qualifying days at a site:

* **heatwave** - Met Office definition: at least 3 consecutive days with daily
  maximum temperature at or above the county threshold (25-28 degC, 1991-2020
  basis). The 2022 update names the 28 degC counties (London plus Surrey,
  Berkshire, Buckinghamshire, Bedfordshire, Hertfordshire, Cambridgeshire) and
  East Riding at 26 degC; the full county map is published only as an image, so
  other counties take a regional approximation (``threshold_source``). ERA5
  understates afternoon peaks by about 1 degC in heatwaves, so a variant with
  every threshold 1 degC lower is reported as a sensitivity.
* **cold spell** - UKHSA cold-weather alert criterion: daily mean temperature
  at or below 2 degC on at least 2 consecutive days (48 h).
* **high wind** - daily maximum gust of at least 22 m/s (about 50 mph).
* **heavy rain** - daily total of at least 25 mm.
* **snow** - daily snowfall of at least 5 mm water equivalent (about 5 cm).

A **national event** is a run of dates (gaps of one day bridged) on which at
least 5 % of sites are inside a site event of that type. Each carries its dates,
the sites affected, the peak value and the demand response at the affected
sites with electricity data: event-day demand over the median of the same
weekday 1 and 2 weeks either side (non-event, non-holiday days), median across
site-days.

Outputs (``code/reports/weather_events/``): ``site_events.csv``,
``national_events.csv``, ``site_thresholds.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import GEO_DIR, INTERIM_DIR, PROJECT_ROOT
from ..experiments.features import _holiday_set
from ..logging_utils import get_logger
from .cooling import daily_demand

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "weather_events"
NATIONAL_MIN_SHARE = 0.05
ERA5_TMAX_BIAS_C = 1.0

#: event type -> (daily column, comparison, threshold or None for the site heat threshold, min run).
CRITERIA = {
    "heatwave": ("t_max", "ge", None, 3),
    "cold spell": ("t_mean", "le", 2.0, 2),
    "high wind": ("gust_max", "ge", 22.0, 1),
    "heavy rain": ("precip_mm", "ge", 25.0, 1),
    "snow": ("snow_mm_we", "ge", 5.0, 1),
}

#: Met Office heatwave thresholds, degC. Named local authorities follow the 2022
#: county list; everything else takes its ONS region's approximation.
HEAT_THRESHOLD_LA = {
    **dict.fromkeys(
        [
            "BRACKNELL FOREST COUNCIL",
            "READING BOROUGH COUNCIL",
            "ROYAL BOROUGH OF WINDSOR AND MAIDENHEAD",
            "WEST BERKSHIRE COUNCIL",
            "WOKINGHAM BOROUGH COUNCIL",
            "BUCKINGHAMSHIRE COUNCIL",
            "BEDFORD BOROUGH COUNCIL",
            "LUTON BOROUGH COUNCIL",
            "HERTFORDSHIRE COUNTY COUNCIL",
            "CAMBRIDGESHIRE COUNTY COUNCIL",
            "PETERBOROUGH CITY COUNCIL",
        ],
        28.0,
    ),
    "EAST RIDING OF YORKSHIRE COUNCIL": 26.0,
}
HEAT_THRESHOLD_REGION = {
    "London": 28.0,
    "South East": 27.0,
    "East of England": 27.0,
    "West Midlands": 26.0,
    "East Midlands": 26.0,
    "South West": 25.0,
    "North West": 25.0,
    "Yorkshire and The Humber": 25.0,
    "North East": 25.0,
}


def heat_thresholds() -> pd.DataFrame:
    s = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    s = s[s["latitude"].notna()]
    la = s["local_authority"].fillna("").str.upper().map(HEAT_THRESHOLD_LA)
    region = s["region_ons"].map(HEAT_THRESHOLD_REGION)
    exact = la.notna() | (s["region_ons"] == "London")
    return pd.DataFrame(
        {
            "site_code": s["site_code"].to_numpy(),
            "region_ons": s["region_ons"].to_numpy(),
            "heat_threshold_c": la.fillna(region).to_numpy(),
            "threshold_source": np.where(exact, "met office 2022 list", "regional approximation"),
        }
    )


def runs(flag: pd.Series, min_len: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """(start, end) of runs of True on consecutive dates, at least ``min_len`` long."""
    d = flag.index[flag.to_numpy()]
    if len(d) == 0:
        return []
    brk = np.r_[True, np.diff(d.values).astype("timedelta64[D]").astype(int) > 1]
    gid = np.cumsum(brk)
    out = []
    for g in np.unique(gid):
        days = d[gid == g]
        if len(days) >= min_len:
            out.append((days[0], days[-1]))
    return out


def site_events(
    daily: pd.DataFrame, thresholds: pd.DataFrame, heat_offset: float = 0.0
) -> pd.DataFrame:
    thr = thresholds.set_index("site_code")["heat_threshold_c"] - heat_offset
    rows = []
    for site, x in daily.groupby("site_code"):
        x = x.set_index("date").sort_index()
        for kind, (col, op, value, min_len) in CRITERIA.items():
            limit = thr.get(site, np.nan) if value is None else value
            v = x[col]
            flag = v >= limit if op == "ge" else v <= limit
            for a, b in runs(flag, min_len):
                seg = v.loc[a:b]
                rows.append(
                    {
                        "site_code": site,
                        "type": kind,
                        "start": a,
                        "end": b,
                        "days": (b - a).days + 1,
                        "threshold": limit,
                        "peak": float(seg.max() if op == "ge" else seg.min()),
                    }
                )
    return pd.DataFrame(rows)


def _expand(ev: pd.DataFrame) -> pd.DataFrame:
    """One row per (site, type, date) inside a site event."""
    parts = [
        pd.DataFrame(
            {"site_code": r.site_code, "type": r.type, "date": pd.date_range(r.start, r.end)}
        )
        for r in ev.itertuples()
    ]
    return pd.concat(parts, ignore_index=True)


def national_events(ev: pd.DataFrame, n_sites: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    days = _expand(ev)
    count = days.groupby(["type", "date"]).size()
    rows = []
    for kind, c in count.groupby(level="type"):
        c = c.droplevel("type")
        full = c.reindex(pd.date_range(c.index.min(), c.index.max()), fill_value=0)
        on = full >= NATIONAL_MIN_SHARE * n_sites
        bridged = on | (
            on.shift(1, fill_value=False) & on.shift(-1, fill_value=False)
        )  # 1-day gaps
        for a, b in runs(bridged, 1):
            sub = days[(days["type"] == kind) & days["date"].between(a, b)]
            rows.append(
                {
                    "type": kind,
                    "start": a,
                    "end": b,
                    "days": (b - a).days + 1,
                    "sites": sub["site_code"].nunique(),
                    "peak_sites_on_one_day": int(full.loc[a:b].max()),
                }
            )
    nat = pd.DataFrame(rows).sort_values(["start", "type"]).reset_index(drop=True)
    nat.insert(
        0,
        "event_id",
        [
            f"{t.replace(' ', '_')}_{s:%Y%m%d}"
            for t, s in zip(nat["type"], nat["start"], strict=True)
        ],
    )
    return nat, days


def demand_response(
    nat: pd.DataFrame, days: pd.DataFrame, daily: pd.DataFrame, kw: pd.DataFrame
) -> pd.DataFrame:
    """Event-day demand over same-weekday reference days, per national event."""
    hol = pd.DatetimeIndex(
        sorted(_holiday_set(range(kw.index.year.min(), kw.index.year.max() + 1)))
    )
    event_days = set(zip(days["site_code"], days["date"], strict=True))
    peak_col = {k: v[0] for k, v in CRITERIA.items()}
    wx = daily.set_index(["site_code", "date"])
    out = []
    for e in nat.itertuples():
        sub = days[(days["type"] == e.type) & days["date"].between(e.start, e.end)]
        ratios, peaks = [], []
        for site, date in zip(sub["site_code"], sub["date"], strict=True):
            peaks.append(wx.loc[(site, date), peak_col[e.type]])
            if (
                site not in kw.columns
                or date not in kw.index
                or date in hol
                or pd.isna(kw.loc[date, site])
            ):
                continue
            refs = [date + pd.Timedelta(days=7 * k) for k in (-2, -1, 1, 2)]
            refs = [
                r for r in refs if r in kw.index and r not in hol and (site, r) not in event_days
            ]
            ref = kw.loc[refs, site].median() if refs else np.nan
            if pd.notna(ref) and ref > 0:
                ratios.append(kw.loc[date, site] / ref)
        op = CRITERIA[e.type][1]
        out.append(
            {
                "event_id": e.event_id,
                "peak_value": float(np.nanmax(peaks) if op == "ge" else np.nanmin(peaks)),
                "demand_site_days": len(ratios),
                "demand_ratio_median": float(np.median(ratios)) if ratios else np.nan,
            }
        )
    return nat.merge(pd.DataFrame(out), on="event_id")


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    daily = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")
    thr = heat_thresholds()
    thr.to_csv(OUT_DIR / "site_thresholds.csv", index=False)
    n_sites = daily["site_code"].nunique()
    kw = daily_demand()
    elec_window = (kw.index.min(), kw.index.max())

    ev = site_events(daily, thr)
    ev.to_csv(OUT_DIR / "site_events.csv", index=False)
    nat, days = national_events(ev, n_sites)
    nat = demand_response(nat, days, daily, kw)
    nat["in_electricity_record"] = nat["end"].ge(elec_window[0]) & nat["start"].le(elec_window[1])
    nat.to_csv(OUT_DIR / "national_events.csv", index=False)

    sens = site_events(daily, thr, heat_offset=ERA5_TMAX_BIAS_C)
    sens_nat, _ = national_events(sens[sens["type"] == "heatwave"], n_sites)
    panel = set(kw.columns)
    summary = {
        "weather_window": [str(daily["date"].min().date()), str(daily["date"].max().date())],
        "electricity_window": [str(elec_window[0].date()), str(elec_window[1].date())],
        "sites": n_sites,
        "national_events": nat["type"].value_counts().to_dict(),
        "national_events_in_electricity_record": nat.loc[nat["in_electricity_record"], "type"]
        .value_counts()
        .to_dict(),
        "site_events": ev["type"].value_counts().to_dict(),
        "site_events_at_sites_with_electricity": ev.loc[ev["site_code"].isin(panel), "type"]
        .value_counts()
        .to_dict(),
        "heatwave_national_events_threshold_minus_1c": len(sens_nat),
        "heatwave_site_events_threshold_minus_1c": int((sens["type"] == "heatwave").sum()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("Weather events: %s", summary["national_events"])
    return {"site_events": ev, "national_events": nat, "summary": summary}
