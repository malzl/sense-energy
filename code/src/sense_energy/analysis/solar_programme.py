"""The hospital solar programme seen from the meter.

Great British Energy's first project funds rooftop solar at NHS sites: GBP 100 million for 187
site postcodes in 78 trusts, announced on 21 March 2025 (list published on GOV.UK, Open Government
Licence, held in ``external/gbe_solar/``). Ten panel sites are on it. Most installations came
after the summer of 2025, which the summer watch (``solar_watch``) cannot see, so the sunshine
slope is estimated here on a spring window (1 March to 8 May, the last day of the record) for
2023 to 2026, with the same design: daytime (06:00-18:00) demand on site x spring x solar
radiation, site x (cold dose, heat dose), specification B's fixed effects, Driscoll-Kraay errors.
CHP sites are kept (their meters record import) and marked.

A break in the slope has a size. Panels of ``k`` kWp generate ``PR x G`` kWh per kWp on a day with
``G`` kWh/m2, nearly all inside the daytime window, so they lower the window's mean demand by
``k x PR x G / 12`` kW. The change in a site's slope, in kW per kWh/m2, therefore measures the
array: ``k = -12 x change / PR`` (``implied_kwp``). The same conversion is applied to the summer
breaks of ``solar_watch`` and compared with the solar generation the estates returns report.

Outputs in ``code/reports/solar_programme/``.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, GEO_DIR, INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import flexibility as fx
from . import heat_drivers as hd
from . import solar_watch as sw
from . import weather_causal as wc

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "solar_programme"
GBE_FILE = EXTERNAL_DIR / "gbe_solar" / "gbe-nhs-solar-projects.csv"
SPRINGS = (2023, 2024, 2025, 2026)
WINDOW = ((3, 1), (5, 8))  # 1 March to 8 May
MIN_DAYS = 45
DAY_WINDOW_H = 12.0
BREAK_PTS, BREAK_Z = sw.BREAK_PTS, sw.BREAK_Z


def _norm(pc: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(pc).upper())


def funded_postcodes(path=GBE_FILE) -> pd.DataFrame:
    """One row per funded site postcode with its trust's award."""
    d = pd.read_csv(path, encoding="latin-1")
    d.columns = [
        "trust_code",
        "trust_name",
        "region",
        "postcodes",
        "trust_sites_funded",
        "trust_funding_gbp",
        "lifetime_savings",
    ]
    rows = []
    for r in d.itertuples():
        for pc in re.split(r"[,;]", str(r.postcodes)):
            if _norm(pc):
                rows.append(
                    {
                        "trust_code": str(r.trust_code).strip(),
                        "trust_name": str(r.trust_name).strip().title(),
                        "postcode_norm": _norm(pc),
                        "trust_sites_funded": int(r.trust_sites_funded),
                        "trust_funding_gbp": float(str(r.trust_funding_gbp).replace(",", "")),
                        "lifetime_savings": str(r.lifetime_savings).strip(),
                    }
                )
    return pd.DataFrame(rows)


def funded_sites(sites: list[str]) -> pd.DataFrame:
    """Panel sites whose postcode is on the funded list."""
    g = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    g = g[g["site_code"].isin(sites)][["site_code", "site_name", "postcode"]].copy()
    g["postcode_norm"] = g["postcode"].map(_norm)
    m = g.merge(funded_postcodes(), on="postcode_norm")
    return m.drop(columns="postcode_norm").drop_duplicates("site_code")


def in_window(date: pd.Series) -> np.ndarray:
    md = date.dt.month * 100 + date.dt.day
    return (
        (md >= WINDOW[0][0] * 100 + WINDOW[0][1]) & (md <= WINDOW[1][0] * 100 + WINDOW[1][1])
    ).to_numpy()


def spring_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Daily rows of the spring windows with daytime demand, every site with at least two
    springs of ``MIN_DAYS`` days."""
    sub = df[in_window(df["date"]) & df["date"].dt.year.isin(SPRINGS)]
    sites = sorted(sub["site_code"].unique())
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    x = sub.merge(sw.daytime_means(w), on=["date", "site_code"], how="inner")
    n = x.groupby(["site_code", x["date"].dt.year]).size().unstack(fill_value=0)
    keep = n.index[(n >= MIN_DAYS).sum(axis=1) >= 2]
    return x[x["site_code"].isin(keep)].reset_index(drop=True)


def fit(x: pd.DataFrame) -> pd.DataFrame:
    """Sunshine slope per site and spring (log points per kWh/m2), Driscoll-Kraay errors."""
    import pyfixest as pf

    sites = sorted(x["site_code"].unique())
    cols, names = {}, {}
    year = x["date"].dt.year.to_numpy()
    for i, s in enumerate(sites):
        m = (x["site_code"] == s).to_numpy(dtype=float)
        for v in ("hdd14", "cdd20"):
            if (x[v].to_numpy() * m).std() > 0:
                cols[f"{v}_s{i}"] = x[v].to_numpy() * m
        for y in SPRINGS:
            k = m * (year == y)
            if k.sum() >= MIN_DAYS:
                n = f"sun_s{i}_y{y}"
                cols[n] = x["solar_kwh_m2"].to_numpy() * k
                names[n] = (s, y)
    data = pd.concat([x[hd.FE + ["site_code", "date"]], pd.DataFrame(cols, index=x.index)], axis=1)
    data["y"] = np.log(x["day_kw"].to_numpy())
    vcov, kw = wc.site_vcov(data)
    f = pf.feols(
        f"y ~ {' + '.join(cols)} | {' + '.join(hd.FE)}", data=data, vcov=vcov, vcov_kwargs=kw
    )
    t = f.tidy()
    rows = [
        {"site_code": s, "spring": y, "slope": t.loc[n, "Estimate"], "se": t.loc[n, "Std. Error"]}
        for n, (s, y) in names.items()
        if n in t.index
    ]
    out = pd.DataFrame(rows)
    out["slope_pct"], out["se_pct"] = 100 * out["slope"], 100 * out["se"]
    kw_mean = x.groupby(["site_code", x["date"].dt.year])["day_kw"].mean().rename("day_kw")
    return out.merge(kw_mean, left_on=["site_code", "spring"], right_index=True, how="left")


def implied_kwp(change_pct: pd.Series, day_kw: pd.Series) -> pd.Series:
    """Array size that explains a change in the sunshine slope (per cent per kWh/m2) at a site
    whose daytime demand is ``day_kw``."""
    return -DAY_WINDOW_H * (change_pct / 100.0) * day_kw / fx.PERFORMANCE_RATIO


def breaks(slopes: pd.DataFrame, last: int = SPRINGS[-1]) -> pd.DataFrame:
    """Change of each site's slope from the previous spring to the last one."""
    p = slopes.pivot(index="site_code", columns="spring", values=["slope_pct", "se_pct", "day_kw"])
    out = pd.DataFrame(index=p.index)
    for y in SPRINGS:
        out[f"sun_{y}_pct"] = p.get(("slope_pct", y), np.nan)
    prev = last - 1
    out["change_pct"] = p[("slope_pct", last)] - p[("slope_pct", prev)]
    out["change_se"] = np.sqrt(p[("se_pct", last)] ** 2 + p[("se_pct", prev)] ** 2)
    out["z"] = out["change_pct"] / out["change_se"]
    out["break_down"] = (out["change_pct"] < BREAK_PTS) & (out["z"] < BREAK_Z)
    out["break_up"] = (out["change_pct"] > -BREAK_PTS) & (out["z"] > -BREAK_Z)
    out["day_kw"] = p[("day_kw", last)]
    out["implied_kwp"] = implied_kwp(out["change_pct"], out["day_kw"])
    out["implied_kwp_se"] = (
        DAY_WINDOW_H * out["change_se"] / 100 * out["day_kw"] / fx.PERFORMANCE_RATIO
    )
    return out.dropna(subset=["change_pct"])


def site_yield(sites: list[str]) -> pd.Series:
    """Annual generation per kWp (kWh) at each site from ERA5 radiation, financial year 2024/25."""
    idx = pd.date_range("2024-04-01", "2025-03-31 23:30", freq="30min", tz="UTC")
    return fx.pv_per_kwp(sites, idx).mean() * 8760.0


def summer_measured() -> pd.DataFrame:
    """The summer breaks of ``solar_watch`` (2023 to 2025) sized in kWp and set against the
    generation the 2024/25 return reports."""
    b = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "solar_watch" / "breaks.csv")
    b = b[b["break_down"] | (b["eric_solar_new_or_grown"] == True)].copy()  # noqa: E712
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[
        b["site_code"].tolist()
    ]
    d = sw.daytime_means(w * 2.0)
    d = d[(d["date"].dt.year == sw.SUMMERS[-1]) & d["date"].dt.month.isin(sw.MONTHS)]
    b["day_kw"] = b["site_code"].map(d.groupby("site_code")["day_kw"].mean())
    b["implied_kwp"] = implied_kwp(b["change_pct"], b["day_kw"])
    # sites whose panels predate the first summer have no change to size: use the level of the
    # last summer's slope against the estate's median slope
    slopes = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "solar_watch" / "slopes.csv")
    base = float(slopes.loc[slopes["summer"] == sw.SUMMERS[-1], "slope_pct"].median())
    b["implied_kwp_level"] = implied_kwp(b[f"sun_{sw.SUMMERS[-1]}_pct"] - base, b["day_kw"])
    y = site_yield(b["site_code"].tolist())
    b["yield_kwh_kwp"] = b["site_code"].map(y)
    b["implied_mwh_year"] = b["implied_kwp"] * b["yield_kwh_kwp"] / 1000.0
    e0 = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").set_index("site_code")
    e1 = pd.read_parquet(INTERIM_DIR / "eric_2024_25_site.parquet").set_index("site_code")
    b["eric_solar_mwh_2023_24"] = b["site_code"].map(e0["solar_generated_kwh"]) / 1000.0
    b["eric_solar_mwh_2024_25"] = b["site_code"].map(e1["solar_generated_kwh"]) / 1000.0
    b["eric_added_mwh"] = b["eric_solar_mwh_2024_25"].fillna(0) - b[
        "eric_solar_mwh_2023_24"
    ].fillna(0)
    b["eric_implied_kwp"] = b["eric_added_mwh"] * 1000.0 / b["yield_kwh_kwp"]
    b["eric_total_kwp"] = b["eric_solar_mwh_2024_25"] * 1000.0 / b["yield_kwh_kwp"]
    # the meter sees only what the site consumes itself: the ratio to the reported array is the
    # self-consumed share where the array is larger than the daytime load
    b["metered_share_of_reported"] = b["implied_kwp_level"] / b["eric_total_kwp"].where(
        b["eric_total_kwp"] > 0
    )
    cols = [
        "site_code",
        "site_name",
        "sun_2023_pct",
        "sun_2025_pct",
        "change_pct",
        "z",
        "break_down",
        "eric_solar_new_or_grown",
        "day_kw",
        "implied_kwp",
        "implied_kwp_level",
        "implied_mwh_year",
        "eric_solar_mwh_2023_24",
        "eric_solar_mwh_2024_25",
        "eric_added_mwh",
        "eric_implied_kwp",
        "eric_total_kwp",
        "metered_share_of_reported",
    ]
    return b[cols].sort_values("implied_kwp", ascending=False)


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    x = spring_panel(df)
    with threadpool_limits(4):
        slopes = fit(x)
    slopes.to_csv(OUT_DIR / "spring_slopes.csv", index=False)
    b = breaks(slopes)
    group = df.drop_duplicates("site_code").set_index("site_code")["group"]
    names = pd.read_parquet(GEO_DIR / "sites_geo.parquet").drop_duplicates("site_code")
    names = names.set_index("site_code")["site_name"]
    b["group"] = group.reindex(b.index)
    b["site_name"] = names.reindex(b.index)
    f = funded_sites(sorted(df["site_code"].unique())).set_index("site_code")
    b["funded"] = b.index.isin(f.index)
    y = site_yield(b.index.tolist())
    b["implied_mwh_year"] = b["implied_kwp"] * y.reindex(b.index) / 1000.0
    b.to_csv(OUT_DIR / "spring_breaks.csv")
    fund = f.join(
        b[
            [
                "group",
                "sun_2023_pct",
                "sun_2024_pct",
                "sun_2025_pct",
                "sun_2026_pct",
                "change_pct",
                "z",
                "break_down",
                "day_kw",
                "implied_kwp",
                "implied_kwp_se",
                "implied_mwh_year",
            ]
        ],
        how="left",
    )
    fund.to_csv(OUT_DIR / "funded_sites.csv")
    measured = summer_measured()
    measured.to_csv(OUT_DIR / "summer_measured.csv", index=False)

    fp = funded_postcodes()
    tested = fund.dropna(subset=["change_pct"])
    both = measured[measured["eric_total_kwp"] > 0]
    summary = {
        "programme": {
            "trusts": int(fp["trust_code"].nunique()),
            "site_postcodes": int(len(fp)),
            "funding_gbp_m": round(
                float(fp.drop_duplicates("trust_code")["trust_funding_gbp"].sum() / 1e6), 1
            ),
        },
        "panel_sites_funded": int(len(fund)),
        "panel_trusts_funded": int(fund["trust_code"].nunique()),
        "funded_sites_tested": int(len(tested)),
        "funded_sites_with_break": int(tested["break_down"].sum()),
        "funded_break_sites": tested.index[tested["break_down"]].tolist(),
        "spring_watch": {
            "sites": int(len(b)),
            "breaks_down": int(b["break_down"].sum()),
            "breaks_up": int(b["break_up"].sum()),
            "breaks_down_not_funded": b.index[b["break_down"] & ~b["funded"]].tolist(),
            "implied_kwp_at_breaks": b.loc[b["break_down"], "implied_kwp"].round(0).to_dict(),
        },
        "summer_measured": {
            "sites": int(len(measured)),
            "sites_with_reported_generation": int(len(both)),
            "metered_vs_reported_kwp": both.set_index("site_code")[
                ["implied_kwp_level", "eric_total_kwp", "metered_share_of_reported", "day_kw"]
            ]
            .round(2)
            .to_dict(orient="index"),
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("solar programme: %s", json.dumps(summary["spring_watch"]))
    return {"summary": summary}
