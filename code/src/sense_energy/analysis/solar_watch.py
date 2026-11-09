"""Watching solar go on roofs: breaks in each site's sunshine response, summer by summer.

Panels behind the meter make a site import less on sunny days; a site that installs them should
see its sunshine slope turn more negative from one summer to the next. For every site without
CHP, the daytime (06:00-18:00 local) mean demand of April-September days is regressed on
site x summer x solar radiation interactions (summers 2023, 2024, 2025), with site x (cold dose,
heat dose) terms and specification B's fixed effects. A site's break is the change of its
sunshine slope from 2023 to 2025; flagged when more negative by over one percentage point per
kWh/m2 and beyond 2.5 standard errors. The flags are checked against the estates returns (solar
generation new or grown between 2023/24 and 2024/25) and the renewable planning database.

Outputs (``code/reports/solar_watch/``): ``slopes.csv`` (site x summer), ``breaks.csv`` (one row
per site), ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import heat_drivers as hd
from . import weather_causal as wc
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "solar_watch"
SUMMERS = (2023, 2024, 2025)
MONTHS = (4, 5, 6, 7, 8, 9)
DAY_HOURS = range(6, 18)
MIN_DAY_PERIODS = 20
BREAK_PTS, BREAK_Z = -1.0, -2.5


def daytime_means(w: pd.DataFrame) -> pd.DataFrame:
    local = pd.to_datetime(w.index, utc=True).tz_convert(wr.TZ)
    keep = np.isin(local.hour, list(DAY_HOURS))
    w, local = w[keep], local[keep]
    day = local.normalize().tz_localize(None).rename("date")
    n = w.notna().groupby(day).sum()
    m = w.groupby(day).mean().where(n >= MIN_DAY_PERIODS)
    out = m.stack().rename("day_kw").rename_axis(["date", "site_code"]).reset_index()
    return out[out["day_kw"] > 0]


def panel(df: pd.DataFrame) -> pd.DataFrame:
    sub = hd.subset(df, clean=False)
    sub = sub[sub["date"].dt.month.isin(MONTHS) & sub["date"].dt.year.isin(SUMMERS)]
    sites = sorted(sub["site_code"].unique())
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    x = sub.merge(daytime_means(w), on=["date", "site_code"], how="inner")
    # every site needs days in at least two summers
    yrs = x.groupby("site_code")["date"].apply(lambda d: d.dt.year.nunique())
    return x[x["site_code"].isin(yrs.index[yrs >= 2])].reset_index(drop=True)


def fit(x: pd.DataFrame) -> pd.DataFrame:
    import pyfixest as pf

    sites = sorted(x["site_code"].unique())
    cols, names = {}, {}
    year = x["date"].dt.year.to_numpy()
    for i, s in enumerate(sites):
        m = (x["site_code"] == s).to_numpy(dtype=float)
        for v in ("hdd14", "cdd20"):
            cols[f"{v}_s{i}"] = x[v].to_numpy() * m
        for y in SUMMERS:
            k = m * (year == y)
            if k.sum() >= 60:  # a summer needs two months of days to carry a slope
                n = f"sun_s{i}_y{y}"
                cols[n] = x["solar_kwh_m2"].to_numpy() * k
                names[n] = (s, y)
    data = pd.concat([x[hd.FE + ["site_code"]], pd.DataFrame(cols, index=x.index)], axis=1)
    data["y"] = np.log(x["day_kw"].to_numpy())
    f = pf.feols(
        f"y ~ {' + '.join(cols)} | {' + '.join(hd.FE)}",
        data=data,
        vcov={"CRV1": "site_code+date_id"},
    )
    t = f.tidy()
    rows = [
        {"site_code": s, "summer": y, "slope": t.loc[n, "Estimate"], "se": t.loc[n, "Std. Error"]}
        for n, (s, y) in names.items()
        if n in t.index
    ]
    out = pd.DataFrame(rows)
    out["slope_pct"], out["se_pct"] = 100 * out["slope"], 100 * out["se"]
    return out


def breaks(slopes: pd.DataFrame) -> pd.DataFrame:
    w = slopes.pivot(index="site_code", columns="summer", values=["slope_pct", "se_pct"])
    out = pd.DataFrame(index=w.index)
    for y in SUMMERS:
        out[f"sun_{y}_pct"] = w.get(("slope_pct", y), np.nan)
        out[f"sun_{y}_se"] = w.get(("se_pct", y), np.nan)
    first = out[f"sun_{SUMMERS[0]}_pct"].fillna(out[f"sun_{SUMMERS[1]}_pct"])
    first_se = out[f"sun_{SUMMERS[0]}_se"].fillna(out[f"sun_{SUMMERS[1]}_se"])
    out["change_pct"] = out[f"sun_{SUMMERS[-1]}_pct"] - first
    out["change_se"] = np.sqrt(out[f"sun_{SUMMERS[-1]}_se"] ** 2 + first_se**2)
    out["z"] = out["change_pct"] / out["change_se"]
    out["break_down"] = (out["change_pct"] < BREAK_PTS) & (out["z"] < BREAK_Z)
    out["break_up"] = (out["change_pct"] > -BREAK_PTS) & (out["z"] > -BREAK_Z)
    ch = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "estates" / "eric_changes.csv").set_index(
        "site_code"
    )
    out["eric_solar_new_or_grown"] = (ch["solar_new"] | ch["solar_grew"]).reindex(out.index)
    out["eric_led_change_pts"] = (ch["led_pct_2425"] - ch["led_pct_2324"]).reindex(out.index)
    ctx = (
        pd.read_parquet(INTERIM_DIR / "site_context.parquet")
        .drop_duplicates("site_code")
        .set_index("site_code")
    )
    out["repd_solar_status"] = ctx["repd_solar_status"].reindex(out.index)
    out["coverage_flag"] = hd.coverage_flags().reindex(out.index)
    sites = (
        pd.read_parquet(INTERIM_DIR / "sites.parquet")
        .drop_duplicates("site_code")
        .set_index("site_code")
    )
    out["site_name"] = sites["site_name"].reindex(out.index)
    return out


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    x = panel(df)
    with threadpool_limits(4):
        slopes = fit(x)
    slopes.to_csv(OUT_DIR / "slopes.csv", index=False)
    b = breaks(slopes)
    b.to_csv(OUT_DIR / "breaks.csv")
    eric = b["eric_solar_new_or_grown"].fillna(False).astype(bool)
    summary = {
        "sites": int(len(b)),
        "pooled_sun_pct_by_summer": slopes.groupby("summer")["slope_pct"]
        .median()
        .round(2)
        .to_dict(),
        "breaks_down": int(b["break_down"].sum()),
        "breaks_up": int(b["break_up"].sum()),
        "eric_solar_sites_in_sample": int(eric.sum()),
        "eric_solar_sites_flagged": int((eric & b["break_down"]).sum()),
        "flagged_without_eric_solar": int((~eric & b["break_down"]).sum()),
        "flagged_sites": b[b["break_down"]].index.tolist(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("solar watch: %s", summary)
    return {"slopes": slopes, "breaks": b, "summary": summary}
