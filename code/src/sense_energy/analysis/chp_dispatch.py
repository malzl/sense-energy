"""How the on-site CHP plants are dispatched: by heat demand, by the power price, or around the
distribution red band.

The meters of the 24 CHP sites record grid import, which is the site's demand minus what the
plant generates; hospital demand itself does not respond to the power price or to the
time-of-use band, so anything in the import that does is the plant. The sites without CHP (the
clean sample of ``heat_drivers``) are the placebo throughout.

1. **Heat-led.** Daily log import on the cold dose, the heat dose and the day's mean power price,
   with site x year-month and site x weekday effects, per CHP site and pooled for each group. A
   heat-led plant runs more on cold days, so import *falls* with the cold dose where it rises at
   sites without CHP.
2. **Price-following.** Hourly log import on the hour's power price (Elexon market index, GBP per
   MWh), with site x date and site x hour-of-day x month effects: within a day, beyond the site's
   usual shape, does import drop when the price is high? Per CHP site and pooled; the non-CHP
   coefficient is the placebo.
3. **Red band.** The same hourly design with an indicator for 16:00-19:00 on working days (the
   core of the DNOs' red time band, when distribution charges peak): a plant run to avoid red-band
   charges makes import drop then by more than on weekends, relative to the same difference at
   sites without CHP.

Outputs (``code/reports/chp_dispatch/``): ``daily.csv``, ``hourly.csv`` (pooled and per-site
coefficients), ``classification.csv`` (one row per CHP site), ``summary.json``.
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
from .hourly_response import hour_means

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "chp_dispatch"
RED_BAND = (16, 17, 18)
PRICE_SCALE = 100.0  # coefficients per GBP 100/MWh
HOURLY_LAG = 48  # Driscoll-Kraay lag for the hourly regressions: two days of hours
Z = 1.96


def prices_hourly() -> pd.DataFrame:
    p = pd.read_parquet(
        INTERIM_DIR / "prices_elexon_mid.parquet", columns=["datetime", "price_gbp_mwh"]
    )
    local = pd.to_datetime(p["datetime"], utc=True).dt.tz_convert(wr.TZ)
    p["date"] = local.dt.normalize().dt.tz_localize(None)
    p["hour"] = local.dt.hour
    return p.groupby(["date", "hour"], as_index=False)["price_gbp_mwh"].mean()


def samples(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    chp = sorted(df.loc[df["group"] == "on-site CHP", "site_code"].unique())
    non = sorted(hd.subset(df)["site_code"].unique())
    return chp, non


def _per_site_terms(sub, terms, sites, prefix):
    cols, names = {}, {}
    for i, s in enumerate(sites):
        m = (sub["site_code"] == s).to_numpy(dtype=float)
        for v in terms:
            n = f"{v}_{prefix}{i}"
            cols[n] = sub[v].to_numpy() * m
            names[n] = (s, v)
    return cols, names


def _tidy(fit, names, extra=()):
    t = fit.tidy()
    rows = []
    for n, (s, v) in names.items():
        if n in t.index:
            rows.append(
                {
                    "site_code": s,
                    "term": v,
                    "coef": t.loc[n, "Estimate"],
                    "se": t.loc[n, "Std. Error"],
                    "p": t.loc[n, "Pr(>|t|)"],
                }
            )
    for n, label in extra:
        if n in t.index:
            rows.append(
                {
                    "site_code": label,
                    "term": n.split("__")[0],
                    "coef": t.loc[n, "Estimate"],
                    "se": t.loc[n, "Std. Error"],
                    "p": t.loc[n, "Pr(>|t|)"],
                }
            )
    return pd.DataFrame(rows)


def daily(df: pd.DataFrame, chp: list[str], non: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    import pyfixest as pf

    dp = prices.groupby("date", as_index=False)["price_gbp_mwh"].mean()
    sub = (
        df[df["site_code"].isin(chp + non)].merge(dp, on="date", how="inner").reset_index(drop=True)
    )
    sub["price"] = sub["price_gbp_mwh"] / PRICE_SCALE
    terms = ("hdd14", "cdd20", "price")
    cols, names = _per_site_terms(sub, terms, chp, "c")
    is_non = sub["site_code"].isin(non).to_numpy(dtype=float)
    extra = []
    for v in terms:
        n = f"{v}__non"
        cols[n] = sub[v].to_numpy() * is_non
        extra.append((n, "no CHP (pooled)"))
    data = pd.concat(
        [
            sub[["fe_month", "fe_dow", "site_code", "date_id", "date"]],
            pd.DataFrame(cols, index=sub.index),
        ],
        axis=1,
    )
    data["y"] = sub["log_kw"]
    vcov, kw = wc.site_vcov(data)  # site-specific terms: Driscoll-Kraay over days
    fit = pf.feols(
        f"y ~ {' + '.join(cols)} | fe_month + fe_dow", data=data, vcov=vcov, vcov_kwargs=kw
    )
    return _tidy(fit, names, extra)


def hourly_panel(df: pd.DataFrame, sites: list[str], prices: pd.DataFrame) -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    hm = hour_means(w)
    hm = hm.merge(prices, on=["date", "hour"], how="inner")
    cal = df[["site_code", "date", "holiday"]].drop_duplicates()
    hm = hm.merge(cal, on=["site_code", "date"], how="inner")
    hm["log_kw"] = np.log(hm["hour_kw"])
    hm["price"] = hm["price_gbp_mwh"] / PRICE_SCALE
    workday = (hm["date"].dt.dayofweek < 5) & (hm["holiday"] == 0)
    hm["red_workday"] = (hm["hour"].isin(RED_BAND) & workday).astype(float)
    hm["fe_sd"] = hm["site_code"] + "|" + hm["date"].dt.strftime("%Y-%m-%d")
    hm["fe_shm"] = (
        hm["site_code"] + "|" + hm["hour"].astype(str) + "|" + hm["date"].dt.strftime("%m")
    )
    hm["date_id"] = hm["date"].dt.strftime("%Y-%m-%d")
    return hm


def hourly(hm: pd.DataFrame, chp: list[str], non: list[str]) -> pd.DataFrame:
    import pyfixest as pf

    terms = ("price", "red_workday")
    cols, names = _per_site_terms(hm, terms, chp, "c")
    is_non = hm["site_code"].isin(non).to_numpy(dtype=float)
    extra = []
    for v in terms:
        n = f"{v}__non"
        cols[n] = hm[v].to_numpy() * is_non
        extra.append((n, "no CHP (pooled)"))
    data = pd.concat(
        [hm[["fe_sd", "fe_shm", "site_code", "date_id"]], pd.DataFrame(cols, index=hm.index)],
        axis=1,
    )
    data["y"] = hm["log_kw"].to_numpy()
    hours = ((hm["date"] - hm["date"].min()).dt.days * 24 + hm["hour"]).to_numpy()
    vcov, kw = wc.site_vcov(data, lag=HOURLY_LAG, time_index=hours)  # Driscoll-Kraay over hours
    fit = pf.feols(f"y ~ {' + '.join(cols)} | fe_sd + fe_shm", data=data, vcov=vcov, vcov_kwargs=kw)
    out = _tidy(fit, names, extra)
    # pooled CHP coefficients (one regression with group-level terms)
    is_chp = hm["site_code"].isin(chp).to_numpy(dtype=float)
    cols2 = {f"{v}__chp": hm[v].to_numpy() * is_chp for v in terms} | {
        f"{v}__non": hm[v].to_numpy() * is_non for v in terms
    }
    data2 = pd.concat(
        [hm[["fe_sd", "fe_shm", "site_code", "date_id"]], pd.DataFrame(cols2, index=hm.index)],
        axis=1,
    )
    data2["y"] = hm["log_kw"].to_numpy()
    f2 = pf.feols(
        f"y ~ {' + '.join(cols2)} | fe_sd + fe_shm", data=data2, vcov={"CRV1": "site_code+date_id"}
    )
    t2 = f2.tidy()
    for v in terms:
        n = f"{v}__chp"
        out = pd.concat(
            [
                out,
                pd.DataFrame(
                    [
                        {
                            "site_code": "CHP (pooled)",
                            "term": v,
                            "coef": t2.loc[n, "Estimate"],
                            "se": t2.loc[n, "Std. Error"],
                            "p": t2.loc[n, "Pr(>|t|)"],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    return out


def classify(dly: pd.DataFrame, hr: pd.DataFrame, chp: list[str]) -> pd.DataFrame:
    def wide(t, term):
        d = t[t["term"] == term].set_index("site_code")
        return d["coef"], d["se"]

    out = pd.DataFrame(index=pd.Index(chp, name="site_code"))
    for t, term, name in (
        (dly, "hdd14", "cold"),
        (dly, "cdd20", "heat"),
        (dly, "price", "price_daily"),
        (hr, "price", "price_hourly"),
        (hr, "red_workday", "red_band"),
    ):
        c, s = wide(t, term)
        out[f"{name}_pct"] = 100 * c.reindex(out.index)
        out[f"{name}_se"] = 100 * s.reindex(out.index)

    def placebo(t, term):
        r = t[(t["site_code"] == "no CHP (pooled)") & (t["term"] == term)].iloc[0]
        return 100 * r["coef"], 100 * r["se"]

    # Each site against the sites without CHP: hospital demand does not follow the price or the
    # red band, and rises on cold days; what the import does beyond that is the plant.
    for name, t, term in (
        ("cold", dly, "hdd14"),
        ("price_hourly", hr, "price"),
        ("red_band", hr, "red_workday"),
    ):
        b, se = placebo(t, term)
        out[f"{name}_vs_no_chp_pct"] = out[f"{name}_pct"] - b
        out[f"{name}_vs_no_chp_se"] = np.sqrt(out[f"{name}_se"] ** 2 + se**2)
    out["heat_led"] = (out["cold_vs_no_chp_pct"] + Z * out["cold_vs_no_chp_se"]) < 0
    out["price_following"] = (
        out["price_hourly_vs_no_chp_pct"] + Z * out["price_hourly_vs_no_chp_se"]
    ) < 0
    out["red_band_avoiding"] = (
        out["red_band_vs_no_chp_pct"] + Z * out["red_band_vs_no_chp_se"]
    ) < 0
    sites = (
        pd.read_parquet(INTERIM_DIR / "sites.parquet")
        .drop_duplicates("site_code")
        .set_index("site_code")
    )
    out["site_name"] = sites["site_name"].reindex(out.index)
    gen = PROJECT_ROOT / "code" / "reports" / "chp" / "generation_2024_25.csv"
    if gen.exists():
        g = pd.read_csv(gen).set_index("site_code")
        for c in ("chp_elec_kw", "capacity_factor", "chp_share", "heat_to_power", "plausible"):
            out[c] = g[c].reindex(out.index)
    return out


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    chp, non = samples(df)
    prices = prices_hourly()
    with threadpool_limits(8):
        dly = daily(df, chp, non, prices)
        dly.to_csv(OUT_DIR / "daily.csv", index=False)
        logger.info("daily done")
        hm = hourly_panel(df, chp + non, prices)
        logger.info("hourly panel: %d site-hours, %d sites", len(hm), hm["site_code"].nunique())
        hr = hourly(hm, chp, non)
        hr.to_csv(OUT_DIR / "hourly.csv", index=False)
    cl = classify(dly, hr, chp)
    cl.to_csv(OUT_DIR / "classification.csv")

    def pooled(t, who, term):
        return round(
            100 * float(t[(t["site_code"] == who) & (t["term"] == term)]["coef"].iloc[0]), 2
        )

    summary = {
        "chp_sites": len(chp),
        "non_chp_sites": len(non),
        "pooled_daily_cold_pct": {"no CHP": pooled(dly, "no CHP (pooled)", "hdd14")},
        "pooled_hourly_price_pct_per_100": {
            "no CHP": pooled(hr, "no CHP (pooled)", "price"),
            "CHP": pooled(hr, "CHP (pooled)", "price"),
        },
        "pooled_red_band_pct": {
            "no CHP": pooled(hr, "no CHP (pooled)", "red_workday"),
            "CHP": pooled(hr, "CHP (pooled)", "red_workday"),
        },
        "heat_led_sites": int(cl["heat_led"].sum()),
        "price_following_sites": int(cl["price_following"].sum()),
        "red_band_avoiding_sites": int(cl["red_band_avoiding"].sum()),
        "sites_with_estimates": int(cl["cold_pct"].notna().sum()),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("CHP dispatch: %s", summary)
    return {"daily": dly, "hourly": hr, "classification": cl, "summary": summary}
