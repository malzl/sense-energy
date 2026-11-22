"""Scores for three extensions of the day-ahead forecast on the whole record.

1. **The trained baseline, trained again** (``experiments.gbm_rolling``). LightGBM trained
   once on 2023 against LightGBM trained again each quarter, and both against the zero-shot
   forecast, on every day from January 2024 to May 2026. Is the zero-shot lead of the proof of
   concept the lead over a model that had grown old?
2. **A correction within the day** (``experiments.forecast_extensions.run_residual``). The
   day-ahead forecast against its correction every two hours by a second zero-shot model of
   its own errors, and against the comparators, for all sites, for the sites with on-site
   generation and for their days with a large step.
3. **Measured and forecast weather as separate inputs** (``run_split_weather``) against the
   one-channel covariate run on the same days.

Every error is the site's summed absolute error over its summed demand, as the mean over the
sites, with a 95 % interval from resampling sites. Every difference between two models is
paired on the same site-days, with a 95 % interval from resampling weeks
(``forecast_anatomy.paired_bootstrap``).

Outputs: ``code/reports/forecast_extensions/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import extreme_events as ee
from . import forecast_anatomy as fa
from . import forecast_record as fr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_extensions"
ZERO_SHOT = "chronos2_ifs"
ROLLING_FILE = fr.OUT_DIR / "forecasts_lightgbm_rolling.parquet"
CORRECTION_FILE = fr.OUT_DIR / "forecasts_intraday_correction.parquet"
SPLIT_GLOB = "forecasts_split_weather*.parquet"
CORRECTION_MODELS = ("day_ahead", "corrected", "error_carried", "value_carried", "direct")
RETRAIN_PAIRS = [
    ("lightgbm_quarterly", "lightgbm_once"),
    (ZERO_SHOT, "lightgbm_quarterly"),
    (ZERO_SHOT, "lightgbm_once"),
]
CORRECTION_PAIRS = [
    ("corrected", "day_ahead"),
    ("corrected", "error_carried"),
    ("corrected", "direct"),
    ("direct", "day_ahead"),
]
SPLIT_PAIRS = [
    ("chronos2_ifs_split", ZERO_SHOT),
    ("chronos2_ifs_fc_only", ZERO_SHOT),
    ("chronos2_ifs_split", "chronos2_ifs_fc_only"),
]
PLANT, PLAIN = "on-site generation", "no on-site generation"


# --------------------------------------------------------------------------- #
# Site-days on a common sample
# --------------------------------------------------------------------------- #


def site_days(f: pd.DataFrame, panel, min_periods: int = 48) -> pd.DataFrame:
    """Per model, site and local day: observed half-hours, demand and absolute error, for the
    site-days that every model has with at least ``min_periods`` half-hours and that pass the
    context rule of the whole-record evaluation."""
    actual = fa.actual_long(panel)
    m = f[["model", "site_code", "target", "point"]].merge(actual, on=["target", "site_code"])
    m = m.dropna(subset=["point", "y"])
    m["ae"] = (m["y"] - m["point"]).abs()
    g = m.groupby(["model", "site_code", "day"], observed=True)
    sd = g.agg(n=("y", "size"), a=("y", "sum"), ae=("ae", "sum")).reset_index()
    sd = sd[sd["n"] >= min_periods]
    sd = sd.merge(fr.context_ok(panel), on=["day", "site_code"], how="left")
    sd = sd[sd["context_ok"].fillna(False).astype(bool)].drop(columns="context_ok")
    every = sd.groupby(["site_code", "day"])["model"].transform("nunique") == sd["model"].nunique()
    return sd[every].reset_index(drop=True)


def error_row(x: pd.DataFrame) -> dict:
    per = fr._site_sums(x, ["ae", "a"])
    err, lo, hi = fr._boot(per, fr.nmae)
    return {
        "sites": len(per),
        "site_days": len(x),
        "days": x["day"].nunique(),
        "nmae_pct": err,
        "nmae_lo": lo,
        "nmae_hi": hi,
    }


def errors(sd: pd.DataFrame, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, mask in groups.items():
        for model, x in sd[mask].groupby("model"):
            if x["site_code"].nunique() >= 3:
                rows.append({"group": name, "model": model} | error_row(x))
    return pd.DataFrame(rows)


def differences(sd: pd.DataFrame, pairs, groups: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    models = set(sd["model"])
    for name, mask in groups.items():
        x = sd[mask]
        if x["day"].nunique() < 3 * fa.BLOCK_DAYS:
            continue
        for a, b in pairs:
            if {a, b} <= models:
                rows.append(
                    {"group": name, "a": a, "b": b}
                    | fa.paired_bootstrap(x, a, b)
                    | {
                        "nmae_a_pct": fa._nmae(x[x["model"] == a]),
                        "nmae_b_pct": fa._nmae(x[x["model"] == b]),
                    }
                )
    return pd.DataFrame(rows)


def site_table(sd: pd.DataFrame) -> pd.DataFrame:
    """Error of each site (rows) for each model (columns), per cent of its demand."""
    per = sd.groupby(["site_code", "model"])[["a", "ae"]].sum()
    return (100 * per["ae"] / per["a"]).unstack("model").reset_index()


def site_differences(sites: pd.DataFrame, pairs) -> pd.DataFrame:
    """The paired differences site by site: the mean is moved by one site with a very large
    error, the median site and the count of sites are not."""
    rows = []
    for a, b in pairs:
        if a in sites and b in sites:
            d = (sites[a] - sites[b]).dropna()
            worst = sites.loc[d.idxmax(), "site_code"]
            rows.append(
                {
                    "a": a,
                    "b": b,
                    "sites": len(d),
                    "mean_diff_pts": float(d.mean()),
                    "median_diff_pts": float(d.median()),
                    "sites_a_better": int((d < 0).sum()),
                    "largest_diff_pts": float(d.max()),
                    "largest_diff_site": worst,
                    "mean_diff_without_it_pts": float(d.drop(d.idxmax()).mean()),
                }
            )
    return pd.DataFrame(rows)


def plant_sites(panel) -> set[str]:
    return set(pd.read_csv(ee.CHP_FILE)["site_code"]) & set(panel.sites)


def record_forecast(model: str = ZERO_SHOT) -> pd.DataFrame:
    f = pd.read_parquet(fa.OUT_DIR / fr.FILES[model], columns=["site_code", "target", "point"])
    return f.assign(model=model)


# --------------------------------------------------------------------------- #
# 1. The trained baseline, trained again
# --------------------------------------------------------------------------- #


def quarter_label(day: pd.Series) -> pd.Series:
    return day.dt.year.astype(str) + " Q" + day.dt.quarter.astype(str)


def retrain(panel) -> dict[str, pd.DataFrame]:
    gbm = pd.read_parquet(ROLLING_FILE)
    targets = pd.DatetimeIndex(gbm["target"].unique())
    zero = record_forecast()
    naive = fr.naive_forecast(panel, targets)
    f = pd.concat([gbm, zero[zero["target"].isin(targets)], naive], ignore_index=True)
    sd = site_days(f, panel)
    groups = {"all": pd.Series(True, index=sd.index)}
    groups |= {str(y): sd["day"].dt.year == y for y in sorted(sd["day"].dt.year.unique())}
    quarter = quarter_label(sd["day"])
    by_quarter = errors(sd, {q: quarter == q for q in sorted(quarter.unique())})
    plant = sd["site_code"].isin(list(plant_sites(panel)))
    groups |= {PLANT: plant, PLAIN: ~plant}
    return {
        "retrain_errors": errors(sd, groups),
        "retrain_quarters": by_quarter.rename(columns={"group": "quarter"}),
        "retrain_differences": differences(sd, RETRAIN_PAIRS, groups),
        "retrain_sites": site_table(sd),
        "retrain_site_differences": site_differences(site_table(sd), RETRAIN_PAIRS),
    }


# --------------------------------------------------------------------------- #
# 2. A correction within the day
# --------------------------------------------------------------------------- #


def correction_long(wide: pd.DataFrame) -> pd.DataFrame:
    keep = wide.dropna(subset=["y"])
    return pd.concat(
        [
            keep[["site_code", "target", "step"]].assign(model=m, point=keep[m].to_numpy())
            for m in CORRECTION_MODELS
        ],
        ignore_index=True,
    )


def by_step(f: pd.DataFrame, panel, plant: set[str]) -> pd.DataFrame:
    """Error by half-hour after the issue (1 to 4), by site group."""
    actual = fa.actual_long(panel)
    m = f.merge(actual, on=["target", "site_code"]).dropna(subset=["point", "y"])
    m["ae"] = (m["y"] - m["point"]).abs()
    m["group"] = np.where(m["site_code"].isin(list(plant)), PLANT, PLAIN)
    per = m.groupby(["group", "model", "step", "site_code"], observed=True).agg(
        a=("y", "sum"), ae=("ae", "sum")
    )
    per["rate"] = 100 * per["ae"] / per["a"]
    return per.groupby(["group", "model", "step"])["rate"].mean().rename("nmae_pct").reset_index()


def correction(panel, config) -> dict[str, pd.DataFrame]:
    f = correction_long(pd.read_parquet(CORRECTION_FILE))
    sd = site_days(f, panel, min_periods=44)
    plant = plant_sites(panel)
    big = fa.day_table(panel, test_start=str(config["test_start"]))[
        ["site_code", "day", "big_step"]
    ]
    sd = sd.merge(big, on=["site_code", "day"], how="left")
    step = sd["big_step"].fillna(False).astype(bool)
    in_plant = sd["site_code"].isin(list(plant))
    groups = {
        "all": pd.Series(True, index=sd.index),
        PLANT: in_plant,
        PLAIN: ~in_plant,
        f"{PLANT}, days with a large step": in_plant & step,
        f"{PLANT}, other days": in_plant & ~step,
    }
    groups |= {str(y): sd["day"].dt.year == y for y in sorted(sd["day"].dt.year.unique())}
    return {
        "correction_errors": errors(sd, groups),
        "correction_differences": differences(sd, CORRECTION_PAIRS, groups),
        "correction_steps": by_step(f, panel, plant),
    }


# --------------------------------------------------------------------------- #
# 3. Measured and forecast weather as separate inputs
# --------------------------------------------------------------------------- #


def split_weather(panel) -> dict[str, pd.DataFrame]:
    files = sorted(fr.OUT_DIR.glob(SPLIT_GLOB))
    split = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    split = split.drop_duplicates(["model", "site_code", "target"])
    targets = pd.DatetimeIndex(split["target"].unique())
    frames = [split]
    for model in (ZERO_SHOT, "chronos2"):
        r = record_forecast(model)
        frames.append(r[r["target"].isin(targets)])
    sd = site_days(pd.concat(frames, ignore_index=True), panel)
    month = sd["day"].dt.month
    plant = sd["site_code"].isin(list(plant_sites(panel)))
    groups = {
        "all": pd.Series(True, index=sd.index),
        "summer (June to August)": month.isin([6, 7, 8]),
        "winter (December to February)": month.isin([12, 1, 2]),
        PLANT: plant,
        PLAIN: ~plant,
    }
    return {
        "split_errors": errors(sd, groups),
        "split_differences": differences(sd, SPLIT_PAIRS, groups),
    }


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, panel = fa.poc_panel()
    out: dict[str, pd.DataFrame] = {}
    if ROLLING_FILE.exists():
        out |= retrain(panel)
    if CORRECTION_FILE.exists():
        out |= correction(panel, config)
    if list(fr.OUT_DIR.glob(SPLIT_GLOB)):
        out |= split_weather(panel)
    for name, table in out.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
    summary = {}
    for key in ("retrain", "correction", "split"):
        if f"{key}_errors" in out:
            e = out[f"{key}_errors"]
            e = e[e["group"] == "all"].set_index("model")
            d = out[f"{key}_differences"]
            d = d[d["group"] == "all"]
            summary[key] = {
                "site_days": int(e["site_days"].iloc[0]),
                "nmae_pct": e["nmae_pct"].astype(float).round(2).to_dict(),
                "differences": {
                    f"{r.a} minus {r.b}": [round(r.diff_pts, 2), round(r.lo, 2), round(r.hi, 2)]
                    for r in d.itertuples()
                },
            }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, **out}
