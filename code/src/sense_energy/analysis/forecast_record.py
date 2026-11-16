"""The forecast findings on the whole record: do they hold outside the one test year?

The proof of concept scores day-ahead forecasts on 69 days between June 2025 and May 2026. A
zero-shot model has no training period, so it can be scored on every day of the meter record:
1,207 days from January 2023, three summers and four winters
(``forecast_probe.run_record``: Chronos-2 without covariates, and with the known-ahead
covariates and the weather forecast of the day before). This module scores those runs.

* **By period.** Error against the seasonal naive forecast, the share of the achievable peak
  cut that a battery gets when it plans on the forecast, and the share of the error at the
  sites with on-site generation, for each calendar year, for the proof of concept's window and
  for its 69 days (which must reproduce the stored numbers).
* **Event days.** Heat-wave and cold-spell days against the other days of the same site and
  calendar month: error, bias and captured peak cut, for the catalogue's events (the
  definition of ``extreme_events``) and for the site's own temperature percentiles, per summer
  and winter. Intervals from resampling sites.
* **The decision.** The same with a battery of half and twice the power and of one and four
  hours, to see what the value results owe to the battery that was chosen.

Outputs: ``code/reports/forecast_record/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import event_audit as ea
from . import extreme_events as ee
from . import flexibility as fx
from . import forecast_anatomy as fa

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_record"
FILES = {
    "chronos2": "forecasts_record_none.parquet",
    "chronos2_ifs": "forecasts_record_ifs.parquet",
}
NAIVE = "seasonal_naive"
POC_START, POC_END, POC_STRIDE = "2025-06-01", "2026-05-08", 5
MIN_CONTEXT = 0.8  # observed share of the six weeks before the issue
BATTERIES = [(0.2, 2.0), (0.1, 2.0), (0.4, 2.0), (0.2, 1.0), (0.2, 4.0)]
N_BOOT = 1000
EVENT_SETS = ("catalogue", "site percentiles")


# --------------------------------------------------------------------------- #
# Site-days
# --------------------------------------------------------------------------- #


def load_forecasts() -> pd.DataFrame:
    frames = []
    for model, name in FILES.items():
        path = fa.OUT_DIR / name
        if path.exists():
            frames.append(pd.read_parquet(path).assign(model=model))
    if not frames:
        raise FileNotFoundError("no whole-record forecasts; run forecast_probe.run_record")
    return pd.concat(frames, ignore_index=True)


def naive_forecast(panel, targets: pd.DatetimeIndex) -> pd.DataFrame:
    """The same half-hour one week before, as a forecast in the long format."""
    y = panel.y_raw[panel.sites].shift(336)
    y = y[y.index.isin(targets)]
    out = y.stack().rename("point").rename_axis(["target", "site_code"]).reset_index()
    return out.assign(model=NAIVE)


def context_ok(panel) -> pd.DataFrame:
    """Per site and day: at least ``MIN_CONTEXT`` of the half-hours of the six weeks before
    the day are observed (the model's context is not mostly filled-in values)."""
    day = pd.DatetimeIndex(panel.local_date.astype("datetime64[ns]"))
    share = panel.y_raw[panel.sites].notna().groupby(day).mean()
    ok = share.rolling(42, min_periods=42).mean().shift(1) >= MIN_CONTEXT
    return ok.stack().rename("context_ok").rename_axis(["day", "site_code"]).reset_index()


def matrices(f: pd.DataFrame, actual: pd.DataFrame) -> dict:
    """Site-days with 48 observed and forecast half-hours of one model: the key and the
    matrices (kW) of the demand, the forecast and its 70 % quantile (if present)."""
    cols = ["site_code", "target", "point"] + (["q70"] if "q70" in f.columns else [])
    m = f[cols].merge(actual, on=["target", "site_code"], how="inner")
    m = m.dropna(subset=["point", "y"]).sort_values(["site_code", "day", "target"])
    m["slot"] = m.groupby(["site_code", "day"]).cumcount()
    m = m[m.groupby(["site_code", "day"])["slot"].transform("size") == 48]
    key = m[["site_code", "day"]].drop_duplicates().reset_index(drop=True)
    row = np.repeat(np.arange(len(key)), 48)
    slot = m["slot"].to_numpy()

    def mat(col):
        a = np.full((len(key), 48), np.nan)
        a[row, slot] = m[col].to_numpy() * 2.0
        return a

    out = {"key": key, "act": mat("y"), "point": mat("point")}
    if "q70" in m.columns and m["q70"].notna().all():
        out["q70"] = mat("q70")
    return out


def value_columns(mx: dict, mean_kw: pd.Series, share: float, hours: float) -> pd.DataFrame:
    """Peak cut under perfect foresight and with a plan on the forecast, its 70 % quantile and
    the forecast with the day's true level, for a battery of ``share`` of the site's mean
    demand and ``hours`` of storage."""
    act, point = mx["act"], mx["point"]
    p_kw = share * mx["key"]["site_code"].map(mean_kw).to_numpy()
    e_kwh = p_kw * hours
    raw = act.max(axis=1)
    out = pd.DataFrame(
        {"raw_peak_kw": raw, "perfect_kw": raw - fx.plan_threshold(act, p_kw, e_kwh)}
    )
    plans = {"point": point, "level": point * (act.sum(axis=1) / point.sum(axis=1))[:, None]}
    if "q70" in mx:
        plans["q70"] = mx["q70"]
    for name, fc in plans.items():
        out[f"cut_{name}_kw"] = raw - fx.realised_peak(
            fx.plan_threshold(fc, p_kw, e_kwh), act, p_kw, e_kwh
        )
    return out


def site_days(f: pd.DataFrame, panel, batteries=BATTERIES) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per model, site and day: demand, absolute error, bias and the peak cuts with the
    reference battery; and the same peak cuts for every battery in ``batteries`` (long)."""
    actual = fa.actual_long(panel)
    mean_kw = (panel.y_raw[panel.sites] * 2.0).mean()
    targets = pd.DatetimeIndex(f["target"].unique())
    frames = {m: g for m, g in f.groupby("model")}
    frames[NAIVE] = naive_forecast(panel, targets)
    ok = context_ok(panel)
    rows, bat = [], []
    for model, g in frames.items():
        mx = matrices(g, actual)
        key = mx["key"].assign(
            model=model,
            a=mx["act"].sum(axis=1) / 2.0,
            ae=np.abs(mx["act"] - mx["point"]).sum(axis=1) / 2.0,
            bias_pct=100 * (mx["point"].sum(axis=1) / mx["act"].sum(axis=1) - 1),
        )
        keep = key.merge(ok, on=["day", "site_code"], how="left")["context_ok"].fillna(False)
        keep = keep.to_numpy()
        for share, hours in batteries:
            v = value_columns(mx, mean_kw, share, hours)
            v = pd.concat([key[["site_code", "day", "model"]], v], axis=1)[keep]
            bat.append(v.assign(battery_share=share, battery_hours=hours))
        ref = bat[-len(batteries)].drop(columns=["battery_share", "battery_hours"])
        rows.append(key[keep].merge(ref, on=["site_code", "day", "model"]))
        logger.info("%s: %d site-days", model, int(keep.sum()))
    return pd.concat(rows, ignore_index=True), pd.concat(bat, ignore_index=True)


# --------------------------------------------------------------------------- #
# Statistics with intervals from resampling sites
# --------------------------------------------------------------------------- #


def _site_sums(x: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return x.groupby("site_code")[cols].sum(min_count=1)


def _boot(per: pd.DataFrame, stat, seed: int = 0) -> tuple[float, float, float]:
    """``stat`` of a per-site table with the 95 % interval from resampling sites."""
    if len(per) < 5:
        return float(stat(per)), np.nan, np.nan
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(per), (N_BOOT, len(per)))
    vals = np.array([stat(per.iloc[i]) for i in pick])
    lo, hi = np.nanquantile(vals, [0.025, 0.975])
    return float(stat(per)), float(lo), float(hi)


def nmae(per: pd.DataFrame, e: str = "ae", a: str = "a") -> float:
    """Mean over sites of the site's error over its demand, per cent."""
    return float(100 * (per[e] / per[a]).mean())


def captured(per: pd.DataFrame, cut: str = "cut_point_kw", perfect: str = "perfect_kw") -> float:
    return float(100 * per[cut].sum() / per[perfect].sum())


# --------------------------------------------------------------------------- #
# 1. By period
# --------------------------------------------------------------------------- #


def windows(sd: pd.DataFrame) -> dict[str, pd.Series]:
    day = sd["day"]
    poc_days = pd.date_range(POC_START, POC_END, freq=f"{POC_STRIDE}D")
    out = {"whole record": pd.Series(True, index=sd.index)}
    for year in sorted(day.dt.year.unique()):
        out[str(year)] = day.dt.year == year
    out["proof-of-concept window, every day"] = day.between(POC_START, POC_END)
    out["proof-of-concept window, its 69 days"] = day.isin(poc_days)
    return out


def by_period(sd: pd.DataFrame, plant_sites: set[str], big: pd.DataFrame) -> pd.DataFrame:
    sd = sd.merge(big, on=["site_code", "day"], how="left")
    naive = sd[sd["model"] == NAIVE].set_index(["site_code", "day"])["ae"].rename("ae_naive")
    rows = []
    for name, mask in windows(sd).items():
        for model, x in sd[mask].groupby("model"):
            x = x.join(naive, on=["site_code", "day"])
            x = x.dropna(subset=["ae_naive"]) if model != NAIVE else x
            cols = ["ae", "a", "ae_naive", "perfect_kw", "cut_point_kw", "cut_level_kw"]
            cols += ["cut_q70_kw"] if x["cut_q70_kw"].notna().any() else []
            per = _site_sums(x, cols)
            err, lo, hi = _boot(per, nmae)
            cap, clo, chi = _boot(per, captured)
            plant = per.index.isin(list(plant_sites))
            step = x["big_step"].fillna(False).astype(bool)
            in_plant = x["site_code"].isin(list(plant_sites))
            rows.append(
                {
                    "window": name,
                    "model": model,
                    "sites": len(per),
                    "site_days": len(x),
                    "days": x["day"].nunique(),
                    "nmae_pct": err,
                    "nmae_lo": lo,
                    "nmae_hi": hi,
                    "naive_nmae_pct": nmae(per, "ae_naive"),
                    "skill_pct": 100 * (1 - (per["ae"] / per["ae_naive"]).median()),
                    "captured_point_pct": cap,
                    "captured_lo": clo,
                    "captured_hi": chi,
                    "captured_q70_pct": captured(per, "cut_q70_kw")
                    if "cut_q70_kw" in per
                    else np.nan,
                    "captured_level_pct": captured(per, "cut_level_kw"),
                    "plant_error_share_pct": 100 * per.loc[plant, "ae"].sum() / per["ae"].sum(),
                    "plant_demand_share_pct": 100 * per.loc[plant, "a"].sum() / per["a"].sum(),
                    "top3_error_share_pct": 100 * per["ae"].nlargest(3).sum() / per["ae"].sum(),
                    "large_step_day_share_pct": 100 * step.mean(),
                    "large_step_error_share_pct": 100 * x.loc[step, "ae"].sum() / x["ae"].sum(),
                    "plant_large_step_day_share_pct": 100 * (step & in_plant).mean(),
                    "plant_large_step_error_share_pct": 100
                    * x.loc[step & in_plant, "ae"].sum()
                    / x["ae"].sum(),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Event days against the other days of the same site and month
# --------------------------------------------------------------------------- #


def event_flags(panel, events: str) -> pd.DataFrame:
    """Site x day flags of heat-wave and cold-spell days under one definition."""
    inp = ea.load_inputs()
    codes = [s for s in panel.sites if s in inp.codes]
    masks = ee.event_masks(inp.days, codes, ea.site_events(inp, events))
    out = []
    for kind, m in masks.items():
        x = m.stack()
        x = x[x].rename_axis(["day", "site_code"]).reset_index()[["day", "site_code"]]
        out.append(x.assign(event=kind))
    return pd.concat(out, ignore_index=True)


def _season(day: pd.Series, kind: str) -> pd.Series:
    year = day.dt.year - ((day.dt.month <= 6) & (kind == "cold spell"))
    if kind == "heatwave":
        return "summer " + year.astype(str)
    return "winter " + year.astype(str) + "/" + ((year + 1) % 100).astype(str).str.zfill(2)


def event_days(sd: pd.DataFrame, flags: pd.DataFrame, plant_sites: set[str]) -> pd.DataFrame:
    """Error, bias and captured peak cut on event days and on the other days of the same site
    and calendar month, with the difference and its interval from resampling sites; by event
    type, season, model and site group."""
    x = sd.merge(flags, on=["day", "site_code"], how="left")
    x["event"] = x["event"].fillna("other")
    x["month"] = x["day"].dt.to_period("M").astype(str)
    x["plant"] = np.where(
        x["site_code"].isin(list(plant_sites)), "on-site generation", "no on-site generation"
    )
    rows = []
    for kind in ee.TYPES:
        k = x[x["event"].isin([kind, "other"])]
        both = k.groupby(["model", "site_code", "month"])["event"].transform("nunique") == 2
        k = k[both].assign(season=_season(k.loc[both, "day"], kind))
        groups = [("all", "all sites", k)]
        groups += [(s, "all sites", g) for s, g in k.groupby("season")]
        groups += [("all", p, g) for p, g in k.groupby("plant")]
        for season, sites, g in groups:
            for model, m in g.groupby("model"):
                e, o = m[m["event"] == kind], m[m["event"] == "other"]
                cols = ["ae", "a", "perfect_kw", "cut_point_kw"]
                per = _site_sums(e, cols).join(
                    _site_sums(o, cols), lsuffix="_e", rsuffix="_o", how="inner"
                )
                if len(per) < 5:
                    continue

                def d_err(p):
                    return nmae(p, "ae_e", "a_e") - nmae(p, "ae_o", "a_o")

                def d_cap(p):
                    return captured(p, "cut_point_kw_e", "perfect_kw_e") - captured(
                        p, "cut_point_kw_o", "perfect_kw_o"
                    )

                de, de_lo, de_hi = _boot(per, d_err)
                dc, dc_lo, dc_hi = _boot(per, d_cap)
                rows.append(
                    {
                        "type": kind,
                        "season": season,
                        "sites_group": sites,
                        "model": model,
                        "sites": len(per),
                        "event_site_days": len(e),
                        "other_site_days": len(o),
                        "nmae_event_pct": nmae(per, "ae_e", "a_e"),
                        "nmae_other_pct": nmae(per, "ae_o", "a_o"),
                        "nmae_diff_pts": de,
                        "nmae_diff_lo": de_lo,
                        "nmae_diff_hi": de_hi,
                        "bias_event_pct": float(e["bias_pct"].median()),
                        "bias_other_pct": float(o["bias_pct"].median()),
                        "captured_event_pct": captured(per, "cut_point_kw_e", "perfect_kw_e"),
                        "captured_other_pct": captured(per, "cut_point_kw_o", "perfect_kw_o"),
                        "captured_diff_pts": dc,
                        "captured_diff_lo": dc_lo,
                        "captured_diff_hi": dc_hi,
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. The decision
# --------------------------------------------------------------------------- #


def decision(bat: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Captured peak cut by battery: overall, and on heat-wave days against the other days of
    the same site and month."""
    rows = []
    heat = flags[flags["event"] == "heatwave"]
    for (share, hours, model), x in bat.groupby(["battery_share", "battery_hours", "model"]):
        cols = ["perfect_kw", "cut_point_kw"] + (
            ["cut_q70_kw"] if x["cut_q70_kw"].notna().any() else []
        )
        per = _site_sums(x, cols)
        cap, lo, hi = _boot(per, captured)
        k = x.merge(heat, on=["day", "site_code"], how="left")
        k["event"] = k["event"].fillna("other")
        k["month"] = k["day"].dt.to_period("M").astype(str)
        k = k[k.groupby(["site_code", "month"])["event"].transform("nunique") == 2]
        e, o = k[k["event"] == "heatwave"], k[k["event"] == "other"]
        rows.append(
            {
                "battery_share_of_mean_demand": share,
                "battery_hours": hours,
                "model": model,
                "captured_point_pct": cap,
                "lo": lo,
                "hi": hi,
                "captured_q70_pct": captured(per, "cut_q70_kw") if "cut_q70_kw" in per else np.nan,
                "perfect_cut_share_of_peak_pct": 100
                * x["perfect_kw"].sum()
                / x["raw_peak_kw"].sum()
                if "raw_peak_kw" in x
                else np.nan,
                "heatwave_captured_pct": 100 * e["cut_point_kw"].sum() / e["perfect_kw"].sum(),
                "other_days_captured_pct": 100 * o["cut_point_kw"].sum() / o["perfect_kw"].sum(),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. Are the differences between the stored models real?
# --------------------------------------------------------------------------- #

MODEL_PAIRS = (
    ("chronos2_ifs", "seasonal_naive", "best model against seasonal naive"),
    ("chronos2", "lightgbm", "zero-shot against trained, no covariates"),
    ("chronos2_ifs", "lightgbm_ifs", "zero-shot against trained, with covariates"),
    ("chronos2_ifs", "tnp", "zero-shot against a neural process trained on the panel"),
    ("chronos2_ifs", "timesfm3_ifs", "the two best zero-shot models"),
    ("chronos2_ifs", "chronos2", "covariates and weather forecast against none"),
    ("chronos2_ifs", "chronos2_era5", "weather forecast against observed weather"),
    ("lightgbm_ifs", "lightgbm_ifs_noid", "trained model with against without site identity"),
)


def model_differences(panel) -> pd.DataFrame:
    """Paired differences in error between stored models of the proof of concept (the 69 test
    days), with 95 % intervals from resampling weeks (``forecast_anatomy.paired_bootstrap``)."""
    actual = fa.actual_long(panel)
    names = sorted({m for a, b, _ in MODEL_PAIRS for m in (a, b)})
    frames = []
    for name in names:
        if name == "seasonal_naive":
            f = pd.read_parquet(fa.POC / "forecasts_baselines.parquet")
            f = f[f["model"] == name]
        else:
            path = fa.POC / f"forecasts_{name}.parquet"
            if not path.exists():
                continue
            f = pd.read_parquet(path).assign(model=name)
        m = f[["model", "site_code", "target", "point"]].merge(actual, on=["target", "site_code"])
        m["ae"] = (m["y"] - m["point"]).abs()
        g = m.groupby(["model", "site_code", "day"], observed=True)
        frames.append(g.agg(n=("y", "size"), a=("y", "sum"), ae=("ae", "sum")).reset_index())
    sd = pd.concat(frames, ignore_index=True)
    sd = sd[sd["n"] >= fa.MIN_PERIODS]
    rows = []
    for a, b, label in MODEL_PAIRS:
        if {a, b} <= set(sd["model"]):
            x = sd[sd["model"].isin([a, b])]
            both = x.groupby(["site_code", "day"])["model"].transform("nunique") == 2
            d = fa.paired_bootstrap(x[both], a, b)
            rows.append(
                {
                    "comparison": label,
                    "a": a,
                    "b": b,
                    "nmae_a_pct": fa._nmae(x[both & (x["model"] == a)]),
                    "nmae_b_pct": fa._nmae(x[both & (x["model"] == b)]),
                    **d,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config, panel = fa.poc_panel()
    f = load_forecasts()
    sd, bat = site_days(f, panel)
    sd.to_parquet(OUT_DIR / "site_days.parquet", index=False)
    plant = set(pd.read_csv(ee.CHP_FILE)["site_code"]) & set(panel.sites)
    days = fa.day_table(panel, test_start=str(config["test_start"]))
    big = days[["site_code", "day", "big_step"]]
    periods = by_period(sd, plant, big)
    periods.to_csv(OUT_DIR / "by_period.csv", index=False)
    events = []
    flags = {}
    for name in EVENT_SETS:
        flags[name] = event_flags(panel, name)
        events.append(event_days(sd, flags[name], plant).assign(definition=name))
    events = pd.concat(events, ignore_index=True)
    events.to_csv(OUT_DIR / "event_days.csv", index=False)
    dec = decision(bat, flags["catalogue"])
    dec.to_csv(OUT_DIR / "decision.csv", index=False)
    model_differences(panel).to_csv(OUT_DIR / "model_differences.csv", index=False)
    best = "chronos2_ifs" if "chronos2_ifs" in set(sd["model"]) else "chronos2"
    whole = periods[(periods["window"] == "whole record") & (periods["model"] == best)].iloc[0]
    summary = {
        "models": sorted(sd["model"].unique()),
        "best": best,
        "site_days": int(whole["site_days"]),
        "days": int(whole["days"]),
        "whole_record": whole.drop(["window", "model"]).astype(float).round(2).to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return {"summary": summary, "by_period": periods, "events": events, "decision": dec}
