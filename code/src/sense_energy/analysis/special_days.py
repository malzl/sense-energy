"""A rule for special days on top of a zero-shot forecast.

The zero-shot forecast is weakest on public holidays: a model that reads six weeks of history
has seen at most one of them. Arora and Taylor (2013, IEEE Trans. Power Systems 28; 2018, EJOR
266) handle anomalous load with a rule: each special day refers to a *corresponding past
special day*, the most recent day of the same kind at the same time of year, weekday with
weekday and weekend with weekend, and the days next to a special day (proximity days) are
special days of their own kinds. Their rule sets the yearly lag of a seasonal ARMA model. Here
the same rule corrects a forecast that has no seasonal lag to set:

    forecast for the special day x (demand / forecast on the corresponding past special day)

by half-hour (the ratio smoothed over five half-hours and held between 0.4 and 2), or as one
ratio for the day. Only the past is used: the corresponding day lies months before the issue
time. The benchmark of the papers, the load of the most recent Sunday, is scored beside it.

Kinds of day (England):

* bank holidays, by name (substitute days are a name of their own);
* the Christmas period: 24 December, 27 to 31 December and 2 January where they are no bank
  holiday;
* the working day before and the working day after a bank holiday outside that period.

Scored on the whole-record forecasts of ``forecast_probe.run_record`` from 2024 on, so that
every day can have a corresponding past day in the record. Outputs:
``code/reports/special_days/``.
"""

from __future__ import annotations

import json

import holidays
import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import forecast_anatomy as fa
from . import forecast_record as fr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "special_days"
EVAL_START = "2024-01-01"
RATIO_LIMITS = (0.4, 2.0)
SMOOTH = 5  # half-hours
KIND_ORDER = (
    "bank holiday",
    "Christmas period",
    "day before a bank holiday",
    "day after a bank holiday",
)
RULES = ("forecast", "rule, by half-hour", "rule, one ratio for the day", "most recent Sunday")


# --------------------------------------------------------------------------- #
# The calendar and the rule
# --------------------------------------------------------------------------- #


def special_calendar(start: str, end: str) -> pd.DataFrame:
    """Every special day between ``start`` and ``end``: its kind, the name that a
    corresponding past day must share, and whether it falls on a weekend."""
    days = pd.date_range(start, end, freq="D")
    hol = holidays.UK(subdiv="ENG", years=range(days.year.min() - 1, days.year.max() + 2))
    # a substitute day (a holiday that fell on a weekend, observed on a weekday) is a kind of
    # its own: it is an ordinary bank-holiday weekday, not the holiday itself
    name = {
        pd.Timestamp(d): "substitute bank holiday" if "(observed)" in n or "substitute" in n else n
        for d, n in hol.items()
    }
    rows = {}
    for d in days:
        if d in name:
            rows[d] = ("bank holiday", name[d])
    for d in days:
        if d in rows:
            continue
        md = (d.month, d.day)
        if md == (12, 24):
            rows[d] = ("Christmas period", "Christmas Eve")
        elif d.month == 12 and 27 <= d.day <= 31:
            rows[d] = ("Christmas period", "between Christmas and New Year")
        elif md == (1, 2):
            rows[d] = ("Christmas period", "day after New Year")
    working = [d for d in days if d.dayofweek < 5 and d not in rows]
    for d in working:
        nxt, prev = d + pd.Timedelta(days=1), d - pd.Timedelta(days=1)
        while nxt.dayofweek >= 5:
            nxt += pd.Timedelta(days=1)
        while prev.dayofweek >= 5:
            prev -= pd.Timedelta(days=1)
        if rows.get(nxt, ("",))[0] == "bank holiday":
            rows[d] = ("day before a bank holiday", "before " + name[nxt])
        elif rows.get(prev, ("",))[0] == "bank holiday":
            rows[d] = ("day after a bank holiday", "after " + name[prev])
    out = pd.DataFrame(
        [
            {"day": d, "kind": k, "name": n, "weekend": d.dayofweek >= 5}
            for d, (k, n) in rows.items()
        ]
    )
    return out.sort_values("day").reset_index(drop=True)


def corresponding_past_day(cal: pd.DataFrame, available: set) -> pd.Series:
    """For each special day the corresponding past special day of the rule: the most recent
    earlier day with the same name and the same weekday or weekend class that is in
    ``available``; without one of the same class, the most recent with the same name."""
    past = {}
    for _, g in cal.groupby("name"):
        g = g.sort_values("day")
        for r in g.itertuples():
            earlier = g[(g["day"] < r.day) & g["day"].isin(available)]
            same = earlier[earlier["weekend"] == r.weekend]
            pick = same if len(same) else earlier
            past[r.day] = pick["day"].iloc[-1] if len(pick) else pd.NaT
    return cal["day"].map(past)


def _smooth(x: np.ndarray, width: int = SMOOTH) -> np.ndarray:
    """Centred moving mean along the half-hours of a day (edges: the values that exist)."""
    pad = width // 2
    padded = np.pad(x, ((0, 0), (pad, pad)), mode="edge")
    kernel = np.ones(width) / width
    return np.apply_along_axis(lambda v: np.convolve(v, kernel, mode="valid"), 1, padded)


def adjust(point: np.ndarray, past_act: np.ndarray, past_point: np.ndarray, by: str) -> np.ndarray:
    """Forecasts (rows x 48) corrected with the anomaly of the corresponding past days."""
    with np.errstate(all="ignore"):
        if by == "half-hour":
            ratio = _smooth(past_act) / _smooth(past_point)
        else:
            ratio = (past_act.sum(axis=1) / past_point.sum(axis=1))[:, None] * np.ones(48)
    ratio = np.clip(np.where(np.isfinite(ratio), ratio, 1.0), *RATIO_LIMITS)
    return point * ratio


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def special_site_days(f: pd.DataFrame, panel, model: str) -> pd.DataFrame:
    """Per site and special day from ``EVAL_START`` on: demand and the absolute error of the
    forecast, of the two forms of the rule and of the most recent Sunday."""
    actual = fa.actual_long(panel)
    mx = fr.matrices(f[f["model"] == model], actual)
    key, act, point = mx["key"], mx["act"], mx["point"]
    cal = special_calendar(str(key["day"].min().date()), str(key["day"].max().date()))
    kind = cal.set_index("day")
    row = {(s, d): i for i, (s, d) in enumerate(zip(key["site_code"], key["day"], strict=True))}
    rows = []
    for site, g in key.groupby("site_code"):
        have = set(g["day"])
        past = corresponding_past_day(cal, have)
        for d, p in zip(cal["day"], past, strict=True):
            if d < pd.Timestamp(EVAL_START) or (site, d) not in row:
                continue
            i = row[(site, d)]
            sunday = d - pd.Timedelta(days=(d.dayofweek + 1) % 7 or 7)
            if sunday >= d - pd.Timedelta(days=1):
                sunday -= pd.Timedelta(days=7)  # Sunday itself is not complete at the issue
            rec = {
                "model": model,
                "site_code": site,
                "day": d,
                "kind": kind.loc[d, "kind"],
                "name": kind.loc[d, "name"],
                "past_day": p,
                "a": act[i].sum() / 2.0,
                "ae_forecast": np.abs(act[i] - point[i]).sum() / 2.0,
                "bias_pct": 100 * (point[i].sum() / act[i].sum() - 1),
            }
            if pd.notna(p):
                j = row[(site, p)]
                for name, by in (("half-hour", "half-hour"), ("day", "day")):
                    adj = adjust(point[i][None], act[j][None], point[j][None], by)[0]
                    rec[f"ae_rule_{name}"] = np.abs(act[i] - adj).sum() / 2.0
            if (site, sunday) in row:
                rec["ae_sunday"] = np.abs(act[i] - act[row[(site, sunday)]]).sum() / 2.0
            rows.append(rec)
    return pd.DataFrame(rows)


def score_table(sd: pd.DataFrame, groups: pd.Series | None = None) -> pd.DataFrame:
    """Error of the forecast and of the rule by kind of special day, on the site-days that
    have a corresponding past day, with the difference and its interval from resampling
    sites."""
    x = sd.dropna(subset=["ae_rule_half-hour"]).copy()
    x["sites_group"] = "all sites" if groups is None else x["site_code"].map(groups)
    rows = []
    sets = [("all special days", x), *x.groupby("kind")]
    for model in x["model"].unique():
        for kind, g in sets:
            g = g[g["model"] == model]
            for group, gg in [
                ("all sites", g),
                *(g.groupby("sites_group") if groups is not None else []),
            ]:
                cols = ["a", "ae_forecast", "ae_rule_half-hour", "ae_rule_day", "ae_sunday"]
                per = gg.groupby("site_code")[cols].sum(min_count=1)
                if len(per) < 5:
                    continue

                def diff(p, col="ae_rule_half-hour"):
                    return fr.nmae(p, col) - fr.nmae(p, "ae_forecast")

                d, lo, hi = fr._boot(per, diff)
                rows.append(
                    {
                        "model": model,
                        "kind": kind,
                        "sites_group": group,
                        "sites": len(per),
                        "site_days": len(gg),
                        "days": gg["day"].nunique(),
                        "nmae_forecast_pct": fr.nmae(per, "ae_forecast"),
                        "nmae_rule_half_hour_pct": fr.nmae(per, "ae_rule_half-hour"),
                        "nmae_rule_day_pct": fr.nmae(per, "ae_rule_day"),
                        "nmae_sunday_pct": fr.nmae(per.dropna(subset=["ae_sunday"]), "ae_sunday"),
                        "rule_minus_forecast_pts": d,
                        "lo": lo,
                        "hi": hi,
                        "median_bias_forecast_pct": float(gg["bias_pct"].median()),
                    }
                )
    return pd.DataFrame(rows)


def by_name(sd: pd.DataFrame) -> pd.DataFrame:
    """The same by named special day (best model), to see which days the rule helps."""
    x = sd.dropna(subset=["ae_rule_half-hour"])
    rows = []
    for (model, kind, name), g in x.groupby(["model", "kind", "name"]):
        per = g.groupby("site_code")[["a", "ae_forecast", "ae_rule_half-hour"]].sum()
        rows.append(
            {
                "model": model,
                "kind": kind,
                "name": name,
                "site_days": len(g),
                "days": g["day"].nunique(),
                "nmae_forecast_pct": fr.nmae(per, "ae_forecast"),
                "nmae_rule_pct": fr.nmae(per, "ae_rule_half-hour"),
                "median_bias_forecast_pct": float(g["bias_pct"].median()),
            }
        )
    return pd.DataFrame(rows)


def overall_effect(sd: pd.DataFrame, record: pd.DataFrame) -> pd.DataFrame:
    """What the rule changes over all days from ``EVAL_START`` on: the error of every day with
    the rule applied on the bank holidays that have a corresponding past day, and with the rule
    applied on every kind of special day."""
    rows = []
    for model, g in sd.groupby("model"):
        base = record[(record["model"] == model) & (record["day"] >= EVAL_START)]
        per = base.groupby("site_code")[["a", "ae"]].sum()
        ok = g.dropna(subset=["ae_rule_half-hour"])
        gain = ok["ae_rule_half-hour"] - ok["ae_forecast"]
        holiday = ok["kind"] == "bank holiday"
        for name, mask in (("ae_rule_holidays", holiday), ("ae_rule_all", holiday | True)):
            add = gain[mask].groupby(ok.loc[mask, "site_code"]).sum()
            per[name] = per["ae"] + add.reindex(per.index).fillna(0.0)
        special = g.groupby("site_code")["ae_forecast"].sum().reindex(per.index).fillna(0.0)
        hol = g[g["kind"] == "bank holiday"]
        hol_err = hol.groupby("site_code")["ae_forecast"].sum().reindex(per.index).fillna(0.0)

        def d_hol(p):
            return fr.nmae(p, "ae_rule_holidays") - fr.nmae(p, "ae")

        d, lo, hi = fr._boot(per, d_hol)
        rows.append(
            {
                "model": model,
                "sites": len(per),
                "site_days": len(base),
                "special_share_of_days_pct": 100 * len(g) / len(base),
                "special_share_of_error_pct": 100 * special.sum() / per["ae"].sum(),
                "holiday_share_of_days_pct": 100 * len(hol) / len(base),
                "holiday_share_of_error_pct": 100 * hol_err.sum() / per["ae"].sum(),
                "nmae_all_days_pct": fr.nmae(per, "ae"),
                "nmae_rule_on_bank_holidays_pct": fr.nmae(per, "ae_rule_holidays"),
                "nmae_rule_on_all_special_days_pct": fr.nmae(per, "ae_rule_all"),
                "holiday_rule_change_pts": d,
                "lo": lo,
                "hi": hi,
            }
        )
    return pd.DataFrame(rows)


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, panel = fa.poc_panel()
    f = fr.load_forecasts()
    sd = pd.concat(
        [special_site_days(f, panel, m) for m in sorted(f["model"].unique())], ignore_index=True
    )
    sd.to_csv(OUT_DIR / "special_site_days.csv", index=False)
    plant = set(pd.read_csv(fa.OUT_DIR.parent / "chp_dispatch" / "classification.csv")["site_code"])
    groups = pd.Series(
        {s: "on-site generation" if s in plant else "no on-site generation" for s in panel.sites}
    )
    scores = score_table(sd, groups)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    names = by_name(sd)
    names.to_csv(OUT_DIR / "scores_by_name.csv", index=False)
    record = pd.read_parquet(
        fr.OUT_DIR / "site_days.parquet", columns=["model", "site_code", "day", "a", "ae"]
    )
    overall = overall_effect(sd, record)
    overall.to_csv(OUT_DIR / "overall.csv", index=False)
    summary = {
        "evaluation_start": EVAL_START,
        "scores": scores[scores["sites_group"] == "all sites"].round(2).to_dict(orient="records"),
        "overall": overall.round(2).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return {"summary": summary, "scores": scores, "names": names, "overall": overall}
