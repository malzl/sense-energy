"""Where the forecast range fails, and an online correction for it.

The day-ahead forecasts of the whole record (``forecast_probe.run_record``) carry five
quantiles. The 10 % and 90 % quantiles should hold the demand 80 % of the time, with 10 % of
the half-hours below and 10 % above. This module

1. counts how often the demand falls below, inside and above that range, for the whole record
   and by season, time of day, site group, day type, year and event days (``coverage``), with
   the share of the demand below each of the five quantiles (``reliability``);
2. corrects the two bounds online, after Angelopoulos, Candes and Tibshirani (2023, conformal
   PID control). The score of a bound is the distance of the demand beyond it, in units of the
   day's mean forecast. For each site, bound and half-hour of the day a tracked offset moves
   up by ``eta * (1 - alpha)`` after a miss and down by ``eta * alpha`` after a hit (quantile
   tracking, their P control); the second variant adds their tangent integrator of the
   coverage error (PI control). The learning rate is their default, 0.1 times a trailing scale
   of the scores; the scale is the 95th percentile of the absolute scores and not the maximum,
   because a plant trip gives one score that is far outside the rest. Their third part, a
   model that forecasts the scores, is not used. The comparator is a fixed rule: the 90th
   percentile of the scores of the last 90 days.

Feedback is delayed as in operation: the forecast for day D is issued at 16:30 on D-1, so the
half-hours up to 16:00 use the scores up to D-1 and the later half-hours the scores up to D-2.
The first ``WARM_UP`` days are not scored for any method.

Outputs: ``code/reports/forecast_range/``.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..experiments.features import _holiday_set
from ..logging_utils import get_logger
from . import extreme_events as ee
from . import forecast_anatomy as fa
from . import forecast_record as fr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_range"
QCOLS = ["q10", "q30", "q50", "q70", "q90"]
LEVELS = (0.1, 0.3, 0.5, 0.7, 0.9)
ALPHA = 0.1  # nominal share outside each bound
WARM_UP = 90  # days at the start that no method is scored on
LATE_SLOT = 32  # first half-hour (16:00-16:30) whose score of D-1 is not known at the issue
LEARNING_RATE = 0.1
LEARNING_RATES = (0.02, 0.05, 0.1, 0.2, 0.5)
SCALE_DAYS = 30
ROLLING_DAYS = 90
MIN_HALF_HOURS = 240  # per site in a group
N_BOOT = 1000
RAW, ROLLING, TRACKED, INTEGRATED = (
    "model",
    "rolling quantile",
    "quantile tracking",
    "tracking with integrator",
)
METHODS = (RAW, ROLLING, TRACKED, INTEGRATED)
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring"} | {
    6: "summer",
    7: "summer",
    8: "summer",
    9: "autumn",
    10: "autumn",
    11: "autumn",
}
SEASON_ORDER = ("winter", "spring", "summer", "autumn")


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


def cube(f: pd.DataFrame, panel) -> dict:
    """Dense arrays (days x sites x 48) of the demand and the five quantiles of one model, on
    the site-days of the whole-record evaluation (48 observed half-hours and a context that is
    at least 80 % observed); missing elsewhere."""
    actual = fa.actual_long(panel)
    m = f[["site_code", "target", *QCOLS]].merge(actual, on=["target", "site_code"], how="inner")
    m = m.dropna(subset=["y", *QCOLS]).sort_values(["site_code", "day", "target"])
    m["slot"] = m.groupby(["site_code", "day"]).cumcount()
    m = m[m.groupby(["site_code", "day"])["slot"].transform("size") == 48]
    m = m.merge(fr.context_ok(panel), on=["day", "site_code"], how="left")
    m = m[m["context_ok"].fillna(False).astype(bool)]
    days = pd.date_range(m["day"].min(), m["day"].max(), freq="D")
    sites = sorted(m["site_code"].unique())
    di, si = days.get_indexer(m["day"]), pd.Index(sites).get_indexer(m["site_code"])
    out: dict = {"days": days, "sites": sites}
    for c in ["y", *QCOLS]:
        a = np.full((len(days), len(sites), 48), np.nan, "float32")
        a[di, si, m["slot"].to_numpy()] = m[c].to_numpy()
        out[c] = a
    return out


def scores(c: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The day's scale (days x sites x 1) and the scores of the lower and the upper bound:
    how far the demand is beyond the bound, in units of the scale (negative inside)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        scale = np.nanmean(c["q50"], axis=2, keepdims=True)
    scale = np.where(scale > 0, scale, np.nan)
    return scale, (c["q10"] - c["y"]) / scale, (c["y"] - c["q90"]) / scale


# --------------------------------------------------------------------------- #
# Online correction of a bound
# --------------------------------------------------------------------------- #


def score_scale(score: np.ndarray, days: int = SCALE_DAYS) -> np.ndarray:
    """Per day and site: the 95th percentile of the absolute scores of the day, as a mean over
    the ``days`` before the day (known when the day's scores come in). (days x sites)"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        daily = np.nanpercentile(np.abs(score), 95, axis=2)
    b = pd.DataFrame(daily).rolling(days, min_periods=1).mean().shift(1)
    b = b.ffill().bfill()
    return b.fillna(b.stack().median() if b.notna().any().any() else 1.0).to_numpy()


def _delayed(early: np.ndarray, late: np.ndarray) -> np.ndarray:
    """The offset in use on a day: ``early`` for the half-hours before ``LATE_SLOT``."""
    return np.where(np.arange(early.shape[-1]) >= LATE_SLOT, late, early)


def track(
    score: np.ndarray,
    learning_rate: float = LEARNING_RATE,
    integrator: bool = False,
    alpha: float = ALPHA,
    delta: float = 0.01,
) -> np.ndarray:
    """Offsets (days x sites x 48) for one bound from quantile tracking, with delayed
    feedback. ``score`` is missing where a half-hour was not observed; such a half-hour gives
    no update. With ``integrator`` the tangent integrator of the summed coverage error is
    added (constants as in Appendix B of the paper, the score scale as K_I)."""
    D, S, K = score.shape
    b = score_scale(score)
    c_sat = 2 / np.pi * (np.ceil(np.log(max(D, 3)) * delta) - 1 / np.log(max(D, 3)))
    p_part = np.zeros((S, K))
    err_sum, n = np.zeros((S, K)), np.zeros((S, K))
    through_1 = np.zeros((S, K))  # with the scores up to the day before
    through_2 = np.zeros((S, K))  # with the scores up to two days before
    used = np.full(score.shape, np.nan, "float32")
    for d in range(D):
        q = _delayed(through_1, through_2)
        used[d] = q
        s = score[d]
        ok = np.isfinite(s)
        g = np.where(ok, (s > q) - alpha, 0.0)
        p_part = p_part + learning_rate * b[d][:, None] * g
        err_sum, n = err_sum + g, n + ok
        through_2 = through_1
        through_1 = p_part
        if integrator:
            arg = err_sum * np.log(n + 1) / (c_sat * (n + 1))
            through_1 = p_part + b[d][:, None] * np.tan(np.clip(arg, -1.47, 1.47))
    return used


def rolling_quantile(score: np.ndarray, days: int = ROLLING_DAYS, alpha: float = ALPHA):
    """Offsets from the (1 - alpha) quantile of the scores of the ``days`` before, with the
    same delay; zero until 30 scores are there."""
    D = score.shape[0]
    early = np.zeros(score.shape, "float32")
    for d in range(1, D):
        w = score[max(0, d - days) : d]
        n = np.isfinite(w).sum(axis=0)
        w = np.sort(np.where(np.isfinite(w), w, np.inf), axis=0)  # missing scores last
        pos = (1 - alpha) * np.maximum(n - 1, 0)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        q_lo = np.take_along_axis(w, lo[None], axis=0)[0]
        q_hi = np.take_along_axis(w, hi[None], axis=0)[0]
        with np.errstate(invalid="ignore"):
            q = q_lo + (pos - lo) * (q_hi - q_lo)
        early[d] = np.where(n >= 30, q, 0.0)
    late = np.concatenate([np.zeros_like(early[:1]), early[:-1]])
    return _delayed(early, late)


def bounds(c: dict, method: str, learning_rate: float = LEARNING_RATE) -> tuple:
    """Lower and upper bound (days x sites x 48) of one method (kept with the cube)."""
    if method == RAW:
        return c["q10"], c["q90"]
    cache = c.setdefault("_bounds", {})
    key = (method, learning_rate if method != ROLLING else None)
    if key not in cache:
        scale, s_lo, s_hi = scores(c)
        if method == ROLLING:
            off_lo, off_hi = rolling_quantile(s_lo), rolling_quantile(s_hi)
        else:
            kw = {"learning_rate": learning_rate, "integrator": method == INTEGRATED}
            off_lo, off_hi = track(s_lo, **kw), track(s_hi, **kw)
        cache[key] = (c["q10"] - off_lo * scale, c["q90"] + off_hi * scale)
    return cache[key]


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #


def site_stats(c: dict, lower: np.ndarray, upper: np.ndarray, mask: np.ndarray) -> pd.DataFrame:
    """Per site, over the half-hours in ``mask`` (broadcast to days x sites x 48): shares
    below, inside and above the range, its width and the quantile loss of the two bounds, the
    last two in per cent of the demand."""
    y = c["y"]
    ok = np.isfinite(y) & np.isfinite(lower) & np.isfinite(upper) & mask
    n = ok.sum(axis=(0, 2)).astype(float)

    def total(x):
        return np.where(ok, x, 0.0).sum(axis=(0, 2), dtype="float64")

    with np.errstate(invalid="ignore", divide="ignore"):
        below, above = total(y < lower) / n, total(y > upper) / n
        demand = total(y)
        loss = total(
            np.maximum(ALPHA * (y - lower), (ALPHA - 1) * (y - lower))
            + np.maximum((1 - ALPHA) * (y - upper), -ALPHA * (y - upper))
        )
        out = pd.DataFrame(
            {
                "site_code": c["sites"],
                "half_hours": n,
                "below_pct": 100 * below,
                "above_pct": 100 * above,
                "inside_pct": 100 * (1 - below - above),
                "width_pct": 100 * total(upper - lower) / demand,
                "loss_pct": 100 * loss / demand,
            }
        )
    return out[out["half_hours"] >= MIN_HALF_HOURS].reset_index(drop=True)


def _mean_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return (float(np.mean(x)) if len(x) else np.nan), np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), (N_BOOT, len(x)))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def summary_row(per: pd.DataFrame) -> dict:
    row: dict = {"sites": len(per), "half_hours": int(per["half_hours"].sum())}
    for col in ("below_pct", "above_pct", "inside_pct", "width_pct", "loss_pct"):
        mean, lo, hi = _mean_ci(per[col].to_numpy())
        row |= {col: mean, f"{col[:-4]}_lo": lo, f"{col[:-4]}_hi": hi}
    return row


def groupings(c: dict, plant: set[str], flags: pd.DataFrame) -> list[tuple[str, str, np.ndarray]]:
    """(grouping, group, mask) with masks that broadcast to days x sites x 48."""
    days, sites = c["days"], np.asarray(c["sites"])
    D, S = len(days), len(sites)
    by_day = lambda m: np.asarray(m, bool)[:, None, None]  # noqa: E731
    by_site = lambda m: np.asarray(m, bool)[None, :, None]  # noqa: E731
    out = [("all", "all", np.ones((1, 1, 1), bool))]
    season = days.month.map(SEASONS)
    out += [("season", s, by_day(season == s)) for s in SEASON_ORDER]
    hour = np.arange(48) // 8
    out += [
        ("time of day", f"{4 * h:02d}:00-{4 * h + 4:02d}:00", (hour == h)[None, None, :])
        for h in range(6)
    ]
    has_plant = np.isin(sites, list(plant))
    out += [("site group", "on-site generation", by_site(has_plant))]
    out += [("site group", "no on-site generation", by_site(~has_plant))]
    holiday = np.isin(
        days.to_numpy().astype("datetime64[D]"), list(_holiday_set(range(2022, 2028)))
    )
    weekend = (days.dayofweek >= 5) & ~holiday
    out += [("day type", "working day", by_day(~weekend & ~holiday))]
    out += [("day type", "weekend", by_day(weekend)), ("day type", "bank holiday", by_day(holiday))]
    out += [("year", str(y), by_day(days.year == y)) for y in sorted(set(days.year))]
    event = np.zeros((D, S), bool)
    for kind in ee.TYPES:
        k = flags[flags["event"] == kind]
        di, si = days.get_indexer(k["day"]), pd.Index(sites).get_indexer(k["site_code"])
        keep = (di >= 0) & (si >= 0)
        m = np.zeros((D, S), bool)
        m[di[keep], si[keep]] = True
        event |= m
        out.append(("event", f"{kind} day", m[:, :, None]))
    out.append(("event", "other day", ~event[:, :, None]))
    return out


def scored_days(c: dict) -> np.ndarray:
    return (np.arange(len(c["days"])) >= WARM_UP)[:, None, None]


def coverage(c: dict, plant: set[str], flags: pd.DataFrame, methods=METHODS) -> pd.DataFrame:
    rows = []
    groups = groupings(c, plant, flags)
    for method in methods:
        lower, upper = bounds(c, method)
        for grouping, group, mask in groups:
            per = site_stats(c, lower, upper, mask & scored_days(c))
            if len(per):
                rows.append(
                    {"method": method, "grouping": grouping, "group": group} | summary_row(per)
                )
    return pd.DataFrame(rows)


def season_hour(c: dict, methods=(RAW, TRACKED)) -> pd.DataFrame:
    """Shares below and above the range for each season and hour of the day."""
    season = c["days"].month.map(SEASONS)
    rows = []
    for method in methods:
        lower, upper = bounds(c, method)
        for s in SEASON_ORDER:
            for h in range(24):
                mask = np.asarray(season == s)[:, None, None] & (np.arange(48) // 2 == h)
                per = site_stats(c, lower, upper, mask & scored_days(c))
                rows.append(
                    {"method": method, "season": s, "hour": h}
                    | per[["below_pct", "above_pct", "inside_pct"]].mean().to_dict()
                )
    return pd.DataFrame(rows)


def reliability(c: dict, plant: set[str]) -> pd.DataFrame:
    """Share of the demand below each quantile of the model, by site group."""
    has_plant = np.isin(np.asarray(c["sites"]), list(plant))
    ok = np.isfinite(c["y"]) & scored_days(c)
    n = ok.sum(axis=(0, 2))
    rows = []
    for level, col in zip(LEVELS, QCOLS, strict=True):
        with np.errstate(invalid="ignore", divide="ignore"):
            share = 100 * (ok & (c["y"] < c[col])).sum(axis=(0, 2)) / n
        for group, m in (
            ("all sites", n > 0),
            ("on-site generation", has_plant),
            ("no on-site generation", ~has_plant),
        ):
            mean, lo, hi = _mean_ci(share[m & (n >= MIN_HALF_HOURS)])
            rows.append(
                {"group": group, "nominal_pct": 100 * level, "below_pct": mean, "lo": lo, "hi": hi}
            )
    return pd.DataFrame(rows)


def by_site(c: dict, methods=METHODS) -> pd.DataFrame:
    frames = []
    for method in methods:
        lower, upper = bounds(c, method)
        frames.append(site_stats(c, lower, upper, scored_days(c)).assign(method=method))
    return pd.concat(frames, ignore_index=True)


def over_time(c: dict, plant: set[str], methods=(RAW, ROLLING, TRACKED), days: int = 30):
    """Share inside the range in windows of ``days``, as the mean over the sites, by group."""
    has_plant = np.isin(np.asarray(c["sites"]), list(plant))
    rows = []
    for method in methods:
        lower, upper = bounds(c, method)
        ok = np.isfinite(c["y"]) & np.isfinite(lower) & np.isfinite(upper)
        inside = (ok & (c["y"] >= lower) & (c["y"] <= upper)).sum(axis=2)
        n = ok.sum(axis=2)
        roll = lambda a: pd.DataFrame(a).rolling(days, min_periods=days // 2).sum()  # noqa: E731
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = (100 * roll(inside) / roll(n).where(roll(n) >= 48 * days // 2)).to_numpy()
        for group, m in (("on-site generation", has_plant), ("no on-site generation", ~has_plant)):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean = np.nanmean(rate[:, m], axis=1)
            rows.append(
                pd.DataFrame(
                    {"day": c["days"], "method": method, "group": group, "inside_pct": mean}
                )
            )
    out = pd.concat(rows, ignore_index=True)
    return out[out["day"] >= c["days"][WARM_UP]].dropna()


def learning_rates(c: dict, rates=LEARNING_RATES) -> pd.DataFrame:
    rows = []
    for method in (TRACKED, INTEGRATED):
        for lr in rates:
            lower, upper = bounds(c, method, lr)
            per = site_stats(c, lower, upper, scored_days(c))
            rows.append({"method": method, "learning_rate": lr} | summary_row(per))
    return pd.DataFrame(rows)


def spread(c: dict, methods=METHODS) -> pd.DataFrame:
    """How even the coverage is: the spread of the share inside the range over the cells of
    site x season x four-hour block, and the share of sites within five points of 80 %."""
    season = c["days"].month.map(SEASONS)
    rows = []
    for method in methods:
        lower, upper = bounds(c, method)
        cells = []
        for s in SEASON_ORDER:
            for h in range(6):
                mask = np.asarray(season == s)[:, None, None] & (np.arange(48) // 8 == h)
                cells.append(site_stats(c, lower, upper, mask & scored_days(c))["inside_pct"])
        cells = pd.concat(cells)
        site = site_stats(c, lower, upper, scored_days(c))["inside_pct"]
        rows.append(
            {
                "method": method,
                "cells": len(cells),
                "cell_sd_pts": float(cells.std()),
                "cells_within_5_pct": float(100 * cells.between(75, 85).mean()),
                "cell_worst_decile_pct": float(cells.quantile(0.1)),
                "sites_within_5_pct": float(100 * site.between(75, 85).mean()),
                "site_min_pct": float(site.min()),
                "site_max_pct": float(site.max()),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _, panel = fa.poc_panel()
    plant = set(pd.read_csv(ee.CHP_FILE)["site_code"]) & set(panel.sites)
    flags = fr.event_flags(panel, "catalogue")
    f = fr.load_forecasts()
    tables: dict[str, list] = {k: [] for k in ("coverage", "spread", "learning_rates")}
    main = None
    for model, g in f.groupby("model"):
        c = cube(g, panel)
        tables["coverage"].append(coverage(c, plant, flags).assign(model=model))
        tables["spread"].append(spread(c).assign(model=model))
        tables["learning_rates"].append(learning_rates(c).assign(model=model))
        if model == "chronos2_ifs" or main is None:
            main = (model, c)
        logger.info("%s: coverage done", model)
    out = {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}
    model, c = main
    out["reliability"] = reliability(c, plant)
    out["season_hour"] = season_hour(c)
    out["by_site"] = by_site(c)
    out["over_time"] = over_time(c, plant)
    for name, table in out.items():
        table.to_csv(OUT_DIR / f"{name}.csv", index=False)
    cov = out["coverage"]
    whole = cov[(cov["model"] == model) & (cov["grouping"] == "all")].set_index("method")
    cols = ["below_pct", "above_pct", "inside_pct", "width_pct", "loss_pct"]
    summary = {
        "model": model,
        "days_scored": int(len(c["days"]) - WARM_UP),
        "sites": len(c["sites"]),
        "whole_record": {
            m: whole.loc[m, cols].astype(float).round(2).to_dict() for m in whole.index
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {"summary": summary, **out}
