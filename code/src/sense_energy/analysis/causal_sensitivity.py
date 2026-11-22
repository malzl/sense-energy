"""How wrong can the event effects be? Bounds under unobserved confounding.

Frauen, Melnychuk and Feuerriegel, "Sharp Bounds for Generalized Causal Sensitivity Analysis"
(NeurIPS 2023). Their generalized marginal sensitivity model (GMSM) bounds how much unobserved
confounders U can shift the treatment assignment: for a binary treatment A with observed
propensity pi(x) = P(A = 1 | x) it is the marginal sensitivity model

    1 / Gamma  <=  [pi(x) / (1 - pi(x))] / [pi(x, u) / (1 - pi(x, u))]  <=  Gamma ,

so Gamma = 1 is no unobserved confounding and Gamma = 2 lets a hidden factor double or halve
the odds of treatment. In the weighted GMSM (their Definition 4, with the weight
q(a, x) = P(a | x) for a binary treatment)

    s-(a, x) = 1 / ((1 - Gamma) q + Gamma),      s+(a, x) = 1 / ((1 - 1/Gamma) q + 1/Gamma),

and the sharp upper bound on E[Y | x, do(a)] (their Theorem 1 and Corollary 1) reweights the
observed conditional distribution of Y: weight 1/s+ on its lower Gamma / (1 + Gamma) share and
1/s- on the rest. The lower bound puts 1/s- on the lower 1 / (1 + Gamma) share and 1/s+ on the
rest. With a sample from the conditional distribution these are weighted means of the sorted
sample (their equation 8). For a binary treatment this is the bound of Dorn and Guo.

Here:

* treatment A: the site-day is inside a heat wave (or a cold spell) at the site;
* outcome Y: the reaction of the day, 100 x log(demand / reference), the measure of
  ``extreme_events``;
* observed x: the site and the season (the reference days already hold weekday and the weeks
  around the day fixed);
* P(Y | site, A = 0): the site's ordinary days of the season;
* P(Y | site, A = 1): the site's mean on its event days plus the pooled, standardised
  deviations of all event days, scaled with the site's own day-to-day spread (a site has few
  event days, so the shape is pooled and the location and scale are the site's);
* the effect: the mean over sites of E[Y | do(1)] - E[Y | do(0)]; intervals from resampling
  sites.

The breakdown value is the Gamma at which the lower bound of the effect reaches zero. To judge
it, the same odds ratio is computed for what we do observe: how much weekends, school
holidays and the calendar month shift the odds that a day is an event day.

Outputs: ``code/reports/causal_sensitivity/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import event_audit as ea
from . import extreme_events as ee

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "causal_sensitivity"
GAMMAS = (1.0, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0, 10.0)
FINE = np.round(np.r_[np.arange(1.0, 3.0, 0.02), np.arange(3.0, 30.01, 0.25)], 3)
SEASONS = {"heatwave": (5, 6, 7, 8, 9), "cold spell": (11, 12, 1, 2, 3)}
MIN_ORDINARY, MIN_EVENT = 60, 3
N_BOOT = 1000


# --------------------------------------------------------------------------- #
# The bounds
# --------------------------------------------------------------------------- #


def gmsm_scales(gamma: float, q: float) -> tuple[float, float]:
    """s- and s+ of the weighted GMSM for sensitivity parameter ``gamma`` and weight ``q``."""
    return 1.0 / ((1.0 - gamma) * q + gamma), 1.0 / ((1.0 - 1.0 / gamma) * q + 1.0 / gamma)


def _weighted_mean(y_sorted: np.ndarray, cut: float, w_low: float, w_high: float) -> float:
    """Mean of a sorted sample with weight ``w_low`` on its lower ``cut`` share and ``w_high``
    on the rest; the value that straddles the cut is split."""
    k = len(y_sorted)
    edges = np.arange(k + 1) / k
    low = np.clip(cut - edges[:-1], 0.0, 1.0 / k)  # share of each value below the cut
    return float(np.sum(y_sorted * (w_low * low + w_high * (1.0 / k - low))))


def mean_bounds(y: np.ndarray, gamma: float, q: float) -> tuple[float, float]:
    """Sharp lower and upper bound on E[Y | x, do(a)] from a sample of P(Y | x, a)."""
    ys = np.sort(np.asarray(y, dtype=float))
    s_minus, s_plus = gmsm_scales(gamma, q)
    upper = _weighted_mean(ys, gamma / (1.0 + gamma), 1.0 / s_plus, 1.0 / s_minus)
    lower = _weighted_mean(ys, 1.0 / (1.0 + gamma), 1.0 / s_minus, 1.0 / s_plus)
    return lower, upper


def effect_bounds(y1: np.ndarray, y0: np.ndarray, pi: float, gamma: float) -> tuple[float, float]:
    """Bounds on E[Y | do(1)] - E[Y | do(0)] for one unit with treatment propensity ``pi``."""
    lo1, hi1 = mean_bounds(y1, gamma, pi)
    lo0, hi0 = mean_bounds(y0, gamma, 1.0 - pi)
    return lo1 - hi0, hi1 - lo0


# --------------------------------------------------------------------------- #
# Data: the conditional distributions per site
# --------------------------------------------------------------------------- #


def site_samples(r: pd.DataFrame, sites: pd.DataFrame, kind: str) -> dict[str, dict]:
    """Per site without on-site generation: the reactions of its ordinary days of the season
    (A = 0), a sample for its event days (A = 1) and its propensity."""
    keep = set(sites.loc[sites["clean"] & (sites["chp"] == ea.PLAIN), "site_code"])
    x = r[r["site_code"].isin(keep) & r["date"].dt.month.isin(SEASONS[kind])].copy()
    x = x[x["kw"].notna() & x["ref_kw"].notna()]
    with np.errstate(all="ignore"):
        x["y"] = 100 * np.log(x["kw"] / x["ref_kw"])
    x = x[np.isfinite(x["y"])]
    ordinary = {s: g["y"].to_numpy() for s, g in x[x["type"] == "none"].groupby("site_code")}
    event = {s: g["y"].to_numpy() for s, g in x[x["type"] == kind].groupby("site_code")}
    units, z = {}, []
    for s, y1 in event.items():
        y0 = ordinary.get(s)
        if y0 is None or len(y0) < MIN_ORDINARY or len(y1) < MIN_EVENT:
            continue
        sd0 = float(np.std(y0, ddof=1))
        z.append((y1 - y1.mean()) / sd0 * np.sqrt(len(y1) / (len(y1) - 1)))
        units[s] = {"y0": y0, "mean1": float(y1.mean()), "sd0": sd0, "n1": len(y1)}
        units[s]["pi"] = len(y1) / (len(y1) + len(y0))
    pooled = np.concatenate(z) if z else np.array([])
    for u in units.values():
        u["y1"] = u["mean1"] + u["sd0"] * pooled
    return units


def bounds_table(units: dict[str, dict], gammas=GAMMAS, seed: int = 0) -> pd.DataFrame:
    """Bounds on the mean effect over sites for each Gamma, with the 95 % interval of each
    bound from resampling sites and the share of sites whose own lower bound is above zero."""
    names = list(units)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(names), (N_BOOT, len(names)))
    rows = []
    for g in gammas:
        b = np.array(
            [effect_bounds(units[s]["y1"], units[s]["y0"], units[s]["pi"], g) for s in names]
        )
        lo, hi = b[:, 0], b[:, 1]
        rows.append(
            {
                "gamma": g,
                "sites": len(names),
                "effect_lo": float(lo.mean()),
                "effect_hi": float(hi.mean()),
                "ci_lo": float(np.quantile(lo[pick].mean(axis=1), 0.025)),
                "ci_hi": float(np.quantile(hi[pick].mean(axis=1), 0.975)),
                "sites_with_lower_bound_above_zero": float((lo > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def breakdown(units: dict[str, dict]) -> dict:
    """The Gamma at which the lower bound of the mean effect reaches zero, and the Gamma at
    which the lower end of its interval does."""
    t = bounds_table(units, FINE)
    point = t.loc[t["effect_lo"] <= 0, "gamma"]
    ci = t.loc[t["ci_lo"] <= 0, "gamma"]
    return {
        "gamma_point": float(point.iloc[0]) if len(point) else float("inf"),
        "gamma_interval": float(ci.iloc[0]) if len(ci) else float("inf"),
        "effect_at_gamma_1": float(t["effect_lo"].iloc[0]),
    }


# --------------------------------------------------------------------------- #
# What would a Gamma of that size be? Odds ratios of observed factors
# --------------------------------------------------------------------------- #


def school_holidays(days: pd.DatetimeIndex) -> np.ndarray:
    """Approximate school holidays in England (fixed windows, not the term dates of each
    authority): the summer break, two weeks at Christmas, the half-terms and Easter."""
    md = days.month * 100 + days.day
    return (
        ((md >= 722) & (md <= 902))
        | (md >= 1221)
        | (md <= 104)
        | ((md >= 1024) & (md <= 1101))
        | ((md >= 212) & (md <= 220))
        | ((md >= 525) & (md <= 602))
        | ((md >= 329) & (md <= 414))
    )


def benchmarks(r: pd.DataFrame, sites: pd.DataFrame, kind: str) -> pd.DataFrame:
    """For observed factors: by how much the odds that a day is an event day differ between
    the levels of the factor and the season as a whole, the way Gamma measures a hidden
    factor. Pooled over the sites (the propensity of a single site is too small for stable
    odds within levels)."""
    keep = set(sites.loc[sites["clean"] & (sites["chp"] == ea.PLAIN), "site_code"])
    x = r[r["site_code"].isin(keep) & r["date"].dt.month.isin(SEASONS[kind])].copy()
    x = x[x["type"].isin(["none", kind])]
    a = (x["type"] == kind).to_numpy()
    odds_all = a.mean() / (1 - a.mean())
    days = pd.DatetimeIndex(x["date"])
    factors = {
        "weekend": np.where(days.dayofweek >= 5, "weekend", "weekday"),
        "school holidays (approximate)": np.where(school_holidays(days), "holiday", "term"),
        "calendar month": days.month.astype(str),
    }
    rows = []
    for name, level in factors.items():
        worst = 1.0
        for lv in np.unique(level):
            p = a[level == lv].mean()
            if p in (0.0, 1.0):
                continue
            ratio = (p / (1 - p)) / odds_all
            worst = max(worst, ratio, 1 / ratio)
        rows.append(
            {"type": kind, "factor": name, "levels": len(np.unique(level)), "odds_ratio": worst}
        )
    return pd.DataFrame(rows)


def observed_adjustment(r: pd.DataFrame, sites: pd.DataFrame, kind: str) -> dict:
    """Does holding the observed calendar factors fixed change the effect? The reaction of the
    season's days on the event indicator with site effects, without and with weekend, school
    holidays and calendar month; standard errors clustered by date."""
    import statsmodels.api as sm

    keep = set(sites.loc[sites["clean"] & (sites["chp"] == ea.PLAIN), "site_code"])
    x = r[r["site_code"].isin(keep) & r["date"].dt.month.isin(SEASONS[kind])].copy()
    x = x[x["type"].isin(["none", kind]) & x["kw"].notna() & x["ref_kw"].notna()]
    with np.errstate(all="ignore"):
        x["y"] = 100 * np.log(x["kw"] / x["ref_kw"])
    x = x[np.isfinite(x["y"]) & (x["y"].abs() < 150)]
    days = pd.DatetimeIndex(x["date"])
    d = pd.DataFrame({"event": (x["type"] == kind).astype(float).to_numpy()}, index=x.index)
    d["weekend"] = (days.dayofweek >= 5).astype(float)
    d["school_holiday"] = school_holidays(days).astype(float)
    month = pd.get_dummies(days.month, prefix="m", drop_first=True, dtype=float)
    month.index = x.index
    full = pd.concat([d, month], axis=1)

    def fit(cols):
        z = full[cols] - full[cols].groupby(x["site_code"]).transform("mean")
        y = x["y"] - x["y"].groupby(x["site_code"]).transform("mean")
        f = sm.OLS(y, z).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["date"])[0]})
        ci = f.conf_int().loc["event"]
        return float(f.params["event"]), float(ci[0]), float(ci[1])

    plain, adjusted = fit(["event"]), fit(list(full.columns))
    return {
        "type": kind,
        "site_days": len(x),
        "effect_pct": plain[0],
        "lo": plain[1],
        "hi": plain[2],
        "effect_adjusted_pct": adjusted[0],
        "adjusted_lo": adjusted[1],
        "adjusted_hi": adjusted[2],
    }


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    inp = ea.load_inputs()
    r, _, _, sites, _, _ = ea.prepare(inp, ea.Spec())
    tables, per_site, bench, adjust, summary = [], [], [], [], {}
    for kind in ee.TYPES:
        units = site_samples(r, sites, kind)
        t = bounds_table(units).assign(type=kind)
        tables.append(t)
        for g in GAMMAS:
            for s, u in units.items():
                lo, hi = effect_bounds(u["y1"], u["y0"], u["pi"], g)
                per_site.append(
                    {
                        "type": kind,
                        "site_code": s,
                        "gamma": g,
                        "lo": lo,
                        "hi": hi,
                        "pi": u["pi"],
                        "event_days": u["n1"],
                    }
                )
        b = benchmarks(r, sites, kind)
        bench.append(b)
        adjust.append(observed_adjustment(r, sites, kind))
        summary[kind] = {
            "sites": len(units),
            "median_propensity": float(np.median([u["pi"] for u in units.values()])),
            **breakdown(units),
            "benchmarks": b.set_index("factor")["odds_ratio"].round(2).to_dict(),
        }
    bounds = pd.concat(tables, ignore_index=True)
    bounds.to_csv(OUT_DIR / "bounds.csv", index=False)
    pd.DataFrame(per_site).to_csv(OUT_DIR / "bounds_sites.csv", index=False)
    pd.concat(bench, ignore_index=True).to_csv(OUT_DIR / "benchmarks.csv", index=False)
    pd.DataFrame(adjust).to_csv(OUT_DIR / "observed_adjustment.csv", index=False)
    for row in adjust:
        summary[row["type"]]["effect_with_calendar_held_fixed"] = round(
            row["effect_adjusted_pct"], 2
        )
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return {"summary": summary, "bounds": bounds}
