"""Weather-normalised savings of the installations reported between the two estates returns.

ERIC 2024/25 records changes against 2023/24 at 45 of the 109 panel sites in both returns: LED
coverage up by ten points or more, solar generation new or grown, heat pumps installed
(``estates_changes``). The returns give the year, not the date, and a raw before-and-after
comparison mixes the installation with the weather. Here each site's electricity is put on a
common weather footing, the way measurement-and-verification practice does it (IPMVP option C,
PRISM's normalised annual consumption):

1. For each site and financial year (April-March: 2023/24, 2024/25, 2025/26), a daily model of
   log mean kW on the cold dose (degrees of daily mean below 14 degC), the heat dose (above
   20 degC), solar radiation, weekday, bank holidays, the Christmas period and the school summer
   holiday, fitted on that year's days (at least 300).
2. Each year's model predicts the site's demand on the *same* reference days: every day of the
   three years with its weather and calendar. The mean is the normalised annual consumption
   (NAC); its change from 2023/24 to 2025/26 (2024/25 is the installation year) is the
   weather-normalised change.
3. Changed sites are compared with the sites whose returns report no change (difference in
   differences across sites, bootstrap intervals), by kind of change, and the LED change is
   used as a dose (percentage points of LED coverage) in a cross-site regression.

Sites with on-site CHP are reported but left out of the comparisons: their import moves with
the plant.

Outputs (``code/reports/installation_savings/``): ``nac.csv`` (site x year: NAC, raw mean,
model fit), ``site_changes.csv`` (one row per site: raw and normalised change, change flags),
``group_effects.csv``, ``led_dose.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import weather_causal as wc

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "installation_savings"
CHANGES = PROJECT_ROOT / "code" / "reports" / "estates" / "eric_changes.csv"
YEARS = {
    "2023/24": ("2023-04-01", "2024-04-01"),
    "2024/25": ("2024-04-01", "2025-04-01"),
    "2025/26": ("2025-04-01", "2026-04-01"),
}
BEFORE, AFTER = "2023/24", "2025/26"
MIN_DAYS = 300
N_BOOT, SEED = 2000, 20261109
GROUPS = ("no change", "LED only", "solar new or grown", "heat pumps installed")


def design(d: pd.DataFrame) -> np.ndarray:
    dow = (
        pd.get_dummies(d["date"].dt.dayofweek, dtype=float)
        .reindex(columns=range(7), fill_value=0.0)
        .to_numpy()[:, 1:]
    )
    cols = [
        np.ones(len(d)),
        d["hdd14"].to_numpy(),
        d["cdd20"].to_numpy(),
        d["solar_kwh_m2"].to_numpy(),
        d["holiday"].to_numpy(),
        d["christmas"].to_numpy(),
        d["school_summer"].to_numpy(),
    ]
    return np.column_stack([*cols, dow]).astype(float)


def fit_year(d: pd.DataFrame) -> dict | None:
    """OLS of log kW on the daily design; returns coefficients, smearing factor and fit."""
    if len(d) < MIN_DAYS:
        return None
    x, y = design(d), np.log(d["kw"].to_numpy())
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    smear = float(np.mean(np.exp(resid)))
    pred = np.exp(x @ beta) * smear
    kw = d["kw"].to_numpy()
    return {
        "beta": beta,
        "smear": smear,
        "days": len(d),
        "r2": float(1 - resid.var() / y.var()),
        "cv_rmse": float(np.sqrt(np.mean((kw - pred) ** 2)) / kw.mean()),
        "raw_mean_kw": float(kw.mean()),
        "cold_slope": float(beta[1]),
        "heat_slope": float(beta[2]),
    }


def nac_table(df: pd.DataFrame) -> pd.DataFrame:
    """NAC per site and financial year under the common reference days."""
    rows = []
    for site, d in df.groupby("site_code"):
        ref = d[(d["date"] >= YEARS[BEFORE][0]) & (d["date"] < YEARS[AFTER][1])]
        if ref.empty:
            continue
        xr = design(ref)
        for fy, (a, b) in YEARS.items():
            m = fit_year(d[(d["date"] >= a) & (d["date"] < b)])
            if m is None:
                continue
            nac = float(np.mean(np.exp(xr @ m["beta"]) * m["smear"]))
            rows.append(
                {
                    "site_code": site,
                    "year": fy,
                    "nac_kw": nac,
                    "raw_mean_kw": m["raw_mean_kw"],
                    "days": m["days"],
                    "r2": m["r2"],
                    "cv_rmse": m["cv_rmse"],
                    "cold_slope": m["cold_slope"],
                    "heat_slope": m["heat_slope"],
                    "reference_days": len(ref),
                }
            )
    return pd.DataFrame(rows)


def site_changes(nac: pd.DataFrame, changes: pd.DataFrame) -> pd.DataFrame:
    w = nac.pivot(index="site_code", columns="year", values=["nac_kw", "raw_mean_kw"])
    out = pd.DataFrame(index=w.index)
    out["nac_change_pct"] = 100 * (w[("nac_kw", AFTER)] / w[("nac_kw", BEFORE)] - 1)
    out["raw_change_pct"] = 100 * (w[("raw_mean_kw", AFTER)] / w[("raw_mean_kw", BEFORE)] - 1)
    out["nac_before_kw"] = w[("nac_kw", BEFORE)]
    out["nac_after_kw"] = w[("nac_kw", AFTER)]
    out = out.dropna(subset=["nac_change_pct"])
    c = changes.set_index("site_code")
    cols = [
        "site_name",
        "chp_site",
        "led_pct_2324",
        "led_pct_2425",
        "solar_new",
        "solar_grew",
        "heat_pumps_new",
        "heat_pumps_more",
        "led_up_10pts",
        "chp_changed",
        "any_change",
        "step_date",
        "step_change",
    ]
    out = out.join(c[cols], how="inner")
    out["led_change_pts"] = (out["led_pct_2425"] - out["led_pct_2324"]).fillna(0)
    solar = out["solar_new"] | out["solar_grew"]
    hp = out["heat_pumps_new"] | out["heat_pumps_more"]
    out["group"] = np.select(
        [hp, solar, out["led_up_10pts"], ~out["any_change"]],
        [GROUPS[3], GROUPS[2], GROUPS[1], GROUPS[0]],
        default="other change",
    )
    return out


def bootstrap_diff(a: np.ndarray, b: np.ndarray, rng, stat=np.median) -> tuple[float, float, float]:
    d = [stat(rng.choice(a, len(a))) - stat(rng.choice(b, len(b))) for _ in range(N_BOOT)]
    return float(stat(a) - stat(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def group_effects(sc: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    base = sc[(sc["group"] == GROUPS[0]) & ~sc["chp_site"]]
    rows = []
    for g in GROUPS:
        d = sc[(sc["group"] == g) & ~sc["chp_site"]]
        if d.empty:
            continue
        row = {
            "group": g,
            "sites": len(d),
            "median_nac_change_pct": float(d["nac_change_pct"].median()),
            "median_raw_change_pct": float(d["raw_change_pct"].median()),
        }
        if g != GROUPS[0] and len(d) >= 3:
            est, lo, hi = bootstrap_diff(
                d["nac_change_pct"].to_numpy(), base["nac_change_pct"].to_numpy(), rng
            )
            row.update({"did_median_pct": est, "did_lo": lo, "did_hi": hi})
            est_r, lo_r, hi_r = bootstrap_diff(
                d["raw_change_pct"].to_numpy(), base["raw_change_pct"].to_numpy(), rng
            )
            row.update({"did_raw_median_pct": est_r, "did_raw_lo": lo_r, "did_raw_hi": hi_r})
        rows.append(row)
    return pd.DataFrame(rows)


def led_dose(sc: pd.DataFrame) -> pd.DataFrame:
    """Cross-site OLS (HC3) of the normalised change on the LED change in percentage points,
    with solar and heat-pump indicators; non-CHP sites in both returns."""
    import statsmodels.api as sm

    d = sc[~sc["chp_site"]].copy()
    d = d[d["nac_change_pct"].abs() < 60]  # metering breaks, not installations
    x = pd.DataFrame(
        {
            "led_change_10pts": d["led_change_pts"] / 10.0,
            "solar": (d["solar_new"] | d["solar_grew"]).astype(float),
            "heat_pumps": (d["heat_pumps_new"] | d["heat_pumps_more"]).astype(float),
        }
    )
    m = sm.OLS(d["nac_change_pct"], sm.add_constant(x)).fit(cov_type="HC3")
    t = pd.DataFrame(
        {
            "coef": m.params,
            "se": m.bse,
            "lo": m.conf_int()[0],
            "hi": m.conf_int()[1],
            "p": m.pvalues,
        }
    )
    t["sites"] = int(m.nobs)
    return t.rename_axis("term").reset_index()


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    nac = nac_table(df)
    nac.to_csv(OUT_DIR / "nac.csv", index=False)
    changes = pd.read_csv(CHANGES)
    sc = site_changes(nac, changes)
    sc.to_csv(OUT_DIR / "site_changes.csv")
    ge = group_effects(sc)
    ge.to_csv(OUT_DIR / "group_effects.csv", index=False)
    ld = led_dose(sc)
    ld.to_csv(OUT_DIR / "led_dose.csv", index=False)
    fit = nac.groupby("year")[["r2", "cv_rmse"]].median()
    summary = {
        "sites_with_nac_both_years": int(len(sc)),
        "chp_sites": int(sc["chp_site"].sum()),
        "group_counts": sc.loc[~sc["chp_site"], "group"].value_counts().to_dict(),
        "median_model_r2": {k: round(v, 3) for k, v in fit["r2"].items()},
        "median_model_cv_rmse": {k: round(v, 3) for k, v in fit["cv_rmse"].items()},
        "group_effects": ge.round(2).to_dict("records"),
        "led_per_10pts": ld.set_index("term")
        .loc["led_change_10pts", ["coef", "lo", "hi", "p"]]
        .round(3)
        .to_dict(),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("installation savings: %s", json.dumps(summary)[:700])
    return {"nac": nac, "sites": sc, "groups": ge, "led": ld, "summary": summary}
