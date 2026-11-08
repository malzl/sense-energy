"""Causal machine learning on the heat response: who responds, learned rather than tabulated.

The fixed-effect regressions gave each site its own heat slope and found that no descriptor
explains the spread. Here the question is put to orthogonal learners that search the covariate
space themselves:

1. **Residualisation.** Log demand (Y), the heat dose (T, degrees of daily mean above 20 degC)
   and the controls (cold dose, sunshine) are demeaned on specification B's fixed effects (site x
   year-month, site x weekday, date) by alternating projections; the controls are then partialled
   out of Y and T. What remains is within-site, within-month, within-day variation, as in the
   regressions.
2. **R-learner.** The conditional effect tau(x) minimises sum of T_res^2 (Y_res / T_res - tau(x))^2:
   a LightGBM regression of Y_res / T_res on x with weights T_res^2, cross-fitted over site folds
   so every site's effect is predicted by a model that never saw it.
3. **Causal forest** (econml ``CausalForestDML``) on the same residuals, with site-grouped
   cross-fitting, honest trees and per-site confidence intervals.
4. **Covariates x**: the site's estates and context descriptors (use, urban form, density, size,
   age, fuel, LED, intensity, peak W/m2, heat pumps, overheating, solar, tenure, load shape, the
   DEC cooling and heating fields) and three calendar modifiers of the day (weekend, bank
   holiday, school summer holiday); a second run adds the site's own diurnal signature (daytime
   sunshine slope, afternoon excess), which the regressions found to be the one thing that
   explains the spread.
5. **Checks**: the site-mean learned effect against the site's fixed-effect slope (correlation,
   calibration), the share of the slopes' true variance the learned effects recover, covariate
   importance, and a placebo with the site covariates shuffled across sites.

Outputs (``code/reports/heat_causal_ml/``): ``cate_site.csv``, ``importance.csv``,
``calendar_effects.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import PROJECT_ROOT
from ..logging_utils import get_logger
from . import heat_drivers as hd
from . import weather_causal as wc
from . import weather_heterogeneity as wh

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "heat_causal_ml"
FE = wc.SPECS["B"]["fe"]
CONTROLS = ["hdd14", "solar_kwh_m2"]
TREATMENT = "cdd20"
CALENDAR = ["is_weekend", "holiday", "school_summer"]
SITE_NUMERIC = [
    "log_density",
    "log_mean_kw",
    "mean_build_year",
    "gas_to_elec",
    "led_pct",
    "log_elec_per_m2",
    "peak_w_m2",
    "night_day_ratio",
    "weekend_ratio",
    "winter_summer_ratio",
    "summer_tmax",
    "dec_ac_share",
    "dec_cooled_share",
    "dec_electric_heating_share",
    "log_dec_elec_kwh_m2",
    "dec_operational_rating",
]
SITE_CATEGORICAL = [
    "use",
    "urban",
    "tenure_group",
    "heat_pumps_any",
    "overheats",
    "solar_eric",
    "dec_main_environment",
]
SIGNATURE = ["sun_day", "heat_afternoon_excess"]
N_FOLDS = 5
SEED = 20261108
MIN_ABS_T = 0.05  # residual dose below this carries no information on the slope


# --------------------------------------------------------------------------- #
# Residuals
# --------------------------------------------------------------------------- #


def demean_fe(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Alternating-projection demeaning of ``cols`` on the specification-B fixed effects."""
    from pyfixest.core.demean import demean

    flist = np.column_stack([pd.factorize(df[f])[0] for f in FE]).astype(np.uint64)
    x = df[cols].to_numpy(dtype=np.float64)
    out, ok = demean(x, flist, np.ones(len(df)), tol=1e-8)
    if not ok:
        logger.warning("demeaning did not converge")
    return pd.DataFrame(out, columns=cols, index=df.index)


def residualise(df: pd.DataFrame) -> pd.DataFrame:
    """Y and T net of the fixed effects and the controls (Frisch-Waugh-Lovell)."""
    cols = ["log_kw", TREATMENT, *CONTROLS]
    r = demean_fe(df, cols)
    w = r[CONTROLS].to_numpy()
    for c in ("log_kw", TREATMENT):
        beta, *_ = np.linalg.lstsq(w, r[c].to_numpy(), rcond=None)
        r[c] = r[c] - w @ beta
    return r.rename(columns={"log_kw": "y_res", TREATMENT: "t_res"})


# --------------------------------------------------------------------------- #
# Covariates
# --------------------------------------------------------------------------- #


def site_covariates(desc: pd.DataFrame, features: pd.DataFrame, signature: bool) -> pd.DataFrame:
    """A numeric design per site: standardised numerics with missing indicators, one-hot
    categoricals; optionally the diurnal signature."""
    d = desc.copy()
    if signature:
        for c in SIGNATURE:
            d[c] = features[c].reindex(d.index)
    numeric = SITE_NUMERIC + (SIGNATURE if signature else [])
    parts = []
    for c in numeric:
        v = pd.to_numeric(d[c], errors="coerce")
        z = (v - v.mean()) / v.std() if v.std() > 0 else v * 0
        parts.append(z.fillna(0).rename(c))
        if v.isna().any():
            parts.append(v.isna().astype(float).rename(f"{c}__missing"))
    for c in SITE_CATEGORICAL:
        dm = pd.get_dummies(d[c].astype("string"), prefix=c, dtype=float)
        parts.append(dm)
    x = pd.concat(parts, axis=1)
    x = x.loc[:, x.std() > 0]
    x.columns = safe_names(list(x.columns))
    return x


def safe_names(names: list[str]) -> list[str]:
    """LightGBM-safe, unique column names (letters, digits and underscores only)."""
    import re

    out, seen = [], set()
    for n in names:
        base = re.sub(r"[^0-9A-Za-z_]+", "_", str(n)).strip("_") or "x"
        name, k = base, 1
        while name in seen:
            k += 1
            name = f"{base}_{k}"
        seen.add(name)
        out.append(name)
    return out


def design(df: pd.DataFrame, x_site: pd.DataFrame) -> pd.DataFrame:
    """Site covariates broadcast to site-days plus the calendar modifiers."""
    x = x_site.reindex(df["site_code"]).to_numpy()
    out = pd.DataFrame(x, columns=x_site.columns, index=df.index)
    out["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(float).to_numpy()
    out["holiday"] = df["holiday"].to_numpy(dtype=float)
    out["school_summer"] = df["school_summer"].to_numpy(dtype=float)
    return out


# --------------------------------------------------------------------------- #
# Learners
# --------------------------------------------------------------------------- #


def site_folds(sites: np.ndarray, n_folds: int = N_FOLDS, seed: int = SEED) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    u = np.array(sorted(set(sites)))
    rng.shuffle(u)
    return {s: i % n_folds for i, s in enumerate(u)}


def r_learner(
    x: pd.DataFrame, y_res: np.ndarray, t_res: np.ndarray, sites: np.ndarray
) -> np.ndarray:
    """Cross-fitted R-learner with LightGBM: tau(x) for every row, predicted out of fold."""
    import lightgbm as lgb

    folds = site_folds(sites)
    fold = np.array([folds[s] for s in sites])
    keep = np.abs(t_res) > MIN_ABS_T
    target = np.where(keep, y_res / np.where(keep, t_res, 1.0), 0.0)
    weight = np.where(keep, t_res**2, 0.0)
    tau = np.full(len(x), np.nan)
    params = dict(
        objective="regression",
        learning_rate=0.03,
        n_estimators=600,
        num_leaves=15,
        min_child_samples=200,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=5.0,
        verbose=-1,
        n_jobs=8,
        random_state=SEED,
    )
    for k in range(N_FOLDS):
        tr = (fold != k) & keep
        m = lgb.LGBMRegressor(**params)
        m.fit(x[tr], target[tr], sample_weight=weight[tr])
        tau[fold == k] = m.predict(x[fold == k])
    return tau


def causal_forest(x: pd.DataFrame, y_res: np.ndarray, t_res: np.ndarray, sites: np.ndarray):
    """econml CausalForestDML on the residuals, site-grouped cross-fitting."""
    import warnings

    from econml.dml import CausalForestDML
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import GroupKFold

    warnings.simplefilter("ignore", ConvergenceWarning)
    # The residuals are already net of the fixed effects and controls; the nuisance models
    # only mop up what the site covariates and calendar explain of them, so ridge suffices.
    est = CausalForestDML(
        model_y=RidgeCV(alphas=np.logspace(-2, 3, 12)),
        model_t=RidgeCV(alphas=np.logspace(-2, 3, 12)),
        n_estimators=600,
        min_samples_leaf=200,
        max_samples=0.4,
        honest=True,
        cv=GroupKFold(n_splits=N_FOLDS),
        random_state=SEED,
        n_jobs=8,
    )
    est.fit(y_res, t_res, X=x.to_numpy(), groups=sites)
    return est


def importance(x: pd.DataFrame, est) -> pd.Series:
    return pd.Series(est.feature_importances_, index=x.columns).sort_values(ascending=False)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


def site_summary(df: pd.DataFrame, tau: np.ndarray, label: str) -> pd.DataFrame:
    return pd.Series(tau, index=df.index).groupby(df["site_code"]).mean().rename(label).to_frame()


def agreement(cate: pd.Series, slope: pd.Series, se: pd.Series) -> dict:
    """Correlation and calibration of the learned site effects against the fixed-effect slopes,
    and the share of the slopes' true variance (DerSimonian-Laird) the learned effects recover."""
    d = pd.concat([cate.rename("cate"), slope.rename("slope"), se.rename("se")], axis=1).dropna()
    d = d[d["se"] > 0]
    corr = float(d[["cate", "slope"]].corr().iloc[0, 1])
    b = np.polyfit(d["cate"], d["slope"], 1)
    tau2 = wh.random_effects(d["slope"], d["se"])["tau"] ** 2
    resid = d["slope"] - (b[0] * d["cate"] + b[1])
    tau2_resid = wh.random_effects(resid, d["se"])["tau"] ** 2
    return {
        "sites": int(len(d)),
        "corr": round(corr, 3),
        "calibration_slope": round(float(b[0]), 3),
        "sd_cate_pct": round(100 * float(d["cate"].std()), 2),
        "sd_true_slope_pct": round(100 * float(np.sqrt(tau2)), 2),
        "share_true_variance_recovered": round(1 - tau2_resid / tau2, 3) if tau2 > 0 else np.nan,
    }


def calendar_effects(x: pd.DataFrame, est, base: pd.DataFrame) -> pd.DataFrame:
    """The forest's average effect with each calendar modifier switched on and off."""
    rows = []
    for c in CALENDAR:
        for v in (0.0, 1.0):
            xx = base.copy()
            xx[c] = v
            eff = est.effect(xx.to_numpy())
            rows.append({"modifier": c, "value": v, "effect_pct": 100 * float(np.mean(eff))})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = hd.subset(wc.panel()).reset_index(drop=True)
    desc = hd.descriptors(df)
    features = pd.read_csv(hd.OUT_DIR / "site_features.csv").set_index("site_code")
    res = residualise(df)
    y_res, t_res = res["y_res"].to_numpy(), res["t_res"].to_numpy()
    sites = df["site_code"].to_numpy()
    slope, se = features["heat"], features["heat_se"]

    out, summary = {}, {"sites": int(df["site_code"].nunique()), "rows": int(len(df))}
    with threadpool_limits(8):
        for label, signature in (("observables", False), ("observables + signature", True)):
            x_site = site_covariates(desc, features, signature)
            x = design(df, x_site)
            tau_r = r_learner(x, y_res, t_res, sites)
            est = causal_forest(x, y_res, t_res, sites)
            tau_f = est.effect(x.to_numpy())
            lo, hi = est.effect_interval(x.to_numpy(), alpha=0.05)
            s = site_summary(df, tau_r, "tau_rlearner").join(site_summary(df, tau_f, "tau_forest"))
            s = s.join(site_summary(df, lo, "tau_forest_lo")).join(
                site_summary(df, hi, "tau_forest_hi")
            )
            s["slope_fe"], s["slope_fe_se"] = slope.reindex(s.index), se.reindex(s.index)
            s["set"] = label
            out[label] = s
            imp = importance(x, est)
            imp.rename("importance").rename_axis("covariate").reset_index().assign(
                set=label
            ).to_csv(
                OUT_DIR / f"importance_{'signature' if signature else 'observables'}.csv",
                index=False,
            )
            # Placebo: site covariates shuffled across sites, calendar kept.
            rng = np.random.default_rng(SEED)
            perm = x_site.copy()
            perm[:] = x_site.to_numpy()[rng.permutation(len(x_site))]
            tau_p = r_learner(design(df, perm), y_res, t_res, sites)
            sp = site_summary(df, tau_p, "tau_placebo")
            summary[label] = {
                "r_learner": agreement(s["tau_rlearner"], slope, se),
                "forest": agreement(s["tau_forest"], slope, se),
                "placebo_r_learner": agreement(sp["tau_placebo"], slope, se),
                "forest_top_covariates": imp.head(8).round(3).to_dict(),
                "forest_mean_effect_pct": round(100 * float(np.mean(tau_f)), 2),
                "calendar": calendar_effects(x, est, x).to_dict("records"),
                "sites_significant_positive": int((s["tau_forest_lo"] > 0).sum()),
                "sites_significant_negative": int((s["tau_forest_hi"] < 0).sum()),
            }
            logger.info("%s: %s", label, json.dumps(summary[label])[:400])
    cate = pd.concat(out.values()).reset_index().rename(columns={"index": "site_code"})
    cate.to_csv(OUT_DIR / "cate_site.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return {"cate": cate, "summary": summary}
