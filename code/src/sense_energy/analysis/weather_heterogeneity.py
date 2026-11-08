"""Who responds to cold and heat, and how much the response varies between sites.

All on sites without CHP, with the doses of ``weather_causal`` (degrees of daily mean below
14 degC and above 20 degC) and its specification B (site x year-month, site x weekday and date
fixed effects), so every national day shock is removed before any between-site comparison.

1. **Site slopes.** One regression with the doses interacted with every site: 110 cold slopes
   and 110 heat slopes with their standard errors. The spread of true slopes is separated from
   sampling noise by a random-effects (DerSimonian-Laird) decomposition: tau is the SD of the
   true slopes, I2 the share of the observed variance that is real; empirical-Bayes shrinkage
   gives each site a posterior slope for ranking and mapping.
2. **Moderators**, one at a time and jointly. Categorical: building use (ERIC site type collapsed
   to inpatient / mental health / non-inpatient / support), demand-shape cluster (the k = 4
   electricity clusters), trust type. Continuous, standardised within the sample: log mean
   demand, mean build year (ERIC), summer mean daily maximum (local climate), gas-to-electricity
   ratio (how heat is supplied), LED share. Each enters as dose x moderator interactions on top
   of the main doses; the joint model carries all of them and reports which survive.
3. **Hour-of-day response by group**: the ``weather_response`` hourly curve for the cold band
   (< 2 degC) and the hot band (>= 22 degC daily mean), by building use, with site x month and
   site x weekday effects (no date effects: too few hot site-days per hour in the small groups)
   and a floor of 150 site-days per band.

Outputs (``code/reports/weather_heterogeneity/``): ``site_slopes.csv``, ``moderators.csv``,
``moderators_joint.csv``, ``hourly_by_use.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import weather_causal as wc
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "weather_heterogeneity"
CLUSTER_DIR = PROJECT_ROOT / "code" / "reports" / "clustering"
FE = wc.SPECS["B"]["fe"]
MIN_DAYS = 300
USE_GROUPS = {
    "General acute hospital": "inpatient, acute",
    "Specialist hospital (acute only)": "inpatient, acute",
    "Mixed service hospital": "inpatient, acute",
    "Community hospital (with inpatient beds)": "inpatient, community",
    "Other inpatient": "inpatient, community",
    "Mental Health (including Specialist services)": "mental health",
    "Mental Health and Learning Disabilities": "mental health",
    "Non inpatient": "non-inpatient",
    "Support Facility": "support",
}
USE_ORDER = (
    "inpatient, acute",
    "inpatient, community",
    "mental health",
    "non-inpatient",
    "support",
)
CONTINUOUS = {
    "log_mean_kw": "site size (log mean kW)",
    "mean_build_year": "mean build year",
    "summer_tmax": "summer mean daily maximum (°C)",
    "gas_to_elec": "gas / electricity ratio",
    "led_pct": "LED share (%)",
}
HOUR_BANDS = {"cold": ("tbin", (-np.inf, 2.0)), "hot": ("tbin", (22.0, np.inf))}
#: The hourly-by-use curves use site x month and site x weekday effects only: with five acute
#: sites, a date effect would leave a hot band's coefficient resting on a handful of site-days.
HOUR_FE = ["fe_month", "fe_dow"]
MIN_BAND_SITE_DAYS = 150


# --------------------------------------------------------------------------- #
# Site descriptors
# --------------------------------------------------------------------------- #


def descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site: group labels and the continuous moderators."""
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").set_index("site_code")
    eric = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").drop_duplicates("site_code")
    eric = eric.set_index("site_code")
    labels = pd.read_csv(CLUSTER_DIR / "elec_site_labels_all_k.csv").set_index("series_id")["k4"]
    clusters = pd.read_csv(CLUSTER_DIR / "elec_site_clusters.csv").set_index("series_id")
    # Clusters ordered by mean size, as in the clustering write-up: 0 = largest.
    order = clusters["mean_kw"].groupby(labels).mean().sort_values(ascending=False).index
    rank = {old: new for new, old in enumerate(order)}
    names = {0: "round-the-clock", 1: "daytime", 2: "office-like", 3: "night-heavy"}

    g = df.groupby("site_code")
    out = pd.DataFrame(index=sorted(df["site_code"].unique()))
    out["days"] = g.size()
    out["mean_kw"] = g["kw"].mean()
    out["log_mean_kw"] = np.log(out["mean_kw"])
    summer = df[df["date"].dt.month.isin([6, 7, 8])]
    out["summer_tmax"] = summer.groupby("site_code")["t_max"].mean()
    out["use"] = eric["site_type"].reindex(out.index).map(USE_GROUPS)
    out["use"] = out["use"].fillna(
        sites["site_use_type"]
        .reindex(out.index)
        .map(
            lambda u: (
                "mental health"
                if isinstance(u, str) and "Mental" in u
                else "inpatient, community"
                if isinstance(u, str) and "inpatient beds" in u
                else "non-inpatient"
                if isinstance(u, str)
                else np.nan
            )
        )
    )
    out["trust_type"] = (
        sites["organisation_type"]
        .reindex(out.index)
        .str.replace(r"ACUTE - .*", "acute", regex=True)
        .str.lower()
    )
    out["cluster"] = labels.reindex(out.index).map(rank).map(names)
    out["mean_build_year"] = eric["mean_build_year"].reindex(out.index)
    out["gas_to_elec"] = eric["gas_to_elec"].reindex(out.index)
    out["led_pct"] = eric["led_pct"].reindex(out.index)
    out["region"] = df.groupby("site_code")["region_ons"].first()
    return out.rename_axis("site_code")


# --------------------------------------------------------------------------- #
# 1. Site slopes
# --------------------------------------------------------------------------- #


def site_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Dose x site interactions in one specification-B regression."""
    import pyfixest as pf

    sub = df[df["group"].isin(wc.GROUPS["no CHP"])].copy()
    counts = sub.groupby("site_code").size()
    sub = sub[sub["site_code"].isin(counts.index[counts >= MIN_DAYS])]
    sites = sorted(sub["site_code"].unique())
    cols, names = {}, {}
    for i, s in enumerate(sites):
        m = (sub["site_code"] == s).to_numpy(dtype=float)
        for v in wc.DOSE:
            n = f"{v}_s{i}"  # site codes are not formula-safe names
            cols[n] = sub[v].to_numpy() * m
            names[(v, s)] = n
    data = pd.concat([sub[FE + ["site_code"]], pd.DataFrame(cols, index=sub.index)], axis=1)
    data["log_kw"] = sub["log_kw"]
    fml = f"log_kw ~ {' + '.join(cols)} | {' + '.join(FE)}"
    fit = pf.feols(fml, data=data, vcov={"CRV1": "site_code+date_id"})
    t = fit.tidy()
    rows = []
    for s in sites:
        for v in wc.DOSE:
            n = names[(v, s)]
            if n not in t.index:  # dropped as collinear (a site with no variation in the dose)
                continue
            r = t.loc[n]
            rows.append({"site_code": s, "dose": v, "slope": r["Estimate"], "se": r["Std. Error"]})
    out = pd.DataFrame(rows)
    out["slope_pct"] = 100 * np.expm1(out["slope"])
    out["se_pct"] = 100 * out["se"]
    return out


def random_effects(slopes: pd.Series, ses: pd.Series) -> dict:
    """DerSimonian-Laird: between-site SD of the true slopes and the share of observed variance
    that is real, plus the inverse-variance pooled mean."""
    w = 1 / ses**2
    mu = float((w * slopes).sum() / w.sum())
    q = float((w * (slopes - mu) ** 2).sum())
    k = len(slopes)
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c)
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"pooled": mu, "tau": float(np.sqrt(tau2)), "i2": i2, "q": q, "df": k - 1}


def shrink(slopes: pd.DataFrame) -> pd.DataFrame:
    """Empirical-Bayes posterior slope per site: a precision-weighted blend of the site's own
    estimate and the pooled mean, with tau from the random-effects decomposition."""
    # A clustered variance can come out negative for a site with few effective clusters, and
    # pyfixest then reports a NaN standard error; such sites are dropped rather than imputed.
    usable = slopes["se"].notna() & np.isfinite(slopes["se"]) & (slopes["se"] > 0)
    if (~usable).any():
        logger.warning(
            "dropping %d site-dose slopes with undefined standard errors: %s",
            int((~usable).sum()),
            slopes.loc[~usable, ["site_code", "dose"]].to_dict("records"),
        )
    slopes = slopes[usable]
    out = []
    for _, d in slopes.groupby("dose"):
        re = random_effects(d["slope"], d["se"])
        tau2 = re["tau"] ** 2
        b = tau2 / (tau2 + d["se"] ** 2)
        d = d.assign(
            posterior=b * d["slope"] + (1 - b) * re["pooled"],
            shrinkage=1 - b,
            pooled=re["pooled"],
            tau=re["tau"],
            i2=re["i2"],
        )
        d["posterior_pct"] = 100 * np.expm1(d["posterior"])
        out.append(d)
    return pd.concat(out, ignore_index=True)


# --------------------------------------------------------------------------- #
# 2. Moderators
# --------------------------------------------------------------------------- #


def _interactions(sub: pd.DataFrame, design: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Dose x column products for every column of ``design`` (indexed like ``sub``)."""
    cols, names = {}, []
    for c in design.columns:
        for v in wc.DOSE:
            n = f"{v}_x_{c}"
            cols[n] = sub[v].to_numpy() * design[c].to_numpy(dtype=float)
            names.append(n)
    return pd.DataFrame(cols, index=sub.index), names


def _fit_moderated(sub: pd.DataFrame, design: pd.DataFrame, label: str) -> pd.DataFrame:
    import pyfixest as pf

    inter, names = _interactions(sub, design)
    data = pd.concat([sub[FE + ["site_code", "log_kw"] + list(wc.DOSE)], inter], axis=1)
    fml = f"log_kw ~ {' + '.join(list(wc.DOSE) + names)} | {' + '.join(FE)}"
    fit = pf.feols(fml, data=data, vcov={"CRV1": "site_code+date_id"})
    t = fit.tidy()
    rows = []
    kept = [n for n in names if n in t.index]  # collinear interactions are dropped by pyfixest
    for n in list(wc.DOSE) + kept:
        r = t.loc[n]
        rows.append(
            {
                "model": label,
                "term": n,
                "dose": n.split("_x_")[0] if "_x_" in n else n,
                "level": n.split("_x_", 1)[1] if "_x_" in n else "base",
                "coef": r["Estimate"],
                "se": r["Std. Error"],
                "lo": r["2.5%"],
                "hi": r["97.5%"],
                "p": r["Pr(>|t|)"],
            }
        )
    out = pd.DataFrame(rows)
    for c in ("coef", "lo", "hi"):
        out[f"{c}_pct"] = 100 * out[c]  # small effects: log points x 100 ~ percentage points
    # Joint test of the interactions for this moderator.
    idx = [i for i, n in enumerate(fit._coefnames) if n in kept]
    r_mat = np.zeros((len(idx), len(fit._coefnames)))
    r_mat[np.arange(len(idx)), idx] = 1.0
    w = fit.wald_test(R=r_mat)
    out["p_joint"] = float(w["pvalue"])
    out["sites"] = sub["site_code"].nunique()
    return out


def _categorical_design(d: pd.Series, order: list[str]) -> pd.DataFrame:
    """Dummies for every level but the first (the reference)."""
    x = pd.get_dummies(pd.Categorical(d, categories=order), dtype=float)
    x.index = d.index  # get_dummies on a Categorical drops the input index
    x.columns = [str(c).replace(" ", "_").replace(",", "").replace("-", "_") for c in x.columns]
    return x.iloc[:, 1:]


def _standardised(d: pd.Series) -> pd.Series:
    return (d - d.mean()) / d.std()


def moderators(df: pd.DataFrame, desc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = df[df["group"].isin(wc.GROUPS["no CHP"])].copy()
    sub = sub[sub["site_code"].isin(desc.index)]
    rows = []
    # Categorical, one at a time
    cats = {
        "building use": ("use", list(USE_ORDER)),
        "demand-shape cluster": (
            "cluster",
            ["round-the-clock", "daytime", "office-like", "night-heavy"],
        ),
        "trust type": (
            "trust_type",
            ["acute", "community", "mental health and learning disability", "ambulance"],
        ),
    }
    for label, (col, order) in cats.items():
        site_val = desc[col].reindex(sub["site_code"]).to_numpy()
        keep = pd.notna(site_val)
        s = sub[keep].copy()
        design = _categorical_design(pd.Series(site_val[keep], index=s.index), order)
        design = design.loc[:, design.sum() > 0]
        rows.append(_fit_moderated(s, design, label).assign(kind="categorical", reference=order[0]))
    # Continuous, one at a time
    for col, label in CONTINUOUS.items():
        site_val = desc[col].reindex(sub["site_code"]).to_numpy()
        keep = pd.notna(site_val)
        s = sub[keep].copy()
        z = _standardised(pd.Series(site_val[keep], index=s.index))
        design = pd.DataFrame({col: z})
        rows.append(_fit_moderated(s, design, label).assign(kind="continuous", reference="per SD"))
    one = pd.concat(rows, ignore_index=True)

    # Joint: building use dummies + all continuous moderators, on sites with every descriptor.
    cont = list(CONTINUOUS)
    full = desc.dropna(subset=["use"] + cont)
    sj = sub[sub["site_code"].isin(full.index)].copy()
    codes = sj["site_code"].to_numpy()
    use_vals = pd.Series(full["use"].reindex(codes).to_numpy(), index=sj.index)
    use = _categorical_design(use_vals, list(USE_ORDER))
    use = use.loc[:, use.sum() > 0]
    zs = pd.DataFrame(
        {
            c: _standardised(pd.Series(full[c].reindex(codes).to_numpy(), index=sj.index))
            for c in cont
        },
        index=sj.index,
    )
    design = pd.concat([use, zs], axis=1)
    assert len(design) == len(sj), (len(design), len(sj))
    joint = _fit_moderated(sj, design, "joint").assign(kind="joint", reference=USE_ORDER[0])
    return one, joint


# --------------------------------------------------------------------------- #
# 3. Hour-of-day response by building use
# --------------------------------------------------------------------------- #


def hourly_by_use(df: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    """Mean demand in each local hour on cold (< 2 degC) and hot (>= 22 degC) days against the
    14-16 degC reference, by building use, with site x month, site x weekday and date effects."""
    import pyfixest as pf

    from ..config import PROCESSED_DIR

    sub = df[df["group"].isin(wc.GROUPS["no CHP"])].copy()
    sub["use"] = desc["use"].reindex(sub["site_code"]).to_numpy()
    sub = sub.dropna(subset=["use"])
    w = (
        pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[
            sorted(sub["site_code"].unique())
        ]
        * 2.0
    )
    local = pd.to_datetime(w.index, utc=True).tz_convert(wr.TZ)
    day, hour = local.normalize().tz_localize(None), local.hour
    bands = wr.bin_dummies(sub, "daily mean")
    cold_cols = [c for c in bands.columns if wr.EDGES[int(c[2:])] < 2.0]
    hot_cols = [c for c in bands.columns if wr.EDGES[int(c[2:])] >= 22.0]
    rows = []
    for h in range(24):
        m = hour == h
        hk = (
            w[m]
            .groupby(day[m])
            .mean()
            .stack()
            .rename("hour_kw")
            .rename_axis(["date", "site_code"])
            .reset_index()
        )
        x = sub.merge(hk[hk["hour_kw"] > 0], on=["date", "site_code"], how="inner")
        b = wr.bin_dummies(x, "daily mean")
        for use, xu in x.groupby("use"):
            if xu["site_code"].nunique() < 3:
                continue
            bu = b.loc[xu.index]
            bu = bu.loc[:, bu.sum() > 0]
            data = pd.concat([xu[FE + ["site_code"]], bu], axis=1)
            data["log_y"] = np.log(xu["hour_kw"])
            fit = pf.feols(
                f"log_y ~ {' + '.join(bu.columns)} | {' + '.join(HOUR_FE)}",
                data=data,
                vcov={"CRV1": "site_code+date_id"},
            )
            coef, vcov, names = fit.coef(), fit._vcov, list(fit._coefnames)
            for band, cols in (("cold", cold_cols), ("hot", hot_cols)):
                # Observation-weighted average of the band's bin coefficients.
                present = [c for c in cols if c in names]
                if not present:
                    continue
                wts = bu[present].sum().to_numpy(dtype=float)
                wts = wts / wts.sum()
                idx = [names.index(c) for c in present]
                n_band = int(bu[present].to_numpy().sum())
                if n_band < MIN_BAND_SITE_DAYS:
                    continue
                est = float(wts @ coef.to_numpy()[idx])
                se = float(np.sqrt(wts @ vcov[np.ix_(idx, idx)] @ wts))
                rows.append(
                    {
                        "hour": h,
                        "use": use,
                        "band": band,
                        "coef": est,
                        "se": se,
                        "coef_pct": 100 * np.expm1(est),
                        "lo_pct": 100 * np.expm1(est - 1.96 * se),
                        "hi_pct": 100 * np.expm1(est + 1.96 * se),
                        "sites": xu["site_code"].nunique(),
                        "site_days": int(bu[present].to_numpy().sum()),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    desc = descriptors(df[df["group"].isin(wc.GROUPS["no CHP"])])
    desc.to_csv(OUT_DIR / "site_descriptors.csv")
    with threadpool_limits(4):
        slopes = shrink(site_slopes(df)).merge(desc.reset_index(), on="site_code", how="left")
        slopes.to_csv(OUT_DIR / "site_slopes.csv", index=False)
        logger.info("site slopes done")
        one, joint = moderators(df, desc)
        one.to_csv(OUT_DIR / "moderators.csv", index=False)
        joint.to_csv(OUT_DIR / "moderators_joint.csv", index=False)
        logger.info("moderators done")
        hours = hourly_by_use(df, desc)
        hours.to_csv(OUT_DIR / "hourly_by_use.csv", index=False)

    re = {}
    for v, d in slopes.groupby("dose"):
        re[v] = {
            "sites": int(len(d)),
            "pooled_pct": round(100 * float(np.expm1(d["pooled"].iloc[0])), 2),
            "tau_pct": round(100 * float(d["tau"].iloc[0]), 2),
            "i2": round(float(d["i2"].iloc[0]), 3),
            "raw_sd_pct": round(float(d["slope_pct"].std()), 2),
            "posterior_q10_q50_q90_pct": d["posterior_pct"]
            .quantile([0.1, 0.5, 0.9])
            .round(2)
            .tolist(),
            "sites_significant_positive": int(((d["slope"] - 1.96 * d["se"]) > 0).sum()),
            "sites_significant_negative": int(((d["slope"] + 1.96 * d["se"]) < 0).sum()),
        }
    by_use = {
        f"{v} | {u}": round(float(d["posterior_pct"].median()), 2)
        for (v, u), d in slopes.dropna(subset=["use"]).groupby(["dose", "use"])
    }
    summary = {
        "sites": int(desc.shape[0]),
        "random_effects": re,
        "posterior_median_by_use": by_use,
        "moderator_joint_p": {
            m: round(float(d["p_joint"].iloc[0]), 3) for m, d in one.groupby("model")
        },
        "joint_model_p": round(float(joint["p_joint"].iloc[0]), 3),
        "joint_model_sites": int(joint["sites"].iloc[0]),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("heterogeneity: %s", summary["random_effects"])
    return {
        "slopes": slopes,
        "moderators": one,
        "joint": joint,
        "hourly": hours,
        "descriptors": desc,
        "summary": summary,
    }
