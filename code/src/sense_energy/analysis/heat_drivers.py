"""What moves the heat slope: sunshine, time of day, the week, and the site's context.

``weather_heterogeneity`` found a real, wide spread in the heat slope (the change in electricity
per degree of daily mean above 20 degC) that none of the estates descriptors explained, with 39
of 106 sites *falling* on hot days. This module takes that apart on the same panel (sites
without CHP, specification B: site x year-month, site x weekday and date effects, standard
errors clustered by site and date):

1. **Sunshine.** Hot days are sunny days. A site with solar panels behind the meter, or with
   daylight-responsive lighting, imports less on sunny days whatever the temperature; against
   temperature alone that reads as a negative heat slope. The dose x site regression is
   re-estimated with a site-specific sunshine term (ERA5 surface solar radiation, kWh/m2/day),
   so every site gets a heat slope net of sunshine and a sunshine slope of its own.
2. **Time of day and the week.** The same regression on the mean demand of four six-hour
   blocks (night, morning, afternoon, evening) separates a cooling signature (heat lifts the
   afternoon and evening) from a sunshine signature (sunshine lowers the daytime and never the
   night, so the night block is a placebo for the sunshine slope). A weekend x heat x site term
   asks whether a site's heat response runs with the working week or all week.
3. **Moderators** beyond the estates descriptors: urban form (ONS rural-urban class) and the
   population density of the site's LSOA, electricity intensity (kWh/m2), peak demand per m2,
   the load shape (night/day and weekend ratios), tenure, heat pumps, overheating reports and
   solar generation in the estates return. Each enters as dose x moderator interactions (with
   sunshine x moderator alongside), one at a time and jointly. A random-effects meta-regression
   of the site heat slopes on descriptor sets then reports the share of the true between-site
   variance each set explains.
4. **A typology** by significance at 5 %: rises with heat / rises with heat and falls with
   sunshine / falls with sunshine only / falls with heat / neither, with flags for an
   afternoon-peaked response and a weekday-only response.

Outputs (``code/reports/heat_drivers/``): ``site_terms.csv``, ``block_slopes.csv``,
``site_features.csv``, ``site_descriptors.csv``, ``moderators_heat.csv``,
``moderators_heat_joint.csv``, ``variance_explained.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..data.site_context import URBAN_ORDER
from ..logging_utils import get_logger
from . import weather_causal as wc
from . import weather_heterogeneity as wh
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "heat_drivers"
FE = wc.SPECS["B"]["fe"]
SUN = "solar_kwh_m2"
TERMS = ("hdd14", "cdd20", SUN)
TERM_NAMES = {"hdd14": "cold", "cdd20": "heat", SUN: "sunshine"}
BLOCKS = {0: "night (00–06)", 1: "morning (06–12)", 2: "afternoon (12–18)", 3: "evening (18–24)"}
#: A six-hour block needs ten of its twelve half-hours (a 23-hour day has ten in one block).
MIN_BLOCK_PERIODS = 10
#: Sites whose metered electricity is flagged ``meter_gaps`` in the ERIC validation (fewer than
#: 80 % of the year's half-hours carried by a stable meter set) are left out: meters that come
#: and go within a month put swings into the within-month variation that the clustered standard
#: errors do not see, and they surfaced as the most extreme negative heat slopes.
EXCLUDE_FLAGS = ("meter_gaps",)
#: The heat dose split in two: degrees of daily maximum above 24 degC (hot afternoons, where
#: cooling plant works) and degrees of daily minimum above 12 degC (warm nights, where heating
#: that still runs in the warm season switches off).
HOT_AFTERNOON_BASE, WARM_NIGHT_BASE = 24.0, 12.0
NIGHT_DAY_TERMS = ("hdd14", "hot_afternoon", "warm_night", SUN)
NIGHT_DAY_NAMES = {
    "hdd14": "cold (per °C of daily mean below 14 °C)",
    "hot_afternoon": "hot afternoon (per °C of daily maximum above 24 °C)",
    "warm_night": "warm night (per °C of daily minimum above 12 °C)",
    SUN: "sunshine (per kWh/m²)",
    "cdd20": "heat (per °C of daily mean above 20 °C)",
}
Z = 1.96
TYPE_ORDER = (
    "rises with heat",
    "rises with heat, falls with sunshine",
    "falls with sunshine only",
    "falls with heat",
    "neither",
)
AC_ORDER = ("no air conditioning", "part air-conditioned", "mostly air-conditioned")
DEC_ENV_ORDER = (
    "heating, natural ventilation",
    "heating, mechanical ventilation",
    "mixed mode",
    "air conditioning",
)
CATEGORICAL = {
    "building use": ("use", list(wh.USE_ORDER)),
    "air conditioning (DEC)": ("dec_ac_class", list(AC_ORDER)),
    "building environment (DEC)": ("dec_main_environment", list(DEC_ENV_ORDER)),
    "urban form": ("urban", list(URBAN_ORDER)),
    "tenure": ("tenure_group", ["freehold", "leased", "other"]),
    "heat pumps (ERIC)": ("heat_pumps_any", ["none", "installed"]),
    "overheating reported (ERIC)": ("overheats", ["none", "reported"]),
    "solar generation (ERIC)": ("solar_eric", ["none", "generates"]),
}
CONTINUOUS = {
    "air-conditioned floor share (DEC)": "dec_ac_share",
    "cooled floor share, AC or mixed mode (DEC)": "dec_cooled_share",
    "electric-heating floor share (DEC)": "dec_electric_heating_share",
    "building electricity (log kWh/m², DEC)": "log_dec_elec_kwh_m2",
    "operational rating (DEC)": "dec_operational_rating",
    "population density (log)": "log_density",
    "electricity intensity (log kWh/m²)": "log_elec_per_m2",
    "peak demand (W/m²)": "peak_w_m2",
    "night / day ratio": "night_day_ratio",
    "weekend / weekday ratio": "weekend_ratio",
    "winter / summer ratio": "winter_summer_ratio",
    "site size (log mean kW)": "log_mean_kw",
    "mean build year": "mean_build_year",
    "gas / electricity ratio": "gas_to_elec",
    "summer mean daily maximum (°C)": "summer_tmax",
}
#: Descriptor sets for the meta-regression of the site heat slopes.
SETS = {
    "building use": (["use"], []),
    "urban form and density": (["urban"], ["log_density"]),
    "estates descriptors": (
        ["heat_pumps_any", "overheats", "solar_eric", "tenure_group"],
        ["log_elec_per_m2", "peak_w_m2", "mean_build_year", "gas_to_elec", "led_pct"],
    ),
    "load shape": ([], ["night_day_ratio", "weekend_ratio", "winter_summer_ratio", "log_mean_kw"]),
    "all descriptors": (
        ["use", "urban", "heat_pumps_any", "overheats", "solar_eric", "tenure_group"],
        [
            "log_density",
            "log_elec_per_m2",
            "peak_w_m2",
            "mean_build_year",
            "gas_to_elec",
            "led_pct",
            "night_day_ratio",
            "weekend_ratio",
            "winter_summer_ratio",
            "log_mean_kw",
        ],
    ),
    "cooling plant (DEC)": (["dec_ac_class"], ["dec_cooled_share"]),
    "heating and intensity (DEC)": ([], ["dec_electric_heating_share", "log_dec_elec_kwh_m2"]),
    "all descriptors and DEC": (
        [
            "use",
            "urban",
            "heat_pumps_any",
            "overheats",
            "solar_eric",
            "tenure_group",
            "dec_ac_class",
        ],
        [
            "log_density",
            "log_elec_per_m2",
            "peak_w_m2",
            "mean_build_year",
            "gas_to_elec",
            "led_pct",
            "night_day_ratio",
            "weekend_ratio",
            "winter_summer_ratio",
            "log_mean_kw",
            "dec_cooled_share",
            "dec_electric_heating_share",
            "log_dec_elec_kwh_m2",
        ],
    ),
    "diurnal signature": ([], ["heat_afternoon_excess", "sun_day"]),
}
LEVELS = {
    "use": list(wh.USE_ORDER),
    "dec_ac_class": list(AC_ORDER),
    "dec_main_environment": list(DEC_ENV_ORDER),
    "urban": list(URBAN_ORDER),
    "tenure_group": ["freehold", "leased", "other"],
    "heat_pumps_any": ["none", "installed"],
    "overheats": ["none", "reported"],
    "solar_eric": ["none", "generates"],
}


# --------------------------------------------------------------------------- #
# Panel and site designs
# --------------------------------------------------------------------------- #


def coverage_flags() -> pd.Series:
    path = INTERIM_DIR / "site_coverage_flags.parquet"
    if not path.exists():
        return pd.Series(dtype=object)
    return (
        pd.read_parquet(path).drop_duplicates("site_code").set_index("site_code")["coverage_flag"]
    )


def subset(df: pd.DataFrame, clean: bool = True) -> pd.DataFrame:
    """Sites without CHP with at least ``weather_heterogeneity.MIN_DAYS`` days; with ``clean``,
    without the sites flagged in ``EXCLUDE_FLAGS``."""
    sub = df[df["group"].isin(wc.GROUPS["no CHP"])].copy()
    counts = sub.groupby("site_code").size()
    sub = sub[sub["site_code"].isin(counts.index[counts >= wh.MIN_DAYS])]
    if clean:
        flags = coverage_flags().reindex(sub["site_code"].unique())
        drop = flags.index[flags.isin(EXCLUDE_FLAGS)]
        if len(drop):
            logger.info("leaving out %d sites flagged %s", len(drop), EXCLUDE_FLAGS)
        sub = sub[~sub["site_code"].isin(drop)]
    return sub


def with_night_day(sub: pd.DataFrame) -> pd.DataFrame:
    """Add the daily minimum and the two halves of the heat dose."""
    out = sub
    if "t_min" not in out.columns:
        tmin = pd.read_parquet(
            INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_min"]
        )
        out = out.merge(tmin, on=["site_code", "date"], how="left")
    out = out.copy()
    out["hot_afternoon"] = np.clip(out["t_max"] - HOT_AFTERNOON_BASE, 0.0, None)
    out["warm_night"] = np.clip(out["t_min"] - WARM_NIGHT_BASE, 0.0, None)
    return out.dropna(subset=["t_min"])


def site_design(
    sub: pd.DataFrame, terms: tuple[str, ...], weekend: bool = False
) -> tuple[pd.DataFrame, dict, list[str]]:
    """Term x site columns under formula-safe names; with ``weekend``, a heat x weekend x site
    column as well."""
    sites = sorted(sub["site_code"].unique())
    cols, names = {}, {}
    we = (sub["date"].dt.dayofweek >= 5).to_numpy(dtype=float)
    for i, s in enumerate(sites):
        m = (sub["site_code"] == s).to_numpy(dtype=float)
        for v in terms:
            n = f"{v}_s{i}"
            cols[n] = sub[v].to_numpy() * m
            names[(v, s)] = n
        if weekend:
            n = f"cdd20we_s{i}"
            cols[n] = sub["cdd20"].to_numpy() * m * we
            names[("cdd20we", s)] = n
    return pd.DataFrame(cols, index=sub.index), names, sites


def fit_sites(sub: pd.DataFrame, y: np.ndarray, terms: tuple[str, ...], weekend: bool = False):
    import pyfixest as pf

    design, names, sites = site_design(sub, terms, weekend)
    data = pd.concat([sub[FE + ["site_code"]], design], axis=1)
    data["y"] = np.asarray(y, dtype=float)
    fml = f"y ~ {' + '.join(design.columns)} | {' + '.join(FE)}"
    fit = pf.feols(fml, data=data, vcov={"CRV1": "site_code+date_id"})
    return fit, names, sites


def extract(fit, names: dict, sites: list[str], terms: tuple[str, ...], weekend: bool = False):
    """Long table of site x term slopes; with ``weekend``, the weekend heat slope (weekday + the
    difference, variance from the fit's covariance) and the difference itself."""
    coef, se = fit.coef(), fit.se()
    cn, v = list(fit._coefnames), fit._vcov
    rows = []
    for s in sites:
        for t in terms:
            n = names[(t, s)]
            if n in coef.index:  # dropped as collinear otherwise
                rows.append({"site_code": s, "term": t, "slope": coef[n], "se": se[n]})
        if weekend:
            a, b = names[("cdd20", s)], names[("cdd20we", s)]
            if a in coef.index and b in coef.index:
                i, j = cn.index(a), cn.index(b)
                var = v[i, i] + v[j, j] + 2 * v[i, j]
                rows.append(
                    {
                        "site_code": s,
                        "term": "cdd20_weekend",
                        "slope": coef[a] + coef[b],
                        "se": np.sqrt(var) if var > 0 else np.nan,
                    }
                )
                rows.append(
                    {"site_code": s, "term": "cdd20_weekend_diff", "slope": coef[b], "se": se[b]}
                )
    out = pd.DataFrame(rows)
    out["slope_pct"] = 100 * out["slope"]
    out["se_pct"] = 100 * out["se"]
    return out


# --------------------------------------------------------------------------- #
# 1. Daily slopes with a site sunshine term; 2. blocks of the day
# --------------------------------------------------------------------------- #


def site_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Cold, heat and sunshine slopes per site, and the weekend heat slope, in one regression."""
    sub = subset(df)
    fit, names, sites = fit_sites(sub, sub["log_kw"].to_numpy(), TERMS, weekend=True)
    return extract(fit, names, sites, TERMS, weekend=True)


def block_demand(sites: list[str]) -> pd.DataFrame:
    """Mean kW of each site in each six-hour local block of each local day."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")[sites] * 2.0
    return block_means(w)


def block_means(w: pd.DataFrame) -> pd.DataFrame:
    """``w``: half-hourly kW, UTC index, one column per site."""
    local = pd.to_datetime(w.index, utc=True).tz_convert(wr.TZ)
    day = local.normalize().tz_localize(None).rename("date")
    block = pd.Index(local.hour // 6, name="block")
    n = w.notna().groupby([day, block]).sum()
    mean = w.groupby([day, block]).mean().where(n >= MIN_BLOCK_PERIODS)
    out = mean.stack().rename("block_kw").rename_axis(["date", "block", "site_code"]).reset_index()
    return out[out["block_kw"] > 0]


def block_slopes(df: pd.DataFrame) -> pd.DataFrame:
    """Cold, heat and sunshine slopes per site for each block of the day."""
    sub = subset(df)
    bd = block_demand(sorted(sub["site_code"].unique()))
    rows = []
    for b, name in BLOCKS.items():
        x = sub.merge(
            bd.loc[bd["block"] == b, ["date", "site_code", "block_kw"]],
            on=["date", "site_code"],
            how="inner",
        )
        fit, names, sites = fit_sites(x, np.log(x["block_kw"].to_numpy()), TERMS)
        rows.append(extract(fit, names, sites, TERMS).assign(block=b, block_name=name))
        logger.info("block %s done", name)
    return pd.concat(rows, ignore_index=True)


def plain_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Cold and heat slopes per site against temperature alone (the ``weather_heterogeneity``
    model on this module's sample)."""
    sub = subset(df)
    fit, names, sites = fit_sites(sub, sub["log_kw"].to_numpy(), ("hdd14", "cdd20"))
    return extract(fit, names, sites, ("hdd14", "cdd20"))


def night_day_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Per site: cold, hot-afternoon, warm-night and sunshine slopes in one regression."""
    sub = with_night_day(subset(df))
    fit, names, sites = fit_sites(sub, sub["log_kw"].to_numpy(), NIGHT_DAY_TERMS)
    return extract(fit, names, sites, NIGHT_DAY_TERMS)


def sign_groups(f: pd.DataFrame) -> pd.Series:
    """Sites by the sign of their sunshine-controlled daily heat slope at 5 %."""
    z = _z(f, "heat")
    return pd.Series(
        np.select([z > Z, z < -Z], ["rises with heat", "falls with heat"], default="neither"),
        index=f.index,
        name="sign",
    )


def group_decomposition(df: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """The heat dose for the three sign groups, two ways: the daily mean above 20 degC, and the
    split into hot afternoons and warm nights; cold and sunshine alongside, all x group."""
    import pyfixest as pf

    sub = with_night_day(subset(df))
    sub["sign"] = groups.reindex(sub["site_code"]).to_numpy()
    sub = sub.dropna(subset=["sign"])
    rows = []
    for model, terms in (
        ("daily mean", ("hdd14", "cdd20", SUN)),
        ("night and day", NIGHT_DAY_TERMS),
    ):
        cols, meta = {}, {}
        for gi, g in enumerate(("rises with heat", "falls with heat", "neither")):
            m = (sub["sign"] == g).to_numpy(dtype=float)
            if m.sum() == 0:
                continue
            for v in terms:
                n = f"{v}_g{gi}"
                cols[n] = sub[v].to_numpy() * m
                meta[n] = (g, v)
        x = pd.DataFrame(cols, index=sub.index)
        data = pd.concat([sub[FE + ["site_code", "log_kw"]], x], axis=1)
        fit = pf.feols(
            f"log_kw ~ {' + '.join(x.columns)} | {' + '.join(FE)}",
            data=data,
            vcov={"CRV1": "site_code+date_id"},
        )
        t = fit.tidy()
        for n, (g, v) in meta.items():
            if n not in t.index:
                continue
            r = t.loc[n]
            rows.append(
                {
                    "model": model,
                    "sign": g,
                    "term": v,
                    "term_name": NIGHT_DAY_NAMES[v],
                    "coef": r["Estimate"],
                    "se": r["Std. Error"],
                    "lo": r["2.5%"],
                    "hi": r["97.5%"],
                    "p": r["Pr(>|t|)"],
                    "sites": int(sub.loc[sub["sign"] == g, "site_code"].nunique()),
                }
            )
    out = pd.DataFrame(rows)
    for c in ("coef", "lo", "hi", "se"):
        out[f"{c}_pct"] = 100 * out[c]
    return out


# --------------------------------------------------------------------------- #
# Features and typology
# --------------------------------------------------------------------------- #


def _wide(long: pd.DataFrame, term: str, prefix: str, out: pd.DataFrame, suffix: str = ""):
    d = long[long["term"] == term].set_index("site_code")
    out[f"{prefix}{suffix}"] = d["slope"].reindex(out.index)
    out[f"{prefix}{suffix}_se"] = d["se"].reindex(out.index)


def features(
    terms: pd.DataFrame, blocks: pd.DataFrame, plain: pd.DataFrame, night_day: pd.DataFrame
) -> pd.DataFrame:
    """One row per site: the plain heat slope (temperature only), the heat and sunshine slopes
    net of each other, weekend heat, block slopes, the hot-afternoon and warm-night slopes, and
    the derived signatures."""
    out = pd.DataFrame(index=pd.Index(sorted(terms["site_code"].unique()), name="site_code"))
    _wide(plain, "cdd20", "heat_plain", out)
    _wide(terms, "cdd20", "heat", out)
    _wide(night_day, "hot_afternoon", "hot_afternoon", out)
    _wide(night_day, "warm_night", "warm_night", out)
    _wide(terms, "hdd14", "cold", out)
    _wide(terms, SUN, "sun", out)
    _wide(terms, "cdd20_weekend", "heat_weekend", out)
    _wide(terms, "cdd20_weekend_diff", "heat_weekend_diff", out)
    for b in BLOCKS:
        _wide(blocks[blocks["block"] == b], "cdd20", "heat", out, suffix=f"_b{b}")
        _wide(blocks[blocks["block"] == b], SUN, "sun", out, suffix=f"_b{b}")
    out["heat_afternoon_excess"] = out["heat_b2"] - out["heat_b0"]
    out["heat_afternoon_excess_se"] = np.sqrt(out["heat_b2_se"] ** 2 + out["heat_b0_se"] ** 2)
    # Daytime sunshine slope: precision-weighted mean of the morning and afternoon blocks.
    w1, w2 = 1 / out["sun_b1_se"] ** 2, 1 / out["sun_b2_se"] ** 2
    out["sun_day"] = (w1 * out["sun_b1"] + w2 * out["sun_b2"]) / (w1 + w2)
    out["sun_day_se"] = np.sqrt(1 / (w1 + w2))
    out["sun_night"] = out["sun_b0"]
    out["sun_night_se"] = out["sun_b0_se"]
    return out


def _z(f: pd.DataFrame, col: str) -> pd.Series:
    return f[col] / f[f"{col}_se"]


def typology(f: pd.DataFrame) -> pd.Series:
    rises = _z(f, "heat") > Z
    falls_heat = _z(f, "heat") < -Z
    sun_neg = _z(f, "sun_day") < -Z
    kind = np.select(
        [rises & sun_neg, rises, falls_heat, sun_neg],
        [TYPE_ORDER[1], TYPE_ORDER[0], TYPE_ORDER[3], TYPE_ORDER[2]],
        default=TYPE_ORDER[4],
    )
    return pd.Series(kind, index=f.index, name="type")


def flags(f: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "afternoon_peaked": _z(f, "heat_afternoon_excess") > Z,
            "weekday_only": (_z(f, "heat_weekend_diff") < -Z) & (_z(f, "heat") > Z),
            "night_sunshine_negative": _z(f, "sun_night") < -Z,
            "cooling_afternoon": _z(f, "hot_afternoon") > Z,
            "heating_tail": _z(f, "warm_night") < -Z,
        },
        index=f.index,
    )


# --------------------------------------------------------------------------- #
# Descriptors
# --------------------------------------------------------------------------- #


def descriptors(df: pd.DataFrame) -> pd.DataFrame:
    """The heterogeneity descriptors plus estates, context and load-shape fields."""
    base = wh.descriptors(df)
    eric = pd.read_parquet(INTERIM_DIR / "eric_2023_24_site.parquet").drop_duplicates("site_code")
    eric = eric.set_index("site_code").reindex(base.index)
    ctx = pd.read_parquet(INTERIM_DIR / "site_context.parquet").drop_duplicates("site_code")
    ctx = ctx.set_index("site_code").reindex(base.index)
    clusters = pd.read_csv(wh.CLUSTER_DIR / "elec_site_clusters.csv").set_index("series_id")
    clusters = clusters.reindex(base.index)
    out = base.copy()
    for c in ("elec_kwh_per_m2", "max_demand_kw", "gia_m2", "heat_pumps", "tenure"):
        out[c] = eric[c]
    out["overheating_occurrences"] = eric["overheating_occurrences"]
    out["solar_generated_kwh"] = eric["solar_generated_kwh"]
    out["solar_consumed_kwh"] = eric["solar_consumed_kwh"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_elec_per_m2"] = np.log(out["elec_kwh_per_m2"].where(out["elec_kwh_per_m2"] > 0))
        out["peak_w_m2"] = (1000 * out["max_demand_kw"] / out["gia_m2"]).where(
            (out["gia_m2"] > 0) & (out["max_demand_kw"] > 0)
        )
        out["log_density"] = np.log(ctx["density_km2"].where(ctx["density_km2"] > 0))
    out["urban"] = ctx["urban"]
    out["density_km2"] = ctx["density_km2"]
    out["repd_solar_mw"] = ctx["repd_solar_mw"]
    out["repd_solar_status"] = ctx["repd_solar_status"]
    out["tenure_group"] = tenure_group(out["tenure"])
    out["heat_pumps_any"] = np.where(
        out["heat_pumps"].isna(), None, np.where(out["heat_pumps"] > 0, "installed", "none")
    )
    out["overheats"] = np.where(
        out["overheating_occurrences"].isna(),
        None,
        np.where(out["overheating_occurrences"] > 0, "reported", "none"),
    )
    solar = out[["solar_generated_kwh", "solar_consumed_kwh"]].sum(axis=1, min_count=1)
    out["solar_eric"] = np.where(solar.isna(), None, np.where(solar > 0, "generates", "none"))
    for c in ("night_day_ratio", "weekend_ratio", "winter_summer_ratio", "daily_cv"):
        out[c] = clusters[c]
    return out.join(dec_descriptors(out.index, out["gia_m2"]))


def dec_descriptors(index: pd.Index, gia_m2: pd.Series | None = None) -> pd.DataFrame:
    """Display Energy Certificate aggregates per site (``interim/dec_sites.parquet``), when
    built: air-conditioned and cooled floor shares, electric-heating share, building electricity
    intensity, operational rating, the main building environment, and an air-conditioning class
    (none / part / mostly, by floor share). Empty columns when the table is absent."""
    cols = [
        "dec_buildings",
        "dec_floor_area_m2",
        "dec_ac_share",
        "dec_cooled_share",
        "dec_ac_kw_m2",
        "dec_electric_heating_share",
        "dec_elec_kwh_m2",
        "dec_operational_rating",
        "dec_main_environment",
    ]
    path = INTERIM_DIR / "dec_sites.parquet"
    if not path.exists():
        out = pd.DataFrame(index=index, columns=cols, dtype=float)
        out["dec_main_environment"] = None
    else:
        d = pd.read_parquet(path).drop_duplicates("site_code").set_index("site_code")
        out = d.reindex(index)[cols].copy()
    has = out["dec_buildings"].fillna(0) > 0
    out.loc[~has, cols[1:]] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_dec_elec_kwh_m2"] = np.log(
            out["dec_elec_kwh_m2"].where(out["dec_elec_kwh_m2"] > 0)
        )
    out["dec_ac_class"] = pd.Series(
        np.select(
            [out["dec_ac_share"] >= 0.5, out["dec_ac_share"] > 0, out["dec_ac_share"] == 0],
            [AC_ORDER[2], AC_ORDER[1], AC_ORDER[0]],
            default=None,
        ),
        index=index,
    ).where(has)
    out.loc[~has, "dec_main_environment"] = None
    out["dec_main_environment"] = out["dec_main_environment"].where(
        out["dec_main_environment"].isin(DEC_ENV_ORDER)
    )
    if gia_m2 is not None:
        out["dec_coverage"] = (out["dec_floor_area_m2"] / gia_m2).where(gia_m2 > 0)
    return out


def tenure_group(tenure: pd.Series) -> pd.Series:
    t = tenure.astype("string").str.lower()
    free = t.str.startswith("freehold").fillna(False).to_numpy(dtype=bool)
    lease = t.str.contains("lease", na=False).fillna(False).to_numpy(dtype=bool)
    out = np.select([free, lease], ["freehold", "leased"], default="other")
    return pd.Series(out, index=tenure.index).where(tenure.notna())


# --------------------------------------------------------------------------- #
# 3. Moderators with sunshine alongside, and the meta-regression
# --------------------------------------------------------------------------- #


def _fit_moderated(sub: pd.DataFrame, design: pd.DataFrame, label: str) -> pd.DataFrame:
    """Cold, heat and sunshine main terms plus their interactions with every design column;
    the joint test covers the heat interactions."""
    import pyfixest as pf

    cols, names = {}, []
    for c in design.columns:
        for v in TERMS:
            n = f"{v}_x_{c}"
            cols[n] = sub[v].to_numpy() * design[c].to_numpy(dtype=float)
            names.append(n)
    inter = pd.DataFrame(cols, index=sub.index)
    data = pd.concat([sub[FE + ["site_code", "log_kw"] + list(TERMS)], inter], axis=1)
    fml = f"log_kw ~ {' + '.join(list(TERMS) + names)} | {' + '.join(FE)}"
    fit = pf.feols(fml, data=data, vcov={"CRV1": "site_code+date_id"})
    t = fit.tidy()
    kept = [n for n in names if n in t.index]
    rows = []
    for n in list(TERMS) + kept:
        r = t.loc[n]
        term = n.split("_x_")[0] if "_x_" in n else n
        rows.append(
            {
                "model": label,
                "term": n,
                "dose": term,
                "dose_name": TERM_NAMES[term],
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
        out[f"{c}_pct"] = 100 * out[c]
    heat = [n for n in kept if n.startswith("cdd20_x_")]
    idx = [i for i, n in enumerate(fit._coefnames) if n in heat]
    r_mat = np.zeros((len(idx), len(fit._coefnames)))
    r_mat[np.arange(len(idx)), idx] = 1.0
    out["p_joint"] = float(fit.wald_test(R=r_mat)["pvalue"]) if idx else np.nan
    out["sites"] = sub["site_code"].nunique()
    return out


def moderators(df: pd.DataFrame, desc: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sub = subset(df)
    sub = sub[sub["site_code"].isin(desc.index)]
    rows = []
    for label, (col, order) in CATEGORICAL.items():
        vals = desc[col].reindex(sub["site_code"]).to_numpy()
        keep = pd.notna(vals)
        s = sub[keep].copy()
        design = wh._categorical_design(pd.Series(vals[keep], index=s.index), order)
        design = design.loc[:, design.sum() > 0]
        if design.shape[1] == 0:
            continue
        rows.append(_fit_moderated(s, design, label).assign(kind="categorical", reference=order[0]))
    for label, col in CONTINUOUS.items():
        vals = desc[col].reindex(sub["site_code"]).to_numpy(dtype=float)
        keep = np.isfinite(vals)
        s = sub[keep].copy()
        design = pd.DataFrame({col: wh._standardised(pd.Series(vals[keep], index=s.index))})
        rows.append(_fit_moderated(s, design, label).assign(kind="continuous", reference="per SD"))
    one = pd.concat(rows, ignore_index=True)

    # Joint: use + urban + the continuous set, on sites with every descriptor.
    cats = ["use", "urban"]
    cont = ["log_density", "log_elec_per_m2", "peak_w_m2", "night_day_ratio", "weekend_ratio"]
    if desc["dec_cooled_share"].notna().sum() >= 30:
        cont = cont + ["dec_cooled_share", "dec_electric_heating_share"]
    full = desc.dropna(subset=cats + cont)
    sj = sub[sub["site_code"].isin(full.index)].copy()
    codes = sj["site_code"].to_numpy()
    parts = []
    for c in cats:
        d = wh._categorical_design(
            pd.Series(full[c].reindex(codes).to_numpy(), index=sj.index), LEVELS[c]
        )
        parts.append(d.loc[:, d.sum() > 0])
    parts.append(
        pd.DataFrame(
            {
                c: wh._standardised(pd.Series(full[c].reindex(codes).to_numpy(), index=sj.index))
                for c in cont
            },
            index=sj.index,
        )
    )
    design = pd.concat(parts, axis=1)
    assert len(design) == len(sj), (len(design), len(sj))
    joint = _fit_moderated(sj, design, "joint").assign(kind="joint", reference="see levels")
    return one, joint


def meta_regression(y: np.ndarray, se: np.ndarray, x: np.ndarray) -> dict:
    """Random-effects meta-regression with a DerSimonian-Laird residual tau2: how much of the
    true between-site variance of the slopes the covariates explain (``r2_true``)."""
    y, se = np.asarray(y, float), np.asarray(se, float)
    x = np.column_stack([np.ones(len(y)), np.asarray(x, float)])
    k, p = x.shape
    re0 = wh.random_effects(pd.Series(y), pd.Series(se))
    tau2_0 = re0["tau"] ** 2
    w = 1 / se**2
    xtwx_inv = np.linalg.pinv(x.T @ (w[:, None] * x))
    beta_fe = xtwx_inv @ x.T @ (w * y)
    resid = y - x @ beta_fe
    q = float((w * resid**2).sum())
    c = float(w.sum() - np.trace(xtwx_inv @ x.T @ (w[:, None] ** 2 * x)))
    tau2 = max(0.0, (q - (k - p)) / c) if c > 0 else 0.0
    w2 = 1 / (se**2 + tau2)
    cov = np.linalg.pinv(x.T @ (w2[:, None] * x))
    beta = cov @ x.T @ (w2 * y)
    return {
        "beta": beta,
        "se": np.sqrt(np.diag(cov)),
        "tau2": tau2,
        "tau2_unconditional": tau2_0,
        "r2_true": 1 - tau2 / tau2_0 if tau2_0 > 0 else np.nan,
        "sites": k,
        "parameters": p,
    }


def variance_explained(f: pd.DataFrame, desc: pd.DataFrame) -> pd.DataFrame:
    """The share of the true between-site variance of the heat slope explained by each descriptor
    set, after the sunshine control; the first row is the sunshine control itself."""
    rows = []
    both = f.dropna(subset=["heat_plain", "heat_plain_se", "heat", "heat_se"])
    both = both[(both["heat_plain_se"] > 0) & (both["heat_se"] > 0)]
    t0 = wh.random_effects(both["heat_plain"], both["heat_plain_se"])["tau"] ** 2
    t1 = wh.random_effects(both["heat"], both["heat_se"])["tau"] ** 2
    rows.append(
        {
            "set": "site sunshine term",
            "sites": len(both),
            "parameters": 0,
            "tau_before_pct": 100 * np.sqrt(t0),
            "tau_after_pct": 100 * np.sqrt(t1),
            "r2_true": 1 - t1 / t0 if t0 > 0 else np.nan,
        }
    )
    d = f.join(desc[[c for c in desc.columns if c not in f.columns]], how="left")
    for label, (cats, cont) in SETS.items():
        cols = cats + cont
        sub = d.dropna(subset=["heat", "heat_se"] + cols)
        sub = sub[sub["heat_se"] > 0]
        parts = []
        for c in cats:
            dm = wh._categorical_design(sub[c], LEVELS[c])
            parts.append(dm.loc[:, dm.sum() > 0])
        for c in cont:
            parts.append(wh._standardised(sub[c].astype(float)).to_frame(c))
        x = pd.concat(parts, axis=1).to_numpy(dtype=float)
        m = meta_regression(sub["heat"].to_numpy(), sub["heat_se"].to_numpy(), x)
        rows.append(
            {
                "set": label,
                "sites": m["sites"],
                "parameters": m["parameters"] - 1,
                "tau_before_pct": 100 * np.sqrt(m["tau2_unconditional"]),
                "tau_after_pct": 100 * np.sqrt(m["tau2"]),
                "r2_true": m["r2_true"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #


def block_pooled(blocks: pd.DataFrame, term: str) -> dict[str, float]:
    """Inverse-variance pooled slope (%) of ``term`` in each block of the day."""
    out = {}
    for b, name in BLOCKS.items():
        d = blocks[(blocks["block"] == b) & (blocks["term"] == term) & (blocks["se"] > 0)]
        out[name] = round(100 * wh.random_effects(d["slope"], d["se"])["pooled"], 2)
    return out


def excluded_sites(sub_all: pd.DataFrame, kept: pd.Index) -> pd.DataFrame:
    """The flagged sites left out, with their heat slopes from the heterogeneity run (all sites)
    so the reader can see what the exclusion removed."""
    out = pd.DataFrame(
        index=pd.Index(sorted(set(sub_all["site_code"]) - set(kept)), name="site_code")
    )
    out["coverage_flag"] = coverage_flags().reindex(out.index)
    path = wh.OUT_DIR / "site_slopes.csv"
    if path.exists():
        sl = pd.read_csv(path)
        heat = sl[sl["dose"] == "cdd20"].set_index("site_code")
        out["heat_plain_pct"] = heat["slope_pct"].reindex(out.index)
        out["heat_plain_se_pct"] = heat["se_pct"].reindex(out.index)
        out["mean_kw"] = heat["mean_kw"].reindex(out.index)
    return out


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = wc.panel()
    sub = subset(df)
    desc = descriptors(sub)
    desc.to_csv(OUT_DIR / "site_descriptors.csv")
    with threadpool_limits(4):
        plain = plain_terms(df)
        plain.to_csv(OUT_DIR / "plain_terms.csv", index=False)
        logger.info("plain slopes done")
        terms = site_terms(df)
        terms.to_csv(OUT_DIR / "site_terms.csv", index=False)
        logger.info("site terms done")
        night_day = night_day_terms(df)
        night_day.to_csv(OUT_DIR / "night_day_terms.csv", index=False)
        logger.info("night and day terms done")
        blocks = block_slopes(df)
        blocks.to_csv(OUT_DIR / "block_slopes.csv", index=False)
        f = features(terms, blocks, plain, night_day)
        f["type"] = typology(f)
        f["sign"] = sign_groups(f)
        f = f.join(flags(f))
        f = f.join(
            desc[
                [
                    "use",
                    "mean_kw",
                    "region",
                    "urban",
                    "overheats",
                    "solar_eric",
                    "dec_buildings",
                    "dec_ac_share",
                    "dec_cooled_share",
                    "dec_electric_heating_share",
                    "dec_main_environment",
                    "dec_ac_class",
                    "dec_coverage",
                ]
            ]
        )
        f["coverage_flag"] = coverage_flags().reindex(f.index)
        f.to_csv(OUT_DIR / "site_features.csv")
        logger.info("features done")
        groups = group_decomposition(df, f["sign"])
        groups.to_csv(OUT_DIR / "group_decomposition.csv", index=False)
        logger.info("group decomposition done")
        excluded_sites(sub_all=subset(df, clean=False), kept=f.index).to_csv(
            OUT_DIR / "excluded_sites.csv"
        )
        one, joint = moderators(df, desc)
        one.to_csv(OUT_DIR / "moderators_heat.csv", index=False)
        joint.to_csv(OUT_DIR / "moderators_heat_joint.csv", index=False)
        logger.info("moderators done")
    ve = variance_explained(f, desc)
    ve.to_csv(OUT_DIR / "variance_explained.csv", index=False)

    ok = f[(f["heat_plain_se"] > 0) & (f["heat_se"] > 0)]
    re_plain = wh.random_effects(ok["heat_plain"], ok["heat_plain_se"])
    re_adj = wh.random_effects(ok["heat"], ok["heat_se"])
    sun_ok = f[f["sun_se"] > 0]
    re_sun = wh.random_effects(sun_ok["sun"], sun_ok["sun_se"])
    neg_plain = (ok["heat_plain"] + Z * ok["heat_plain_se"]) < 0
    neg_adj = (ok["heat"] + Z * ok["heat_se"]) < 0
    types = f["type"].value_counts().reindex(TYPE_ORDER).fillna(0).astype(int).to_dict()
    demand_share = (
        f.groupby("type")["mean_kw"].sum().reindex(TYPE_ORDER).fillna(0) / f["mean_kw"].sum()
    )
    summary = {
        "sites": int(len(f)),
        "tau_pct": {
            "plain": round(100 * re_plain["tau"], 2),
            "sunshine_controlled": round(100 * re_adj["tau"], 2),
        },
        "pooled_pct": {
            "heat_plain": round(100 * re_plain["pooled"], 2),
            "heat_sunshine_controlled": round(100 * re_adj["pooled"], 2),
            "sunshine_per_kwh_m2": round(100 * re_sun["pooled"], 2),
        },
        "sunshine_tau_pct": round(100 * re_sun["tau"], 2),
        "sites_falling_on_hot_days": {
            "plain": int(neg_plain.sum()),
            "sunshine_controlled": int(neg_adj.sum()),
            "both": int((neg_plain & neg_adj).sum()),
        },
        "sites_sunshine_negative_daytime": int((_z(f, "sun_day") < -Z).sum()),
        "sites_sunshine_positive_daytime": int((_z(f, "sun_day") > Z).sum()),
        "sites_night_sunshine_negative": int(f["night_sunshine_negative"].sum()),
        "mechanisms": {
            "risers_with_afternoon_cooling": int(
                (f["cooling_afternoon"] & (f["sign"] == "rises with heat")).sum()
            ),
            "fallers_with_heating_tail": int(
                (f["heating_tail"] & (f["sign"] == "falls with heat")).sum()
            ),
            "risers": int((f["sign"] == "rises with heat").sum()),
            "fallers": int((f["sign"] == "falls with heat").sum()),
        },
        "group_decomposition_pct": {
            f"{r['model']} | {r['sign']} | {r['term']}": [
                round(r["coef_pct"], 2),
                round(r["se_pct"], 2),
            ]
            for _, r in groups.iterrows()
        },
        "corr_heat_sun": round(float(f[["heat", "sun"]].corr().iloc[0, 1]), 3),
        "corr_heat_plain_vs_adjusted": round(float(f[["heat_plain", "heat"]].corr().iloc[0, 1]), 3),
        "types": types,
        "type_demand_share": {k: round(float(v), 3) for k, v in demand_share.items()},
        "afternoon_peaked_among_risers": int(
            (f["afternoon_peaked"] & f["type"].isin(TYPE_ORDER[:2])).sum()
        ),
        "weekday_only": int(f["weekday_only"].sum()),
        "block_pooled_heat_pct": block_pooled(blocks, "cdd20"),
        "block_pooled_sun_pct": block_pooled(blocks, SUN),
        "dec_sites": int((f["dec_buildings"].fillna(0) > 0).sum()),
        "dec_ac_class_counts": f["dec_ac_class"].value_counts().to_dict(),
        "dec_median_coverage": (
            round(float(f["dec_coverage"].median()), 2) if f["dec_coverage"].notna().any() else None
        ),
        "moderator_joint_p": {
            m: round(float(d["p_joint"].iloc[0]), 3) for m, d in one.groupby("model")
        },
        "joint_model_p": round(float(joint["p_joint"].iloc[0]), 3),
        "joint_model_sites": int(joint["sites"].iloc[0]),
        "variance_explained": {
            r["set"]: (round(float(r["r2_true"]), 3), int(r["sites"])) for _, r in ve.iterrows()
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("heat drivers: %s", json.dumps(summary)[:600])
    return {
        "terms": terms,
        "night_day": night_day,
        "groups": groups,
        "blocks": blocks,
        "features": f,
        "descriptors": desc,
        "moderators": one,
        "joint": joint,
        "variance": ve,
        "summary": summary,
    }
