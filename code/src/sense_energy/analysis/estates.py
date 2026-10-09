"""Analyses that join the half-hourly demand data with ERIC 2023/24 and KH03 beds.

1. **Meter coverage flags** - each site's metered electricity over the ERIC
   financial year (grossed up for missing half-hours) against ERIC's reported
   consumption, and the metered peak against ERIC's maximum demand. A site is
   ``ok`` when the annual ratio is within [0.65, 1.5], ``partial_supply`` below
   (our meters miss part of the site), ``aggregate_supply`` above (one meter feeds
   more than the ERIC site), ``meter_gaps`` when fewer than 80 % of the year's
   half-hours are observed and ``no_eric`` when ERIC has no electricity figure.
2. **Heat-pump headroom** - the site's gas use replaced by air-source heat
   pumps, added half-hour by half-hour to its observed electricity over the same
   year. Heat = gas x boiler efficiency, split into a weather-independent base
   (hot water, sterilisation, catering) and space heating proportional to
   heating degree-hours below a base temperature at the site (ERA5); the base
   share and base temperature are fitted on the project's half-hourly gas
   meters. Heat-pump electricity = heat / COP(T), with the Staffell et al. (2012)
   air-source curve at a 55 degC flow temperature. The new peak is compared with
   ERIC's available capacity (kVA x power factor).
3. **Bed-day intensity** - trust electricity (ERIC sum over sites) per occupied
   general & acute bed-day (KH03, the four quarters of 2023/24).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "estates"
FY = ("2023-04-01", "2024-04-01")
TZ = "Europe/London"
DEFAULTS = {
    "ratio_ok": (0.65, 1.5),
    "min_meter_coverage": 0.8,
    "boiler_efficiency": 0.85,
    "flow_temp_c": 55.0,
    "power_factor": 0.95,
    "base_temp_c": 15.5,
}


def _fy_slice(wide: pd.DataFrame) -> pd.DataFrame:
    loc = wide.index.tz_convert(TZ)
    m = (loc >= pd.Timestamp(FY[0], tz=TZ)) & (loc < pd.Timestamp(FY[1], tz=TZ))
    return wide.loc[m]


def _wide(energy: str, flavour: str = "raw") -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED_DIR / energy / "site" / f"wide_{flavour}.parquet")
    w.index = pd.to_datetime(w.index, utc=True)
    return w


# --------------------------------------------------------------------------- #
# 1. Coverage flags
# --------------------------------------------------------------------------- #


def coverage_flags(eric: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = {**DEFAULTS, **(cfg or {})}
    lo, hi = cfg["ratio_ok"]
    fy = _fy_slice(_wide("elec"))
    cov = fy.notna().mean()
    meter = pd.DataFrame(
        {
            "meter_kwh_fy": fy.sum() / cov.replace(0, np.nan),
            "meter_coverage_fy": cov,
            "meter_peak_kw": fy.max() * 2,
        }
    )
    meter.index.name = "site_code"
    e = eric.drop_duplicates("site_code").set_index("site_code")
    t = meter.join(
        e[
            [
                "elec_kwh",
                "max_demand_kw",
                "capacity_kva",
                "site_type",
                "chp_units",
                "solar_generated_kwh",
            ]
        ],
        how="left",
    )
    t["in_eric"] = t.index.isin(e.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        t["energy_ratio"] = (t["meter_kwh_fy"] / t["elec_kwh"]).where(t["elec_kwh"] > 0)
        t["peak_ratio"] = (t["meter_peak_kw"] / t["max_demand_kw"]).where(t["max_demand_kw"] > 0)
    outside = (t["energy_ratio"] < lo) | (t["energy_ratio"] > hi)
    flag = np.select(
        [
            ~t["in_eric"] | ~(t["elec_kwh"] > 0),
            t["meter_coverage_fy"] < cfg["min_meter_coverage"],
            # with on-site generation the grid import and ERIC's consumption differ by the
            # CHP output in either direction, so a mismatch says nothing about meter coverage
            outside & (t["chp_units"].fillna(0) > 0),
            t["energy_ratio"] < lo,
            t["energy_ratio"] > hi,
        ],
        ["no_eric", "meter_gaps", "chp_mismatch", "partial_supply", "aggregate_supply"],
        default="ok",
    )
    t["coverage_flag"] = flag
    return t.reset_index()


# --------------------------------------------------------------------------- #
# 2. Heat-pump headroom
# --------------------------------------------------------------------------- #


def cop_ashp(temp_c: np.ndarray, flow_temp_c: float = 55.0) -> np.ndarray:
    """Air-source heat pump COP (Staffell et al. 2012): 6.81 - 0.121 dT + 0.000630 dT^2."""
    dt = np.clip(flow_temp_c - np.asarray(temp_c, dtype=float), 15.0, None)
    return np.clip(6.81 - 0.121 * dt + 0.000630 * dt**2, 1.0, None)


def site_temperature(sites: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """ERA5 2 m temperature (degC) per site, interpolated onto a half-hourly index."""
    w = pd.read_parquet(
        INTERIM_DIR / "weather_reanalysis.parquet", columns=["site_code", "datetime", "t2m"]
    )
    w = w[w["site_code"].isin(sites)]
    grid = index.asi8
    out = {}
    for s, g in w.groupby("site_code"):
        g = g.sort_values("datetime")
        x = g["datetime"].to_numpy().astype("datetime64[ns]").astype("int64")
        out[s] = np.interp(grid, x, g["t2m"].to_numpy() - 273.15)
    return pd.DataFrame(out, index=index)


def fit_heat_model(
    base_temps=(12.0, 13.5, 15.5, 17.0, 18.5, 20.0), min_days: int = 180
) -> dict[str, Any]:
    """Fit daily gas = a + b * HDD(T_base) per gas meter (ERA5 at the meter's site); return the
    base temperature with the best median R^2 and the median weather-independent share."""
    gas = pd.read_parquet(PROCESSED_DIR / "gas" / "meter" / "wide_raw.parquet")
    gas.index = pd.to_datetime(gas.index, utc=True)
    idx = pd.read_parquet(PROCESSED_DIR / "gas" / "meter" / "series_index.parquet").set_index(
        "series_id"
    )
    local = gas.index.tz_convert(TZ)
    day = local.normalize()
    daily = gas.groupby(day).sum(min_count=40)
    keep = [c for c in daily.columns if daily[c].notna().sum() >= min_days and daily[c].sum() > 0]
    site_of = {m: str(idx.loc[m, "site_code"]) for m in keep if m in idx.index}
    temp = site_temperature(sorted(set(site_of.values())), gas.index)
    tday = temp.groupby(day).mean()
    rows = []
    for tb in base_temps:
        hdd = np.clip(tb - tday, 0, None)
        for m, s in site_of.items():
            if s not in hdd.columns:
                continue
            d = pd.concat([daily[m].rename("g"), hdd[s].rename("h")], axis=1).dropna()
            if len(d) < min_days or d["h"].std() == 0:
                continue
            b, a = np.polyfit(d["h"], d["g"], 1)
            pred = a + b * d["h"]
            r2 = 1 - ((d["g"] - pred) ** 2).sum() / ((d["g"] - d["g"].mean()) ** 2).sum()
            base_share = float(np.clip(max(0.0, a) * len(d) / d["g"].sum(), 0, 1))
            rows.append(
                {
                    "base_temp_c": tb,
                    "meter": m,
                    "site_code": s,
                    "r2": r2,
                    "base_share": base_share,
                    "n_days": len(d),
                }
            )
    fits = pd.DataFrame(rows)
    by_t = fits.groupby("base_temp_c")["r2"].median()
    best = float(by_t.idxmax())
    sel = fits[
        (fits["base_temp_c"] == best) & (fits["r2"] > 0.3)
    ]  # meters that actually follow the weather
    out = {
        "base_temp_c": best,
        "base_share": float(sel["base_share"].median()),
        "r2_median": float(fits.loc[fits["base_temp_c"] == best, "r2"].median()),
        "n_meters": int((fits["base_temp_c"] == best).sum()),
        "n_meters_weather_driven": int(len(sel)),
        "r2_by_base_temp": {float(k): round(float(v), 3) for k, v in by_t.items()},
        "fits": fits,
    }
    logger.info(
        "heat model: base %.1f degC (median R2 by base: %s), weather-independent share %.2f over %d weather-driven of %d gas meters",
        best,
        out["r2_by_base_temp"],
        out["base_share"],
        out["n_meters_weather_driven"],
        out["n_meters"],
    )
    return out


def heat_pump_headroom(
    eric: pd.DataFrame, model: dict[str, Any], cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """Per site: observed peak, peak with all gas heat from heat pumps, capacity, utilisation."""
    cfg = {**DEFAULTS, **(cfg or {})}
    elec = _fy_slice(_wide("elec", "imputed"))
    e = eric.drop_duplicates("site_code").set_index("site_code")
    sites = [
        s
        for s in elec.columns
        if s in e.index
        and e.loc[s, "gas_kwh"] > 0
        and e.loc[s, "capacity_kva"] > 0
        and elec[s].notna().mean() >= cfg["min_meter_coverage"]
    ]
    temp = site_temperature(sites, elec.index)
    rows, profiles = [], {}
    for s in sites:
        if s not in temp.columns:
            continue
        T = temp[s].to_numpy()
        heat_total = e.loc[s, "gas_kwh"] * cfg["boiler_efficiency"]  # kWh of heat over the year
        hdh = np.clip(model["base_temp_c"] - T, 0, None)
        n = len(T)
        base = model["base_share"] * heat_total / n * np.ones(n)
        space = (
            (1 - model["base_share"]) * heat_total * hdh / hdh.sum()
            if hdh.sum() > 0
            else np.zeros(n)
        )
        heat = base + space  # kWh per half hour
        hp = heat / cop_ashp(T, cfg["flow_temp_c"])
        obs = elec[s].to_numpy()
        obs_filled = np.where(np.isfinite(obs), obs, np.nanmean(obs))
        new = obs_filled + hp
        cap_kw = e.loc[s, "capacity_kva"] * cfg["power_factor"]
        rows.append(
            {
                "site_code": s,
                "site_type": e.loc[s, "site_type"],
                "gas_kwh": e.loc[s, "gas_kwh"],
                "elec_kwh_fy": float(np.nansum(obs_filled)),
                "hp_kwh": float(hp.sum()),
                "peak_now_kw": float(np.nanmax(obs) * 2),
                "peak_hp_kw": float(new.max() * 2),
                "hp_peak_kw": float(hp.max() * 2),
                "hp_mean_kw": float(hp.mean() * 2),
                "capacity_kw": float(cap_kw),
                "seasonal_cop": float(heat.sum() / hp.sum()),
                "max_demand_kw_eric": e.loc[s, "max_demand_kw"],
            }
        )
        profiles[s] = pd.DataFrame(
            {
                "datetime": elec.index,
                "site_code": s,
                "now_kw": obs * 2,
                "hp_kw": new * 2,
                "temp_c": T,
            }
        )
    out = pd.DataFrame(rows)
    out.attrs["profiles"] = profiles
    out["chp_units"] = out["site_code"].map(e["chp_units"]).fillna(0)
    out["util_now"] = out["peak_now_kw"] / out["capacity_kw"]
    out["util_hp"] = out["peak_hp_kw"] / out["capacity_kw"]
    out["hp_peak_to_mean"] = out["hp_peak_kw"] / out["hp_mean_kw"]
    out["elec_increase"] = out["hp_kwh"] / out["elec_kwh_fy"]
    # how much the site's peak actually rises, per kW of mean heat-pump load: below the
    # heat-pump peak-to-mean ratio because the heating peak and today's peak rarely coincide
    out["coincident_increase_factor"] = (out["peak_hp_kw"] - out["peak_now_kw"]) / out["hp_mean_kw"]
    return out


def headroom_status(head: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Why a site is (not) in the main headroom estimate: the metered supply must match
    ERIC (coverage flag ok), the stated capacity must exceed today's peak, and the gas
    must be heating fuel only (no CHP, whose gas also generates electricity)."""
    f = flags.set_index("site_code")["coverage_flag"]
    h = head.copy()
    h["coverage_flag"] = h["site_code"].map(f)
    h["headroom_status"] = np.select(
        [h["coverage_flag"] != "ok", h["util_now"] > 1.05, h["chp_units"] > 0],
        ["meter_not_whole_site", "capacity_implausible", "gas_feeds_chp"],
        default="included",
    )
    return h


def headroom_england(
    eric: pd.DataFrame, peak_to_mean: float, cfg: dict[str, Any] | None = None
) -> pd.DataFrame:
    """All ERIC sites with gas, max demand and capacity, no CHP and a plausible capacity:
    peak with heat pumps = ERIC maximum demand + mean heat-pump load x the coincident
    increase factor calibrated on the profile-modelled panel sites (``peak_to_mean``);
    seasonal COP from the same sites."""
    cfg = {**DEFAULTS, **(cfg or {})}
    e = eric[
        (eric["gas_kwh"] > 0) & (eric["max_demand_kw"] > 0) & (eric["capacity_kva"] > 0)
    ].copy()
    e["capacity_kw"] = e["capacity_kva"] * cfg["power_factor"]
    e = e[(e["chp_units"].fillna(0) == 0) & (e["max_demand_kw"] <= 1.05 * e["capacity_kw"])]
    hp_mean_kw = e["gas_kwh"] * cfg["boiler_efficiency"] / cfg.get("seasonal_cop", 2.55) / 8760
    e["hp_mean_kw"] = hp_mean_kw
    e["peak_hp_kw"] = e["max_demand_kw"] + peak_to_mean * hp_mean_kw
    e["util_now"] = e["max_demand_kw"] / e["capacity_kw"]
    e["util_hp"] = e["peak_hp_kw"] / e["capacity_kw"]
    return e


# --------------------------------------------------------------------------- #
# 3. Bed-day intensity
# --------------------------------------------------------------------------- #


def bed_day_intensity(eric: pd.DataFrame, beds: pd.DataFrame) -> pd.DataFrame:
    tr = eric.groupby("trust_code").agg(
        elec_kwh=("elec_kwh", "sum"),
        gas_kwh=("gas_kwh", "sum"),
        gia_m2=("gia_m2", "sum"),
        trust_type=("trust_type", "first"),
        trust_name=("trust_name", "first"),
    )
    fy = beds[(beds["quarter_end"] >= "2023-06-30") & (beds["quarter_end"] <= "2024-03-31")]
    ga = fy[fy["sector"] == "General & Acute"].groupby("ods_code")["overnight"].mean()
    allb = fy.groupby(["ods_code", "quarter_end"])["overnight"].sum().groupby("ods_code").mean()
    t = tr.join(ga.rename("occupied_ga_beds")).join(allb.rename("occupied_beds_all"))
    t["elec_kwh_per_bed_day"] = (t["elec_kwh"] / (t["occupied_beds_all"] * 366)).where(
        t["occupied_beds_all"] > 10
    )
    t["elec_kwh_per_m2"] = (t["elec_kwh"] / t["gia_m2"]).where(t["gia_m2"] > 0)
    return t.reset_index()


def run(cfg: dict[str, Any] | None = None) -> dict[str, pd.DataFrame]:
    from ..data import eric as eric_data
    from ..data import nhs_beds

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    site, _ = eric_data.build()
    sector, _ = nhs_beds.build()
    flags = coverage_flags(site, cfg)
    flags.to_csv(OUT_DIR / "coverage_flags.csv", index=False)
    flags[["site_code", "coverage_flag", "energy_ratio", "peak_ratio"]].to_parquet(
        INTERIM_DIR / "site_coverage_flags.parquet", index=False
    )
    model = fit_heat_model()
    model["fits"].to_csv(OUT_DIR / "heat_model_fits.csv", index=False)
    raw_head = heat_pump_headroom(site, model, cfg)
    profiles = raw_head.attrs.pop("profiles", {})
    head = headroom_status(raw_head, flags)
    head.to_csv(OUT_DIR / "heat_pump_headroom_panel.csv", index=False)
    inc = set(head.loc[head["headroom_status"] == "included", "site_code"])
    if profiles:
        pd.concat([v for k, v in profiles.items() if k in inc], ignore_index=True).to_parquet(
            OUT_DIR / "heat_pump_profiles.parquet", index=False
        )
    main = head[head["headroom_status"] == "included"]
    factor = float(main["coincident_increase_factor"].median())
    cfg = {**DEFAULTS, **(cfg or {}), "seasonal_cop": float(main["seasonal_cop"].median())}
    eng = headroom_england(site, factor, cfg)
    eng[
        [
            "site_code",
            "trust_name",
            "site_type",
            "max_demand_kw",
            "capacity_kw",
            "gas_kwh",
            "peak_hp_kw",
            "util_now",
            "util_hp",
            "latitude",
            "longitude",
        ]
    ].to_csv(OUT_DIR / "heat_pump_headroom_england.csv", index=False)
    intensity = bed_day_intensity(site, sector)
    intensity.to_csv(OUT_DIR / "trust_intensity.csv", index=False)
    per_site_path = PROJECT_ROOT / "code" / "reports" / "poc" / "scores_per_site.csv"
    if per_site_path.exists():
        sens = scoring_sensitivity(pd.read_csv(per_site_path), flags)
        sens.to_csv(OUT_DIR / "scoring_by_coverage_flag.csv", index=False)
    bvd = beds_vs_demand(sector)
    bvd.to_csv(OUT_DIR / "beds_vs_demand_quarterly.csv", index=False)
    summary = {
        "coverage_flags": flags["coverage_flag"].value_counts().to_dict(),
        "heat_model": {k: v for k, v in model.items() if k != "fits"},
        "hp_peak_to_mean_median": float(main["hp_peak_to_mean"].median()),
        "coincident_increase_factor_median": factor,
        "seasonal_cop_median": cfg["seasonal_cop"],
        "panel_sites_modelled": len(head),
        "panel_status": head["headroom_status"].value_counts().to_dict(),
        "panel_included": len(main),
        "panel_exceed_capacity": int((main["util_hp"] > 1).sum()),
        "panel_util_hp_median": float(main["util_hp"].median()),
        "panel_elec_increase_median": float(main["elec_increase"].median()),
        "england_sites": len(eng),
        "england_exceed_capacity_share": float((eng["util_hp"] > 1).mean()),
        "beds_vs_demand_trusts": int(bvd["trust"].nunique()) if len(bvd) else 0,
        "beds_vs_demand_quarters": int(bvd["quarter_end"].nunique()) if len(bvd) else 0,
    }
    pd.Series(summary, dtype=object).to_json(
        OUT_DIR / "summary.json", indent=2, default_handler=str
    )
    logger.info("estates summary: %s", summary)
    return {
        "flags": flags,
        "headroom": head,
        "headroom_main": main,
        "england": eng,
        "intensity": intensity,
    }


# --------------------------------------------------------------------------- #
# 4. Scoring sensitivity to the coverage flags
# --------------------------------------------------------------------------- #

SUBSETS = {
    "all sites": None,
    "without flagged supplies": ("partial_supply", "aggregate_supply", "chp_mismatch"),
    "verified only": "ok",
}


def scoring_sensitivity(per_site: pd.DataFrame, flags: pd.DataFrame) -> pd.DataFrame:
    """Pooled PoC scores (mean of per-site metrics, median of skills, as in ``poc.score``)
    on site subsets defined by the coverage flags."""
    f = flags.set_index("site_code")["coverage_flag"]
    per = per_site.assign(coverage_flag=per_site["site_code"].map(f).fillna("no_eric"))
    rows = []
    for label, rule in SUBSETS.items():
        if rule is None:
            sub = per
        elif isinstance(rule, tuple):
            sub = per[~per["coverage_flag"].isin(rule)]
        else:
            sub = per[per["coverage_flag"] == rule]
        g = sub.groupby("model")
        t = pd.DataFrame(
            {
                "sites": g["site_code"].nunique(),
                "nmae": g["nmae"].mean(),
                "crps_q": g["crps_q"].mean(),
                "coverage_80": g["coverage_80"].mean(),
                "crps_skill_median": g["crps_skill_vs_naive"].median(),
            }
        ).reset_index()
        t.insert(0, "subset", label)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# 5. Occupied beds against trust demand over the overlapping quarters
# --------------------------------------------------------------------------- #


def beds_vs_demand(beds: pd.DataFrame) -> pd.DataFrame:
    """Quarterly mean electricity demand of each trust in the panel (processed trust totals)
    against its occupied beds (all sectors, overnight) in the quarters both cover."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "trust" / "wide_imputed.parquet")
    w.index = pd.to_datetime(w.index, utc=True)
    idx = pd.read_parquet(PROCESSED_DIR / "elec" / "trust" / "series_index.parquet")
    trusts = pd.read_parquet(INTERIM_DIR / "nhs_activity_trusts.parquet")
    ods = idx.merge(trusts[["organisation_name", "ods_code"]], on="organisation_name", how="left")
    ods = ods.set_index("series_id")["ods_code"]
    loc = w.index.tz_convert(TZ).tz_localize(None)
    quarter_end = loc.to_period("Q").to_timestamp(how="end").normalize()
    qcount = w.notna().groupby(quarter_end).mean()
    qmean = w.groupby(quarter_end).mean().where(qcount >= 0.8) * 2  # kW
    b = beds.groupby(["ods_code", "quarter_end"])["overnight"].sum()
    rows = []
    for t in w.columns:
        code = ods.get(t)
        if code is None or code not in b.index.get_level_values(0):
            continue
        bt = b.loc[code]
        d = pd.concat([qmean[t].rename("demand_kw"), bt.rename("occupied_beds")], axis=1).dropna()
        for q, r in d.iterrows():
            rows.append({"trust": t, "ods_code": code, "quarter_end": q, **r.to_dict()})
    return pd.DataFrame(rows)
