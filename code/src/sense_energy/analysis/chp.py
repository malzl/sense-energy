"""What sets sites with fossil-fuel CHP apart: their meters record grid import (demand minus
CHP output), so the series carries CHP operation as a second, hidden driver.

Per electricity site: size, load factor (mean / 99th percentile), day-to-day variability of
weekday demand, the share of weekdays whose import is more than 30 % above the median of the
surrounding 21 weekdays (CHP trips, maintenance, turn-down), forecast error (Chronos-2 + IFS ENS,
``code/reports/poc``), meter-to-ERIC ratio (``code/reports/estates``) and cooling response
(``code/reports/cooling``). Groups: on-site CHP, acute hospitals without CHP, all other sites.

Outputs (``code/reports/chp/``): ``site_profile.csv``, ``group_summary.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, LOCAL_TZ, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from .cooling import daily_demand

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "chp"
REPORTS = PROJECT_ROOT / "code" / "reports"
JUMP_RATIO, JUMP_WINDOW = 1.3, 21
FY_2024_25 = ("2024-04-01", "2025-04-01")  # ERIC 2024/25 reporting year, local calendar
#: Capacity factors outside this range mark a unit mismatch in the return (MWh entered as kWh,
#: or a unit that ran for part of the year).
PLAUSIBLE_CAPACITY_FACTOR = (0.05, 1.05)
GROUP_ORDER = ("on-site CHP", "acute, no CHP", "other sites")
ACUTE_USES = (
    "General acute hospital",
    "Specialist hospital (acute only)",
    "Mixed service hospital",
)


def site_profile() -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").set_index("site_code").reindex(w.columns)
    chp = sites["fossil_fuel_led_chp_units_operated_on_site"].fillna(0) > 0
    acute = sites["site_use_type"].isin(ACUTE_USES)
    kw = daily_demand()
    wk = kw[kw.index.dayofweek < 5]
    ratio = wk / wk.rolling(JUMP_WINDOW, center=True, min_periods=JUMP_WINDOW // 2).median()

    out = pd.DataFrame(
        {
            "group": np.where(chp, GROUP_ORDER[0], np.where(acute, GROUP_ORDER[1], GROUP_ORDER[2])),
            "site_use_type": sites["site_use_type"],
            "mean_kw": w.mean(),
            "load_factor": w.mean() / w.quantile(0.99),
            "daily_cv": wk.std() / wk.mean(),
            "jump_day_share": (ratio > JUMP_RATIO).sum() / ratio.notna().sum(),
        }
    )
    sc = pd.read_csv(REPORTS / "poc" / "scores_per_site.csv")
    sc = sc[sc["model"] == "chronos2_ifs"].set_index("site_code")
    out["nmae"] = sc["nmae"]
    out["crps_skill"] = sc["crps_skill_vs_naive"]
    flags = pd.read_csv(REPORTS / "estates" / "coverage_flags.csv").set_index("site_code")
    out["eric_ratio"] = flags["energy_ratio"].where(flags["meter_coverage_fy"] >= 0.8)
    cool = pd.read_csv(REPORTS / "cooling" / "cooling_response_site.csv").set_index("site_code")
    out["cooling_pct_per_cdd"] = cool["cdd_pct"]
    return out.rename_axis("site_code").reset_index()


def group_summary(p: pd.DataFrame) -> pd.DataFrame:
    g = p.groupby("group")
    out = g[
        [
            "mean_kw",
            "load_factor",
            "daily_cv",
            "jump_day_share",
            "nmae",
            "crps_skill",
            "cooling_pct_per_cdd",
        ]
    ].median()
    out["sites"] = g.size()
    out["sites_scored"] = g["nmae"].count()
    out["demand_share"] = g["mean_kw"].sum() / p["mean_kw"].sum()
    out["eric_ratio_q10"] = g["eric_ratio"].quantile(0.1)
    out["eric_ratio_q90"] = g["eric_ratio"].quantile(0.9)
    return out.reindex(GROUP_ORDER).rename_axis("group").reset_index()


def generation_2024_25() -> pd.DataFrame:
    """CHP sites in the ERIC 2024/25 return: grid import as metered here over the reporting year
    and as returned, the CHP's electrical output, exports and purchased CHP electricity, hence
    the implied site demand and the CHP share of it, and the unit's capacity factor."""
    path = INTERIM_DIR / "eric_2024_25_site.parquet"
    if not path.exists():
        return pd.DataFrame()
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")  # kWh per half-hour
    local = pd.to_datetime(w.index, utc=True).tz_convert(LOCAL_TZ)
    start, end = (pd.Timestamp(d, tz=LOCAL_TZ) for d in FY_2024_25)
    fy = w[(local >= start) & (local < end)]
    metered = pd.DataFrame(
        {"meter_import_kwh": fy.sum(min_count=1), "meter_coverage": fy.notna().mean()}
    )
    eric = pd.read_parquet(path).drop_duplicates("site_code").set_index("site_code")
    sites = pd.read_parquet(INTERIM_DIR / "sites.parquet").drop_duplicates("site_code")
    sites = sites.set_index("site_code")
    chp = sites.index[sites["fossil_fuel_led_chp_units_operated_on_site"].fillna(0) > 0]
    cols = [
        "site_name",
        "elec_kwh",
        "chp_elec_output_kwh",
        "chp_thermal_output_kwh",
        "chp_gas_input_kwh",
        "chp_elec_purchased_kwh",
        "exported_elec_kwh",
        "chp_elec_kw",
        "chp_thermal_kw",
        "chp_units",
        "gia_m2",
    ]
    out = metered.reindex(chp).join(eric.reindex(chp)[cols])
    export = out["exported_elec_kwh"].fillna(0)
    out["implied_demand_kwh"] = (
        out["elec_kwh"].fillna(0)
        + out["chp_elec_output_kwh"]
        + out["chp_elec_purchased_kwh"].fillna(0)
        - export
    )
    out["chp_share"] = (out["chp_elec_output_kwh"] - export) / out["implied_demand_kwh"]
    out["capacity_factor"] = out["chp_elec_output_kwh"] / (out["chp_elec_kw"] * 8760)
    lo, hi = PLAUSIBLE_CAPACITY_FACTOR
    out["plausible"] = out["capacity_factor"].between(lo, hi)
    out["meter_to_eric_import"] = out["meter_import_kwh"] / out["elec_kwh"]
    out["implied_mean_kw"] = out["implied_demand_kwh"] / 8760
    out["eric_import_mean_kw"] = out["elec_kwh"] / 8760
    out["metered_mean_kw"] = out["meter_import_kwh"] / (out["meter_coverage"] * 8760)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["implied_kwh_m2"] = (out["implied_demand_kwh"] / out["gia_m2"]).where(out["gia_m2"] > 0)
        out["heat_to_power"] = out["chp_thermal_output_kwh"] / out["chp_elec_output_kwh"]
    return out.rename_axis("site_code").reset_index()


def run() -> dict[str, pd.DataFrame]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = site_profile()
    s = group_summary(p)
    p.to_csv(OUT_DIR / "site_profile.csv", index=False)
    s.to_csv(OUT_DIR / "group_summary.csv", index=False)
    g = generation_2024_25()
    if not g.empty:
        g.to_csv(OUT_DIR / "generation_2024_25.csv", index=False)
        ok = g[g["plausible"] & g["chp_elec_output_kwh"].gt(0)]
        logger.info(
            "CHP generation 2024/25: %d sites with output, %d plausible; CHP covers a median %.0f %% "
            "of implied demand (capacity factor median %.2f)",
            int(g["chp_elec_output_kwh"].gt(0).sum()),
            len(ok),
            100 * ok["chp_share"].median(),
            ok["capacity_factor"].median(),
        )
    logger.info("CHP profile: %s", s.set_index("group")["sites"].to_dict())
    return {"profile": p, "summary": s, "generation": g}
