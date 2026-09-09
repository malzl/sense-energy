"""NESO covariates on the demand grid, and static FES attributes per site.

Time series (``interim/neso_covariates_halfhourly.parquet``), one row per
30-minute UTC period:

* ``*_actual`` - outturn: national and England/Wales demand, embedded wind and
  solar generation and capacity, pumped-storage pumping, net interconnector
  import, generation mix and carbon intensity. Ex-post; legal only as lags.
* ``*_da`` - known by the day-ahead cutoff: NESO's half-hourly day-ahead
  national demand forecast (published ~09:45 on D-1), the embedded wind/solar
  forecast issued last before the cutoff, and daily cardinal-point demand
  forecasts (peak/trough) at 1, 2, 7 and 14 days ahead.

Static (``interim/neso_site_covariates.parquet``), one row per site: the FES
regional peak demand and distributed-generation capacity of the site's GSP.

Attribution: "Contains NESO open data".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import GEO_DIR, INTERIM_DIR
from ..logging_utils import get_logger
from .neso import TABLES_DIR, _parse_settlement_date, _settlement_to_utc

logger = get_logger(__name__)

CARDINAL_FILES = {
    1: "demand_forecast_day_ahead_historic.csv",
    2: "demand_forecast_2day_ahead_historic.csv",
    7: "demand_forecast_7day_ahead_historic.csv",
}
CARDINAL_MULTI = "demand_forecast_2_14day_historic.csv"


def _cutoff(target_utc: pd.Series, hour: int) -> pd.Series:
    """The last moment a value may be issued to count as day-ahead-known for ``target``."""
    return target_utc.dt.floor("D") - pd.Timedelta(days=1) + pd.Timedelta(hours=hour)


# --------------------------------------------------------------------------- #
# Day-ahead known
# --------------------------------------------------------------------------- #


def day_ahead_hh_forecast() -> pd.DataFrame:
    """NESO's half-hourly day-ahead ND forecast with its publish time, on the UTC grid.

    Aligned by settlement date and period rather than the file's ``Datetime``
    column, whose clock convention is not documented.
    """
    p = pd.read_csv(TABLES_DIR / "demand_forecast_day_ahead_performance.csv", encoding="utf-8-sig")
    p.columns = [c.strip().lower() for c in p.columns]
    p["datetime"] = _settlement_to_utc(_parse_settlement_date(p["date"]), p["settlement_period"])
    p["published_at"] = pd.to_datetime(p["publish_datetime"], utc=True)
    out = (
        p.dropna(subset=["datetime"])
        .sort_values("published_at")
        .drop_duplicates("datetime", keep="last")
    )
    out = out.rename(
        columns={
            "demand_forecast": "nd_forecast_da_mw",
            "demand_outturn": "nd_outturn_perf_mw",
            "triad_avoidance_estimate": "triad_avoidance_estimate_mw",
            "published_at": "nd_forecast_da_published_at",
        }
    )
    return out[
        [
            "datetime",
            "nd_forecast_da_mw",
            "nd_forecast_da_published_at",
            "triad_avoidance_estimate_mw",
        ]
    ]


def cardinal_point_features(horizons: list[int], cutoff_hour: int) -> pd.DataFrame:
    """Daily forecast peak/trough per target date and horizon, latest issue before the cutoff."""
    frames = []
    for h in horizons:
        if h in CARDINAL_FILES:
            d = pd.read_csv(TABLES_DIR / CARDINAL_FILES[h])
        else:
            d = pd.read_csv(TABLES_DIR / CARDINAL_MULTI)
            d = d[d["DAYSAHEAD"] == h]
        if d.empty:
            logger.warning("no cardinal-point rows for %d days ahead", h)
            continue
        d["target_date"] = pd.to_datetime(d["TARGETDATE"], utc=True)
        d["issued_at"] = pd.to_datetime(d["FORECAST_TIMESTAMP"], utc=True)
        d = d[d["issued_at"] <= _cutoff(d["target_date"], cutoff_hour)]
        latest = d.groupby("target_date")["issued_at"].transform("max")
        d = d[d["issued_at"] == latest]
        g = d.groupby("target_date")
        peak = g.apply(
            lambda x: (
                x.loc[x["CP_TYPE"] == "P", "FORECASTDEMAND"].max()
                if (x["CP_TYPE"] == "P").any()
                else x["FORECASTDEMAND"].max()
            ),
            include_groups=False,
        )
        trough = g.apply(
            lambda x: (
                x.loc[x["CP_TYPE"] == "T", "FORECASTDEMAND"].min()
                if (x["CP_TYPE"] == "T").any()
                else x["FORECASTDEMAND"].min()
            ),
            include_groups=False,
        )
        f = pd.DataFrame(
            {f"nd_forecast_{h}d_peak_da_mw": peak, f"nd_forecast_{h}d_trough_da_mw": trough}
        )
        f[f"nd_forecast_{h}d_issued_at"] = g["issued_at"].max()
        frames.append(f)
    out = pd.concat(frames, axis=1)
    out.index.name = "target_date"
    return out.reset_index()


def embedded_forecast_at_cutoff(cutoff_hour: int) -> pd.DataFrame:
    """Per target period, the embedded wind/solar forecast issued last before the cutoff."""
    a = pd.read_parquet(
        INTERIM_DIR / "embedded_forecast_archive.parquet",
        columns=[
            "issued_at",
            "datetime",
            "embedded_wind_forecast",
            "embedded_solar_forecast",
            "embedded_wind_capacity",
            "embedded_solar_capacity",
        ],
    )
    a = a[a["issued_at"] <= _cutoff(a["datetime"], cutoff_hour)]
    a = a.sort_values("issued_at").drop_duplicates("datetime", keep="last")
    a = a.rename(
        columns={
            "embedded_wind_forecast": "embedded_wind_forecast_da_mw",
            "embedded_solar_forecast": "embedded_solar_forecast_da_mw",
            "embedded_wind_capacity": "embedded_wind_capacity_fc_mw",
            "embedded_solar_capacity": "embedded_solar_capacity_fc_mw",
            "issued_at": "embedded_forecast_da_issued_at",
        }
    )
    return a[
        [
            "datetime",
            "embedded_wind_forecast_da_mw",
            "embedded_solar_forecast_da_mw",
            "embedded_forecast_da_issued_at",
        ]
    ]


# --------------------------------------------------------------------------- #
# Actuals
# --------------------------------------------------------------------------- #


def demand_actuals() -> pd.DataFrame:
    nd = pd.read_parquet(INTERIM_DIR / "national_demand_halfhourly.parquet")
    flows = [c for c in nd.columns if c.endswith("_flow")]
    out = pd.DataFrame(
        {
            "datetime": nd["datetime"],
            "nd_actual_mw": nd["nd"],
            "tsd_actual_mw": nd["tsd"],
            "england_wales_demand_actual_mw": nd["england_wales_demand"],
            "embedded_wind_actual_mw": nd["embedded_wind_generation"],
            "embedded_solar_actual_mw": nd["embedded_solar_generation"],
            "embedded_wind_capacity_mw": nd["embedded_wind_capacity"],
            "embedded_solar_capacity_mw": nd["embedded_solar_capacity"],
            "pump_storage_pumping_actual_mw": nd["pump_storage_pumping"],
            "interconnector_net_import_actual_mw": nd[flows].sum(axis=1, min_count=1),
        }
    )
    out["embedded_wind_cf_actual"] = out["embedded_wind_actual_mw"] / out[
        "embedded_wind_capacity_mw"
    ].replace(0, np.nan)
    out["embedded_solar_cf_actual"] = out["embedded_solar_actual_mw"] / out[
        "embedded_solar_capacity_mw"
    ].replace(0, np.nan)
    return out


def generation_mix(config: dict[str, Any]) -> pd.DataFrame:
    """Half-hourly GB generation mix and carbon intensity. DATETIME is taken as UTC."""
    g = pd.read_csv(TABLES_DIR / "generation_mix_halfhourly.csv")
    g["datetime"] = pd.to_datetime(g["DATETIME"], utc=True)
    cols = {}
    for c in config.get("generation_mix_mw", []):
        if c in g.columns:
            cols[f"genmix_{c.lower()}_actual_mw"] = g[c]
    for c in config.get("generation_mix_pct", []):
        if c in g.columns:
            cols[f"genmix_{c.lower().replace('_perc', '')}_actual_pct"] = g[c]
    if "CARBON_INTENSITY" in g.columns:
        cols["carbon_intensity_actual_gco2_kwh"] = g["CARBON_INTENSITY"]
    out = pd.DataFrame({"datetime": g["datetime"], **cols})
    return out.drop_duplicates("datetime")


# --------------------------------------------------------------------------- #
# Static site attributes from FES
# --------------------------------------------------------------------------- #

DG_TECH_GROUPS = {
    "solar": ("solar",),
    "wind": ("wind",),
    "battery": ("battery", "storage"),
}


def site_fes_attributes(config: dict[str, Any]) -> pd.DataFrame:
    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")[
        ["site_code", "site_name", "organisation_name", "gsp_region", "gsp_group", "gsp_group_neso"]
    ]
    fes_year, base, target = (
        int(config["fes_year"]),
        int(config["fes_base_year"]),
        int(config["fes_target_year"]),
    )
    rd = pd.read_parquet(INTERIM_DIR / "fes_regional_demand.parquet")
    rd = rd[rd["fes_year"] == fes_year]
    dg = pd.read_parquet(INTERIM_DIR / "fes_regional_distributed_generation.parquet")
    dg = dg[(dg["fes_year"] == fes_year) & (dg["year"] == base)]

    # demand summed over demand types per (scenario, gsp, year) - the file splits total demand by type
    peak = rd.groupby(["scenario", "gsp_id", "year"])["demand_peak_mw"].sum()
    dgcap = dg.groupby(["scenario", "gsp_id", "tech"])["capacity_mw"].sum()

    rows = []
    for _, s in sites.iterrows():
        parts = (
            [p.strip("[+]") for p in str(s["gsp_region"]).split("|")]
            if isinstance(s["gsp_region"], str)
            else []
        )
        row = {
            "site_code": s["site_code"],
            "gsp_region": s["gsp_region"],
            "fes_gsp_n_parts": len(parts),
        }
        # base-year peak: same across scenarios; take EE
        row[f"fes_gsp_peak_{base}_mw"] = (
            sum(peak.get(("EE", p, base), np.nan) for p in parts) if parts else np.nan
        )
        for code, name in config["fes_pathways"].items():
            row[f"fes_gsp_peak_{target}_{name}_mw"] = (
                sum(peak.get((code, p, target), np.nan) for p in parts) if parts else np.nan
            )
        for group, keys in DG_TECH_GROUPS.items():
            total = 0.0
            for p in parts:
                sub = dgcap.loc["EE", p] if ("EE", p) in dgcap.index.droplevel(2).unique() else None
                if sub is not None:
                    total += float(
                        sub[[any(k in str(t).lower() for k in keys) for t in sub.index]].sum()
                    )
            row[f"fes_gsp_dg_{group}_{base}_mw"] = total if parts else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    logger.info(
        "FES site attributes for %d sites (%d with a GSP)",
        len(out),
        int(out["fes_gsp_n_parts"].gt(0).sum()),
    )
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build(config: dict[str, Any]) -> dict[str, Any]:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    grid = pd.DataFrame(
        {
            "datetime": pd.date_range(
                config["grid_start"], config["grid_end"], freq="30min", tz="UTC"
            )
        }
    )
    cutoff = int(config.get("day_ahead_cutoff_hour_utc", 16))

    df = grid.merge(demand_actuals(), on="datetime", how="left")
    df = df.merge(generation_mix(config), on="datetime", how="left")
    df = df.merge(day_ahead_hh_forecast(), on="datetime", how="left")
    df = df.merge(embedded_forecast_at_cutoff(cutoff), on="datetime", how="left")
    cp = cardinal_point_features(list(config.get("cardinal_horizons_days", [1, 2, 7, 14])), cutoff)
    df["target_date"] = df["datetime"].dt.floor("D")
    df = df.merge(cp, on="target_date", how="left").drop(columns="target_date")
    df["nd_forecast_da_error_actual_mw"] = df["nd_actual_mw"] - df["nd_forecast_da_mw"]
    df["settlement_date_local"] = df["datetime"].dt.tz_convert("Europe/London").dt.date
    df["day_ahead_cutoff_hour_utc"] = cutoff

    out = INTERIM_DIR / "neso_covariates_halfhourly.parquet"
    df.to_parquet(out, index=False, compression="zstd")
    present = df.drop(columns=["datetime"]).notna().mean().sort_values()
    logger.info(
        "Wrote %s (%s rows, %d columns); least complete: %s",
        out.name,
        f"{len(df):,}",
        df.shape[1],
        present.head(4).round(3).to_dict(),
    )

    site = site_fes_attributes(config)
    out2 = INTERIM_DIR / "neso_site_covariates.parquet"
    site.to_parquet(out2, index=False)
    return {
        "halfhourly": out,
        "site": out2,
        "columns": df.columns.tolist(),
        "completeness": present.round(4).to_dict(),
    }
