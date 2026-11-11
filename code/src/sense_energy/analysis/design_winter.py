"""Design winters: the estate's demand under every winter since 1960.

The headroom work asked whether heat pumps fit under each site's connection in the weather of one
year, 2023/24. A connection is sized for the cold winter, not the average one. Here the reference
year's half-hourly demand is re-weathered with each of 66 weather years (April 1960 to March
2026), day for day:

* today's electricity moves with the site's own hourly cold and heat response
  (``hourly_response``): demand x exp(beta_cold[s, h] x change in cold dose + beta_heat[s, h] x
  change in heat dose), the change being that year's daily mean temperature against the
  reference year's on the same calendar day;
* the heat-pump load is rebuilt from that year's hourly temperature with the headroom model:
  space heat in proportion to heating degree-hours (calibrated on the reference year's gas), a
  weather-independent base, and the air-source COP at the hour's temperature.

Per site and weather year this gives the annual electricity, the peak half-hour today and with
heat pumps, the hours above the connection's capacity and the storage that would hold the site
under it (power: the largest excess; energy: the largest run of excess net of recharging below
capacity). Design statistics are taken over the last 30 weather years, which stand for the
current climate (1-in-20: their 95th percentile), and the full record names the worst winters.

The cold response is extrapolated beyond the temperatures of the demand record on the coldest
days of the old winters; the heat model's proportionality to degree-hours is the same assumption
at a larger range. Both are flagged in the outputs (``extrapolated_days``).

Outputs in ``code/reports/design_winter/``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..data import era5_timeseries as ets
from ..logging_utils import get_logger
from . import estate_counterfactuals as ec
from . import estates

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "design_winter"
REF_YEAR = 2023  # the reference demand year is the headroom year, April 2023 to March 2024
WEATHER_YEARS = tuple(range(1960, 2026))  # financial years by their April
RECENT = 30  # weather years that stand for the current climate
DESIGN_Q = 0.95  # 1-in-20
ROUND_TRIP = 0.85
#: Storage classes by the hours of the connection's capacity the store must hold.
STORE_CLASSES = ((2.0, "battery scale (under 2 h)"), (24.0, "thermal store scale (2 to 24 h)"))
STORE_BEYOND = "beyond storage (over 24 h)"


def year_label(y: int) -> str:
    return f"{y}/{str(y + 1)[2:]}"


def reference() -> pd.DataFrame:
    """Imputed half-hourly demand of the reference year in kW (UTC index), sites as columns."""
    w = estates._fy_slice(estates._wide("elec", "imputed")) * 2.0
    return w


def storage_need(load: np.ndarray, cap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For rows of half-hourly load (kW) and a capacity per row: the largest excess (kW), the
    store (kWh) that covers every run of excess when it recharges below capacity, and the hours
    above capacity."""
    s = np.zeros(load.shape[0])
    e = np.zeros(load.shape[0])
    for t in range(load.shape[1]):
        d = load[:, t] - cap
        s = np.maximum(0.0, s + 0.5 * np.where(d > 0, d, ROUND_TRIP * d))
        e = np.maximum(e, s)
    over = load > cap[:, None]
    return np.clip(load - cap[:, None], 0, None).max(axis=1), e, over.sum(axis=1) / 2.0


def store_class(hours: float) -> str:
    if not np.isfinite(hours) or hours <= 0:
        return "within capacity"
    for limit, name in STORE_CLASSES:
        if hours <= limit:
            return name
    return STORE_BEYOND


def reweather() -> tuple[pd.DataFrame, dict]:
    """Per site and weather year: electricity, peaks and, for the headroom sites, the heat-pump
    case with capacity and storage."""
    prof = ec.smoothed_profiles()
    ref = reference()
    sites = [s for s in prof["heat"].index if s in ref.columns and ref[s].notna().mean() >= 0.8]
    ref = ref[sites]
    daily = pd.read_parquet(INTERIM_DIR / "weather_daily_long.parquet")
    daily = daily[daily["site_code"].isin(sites)]
    tday = daily.pivot(index="date", columns="site_code", values="t_mean")
    sites = [s for s in sites if s in tday.columns]
    ref, tday = ref[sites], tday[sites]

    head = pd.read_csv(
        PROJECT_ROOT / "code" / "reports" / "estates" / "heat_pump_headroom_panel.csv"
    )
    head = head[(head["headroom_status"] == "included") & head["site_code"].isin(sites)]
    head = head.set_index("site_code")
    hp_sites = list(head.index)
    model = json.loads(
        (PROJECT_ROOT / "code" / "reports" / "estates" / "summary.json").read_text()
    )["heat_model"]
    cfg = estates.DEFAULTS
    hourly = ets.hourly_for_sites(hp_sites)
    hourly_ns = hourly.index.as_unit("ns").asi8

    local = ref.index.tz_convert(estates.TZ)
    date_ref = local.normalize().tz_localize(None)
    hour = local.hour.to_numpy()
    kw = ref.to_numpy()
    kw_fill = np.where(np.isfinite(kw), kw, np.nanmean(kw, axis=0))
    bc = prof["cold"].loc[sites].to_numpy()[:, hour].T  # half-hours x sites
    bh = prof["heat"].loc[sites].to_numpy()[:, hour].T
    t_obs = tday.reindex(date_ref).to_numpy()
    c0, h0 = ec.doses(t_obs)
    in_sample_min = tday.loc["2022-12-07":].min().to_numpy()  # coldest day of the demand record
    hp_col = [sites.index(s) for s in hp_sites]
    heat_total = head["gas_kwh"].to_numpy() * cfg["boiler_efficiency"]
    cap = head["capacity_kw"].to_numpy()
    n = len(ref)

    def temps(k: int) -> np.ndarray:
        idx = (ref.index + pd.DateOffset(years=k)).as_unit("ns").asi8
        return np.column_stack([np.interp(idx, hourly_ns, hourly[s].to_numpy()) for s in hp_sites])

    t_ref_hh = temps(0)
    hdh_ref_sum = np.clip(model["base_temp_c"] - t_ref_hh, 0, None).sum(axis=0)

    rows = []
    for y in WEATHER_YEARS:
        k = y - REF_YEAR
        date_y = date_ref + pd.DateOffset(years=k)
        t_y = tday.reindex(date_y).to_numpy()
        if np.isnan(t_y).mean() > 0.01:
            continue
        c1, h1 = ec.doses(t_y)
        kw_y = kw_fill * np.exp(bc * (c1 - c0) + bh * (h1 - h0))
        energy = kw_y.sum(axis=0) / kw_fill.sum(axis=0)
        peak = np.where(np.isfinite(kw), kw_y, np.nan)
        peak_now = np.nanmax(peak, axis=0)
        t_hh = temps(k)
        hdh = np.clip(model["base_temp_c"] - t_hh, 0, None)
        heat = model["base_share"] * heat_total / n + (1 - model["base_share"]) * heat_total * (
            hdh / hdh_ref_sum
        )  # kWh per half-hour
        hp_kw = 2.0 * heat / estates.cop_ashp(t_hh, cfg["flow_temp_c"])
        new = kw_y[:, hp_col] + hp_kw
        p_need, e_need, hours_over = storage_need(new.T, cap)
        winter = np.isin(date_y.month, (12, 1, 2))
        for j, s in enumerate(sites):
            row = {
                "site_code": s,
                "weather_year": y,
                "energy_ratio": energy[j],
                "peak_now_kw": peak_now[j],
                "t_winter_mean": float(np.nanmean(t_y[winter, j])),
                "t_coldest_day": float(np.nanmin(t_y[:, j])),
                "extrapolated_days": int((t_y[:, j] < in_sample_min[j]).sum() // 48),
            }
            if s in head.index:
                i = hp_sites.index(s)
                row.update(
                    {
                        "capacity_kw": cap[i],
                        "hp_kwh": float(hp_kw[:, i].sum() * 0.5),
                        "energy_hp_ratio": float(new[:, i].sum() / kw_fill[:, hp_col[i]].sum()),
                        "peak_hp_kw": float(new[:, i].max()),
                        "util_now": peak_now[j] / cap[i],
                        "util_hp": float(new[:, i].max() / cap[i]),
                        "hours_over": float(hours_over[i]),
                        "store_kw": float(p_need[i]),
                        "store_kwh": float(e_need[i]),
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out["store_hours"] = out["store_kwh"] / out["capacity_kw"]
    chk = out[out["weather_year"] == REF_YEAR].set_index("site_code").loc[hp_sites]
    check = {
        "reference_year_peak_hp_max_rel_diff": float(
            (chk["peak_hp_kw"] / head["peak_hp_kw"] - 1).abs().max()
        ),
        "reference_sites_above_capacity": int((chk["util_hp"] > 1).sum()),
        "in_sample_coldest_day_c_median_site": float(np.median(in_sample_min)),
    }
    return out, check


def design_table(sy: pd.DataFrame) -> pd.DataFrame:
    """Per headroom site: utilisation with heat pumps in the reference year, the median recent
    year, the 1-in-20 winter and the worst winter on record, with the storage that holds it."""
    hp = sy.dropna(subset=["capacity_kw"])
    last = max(WEATHER_YEARS)
    recent = hp[hp["weather_year"] > last - RECENT]
    rows = []
    for s, g in hp.groupby("site_code"):
        r = recent[recent["site_code"] == s]
        worst = g.loc[g["util_hp"].idxmax()]
        ref = g[g["weather_year"] == REF_YEAR].iloc[0]
        rows.append(
            {
                "site_code": s,
                "capacity_kw": ref["capacity_kw"],
                "util_now_ref": ref["util_now"],
                "util_now_design": r["util_now"].quantile(DESIGN_Q),
                "util_hp_ref": ref["util_hp"],
                "util_hp_median": r["util_hp"].median(),
                "util_hp_design": r["util_hp"].quantile(DESIGN_Q),
                "util_hp_worst": worst["util_hp"],
                "worst_year": year_label(int(worst["weather_year"])),
                "store_kw_ref": ref["store_kw"],
                "store_kwh_ref": ref["store_kwh"],
                "store_hours_ref": ref["store_hours"],
                "store_kw_design": r["store_kw"].quantile(DESIGN_Q),
                "store_kwh_design": r["store_kwh"].quantile(DESIGN_Q),
                "store_hours_design": r["store_hours"].quantile(DESIGN_Q),
                "hours_over_design": r["hours_over"].quantile(DESIGN_Q),
                "hp_kwh_sd_pct": 100 * r["hp_kwh"].std() / r["hp_kwh"].mean(),
                "energy_hp_sd_pct": 100 * r["energy_hp_ratio"].std() / r["energy_hp_ratio"].mean(),
            }
        )
    d = pd.DataFrame(rows)
    d["store_class_ref"] = d["store_hours_ref"].map(store_class)
    d["store_class_design"] = d["store_hours_design"].map(store_class)
    return d.sort_values("util_hp_design", ascending=False)


def year_table(sy: pd.DataFrame) -> pd.DataFrame:
    """Per weather year: the panel's winter, today's demand and the heat-pump case."""
    hp = sy.dropna(subset=["capacity_kw"])
    g = sy.groupby("weather_year")
    out = pd.DataFrame(
        {
            "t_winter_mean": g["t_winter_mean"].mean(),
            "t_coldest_day_median": g["t_coldest_day"].median(),
            "energy_change_median_pct": 100 * (g["energy_ratio"].median() - 1),
            "sites": g.size(),
        }
    )
    ref_peak = sy[sy["weather_year"] == REF_YEAR].set_index("site_code")["peak_now_kw"]
    sy = sy.assign(peak_ratio=sy["peak_now_kw"] / sy["site_code"].map(ref_peak))
    out["peak_now_change_median_pct"] = 100 * (
        sy.groupby("weather_year")["peak_ratio"].median() - 1
    )
    h = hp.groupby("weather_year")
    out["hp_sites"] = h.size()
    out["hp_sites_above_capacity"] = h["util_hp"].apply(lambda s: int((s > 1).sum()))
    out["util_hp_median"] = h["util_hp"].median()
    out["hp_energy_vs_ref_pct"] = 100 * (
        h["hp_kwh"].sum() / hp[hp["weather_year"] == REF_YEAR]["hp_kwh"].sum() - 1
    )
    out = out.reset_index()
    out["label"] = out["weather_year"].map(year_label)
    return out


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sy, check = reweather()
    sy.to_parquet(OUT_DIR / "site_years.parquet", index=False)
    years = year_table(sy)
    years.to_csv(OUT_DIR / "years.csv", index=False)
    design = design_table(sy)
    design.to_csv(OUT_DIR / "design_sites.csv", index=False)

    last = max(WEATHER_YEARS)
    recent = years[years["weather_year"] > last - RECENT]
    all_sites = sy[sy["weather_year"] > last - RECENT]
    e_sd = all_sites.groupby("site_code")["energy_ratio"].std() * 100
    ref_row = years[years["weather_year"] == REF_YEAR].iloc[0]
    worst = years.sort_values("hp_sites_above_capacity", ascending=False).head(5)
    cold_rank = years.sort_values("t_winter_mean")
    summary = {
        "check": check,
        "sites": int(sy["site_code"].nunique()),
        "hp_sites": int(len(design)),
        "weather_years": [year_label(int(years["weather_year"].min())), year_label(last)],
        "reference_year": {
            "label": year_label(REF_YEAR),
            "winter_mean_c": round(float(ref_row["t_winter_mean"]), 2),
            "winter_rank_coldest_of_recent": int(
                (recent["t_winter_mean"] < ref_row["t_winter_mean"]).sum() + 1
            ),
            "sites_above_capacity": int(ref_row["hp_sites_above_capacity"]),
        },
        "heat_pump_sites_above_capacity": {
            "reference_year": int((design["util_hp_ref"] > 1).sum()),
            "median_recent_year": int((design["util_hp_median"] > 1).sum()),
            "one_in_twenty": int((design["util_hp_design"] > 1).sum()),
            "worst_on_record": int((design["util_hp_worst"] > 1).sum()),
            "recent_years_min": int(recent["hp_sites_above_capacity"].min()),
            "recent_years_max": int(recent["hp_sites_above_capacity"].max()),
        },
        "today_sites_above_capacity_one_in_twenty": int((design["util_now_design"] > 1).sum()),
        "util_hp_median": {
            "reference_year": round(float(design["util_hp_ref"].median()), 3),
            "one_in_twenty": round(float(design["util_hp_design"].median()), 3),
            "worst_on_record": round(float(design["util_hp_worst"].median()), 3),
        },
        "worst_years_by_sites_above_capacity": worst[
            ["label", "hp_sites_above_capacity", "t_winter_mean", "t_coldest_day_median"]
        ]
        .round(2)
        .to_dict(orient="records"),
        "coldest_winters": cold_rank[["label", "t_winter_mean"]]
        .head(5)
        .round(2)
        .to_dict(orient="records"),
        "budget_risk_recent": {
            "today_energy_sd_pct_median_site": round(float(e_sd.median()), 2),
            "today_energy_range_pct": [
                round(float(recent["energy_change_median_pct"].min()), 2),
                round(float(recent["energy_change_median_pct"].max()), 2),
            ],
            "heat_pump_electricity_range_pct": [
                round(float(recent["hp_energy_vs_ref_pct"].min()), 1),
                round(float(recent["hp_energy_vs_ref_pct"].max()), 1),
            ],
            "with_heat_pumps_energy_sd_pct_median_site": round(
                float(design["energy_hp_sd_pct"].median()), 2
            ),
        },
        "storage_one_in_twenty": design.loc[design["util_hp_design"] > 1, "store_class_design"]
        .value_counts()
        .to_dict(),
        "storage_reference_year": design.loc[design["util_hp_ref"] > 1, "store_class_ref"]
        .value_counts()
        .to_dict(),
        "today_peak_change_pct": {
            "one_in_twenty_median_site": round(
                100
                * float(
                    (
                        all_sites.groupby("site_code")["peak_now_kw"].quantile(DESIGN_Q)
                        / sy[sy["weather_year"] == REF_YEAR].set_index("site_code")["peak_now_kw"]
                        - 1
                    ).median()
                ),
                2,
            )
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("design winter: %s", json.dumps(summary["heat_pump_sites_above_capacity"]))
    return {"summary": summary}
