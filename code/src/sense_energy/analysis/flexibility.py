"""What flexibility is worth at each site, in pounds and kilowatts.

Four questions on the half-hourly panel, all answered from the meter and public prices:

1. **Bill anatomy.** What a site's load shape costs at the wholesale index: the price it pays per
   kWh against the time-average price (the shape premium), and how much of its energy falls in
   the working-day red band (16:00-19:00), where distribution charges sit.
2. **Battery arbitrage.** The value of one kW of two-hour battery cycled against each price case
   with perfect knowledge of the day's prices: a linear programme per day. That is exact for
   Agile, which is published the day before, and an upper bound for the market index, which is
   known only after delivery. The value belongs to the price series, not to the site; the site only
   enters through the no-export limit, which is checked.
3. **Rooftop solar against the real load.** Generation per kWp from ERA5 surface solar radiation
   at the site (performance ratio 0.8 on the horizontal irradiance), sized relative to the site's
   mean demand: self-consumption, the share of demand covered, exports and the wholesale value.
4. **What a forecast is worth.** Peak shaving is the one use where the load forecast matters. A
   battery (20 % of mean demand, two hours) is run against a threshold set from the day-ahead
   forecast and then controlled in real time on the actual load; the realised cut in the day's
   peak is compared with the cut under perfect foresight, for every forecasting model of the
   proof of concept and for thresholds planned on the median or on an upper quantile.

Price cases (p/kWh): the Elexon market index (``wholesale``); the same plus a red-band unit charge
of 2, 5 or 10 p/kWh (scenario values: the real rates differ by network area and voltage and are
not in the project); and the Octopus Agile tariff of the site's region, a published retail
half-hourly tariff with its own peak adder.

Outputs in ``code/reports/flexibility/``.
"""

from __future__ import annotations

import json

import holidays
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from ..config import GEO_DIR, INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..data.prices import load_mid
from ..logging_utils import get_logger
from . import weather_response as wr

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "flexibility"
POC = PROJECT_ROOT / "code" / "reports" / "poc"
TZ = "Europe/London"
RED_HOURS = (16, 17, 18)
RED_ADDERS = (2.0, 5.0, 10.0)  # p/kWh, scenario values
YEARS = ("2023/24", "2024/25", "2025/26")
MIN_COVERAGE = 0.9
ROUND_TRIP = 0.85
DURATION_H = 2.0
BATTERY_SHARE = 0.2  # battery power as a share of the site's mean demand
PERFORMANCE_RATIO = 0.8
SOLAR_SIZES = (0.25, 0.5, 1.0, 2.0, 4.0)  # kWp per kW of mean demand
FORECAST_MODELS = {
    "seasonal_naive": ("forecasts_baselines.parquet", "seasonal_naive"),
    "profile_quantiles": ("forecasts_baselines.parquet", "profile_quantiles"),
    "lightgbm_ifs": ("forecasts_lightgbm_ifs.parquet", None),
    "tabpfn_ts_ifs": ("forecasts_tabpfn_ts_ifs.parquet", None),
    "timesfm3_ifs": ("forecasts_timesfm3_ifs.parquet", None),
    "chronos2": ("forecasts_chronos2.parquet", None),
    "chronos2_ifs": ("forecasts_chronos2_ifs.parquet", None),
}
PLAN_QUANTILES = ("q50", "q60", "q70", "q80", "q90")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


def demand() -> pd.DataFrame:
    """Half-hourly site demand in kW (UTC index)."""
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet") * 2.0
    w.index = pd.to_datetime(w.index, utc=True)
    return w


def calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Local date and hour, working-day and red-band flags and the financial year of each
    half-hour."""
    local = index.tz_convert(TZ)
    date = local.normalize().tz_localize(None)
    hol = set(holidays.UK(subdiv="ENG", years=range(local.year.min(), local.year.max() + 1)))
    working = (local.dayofweek < 5) & ~np.isin(date.date, list(hol))
    start = np.where(local.month >= 4, local.year, local.year - 1)
    fy = np.array([f"{y}/{str(y + 1)[2:]}" for y in start])
    return pd.DataFrame(
        {
            "date": date,
            "hour": local.hour,
            "working": working,
            "red": working & np.isin(local.hour, RED_HOURS),
            "fy": fy,
        },
        index=index,
    )


def price_cases(index: pd.DatetimeIndex, cal: pd.DataFrame) -> pd.DataFrame:
    """Price in p/kWh per half-hour for every case; Agile one column per region."""
    mid = load_mid().set_index("datetime")["price_gbp_mwh"].reindex(index) / 10.0
    out = {"wholesale": mid}
    for a in RED_ADDERS:
        out[f"wholesale + red {a:g}p"] = mid + a * cal["red"].to_numpy()
    agile = pd.read_parquet(
        INTERIM_DIR / "prices_agile.parquet", columns=["datetime", "region", "rate_exc_vat"]
    )
    agile["datetime"] = pd.to_datetime(agile["datetime"], utc=True)
    ag = agile.pivot_table(index="datetime", columns="region", values="rate_exc_vat")
    for r in ag.columns:
        out[f"Agile {r}"] = ag[r].reindex(index)
    return pd.DataFrame(out, index=index)


def site_table(w: pd.DataFrame) -> pd.DataFrame:
    geo = pd.read_parquet(GEO_DIR / "sites_geo.parquet").set_index("site_code")
    t = pd.DataFrame(index=w.columns)
    t["group"] = wr.site_groups().reindex(t.index)
    t["region"] = geo["gsp_group"].reindex(t.index)
    t["site_name"] = geo["site_name"].reindex(t.index)
    t["mean_kw"] = w.mean()
    return t


# --------------------------------------------------------------------------- #
# 1. Bill anatomy
# --------------------------------------------------------------------------- #


def bill_anatomy(w: pd.DataFrame, cal: pd.DataFrame, price: pd.Series) -> pd.DataFrame:
    """Per site and financial year: energy, wholesale cost, price paid against the time-average
    price, and the red-band share of energy."""
    rows = []
    p = price.to_numpy()
    for fy in YEARS:
        m = (cal["fy"] == fy).to_numpy() & np.isfinite(p)
        if not m.any():
            continue
        x, pm, red = w.to_numpy()[m], p[m], cal["red"].to_numpy()[m]
        ok = np.isfinite(x)
        cover = ok.mean(axis=0)
        kwh = np.where(ok, x, 0.0) * 0.5
        energy = kwh.sum(axis=0)
        cost = (kwh * pm[:, None]).sum(axis=0) / 100.0
        time_mean = np.where(ok, pm[:, None], 0.0).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            paid = 100.0 * cost / energy
            red_share = kwh[red].sum(axis=0) / energy
            red_time = (ok & red[:, None]).sum(axis=0) / np.maximum(ok.sum(axis=0), 1)
        for j, s in enumerate(w.columns):
            if cover[j] < MIN_COVERAGE or energy[j] <= 0:
                continue
            scale = 1.0 / cover[j]  # to a full year
            rows.append(
                {
                    "site_code": s,
                    "fy": fy,
                    "coverage": cover[j],
                    "mwh": energy[j] * scale / 1000.0,
                    "wholesale_cost_gbp": cost[j] * scale,
                    "paid_p_kwh": paid[j],
                    "time_mean_p_kwh": time_mean[j],
                    "shape_premium_pct": 100.0 * (paid[j] / time_mean[j] - 1.0),
                    "red_share_energy": red_share[j],
                    "red_share_time": red_time[j],
                    "red_mwh": kwh[red, j].sum() * scale / 1000.0,
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 2. Battery arbitrage
# --------------------------------------------------------------------------- #


def _lp_matrices(n: int) -> tuple[np.ndarray, np.ndarray]:
    """State-of-charge bounds for a 1 kW battery over n half-hours: variables [charge, discharge]."""
    low = np.tril(np.ones((n, n))) * 0.5
    soc = np.hstack([low * ROUND_TRIP, -low])  # kWh stored after each half-hour
    a_ub = np.vstack([soc, -soc])
    b_ub = np.r_[np.full(n, DURATION_H), np.zeros(n)]
    return a_ub, b_ub


def battery_day(price: np.ndarray) -> tuple[float, np.ndarray]:
    """Best cycle of a 1 kW, ``DURATION_H`` h battery over one day of prices (p/kWh), starting
    and ending empty: value in GBP per kW and the net discharge per half-hour (kW per kW)."""
    n = len(price)
    a_ub, b_ub = _lp_matrices(n)
    c = np.r_[price, -price] * 0.5 / 100.0
    r = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=(0.0, 1.0), method="highs")
    if r.status != 0:
        return 0.0, np.zeros(n)
    return float(-r.fun), r.x[n:] - r.x[:n]


def battery_unit(prices: pd.DataFrame, cal: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily value of the unit battery under every price case, and its net-discharge schedules."""
    from joblib import Parallel, delayed

    days = cal.groupby("date", sort=True).indices

    def one(case: str):
        p = prices[case].to_numpy()
        vals, sched = [], np.full(len(p), np.nan)
        for d, idx in days.items():
            pd_ = p[idx]
            if len(idx) < 46 or not np.isfinite(pd_).all():
                continue
            v, net = battery_day(pd_)
            vals.append((case, d, v, float(np.clip(net, 0, None).sum() * 0.5 / DURATION_H)))
            sched[idx] = net
        return vals, sched

    res = Parallel(n_jobs=min(16, len(prices.columns)))(delayed(one)(c) for c in prices.columns)
    daily = pd.DataFrame(
        [v for vals, _ in res for v in vals], columns=["case", "date", "value_gbp_kw", "cycles"]
    )
    sched = pd.DataFrame(
        {c: s for c, (_, s) in zip(prices.columns, res, strict=True)}, index=prices.index
    )
    return daily, sched


def battery_annual(daily: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    fy = cal.drop_duplicates("date").set_index("date")["fy"]
    d = daily.assign(fy=daily["date"].map(fy))
    g = d[d["fy"].isin(YEARS)].groupby(["case", "fy"])
    out = g.agg(days=("date", "size"), value=("value_gbp_kw", "sum"), cycles=("cycles", "mean"))
    out["value_gbp_kw_year"] = out["value"] * 365.0 / out["days"]
    out["value_gbp_kwh_year"] = out["value_gbp_kw_year"] / DURATION_H
    return out.drop(columns="value").reset_index()


def battery_sites(
    w: pd.DataFrame, sites: pd.DataFrame, annual: pd.DataFrame, sched: pd.DataFrame, fy: str
) -> pd.DataFrame:
    """Each site's battery (``BATTERY_SHARE`` of mean demand, two hours) under three cases: annual
    value and the share of its discharge that would exceed the site's load (exports)."""
    a = annual[annual["fy"] == fy].set_index("case")["value_gbp_kw_year"]
    rows = []
    x = w.to_numpy()
    for j, s in enumerate(w.columns):
        p_kw = BATTERY_SHARE * sites.loc[s, "mean_kw"]
        if not np.isfinite(p_kw) or p_kw <= 0:
            continue
        row = {"site_code": s, "battery_kw": p_kw, "battery_kwh": p_kw * DURATION_H}
        cases = {"wholesale": "wholesale", "red5": "wholesale + red 5p"}
        if isinstance(sites.loc[s, "region"], str) and f"Agile {sites.loc[s, 'region']}" in a.index:
            cases["agile"] = f"Agile {sites.loc[s, 'region']}"
        for k, case in cases.items():
            row[f"value_{k}_gbp_year"] = a.get(case, np.nan) * p_kw
            net = sched[case].to_numpy() * p_kw
            ok = np.isfinite(net) & np.isfinite(x[:, j])
            out = np.clip(net[ok], 0, None)
            row[f"export_share_{k}"] = (
                float(np.clip(net[ok] - x[ok, j], 0, None).sum() / out.sum()) if out.sum() else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 3. Rooftop solar
# --------------------------------------------------------------------------- #


def pv_per_kwp(sites: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Half-hourly generation in kW per kWp from ERA5 surface solar radiation at the site. ERA5
    radiation at an hour is the energy of the hour before it, spread here over its two
    half-hours."""
    r = pd.read_parquet(
        INTERIM_DIR / "weather_reanalysis.parquet", columns=["site_code", "datetime", "ssrd"]
    )
    r = r[r["site_code"].isin(sites)]
    wide = r.pivot_table(index="datetime", columns="site_code", values="ssrd")
    wide.index = pd.to_datetime(wide.index, utc=True)
    kw = wide / 3.6e6 * PERFORMANCE_RATIO  # mean kW per kWp over the hour ending at the stamp
    a = kw.copy()
    a.index = a.index - pd.Timedelta(minutes=60)
    b = kw.copy()
    b.index = b.index - pd.Timedelta(minutes=30)
    return pd.concat([a, b]).sort_index().reindex(index)


def solar_sites(
    w: pd.DataFrame, cal: pd.DataFrame, price: pd.Series, sites: pd.DataFrame, fy: str
) -> pd.DataFrame:
    """Per site without CHP and array size: generation, self-consumption, exports and value."""
    m = (cal["fy"] == fy).to_numpy()
    keep = [
        s
        for s in w.columns
        if sites.loc[s, "group"] != "on-site CHP" and w[s][m].notna().mean() >= MIN_COVERAGE
    ]
    pv = pv_per_kwp(keep, w.index)
    keep = [s for s in keep if s in pv.columns and pv[s][m].notna().mean() > 0.99]
    p = price.to_numpy()[m]
    rows = []
    for s in keep:
        load = w[s].to_numpy()[m]
        unit = pv[s].to_numpy()[m]
        ok = np.isfinite(load) & np.isfinite(unit) & np.isfinite(p)
        load, unit, pp = load[ok], unit[ok], p[ok]
        mean_kw = load.mean()
        for k in SOLAR_SIZES:
            gen = unit * k * mean_kw
            used = np.minimum(gen, load)
            rows.append(
                {
                    "site_code": s,
                    "group": sites.loc[s, "group"],
                    "size": k,
                    "kwp": k * mean_kw,
                    "yield_kwh_kwp": unit.mean() * 8760.0,
                    "self_consumption": used.sum() / gen.sum(),
                    "demand_covered": used.sum() / load.sum(),
                    "export_share": 1.0 - used.sum() / gen.sum(),
                    "capture_ratio": (gen * pp).sum() / gen.sum() / pp.mean(),
                    "value_gbp_kwp_year": (used * pp).sum()
                    * 0.5
                    / 100.0
                    / (k * mean_kw)
                    * (17520.0 / len(load)),
                }
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# 4. What a forecast is worth: peak shaving
# --------------------------------------------------------------------------- #


def _feasible(theta, load, p_kw, e_kwh) -> np.ndarray:
    """Whether a full battery can hold import at or below theta through each row of ``load``."""
    soc = e_kwh.copy()
    ok = np.ones(len(theta), dtype=bool)
    for t in range(load.shape[1]):
        over = load[:, t] - theta
        need = np.clip(over, 0, None)
        ok &= need <= p_kw + 1e-9
        ok &= need * 0.5 <= soc + 1e-9
        soc = soc - np.minimum(need * 0.5, soc)
        charge = np.clip(np.minimum(p_kw, -over), 0, None)
        soc = soc + np.minimum(charge * 0.5 * ROUND_TRIP, e_kwh - soc)
    return ok


def plan_threshold(load, p_kw, e_kwh, iters: int = 40) -> np.ndarray:
    """Lowest import ceiling the battery can hold through each row of ``load`` (kW)."""
    hi = np.nanmax(load, axis=1)
    lo = np.clip(hi - p_kw, 0, None)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        f = _feasible(mid, load, p_kw, e_kwh)
        hi = np.where(f, mid, hi)
        lo = np.where(f, lo, mid)
    return hi


def realised_peak(theta, load, p_kw, e_kwh) -> np.ndarray:
    """Peak import when the battery defends ``theta`` in real time on the actual load."""
    soc = e_kwh.copy()
    peak = np.zeros(len(theta))
    for t in range(load.shape[1]):
        over = load[:, t] - theta
        dis = np.minimum(np.minimum(np.clip(over, 0, None), p_kw), soc / 0.5)
        charge = np.minimum(
            np.clip(np.minimum(p_kw, -over), 0, None), (e_kwh - soc) / (0.5 * ROUND_TRIP)
        )
        soc = soc - dis * 0.5 + charge * 0.5 * ROUND_TRIP
        peak = np.maximum(peak, load[:, t] - dis + charge)
    return peak


def load_forecasts(name: str) -> pd.DataFrame:
    file, model = FORECAST_MODELS[name]
    f = pd.read_parquet(POC / file)
    if model is not None:
        f = f[f["model"] == model]
    for c in ("origin", "target"):
        f[c] = pd.to_datetime(f[c], utc=True)
    return f


def forecast_value(w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Peak cut achieved with a forecast-set threshold against the cut under perfect foresight,
    per model and planning quantile, on the proof-of-concept site-days."""
    actual = w.stack().rename("actual").rename_axis(["target", "site_code"]).reset_index()  # kW
    site_rows, rows = [], []
    for name in FORECAST_MODELS:
        f = load_forecasts(name).merge(actual, on=["target", "site_code"], how="left")
        f = f.sort_values(["site_code", "origin", "target"])
        f["slot"] = f.groupby(["site_code", "origin"]).cumcount()
        n_slots = int(f["slot"].max()) + 1
        key = f[["site_code", "origin"]].drop_duplicates().reset_index(drop=True)
        key["row"] = np.arange(len(key))
        f = f.merge(key, on=["site_code", "origin"])

        n_rows = len(key)

        def mat(col, f=f, n_rows=n_rows, n_slots=n_slots):
            a = np.full((n_rows, n_slots), np.nan)
            a[f["row"].to_numpy(), f["slot"].to_numpy()] = f[col].to_numpy()
            return a

        act = mat("actual")
        good = np.isfinite(act).all(axis=1)
        mean_kw = key["site_code"].map(w.mean()).to_numpy()
        p_kw = BATTERY_SHARE * mean_kw
        e_kwh = p_kw * DURATION_H
        best = plan_threshold(act[good], p_kw[good], e_kwh[good])
        raw_peak = act[good].max(axis=1)
        perfect = raw_peak - best
        point = mat("point")[good] * 2.0  # forecasts are kWh per half-hour
        nmae = np.nanmean(np.abs(point - act[good]), axis=1) / mean_kw[good]
        plans = {"point": point}
        for q in PLAN_QUANTILES:
            if q in f.columns and f[q].notna().any():
                fq = mat(q)[good] * 2.0
                if not np.allclose(fq, point, equal_nan=True):  # point-only models repeat the point
                    plans[q] = fq
        for plan, fc in plans.items():
            usable = np.isfinite(fc).all(axis=1)
            theta = plan_threshold(fc[usable], p_kw[good][usable], e_kwh[good][usable])
            got = raw_peak[usable] - realised_peak(
                theta, act[good][usable], p_kw[good][usable], e_kwh[good][usable]
            )
            d = key[good][usable].assign(
                model=name,
                plan=plan,
                achieved_kw=got,
                perfect_kw=perfect[usable],
                raw_peak_kw=raw_peak[usable],
                nmae=nmae[usable],
            )
            site_rows.append(d)
            rows.append(
                {
                    "model": name,
                    "plan": plan,
                    "site_days": int(usable.sum()),
                    "nmae_pct": 100.0 * float(np.mean(nmae[usable])),
                    "captured_share": float(got.sum() / perfect[usable].sum()),
                    "median_site_day_share": float(
                        np.median(got / np.maximum(perfect[usable], 1e-9))
                    ),
                    "peak_cut_pct": 100.0 * float(np.mean(got / raw_peak[usable])),
                    "perfect_cut_pct": 100.0 * float(np.mean(perfect[usable] / raw_peak[usable])),
                    "failed_share": float(np.mean(got < 0.05 * perfect[usable])),
                }
            )
    return pd.DataFrame(rows), pd.concat(site_rows, ignore_index=True)


# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #


def run() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    w = demand()
    cal = calendar(w.index)
    prices = price_cases(w.index, cal)
    sites = site_table(w)

    bill = bill_anatomy(w, cal, prices["wholesale"]).merge(
        sites[["group", "site_name"]], left_on="site_code", right_index=True
    )
    bill.to_csv(OUT_DIR / "bill_anatomy.csv", index=False)

    daily, sched = battery_unit(prices, cal)
    annual = battery_annual(daily, cal)
    annual.to_csv(OUT_DIR / "battery_unit_annual.csv", index=False)
    daily.to_csv(OUT_DIR / "battery_unit_daily.csv", index=False)
    bsites = battery_sites(w, sites, annual, sched, "2024/25").merge(
        sites[["group", "region"]], left_on="site_code", right_index=True
    )
    bsites.to_csv(OUT_DIR / "battery_sites.csv", index=False)

    solar = solar_sites(w, cal, prices["wholesale"], sites, "2024/25")
    solar.to_csv(OUT_DIR / "solar_sites.csv", index=False)
    solar_summary = (
        solar.groupby(["group", "size"])[
            ["self_consumption", "demand_covered", "export_share", "value_gbp_kwp_year"]
        ]
        .median()
        .reset_index()
    )
    solar_summary.to_csv(OUT_DIR / "solar_summary.csv", index=False)

    fv, fv_days = forecast_value(w)
    fv.to_csv(OUT_DIR / "forecast_value.csv", index=False)
    fv_days.to_parquet(OUT_DIR / "forecast_value_site_days.parquet", index=False)

    b = bill[bill["fy"] == "2024/25"]
    ann = annual[annual["fy"] == "2024/25"].set_index("case")
    agile = ann[ann.index.str.startswith("Agile")]["value_gbp_kwh_year"]
    one = solar[solar["size"] == 1.0]
    summary = {
        "bill_2024_25": {
            "sites": int(len(b)),
            "gwh": round(float(b["mwh"].sum() / 1000.0), 1),
            "wholesale_cost_gbp_m": round(float(b["wholesale_cost_gbp"].sum() / 1e6), 2),
            "median_shape_premium_pct": round(float(b["shape_premium_pct"].median()), 2),
            "shape_premium_by_group": b.groupby("group")["shape_premium_pct"]
            .median()
            .round(2)
            .to_dict(),
            "median_red_share_energy": round(float(b["red_share_energy"].median()), 4),
            "red_share_time": round(float(b["red_share_time"].median()), 4),
            "red_gwh": round(float(b["red_mwh"].sum() / 1000.0), 2),
        },
        "battery_gbp_per_kwh_year_2024_25": {
            "wholesale": round(float(ann.loc["wholesale", "value_gbp_kwh_year"]), 1),
            **{
                f"red_{a:g}p": round(
                    float(ann.loc[f"wholesale + red {a:g}p", "value_gbp_kwh_year"]), 1
                )
                for a in RED_ADDERS
            },
            "agile_min": round(float(agile.min()), 1),
            "agile_max": round(float(agile.max()), 1),
            "cycles_per_day_wholesale": round(float(ann.loc["wholesale", "cycles"]), 2),
        },
        "battery_sites": {
            "sites": int(len(bsites)),
            "total_kwh": round(float(bsites["battery_kwh"].sum()), 0),
            "value_wholesale_gbp_year": round(float(bsites["value_wholesale_gbp_year"].sum()), 0),
            "value_red5_gbp_year": round(float(bsites["value_red5_gbp_year"].sum()), 0),
            "max_export_share_red5": round(float(bsites["export_share_red5"].max()), 4),
            "sites_exporting_over_1pct": int((bsites["export_share_red5"] > 0.01).sum()),
        },
        "solar_1kwp_per_kw_2024_25": {
            "sites": int(len(one)),
            "median_yield_kwh_kwp": round(float(one["yield_kwh_kwp"].median()), 0),
            "self_consumption_by_group": one.groupby("group")["self_consumption"]
            .median()
            .round(3)
            .to_dict(),
            "demand_covered_by_group": one.groupby("group")["demand_covered"]
            .median()
            .round(3)
            .to_dict(),
            "median_capture_ratio": round(float(one["capture_ratio"].median()), 3),
            "median_value_gbp_kwp_year": round(float(one["value_gbp_kwp_year"].median()), 1),
        },
        "forecast_value": fv.round(3).to_dict(orient="records"),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    logger.info("flexibility: %s", json.dumps(summary["battery_gbp_per_kwh_year_2024_25"]))
    return {"summary": summary}
