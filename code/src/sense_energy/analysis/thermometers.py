"""Which thermometer? Site temperature from ERA5 (nearest 25 km grid point), ERA5-Land (nearest
9 km land point) and the nearest Met Office station (NOAA ISD, median 13 km), compared on the
same site-days, and the weather effects re-estimated with each.

1. **Agreement**: per site, bias and RMSE of the daily mean and maximum against the station,
   and the correlation of day-to-day anomalies (within site x month), with the station distance.
2. **Effects** on the site-days every available thermometer covers, so estimates differ only by
   the thermometer: specifications A, B and D of ``weather_causal``, the neighbour horse race
   (own dose against the 30-150 km mean) and the neighbours-only dose, the leads test and the
   30+ degC indicator.
3. **Response curves** (A, B, D) on the daily mean and maximum with each thermometer.

Classical measurement error attenuates a slope towards zero, and more so once fixed effects
have removed most of the true variation. If the heat effect rises as the thermometer moves
closer to the site, the 25 km grid point was hiding part of it; if the station and the grid
agree, the attenuation reading is wrong.

Outputs (``code/reports/thermometers/``): ``agreement.csv``, ``effects.csv``,
``response_curve.csv``, ``summary.json``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from . import weather_causal as wc

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "thermometers"
THERMOMETERS = {
    "ERA5": "weather_daily.parquet",
    "ERA5-Land": "weather_daily_land.parquet",
    "stations": "weather_daily_isd.parquet",
}
REFERENCE = "stations"
KEEP = ["site_code", "date", "t_mean", "t_max", "t_eve"]


def tables() -> dict[str, pd.DataFrame]:
    """The available thermometer tables, each as site_code, date, t_mean, t_max, t_eve."""
    out = {}
    for name, file in THERMOMETERS.items():
        path = INTERIM_DIR / file
        if path.exists():
            t = pd.read_parquet(path)
            out[name] = t[KEEP + (["distance_km"] if "distance_km" in t else [])].copy()
    return out


def common_keys(tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    keys = None
    for t in tabs.values():
        k = t[["site_code", "date"]].drop_duplicates()
        keys = k if keys is None else keys.merge(k, on=["site_code", "date"])
    return keys


def _anomaly(t: pd.DataFrame, col: str) -> pd.Series:
    ym = t["date"].dt.strftime("%Y-%m")
    return t[col] - t.groupby([t["site_code"], ym])[col].transform("mean")


def agreement(tabs: dict[str, pd.DataFrame], reference: str = REFERENCE) -> pd.DataFrame:
    """Per-site agreement of every other thermometer with the reference."""
    ref = tabs[reference]
    rows = []
    for name, t in tabs.items():
        if name == reference:
            continue
        m = t.merge(ref, on=["site_code", "date"], suffixes=("", "_ref"))
        for col in ("t_mean", "t_max"):
            m[f"d_{col}"] = m[col] - m[f"{col}_ref"]
        m["anom"], m["anom_ref"] = _anomaly(m, "t_mean"), _anomaly(m, "t_mean_ref")
        g = m.groupby("site_code")
        a = pd.DataFrame(
            {
                "bias_mean_c": g["d_t_mean"].mean(),
                "rmse_mean_c": g["d_t_mean"].apply(lambda s: float(np.sqrt(np.mean(s**2)))),
                "bias_max_c": g["d_t_max"].mean(),
                "rmse_max_c": g["d_t_max"].apply(lambda s: float(np.sqrt(np.mean(s**2)))),
                "anomaly_corr": g.apply(lambda d: d["anom"].corr(d["anom_ref"])),
                "anomaly_rmse_c": g.apply(
                    lambda d: float(np.sqrt(np.mean((d["anom"] - d["anom_ref"]) ** 2)))
                ),
                "days": g.size(),
            }
        )
        if "distance_km" in ref:
            a["station_km"] = m.groupby("site_code")[
                "distance_km_ref" if "distance_km_ref" in m else "distance_km"
            ].mean()
        rows.append(a.assign(thermometer=name).reset_index())
    return pd.concat(rows, ignore_index=True)


def effects(dfs: dict[str, pd.DataFrame], tabs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for name, df in dfs.items():
        no_chp = df[df["group"].isin(wc.GROUPS["no CHP"])]
        s = wc.twfe_specs(no_chp)
        rows.append(s[s["group"] == "no CHP"].assign(block="specs"))
        lag = wc.spatial_lag(df, doses_table=tabs[name])
        rows.append(lag.assign(block="neighbours", group="no CHP"))
        ex = wc.twfe_extreme(no_chp)
        rows.append(
            ex[(ex["group"] == "no CHP") & ex["term"].isin(wc.EXTREME)].assign(block="extreme")
        )
        ll = wc.leads_lags(df)
        lead = ll[(ll["kind"] == "coefficient") & (ll["lag"] < 0)].copy()
        rows.append(lead.assign(block="leads", group="no CHP", spec="B"))
        for r in rows[-4:]:
            r["thermometer"] = name
    return pd.concat(rows, ignore_index=True)


def run() -> dict:
    from threadpoolctl import threadpool_limits

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tabs = tables()
    if len(tabs) < 2:
        raise RuntimeError(f"Need at least two thermometer tables; have {sorted(tabs)}")
    keys = common_keys(tabs)
    tabs = {k: t.merge(keys, on=["site_code", "date"]) for k, t in tabs.items()}
    logger.info("thermometers %s on %s common site-days", sorted(tabs), f"{len(keys):,}")

    with threadpool_limits(4):
        agree = agreement(tabs) if REFERENCE in tabs else agreement(tabs, reference="ERA5")
        agree.to_csv(OUT_DIR / "agreement.csv", index=False)
        dfs = {name: wc.panel(thermometer=t) for name, t in tabs.items()}
        eff = effects(dfs, tabs)
        eff.to_csv(OUT_DIR / "effects.csv", index=False)
        curves = pd.concat(
            [wc.response_curve(df).assign(thermometer=name) for name, df in dfs.items()],
            ignore_index=True,
        )
        curves.to_csv(OUT_DIR / "response_curve.csv", index=False)

    def pick(name: str, block: str, spec: str, term: str, variant: str | None = None):
        d = eff[
            (eff["thermometer"] == name)
            & (eff["block"] == block)
            & (eff["spec"] == spec)
            & (eff["term"] == term)
        ]
        if variant is not None:
            d = d[d["variant"] == variant]
        r = d.iloc[0]
        return [
            round(float(r["coef_pct"]), 2),
            round(float(r["lo_pct"]), 2),
            round(float(r["hi_pct"]), 2),
        ]

    summary = {
        "thermometers": sorted(tabs),
        "common_site_days": int(len(keys)),
        "panel_site_days": {name: int(len(df)) for name, df in dfs.items()},
        "agreement_medians": {
            name: g[["bias_mean_c", "rmse_mean_c", "rmse_max_c", "anomaly_corr", "anomaly_rmse_c"]]
            .median()
            .round(3)
            .to_dict()
            for name, g in agree.groupby("thermometer")
        },
        "effects_pct_per_c": {
            f"{name} | {spec} | {term}": pick(name, "specs", spec, term)
            for name in dfs
            for spec in ("A", "B", "D")
            for term in wc.DOSE
        },
        "neighbours_only_pct_per_c": {
            f"{name} | {spec} | {term}": pick(name, "neighbours", spec, term, "neighbours only")
            for name in dfs
            for spec in ("A", "B")
            for term in ("hdd14_nb", "cdd20_nb")
        },
        "hot30_pct": {f"{name} | B": pick(name, "extreme", "B", "hot30") for name in dfs},
        "leads_joint_p": {
            name: round(
                float(
                    eff[(eff["thermometer"] == name) & (eff["block"] == "leads")][
                        "p_leads_joint"
                    ].iloc[0]
                ),
                3,
            )
            for name in dfs
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("thermometers: %s", summary["effects_pct_per_c"])
    return {"agreement": agree, "effects": eff, "response_curve": curves, "summary": summary}
