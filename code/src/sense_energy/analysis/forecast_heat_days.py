"""Day-ahead forecast skill on hot days.

The heat work says a site's electricity rises on hot afternoons and falls on sunny days, and
that ERA5 reads the hottest days 1.3 degC too cool. Does any of that reach the day-ahead
forecasts? Every model's test-window forecasts (``code/reports/poc/forecasts_<model>.parquet``)
are scored against the observed half-hours, with the target day classed by the site's ERA5
daily maximum: 28 degC and above, 25-28, other June-September days, days at or below 2 degC
daily mean, and the rest. Scores follow ``experiments.poc.score``: nMAE per site (MAE over the
site's mean), CRPS on the quantile grid, 80 % coverage, then the site mean per model and class.

Output: ``code/reports/forecast_heat_days/scores_by_day_class.csv`` and ``per_site.csv``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..experiments.poc import QUANTILE_COLS, pinball
from ..logging_utils import get_logger

logger = get_logger(__name__)

POC = PROJECT_ROOT / "code" / "reports" / "poc"
OUT_DIR = PROJECT_ROOT / "code" / "reports" / "forecast_heat_days"
TZ = "Europe/London"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODELS = (
    "seasonal_naive",
    "profile_quantiles",
    "lightgbm",
    "lightgbm_era5",
    "lightgbm_ifs",
    "lightgbm_ifs_heat",
    "chronos2",
    "chronos2_era5",
    "chronos2_ifs",
    "tabpfn_ts_ifs",
    "tirex2_ifs",
    "timesfm3_ifs",
    "t0beta_ifs",
    "tnp_ifs",
    "pttnp_ifs",
    "gtnp_ifs",
)
CLASS_ORDER = (
    "28 °C and above",
    "25–28 °C",
    "other June–September",
    "at or below 2 °C",
    "rest of year",
)


def day_classes() -> pd.DataFrame:
    """One row per site and local day with its class."""
    w = pd.read_parquet(
        INTERIM_DIR / "weather_daily.parquet", columns=["site_code", "date", "t_max", "t_mean"]
    )
    summer = w["date"].dt.month.isin([6, 7, 8, 9])
    w["day_class"] = np.select(
        [w["t_max"] >= 28, w["t_max"] >= 25, summer, w["t_mean"] <= 2],
        list(CLASS_ORDER[:4]),
        default=CLASS_ORDER[4],
    )
    return w[["site_code", "date", "day_class", "t_max"]]


def actuals() -> pd.DataFrame:
    w = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")
    y = w.stack(future_stack=True).rename("y").reset_index()
    y.columns = ["target", "site_code", "y"]
    y["target"] = pd.to_datetime(y["target"], utc=True)
    return y.dropna(subset=["y"])


def load_forecasts(models=MODELS) -> pd.DataFrame:
    frames = []
    for m in models:
        p = POC / f"forecasts_{m}.parquet"
        if p.exists():
            f = pd.read_parquet(p)
            f["model"] = m
            frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out["target"] = pd.to_datetime(out["target"], utc=True)
    return out


def score(models=MODELS) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = load_forecasts(models).merge(actuals(), on=["target", "site_code"], how="inner")
    f["date"] = f["target"].dt.tz_convert(TZ).dt.normalize().dt.tz_localize(None)
    f = f.merge(day_classes(), on=["site_code", "date"], how="left")
    f["day_class"] = f["day_class"].fillna(CLASS_ORDER[4])
    f["ae"] = (f["y"] - f["point"]).abs()
    qcols = [QUANTILE_COLS[q] for q in QUANTILES]
    pb = np.column_stack(
        [
            pinball(f["y"].to_numpy(), f[c].to_numpy(), q)
            for q, c in zip(QUANTILES, qcols, strict=True)
        ]
    )
    f["crps_q"] = 2 * pb.mean(axis=1)
    f["in80"] = ((f["y"] >= f[QUANTILE_COLS[0.1]]) & (f["y"] <= f[QUANTILE_COLS[0.9]])).astype(
        float
    )
    site_mean = f[f["model"] == f["model"].iloc[0]].groupby("site_code")["y"].mean()
    per = (
        f.groupby(["model", "day_class", "site_code"], observed=True)
        .agg(
            n=("y", "size"),
            mae=("ae", "mean"),
            crps_q=("crps_q", "mean"),
            coverage_80=("in80", "mean"),
        )
        .reset_index()
    )
    per["y_mean_site"] = per["site_code"].map(site_mean)
    per["nmae"] = per["mae"] / per["y_mean_site"]
    per["ncrps"] = per["crps_q"] / per["y_mean_site"]
    per = per[per["n"] >= 48]  # at least one full day in the class
    pooled = (
        per.groupby(["model", "day_class"], observed=True)
        .agg(
            sites=("site_code", "nunique"),
            site_days=("n", lambda s: float(s.sum() / 48)),
            nmae=("nmae", "mean"),
            nmae_median=("nmae", "median"),
            ncrps=("ncrps", "mean"),
            coverage_80=("coverage_80", "mean"),
        )
        .reset_index()
    )
    base = pooled[pooled["model"] == "seasonal_naive"].set_index("day_class")["nmae"]
    pooled["skill_vs_naive"] = 1 - pooled["nmae"] / pooled["day_class"].map(base)
    ref = pooled[pooled["model"] == "lightgbm_ifs"].set_index("day_class")["nmae"]
    pooled["nmae_vs_lightgbm_ifs"] = pooled["nmae"] - pooled["day_class"].map(ref)
    return pooled, per


def run(models=MODELS) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pooled, per = score(models)
    pooled.to_csv(OUT_DIR / "scores_by_day_class.csv", index=False)
    per.to_csv(OUT_DIR / "per_site.csv", index=False)
    hot = pooled[pooled["day_class"] == CLASS_ORDER[0]].set_index("model")["nmae"]
    logger.info("nMAE on 28 degC+ days: %s", hot.round(3).to_dict())
    return pooled
