"""An independent check of "Electricity load forecasting under extreme heat and cold waves"
(Lu, Qin, Zhou, Bai, Wen, Kang and Wang, Nature Communications 17, 10041, 2026) on the paper's
own public data.

The paper proposes ESDF: a diffusion model makes synthetic load and temperature samples for heat
and cold waves, and a network with shared and private encoders (a domain separation network)
learns from the real and the synthetic samples. It reports the error of ESDF against seven
neural baselines on 80 regional load series. The repository
(github.com/hkuedl/Heat-Cold_Wave_Load_Forecasting, MIT licence, commit 0036608 of 14 October
2026) holds the hourly load and temperature of 18 European countries and 19 PJM zones and the
forecast that each of the paper's models made for every day of its test period. This module
reads those files as they are (the paper's code is not run) and asks:

1. **What does the paper score?** The task, rebuilt from the paper's loaders: the 24 hourly loads
   of day D from the hourly load and temperature of D-7 to D-1 and the *observed* temperature of
   D. Load is scaled by the range of the training period, so an error of 0.03 is 3 % of that
   range. The reported error is the error on one window of consecutive days per region (Europe:
   8 days from 25 February 2018; PJM: 15 days from 29 June 2023), the same window for every
   series of the region.
2. **What do the saved forecasts say on the other days?** The same models on every day of the
   test period, on its hot and cold days (target-day temperature beyond the 5th or 95th
   percentile of the training period) and on the days in between.
3. **What do references that the paper does not have do with the same inputs?** The load of the
   same hour one week before, a ridge regression per hour and a gradient-boosted model fitted
   on the training period, and Chronos-2 with no training (seven days of context as the paper's
   models, and four weeks).

Outputs in ``code/reports/esdf_check/``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import EXTERNAL_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

ESDF = EXTERNAL_DIR / "esdf"
DATA = ESDF / "Data" / "reformed_data_updated"
OUT_DIR = PROJECT_ROOT / "code" / "reports" / "esdf_check"
HISTORY_DAYS = 7
LONG_CONTEXT_DAYS = 28


@dataclass(frozen=True)
class Region:
    """One of the paper's regions with the settings of its training script."""

    key: str
    folder: str
    suffix: str
    time_col: str
    train: tuple[str, str]
    test: tuple[str, str]
    first: int  # first sample of the scored window (``day`` in the paper's script)
    days: int  # length of the scored window (``period``)
    wave: str
    results: str
    sheet: str


REGIONS = {
    "europe": Region(
        "europe",
        "real_Europe_data_reformed",
        ".xlsx",
        "Data_Hour",
        ("2015/01/01/00", "2017/12/31/23"),
        ("2017/12/31/00", "2018/12/31/23"),
        19,
        8,
        "cold",
        "europe/results",
        "EU_basic",
    ),
    "pjm": Region(
        "pjm",
        "PJM_reformed_data",
        ".csv",
        "Date_Hour",
        ("2021/03/28/00", "2023/05/31/23"),
        ("2023/05/01/00", "2023/09/30/23"),
        22,
        15,
        "heat",
        "pjm/results",
        "PJM_basic",
    ),
}

#: the paper's saved forecasts: label -> file-name pattern (``{s}`` is the series)
PAPER_MODELS = {
    "ESDF (LSTM)": "results_proposed_{s}_LSTM_1",
    "ESDF (MLP)": "results_proposed_{s}_ANN_1",
    "ESDF (CNN)": "results_proposed_{s}_CNN_1",
    "LSTM": "results_basic_{s}_LSTM",
    "LSTM + synthetic": "results_basic_{s}_LSTM_1",
    "MLP": "results_basic_{s}_ANN",
    "CNN": "results_basic_{s}_CNN",
    "N-BEATS": "results_basic_{s}_nbeats",
    "ImpactNet": "results_basic_{s}_impactnet",
    "Informer": "results_basic_{s}_informer",
    "Autoformer": "results_basic_{s}_autoformer",
}
PROPOSED = "ESDF (LSTM)"
REFERENCES = (
    "Week before",
    "Linear regression",
    "LightGBM",
    "Chronos-2, 7 days",
    "Chronos-2, 4 weeks",
)
SUBSETS = ("paper window", "all test days", "hot and cold days", "other days")


# --------------------------------------------------------------------------- #
# The paper's samples, rebuilt from its loaders
# --------------------------------------------------------------------------- #


def series_names(region: Region) -> list[str]:
    return sorted(p.stem for p in (DATA / region.folder).glob(f"*{region.suffix}"))


def read_series(region: Region, name: str) -> pd.DataFrame:
    """Hourly ``time``, ``load``, ``temp`` in the file's order, with the paper's time handling."""
    path = DATA / region.folder / f"{name}{region.suffix}"
    raw = pd.read_excel(path) if region.suffix == ".xlsx" else pd.read_csv(path)
    stamp = raw[region.time_col].astype(str)
    if region.key == "pjm":  # hour-ending 1..24 (25 on the long day) to hour-beginning 0..23
        day, hour = stamp.str.rsplit("/", n=1).str[0], stamp.str.rsplit("/", n=1).str[1].astype(int)
        hour = np.where(hour == 25, 23, np.where(hour == 1, 0, hour - 1))
        stamp = day + "/" + pd.Series(hour, index=raw.index).astype(str).str.zfill(2)
    return pd.DataFrame(
        {
            "time": pd.to_datetime(stamp, format="%Y/%m/%d/%H"),
            "load": raw["Load"].to_numpy(dtype="float64"),
            "temp": raw["Temperature"].to_numpy(dtype="float64"),
        }
    )


def window(df: pd.DataFrame, span: tuple[str, str]) -> pd.DataFrame:
    lo, hi = (pd.to_datetime(s, format="%Y/%m/%d/%H") for s in span)
    return df[(df["time"] >= lo) & (df["time"] <= hi)].reset_index(drop=True)


def daily_temperature(temp: np.ndarray) -> np.ndarray:
    """The paper's daily temperature: the mean of the day's highest and lowest hour."""
    days = temp[: len(temp) // 24 * 24].reshape(-1, 24)
    return (days.max(axis=1) + days.min(axis=1)) / 2.0


@dataclass
class Samples:
    """Eight-day slices as the paper's loaders make them, in scaled units."""

    load: np.ndarray  # (n, 8, 24): seven days of history and the target day
    temp: np.ndarray  # (n, 8, 24)
    target_day: np.ndarray  # date of the target day
    first_row: np.ndarray  # row of the slice's first hour in the frame
    cold: np.ndarray  # target-day temperature under the 5th percentile of the training period
    hot: np.ndarray  # over the 95th percentile


def make_samples(
    df: pd.DataFrame,
    scale: tuple[float, float, float, float],
    limits: tuple[float, float],
    drop_zero: bool = False,
) -> Samples:
    """Slices ``[24 i, 24 (i + 8))`` for ``i`` from 30 to the paper's upper end. ``scale`` is
    (load max, load min, temperature max, temperature min) of the training period; ``limits``
    are the 5th and 95th percentile of the training period's daily temperature (scaled)."""
    lmax, lmin, tmax, tmin = scale
    load_raw = df["load"].to_numpy()
    load = (load_raw - lmin) / (lmax - lmin)
    temp = (df["temp"].to_numpy() - tmin) / (tmax - tmin)
    t_day = daily_temperature(temp)
    n_days = len(load) // 24
    rows = []
    for i in range(30, n_days - 3 - 30 - 6):
        sl = slice(24 * i, 24 * (i + 8))
        if drop_zero and np.nanmin(load_raw[sl]) <= 1e-7:  # as the PJM loader
            continue
        rows.append(i)
    idx = np.array(rows, dtype=int)
    take = (24 * idx)[:, None] + np.arange(8 * 24)[None, :]
    return Samples(
        load=load[take].reshape(-1, 8, 24),
        temp=temp[take].reshape(-1, 8, 24),
        target_day=df["time"].to_numpy()[24 * (idx + 7)].astype("datetime64[D]"),
        first_row=24 * idx,
        cold=t_day[idx + 7] < limits[0],
        hot=t_day[idx + 7] > limits[1],
    )


def prepare(region: Region, name: str) -> tuple[Samples, Samples, pd.DataFrame, tuple]:
    """Training samples, test samples, the test frame and the scale of one series."""
    df = read_series(region, name)
    train, test = window(df, region.train), window(df, region.test)
    load_min = train["load"].min()
    if region.key == "pjm":  # the PJM scaler takes the lowest load that is not zero
        load_min = train.loc[train["load"] != 0, "load"].min()
    scale = (train["load"].max(), load_min, train["temp"].max(), train["temp"].min())
    t_day = daily_temperature((train["temp"].to_numpy() - scale[3]) / (scale[2] - scale[3]))
    limits = (float(np.percentile(t_day, 5)), float(np.percentile(t_day, 95)))
    drop = region.key == "pjm"
    return (
        make_samples(train, scale, limits, drop),
        make_samples(test, scale, limits, drop),
        test,
        scale,
    )


def paper_forecast(region: Region, name: str, pattern: str) -> np.ndarray | None:
    """A saved forecast of the paper as (n, 24) with its ``true`` column, or ``None``."""
    stem = pattern.format(s=name)
    for path in (ESDF / region.results / stem, ESDF / region.results / f"{stem}.csv"):
        if path.exists():
            r = pd.read_csv(path)
            return np.stack([r["true"].to_numpy(), r["forecasted"].to_numpy()]).reshape(2, -1, 24)
    return None


# --------------------------------------------------------------------------- #
# References with the paper's inputs
# --------------------------------------------------------------------------- #


def features(s: Samples) -> pd.DataFrame:
    """One row per sample and hour, from the seven days of history and the target day's
    temperature only: what the paper's models see."""
    n = len(s.load)
    hist_l, hist_t, tgt_t = s.load[:, :7, :], s.temp[:, :7, :], s.temp[:, 7, :]
    f = {"hour": np.tile(np.arange(24), n)}
    for d in range(7):  # same hour, 7 to 1 days before
        f[f"load_d{7 - d}"] = hist_l[:, d, :].ravel()
        f[f"temp_d{7 - d}"] = hist_t[:, d, :].ravel()
    f["temp"] = tgt_t.ravel()
    day = {
        "load_last": hist_l[:, 6, 23],
        "load_d1_mean": hist_l[:, 6, :].mean(axis=1),
        "load_d1_max": hist_l[:, 6, :].max(axis=1),
        "load_d1_min": hist_l[:, 6, :].min(axis=1),
        "load_week_mean": hist_l.mean(axis=(1, 2)),
        "temp_mean": tgt_t.mean(axis=1),
        "temp_max": tgt_t.max(axis=1),
        "temp_min": tgt_t.min(axis=1),
        "temp_d1_mean": hist_t[:, 6, :].mean(axis=1),
        "temp_week_mean": hist_t.mean(axis=(1, 2)),
    }
    for k, v in day.items():
        f[k] = np.repeat(v, 24)
    return pd.DataFrame(f)


def linear_forecast(train: Samples, test: Samples, alpha: float = 1.0) -> np.ndarray:
    """One ridge regression per hour of the day on the same inputs: the plain reference of
    load forecasting, and one that can go beyond the loads it has seen."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ftr, fte = features(train), features(test)
    y = train.load[:, 7, :]
    out = np.full((len(test.load), 24), np.nan)
    cols = [c for c in ftr.columns if c != "hour"]
    for hour in range(24):
        a, b = ftr[ftr["hour"] == hour][cols].to_numpy(), fte[fte["hour"] == hour][cols].to_numpy()
        ok = np.isfinite(a).all(axis=1) & np.isfinite(y[:, hour])
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(a[ok], y[ok, hour])
        good = np.isfinite(b).all(axis=1)
        out[good, hour] = model.predict(b[good])
    return out


def lightgbm_forecast(train: Samples, test: Samples, seed: int = 0) -> np.ndarray:
    """A gradient-boosted model fitted on the training period with fixed settings."""
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        objective="l1",
        random_state=seed,
        n_jobs=4,
        verbose=-1,
    )
    y = train.load[:, 7, :].ravel()
    ok = np.isfinite(y)
    model.fit(features(train)[ok], y[ok])
    return model.predict(features(test)).reshape(-1, 24)


def _filled(x: np.ndarray) -> np.ndarray:
    return pd.Series(x).interpolate(limit_direction="both").fillna(0.0).to_numpy(dtype="float32")


def chronos_inputs(test: Samples, frame: pd.DataFrame, scale, days: int) -> list[dict]:
    """Chronos-2 inputs: ``days`` of hourly load before the target day, the temperature of
    those days as a past covariate and the target day's temperature as a known covariate."""
    lmax, lmin, tmax, tmin = scale
    load = (frame["load"].to_numpy() - lmin) / (lmax - lmin)
    temp = (frame["temp"].to_numpy() - tmin) / (tmax - tmin)
    out = []
    for row in test.first_row:
        end = row + 24 * 7  # first hour of the target day
        ctx = slice(end - 24 * days, end)
        out.append(
            {
                "target": _filled(load[ctx]),
                "past_covariates": {"temp": _filled(temp[ctx])},
                "future_covariates": {"temp": _filled(temp[end : end + 24])},
            }
        )
    return out


def chronos_forecast(pipe, inputs: list[dict], batch_size: int = 64) -> np.ndarray:
    import torch

    from .zero_shot import _to_array

    out = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            q, _ = pipe.predict_quantiles(
                inputs[start : start + batch_size],
                prediction_length=24,
                quantile_levels=[0.5],
                batch_size=batch_size,
            )
            out.append(_to_array(q, torch).reshape(-1, 24))
    return np.concatenate(out)


# --------------------------------------------------------------------------- #
# Scores
# --------------------------------------------------------------------------- #


def subset_masks(region: Region, s: Samples) -> dict[str, np.ndarray]:
    n = len(s.load)
    win = np.zeros(n, dtype=bool)
    win[region.first : region.first + region.days] = True
    extreme = s.cold | s.hot
    return {
        "paper window": win,
        "all test days": np.ones(n, dtype=bool),
        "hot and cold days": extreme,
        "other days": ~extreme,
    }


def score(y: np.ndarray, f: np.ndarray, mask: np.ndarray, offset: float) -> dict:
    """Error in the paper's units (share of the training period's load range) and as a share of
    the mean load of the scored days. ``offset`` is the training minimum over the training
    range, so ``scaled + offset`` is the load in units of the range and the second number
    compares with errors that are normalised by mean demand."""
    e = (f[mask] - y[mask]).ravel()
    ok = np.isfinite(e)
    if not ok.any():
        return {"days": 0, "mae": np.nan, "rmse": np.nan, "nmae_mean_pct": np.nan}
    e = e[ok]
    mean_load = float(np.nanmean(y[mask])) + offset
    return {
        "days": int(mask.sum()),
        "mae": float(np.abs(e).mean()),
        "rmse": float(np.sqrt((e**2).mean())),
        "nmae_mean_pct": float(100 * np.abs(e).mean() / mean_load),
    }


def run_series(region: Region, name: str, pipe=None) -> tuple[list[dict], dict, pd.DataFrame]:
    """Scores of every model on one series, the check of the rebuilt samples against the paper's
    saved ``true`` column, and the error of every model on every test day."""
    train, test, frame, scale = prepare(region, name)
    y = test.load[:, 7, :]
    offset = float(scale[1] / (scale[0] - scale[1]))  # load = (scaled + offset) x range
    masks = subset_masks(region, test)
    forecasts: dict[str, np.ndarray] = {}
    match = {"region": region.key, "series": name, "samples": len(y), "paper_samples": np.nan}
    for label, pattern in PAPER_MODELS.items():
        got = paper_forecast(region, name, pattern)
        if got is None:
            continue
        true, fc = got
        if label == PROPOSED:
            match["paper_samples"] = len(true)
            if len(true) == len(y):
                match["max_abs_diff_true"] = float(np.nanmax(np.abs(true - y)))
        if len(fc) == len(y):
            forecasts[label] = fc
    forecasts["Week before"] = test.load[:, 0, :]
    forecasts["Linear regression"] = linear_forecast(train, test)
    forecasts["LightGBM"] = lightgbm_forecast(train, test)
    if pipe is not None:
        forecasts["Chronos-2, 7 days"] = chronos_forecast(
            pipe, chronos_inputs(test, frame, scale, HISTORY_DAYS)
        )
        forecasts["Chronos-2, 4 weeks"] = chronos_forecast(
            pipe, chronos_inputs(test, frame, scale, LONG_CONTEXT_DAYS)
        )
    rows, daily = [], []
    for label, fc in forecasts.items():
        daily.append(
            pd.DataFrame(
                {
                    "region": region.key,
                    "series": name,
                    "model": label,
                    "target_day": test.target_day,
                    "mae": np.abs(fc - y).mean(axis=1),
                    "cold": test.cold,
                    "hot": test.hot,
                    "in_window": masks["paper window"],
                }
            )
        )
        for subset, mask in masks.items():
            rows.append(
                {
                    "region": region.key,
                    "series": name,
                    "model": label,
                    "subset": subset,
                    **score(y, fc, mask, offset),
                }
            )
    match["hot_days"], match["cold_days"] = int(test.hot.sum()), int(test.cold.sum())
    match["window_first_day"] = str(test.target_day[region.first])
    match["window_last_day"] = str(test.target_day[region.first + region.days - 1])
    match["window_hot_or_cold_days"] = int((test.hot | test.cold)[masks["paper window"]].sum())
    return rows, match, pd.concat(daily, ignore_index=True)


def published_table(region: Region) -> pd.DataFrame:
    """The paper's results workbook: reported window error per series and model."""
    raw = pd.read_excel(ESDF / "main_results.xlsx", sheet_name=region.sheet, header=None)
    names = raw.iloc[0].ffill().tolist()
    kinds = raw.iloc[1].tolist()
    body = raw.iloc[2:].reset_index(drop=True)
    rows = []
    for j in range(1, raw.shape[1]):
        if kinds[j] != "MAE":
            continue
        for i in range(len(body)):
            series = body.iloc[i, 0]
            if isinstance(series, str) and series.strip().lower() != "average":
                rows.append(
                    {
                        "region": region.key,
                        "series": series.strip(),
                        "model": str(names[j]).strip(),
                        "reported_mae": float(body.iloc[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def summarise(scores: pd.DataFrame) -> pd.DataFrame:
    """Mean error over the series of a region (as the paper averages), and in how many series a
    model has a lower error than ESDF."""
    ref = scores[scores["model"] == PROPOSED].set_index(["region", "series", "subset"])["mae"]
    s = scores.join(ref.rename("mae_esdf"), on=["region", "series", "subset"])
    s["beats_esdf"] = s["mae"] < s["mae_esdf"]
    out = (
        s.groupby(["region", "subset", "model"])
        .agg(
            series=("series", "nunique"),
            days=("days", "mean"),
            mae=("mae", "mean"),
            rmse=("rmse", "mean"),
            nmae_mean_pct=("nmae_mean_pct", "mean"),
            beats_esdf=("beats_esdf", "sum"),
        )
        .reset_index()
    )
    base = out[out["model"] == PROPOSED].set_index(["region", "subset"])["mae"]
    out["mae_vs_esdf_pct"] = 100 * (
        out["mae"] / out.set_index(["region", "subset"]).index.map(base).to_numpy() - 1
    )
    return out


#: the workbook's column names for the paper's models
PUBLISHED_NAMES = {
    "Proposed": PROPOSED,
    "MLP": "MLP",
    "LSTM": "LSTM",
    "CNN": "CNN",
    "NBEATS": "N-BEATS",
    "ImpactNet": "ImpactNet",
    "Informer": "Informer",
    "Autoformer": "Autoformer",
}
MODEL_ORDER = (
    "Week before",
    "MLP",
    "CNN",
    "LSTM",
    "LSTM + synthetic",
    "N-BEATS",
    "ImpactNet",
    "Informer",
    "Autoformer",
    "ESDF (MLP)",
    "ESDF (CNN)",
    PROPOSED,
    "Linear regression",
    "LightGBM",
    "Chronos-2, 7 days",
    "Chronos-2, 4 weeks",
)


#: ISO codes of the European series, for the public holidays of 2018
EUROPE_CODES = {
    "Belgium": "BE",
    "Croatia": "HR",
    "Denmark": "DK",
    "Finland": "FI",
    "France": "FR",
    "Germany": "DE",
    "Hungary": "HU",
    "Ireland": "IE",
    "Italy": "IT",
    "Latvia": "LV",
    "Lithuania": "LT",
    "Netherlands": "NL",
    "Norway": "NO",
    "Poland": "PL",
    "Romania": "RO",
    "Slovenia": "SI",
    "Sweden": "SE",
    "Switzerland": "CH",
}


def holiday_table() -> pd.DataFrame:
    """Error of every model on the public holidays of the European test period against its
    other days. No model of the paper gets the calendar; the references here do not get it
    either, so the table shows what the missing input costs."""
    import holidays

    d = pd.read_csv(OUT_DIR / "daily_errors.csv", parse_dates=["target_day"])
    d = d[d["region"] == "europe"].copy()
    flags = {
        name: set(holidays.country_holidays(code, years=[2018]).keys())
        for name, code in EUROPE_CODES.items()
    }
    d["holiday"] = [
        day.date() in flags.get(series, set())
        for series, day in zip(d["series"], d["target_day"], strict=True)
    ]
    per = (
        d.groupby(["model", "series", "holiday"])["mae"].agg(["mean", "sum", "size"]).reset_index()
    )
    wide = per.pivot(index=["model", "series"], columns="holiday", values=["mean", "sum", "size"])
    out = pd.DataFrame(
        {
            "holiday_mae_pct": 100 * wide[("mean", True)],
            "other_days_mae_pct": 100 * wide[("mean", False)],
            "holiday_share_of_days_pct": 100
            * wide[("size", True)]
            / (wide[("size", True)] + wide[("size", False)]),
            "holiday_share_of_error_pct": 100
            * wide[("sum", True)]
            / (wide[("sum", True)] + wide[("sum", False)]),
        }
    )
    out = out.groupby("model").mean()
    out["holiday_over_other"] = out["holiday_mae_pct"] / out["other_days_mae_pct"]
    return out.reindex([m for m in MODEL_ORDER if m in out.index]).reset_index()


def tables() -> dict[str, pd.DataFrame]:
    """Tables for the write-up from the saved scores: per region the error of every model by
    subset (per cent of the training range) with the number of series in which it is below
    ESDF, and the paper's reported window error next to the one recomputed from its saved
    forecasts."""
    summary = pd.read_csv(OUT_DIR / "summary.csv")
    published = pd.read_csv(OUT_DIR / "published_window_errors.csv")
    out = {}
    for key in REGIONS:
        t = summary[summary["region"] == key]
        mae = 100 * t.pivot(index="model", columns="subset", values="mae")
        wins = t.pivot(index="model", columns="subset", values="beats_esdf")
        wide = pd.DataFrame(
            {
                "window_mae_pct": mae["paper window"],
                "all_days_mae_pct": mae["all test days"],
                "hot_cold_days_mae_pct": mae["hot and cold days"],
                "other_days_mae_pct": mae["other days"],
                "window_vs_esdf_pct": 100
                * (mae["paper window"] / mae.loc[PROPOSED, "paper window"] - 1),
                "all_days_vs_esdf_pct": 100
                * (mae["all test days"] / mae.loc[PROPOSED, "all test days"] - 1),
                "series_below_esdf_window": wins["paper window"],
                "series_below_esdf_all_days": wins["all test days"],
                "series": t.groupby("model")["series"].max(),
            }
        )
        wide = wide.reindex([m for m in MODEL_ORDER if m in wide.index]).reset_index()
        wide.to_csv(OUT_DIR / f"table_{key}.csv", index=False)
        out[key] = wide
    window = summary[summary["subset"] == "paper window"].set_index(["region", "model"])["mae"]
    rep = published.groupby(["region", "model"])["reported_mae"].mean().reset_index()
    rep["model"] = rep["model"].map(PUBLISHED_NAMES)
    rep["recomputed_mae"] = [
        window.get((r, m), np.nan) for r, m in zip(rep["region"], rep["model"], strict=True)
    ]
    rep["difference_pct"] = 100 * (rep["recomputed_mae"] / rep["reported_mae"] - 1)
    rep.to_csv(OUT_DIR / "reproduction.csv", index=False)
    out["reproduction"] = rep
    out["holidays"] = holiday_table()
    out["holidays"].to_csv(OUT_DIR / "table_holidays_europe.csv", index=False)
    return out


def run(chronos: bool = True, threads: int = 8, limit: int | None = None) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pipe = None
    if chronos:
        import torch

        from ..models.foundation import load_chronos2

        pipe = load_chronos2("cpu")
        torch.set_num_threads(threads)
    rows, matches, days = [], [], []
    for region in REGIONS.values():
        names = series_names(region)[:limit] if limit else series_names(region)
        for name in names:
            t0 = time.time()
            r, m, d = run_series(region, name, pipe)
            rows.extend(r)
            matches.append(m)
            days.append(d)
            logger.info(
                "%s %s: %d samples in %.0f s", region.key, name, m["samples"], time.time() - t0
            )
    scores = pd.DataFrame(rows)
    match = pd.DataFrame(matches)
    summary = summarise(scores)
    published = pd.concat([published_table(r) for r in REGIONS.values()], ignore_index=True)
    pd.concat(days, ignore_index=True).to_csv(OUT_DIR / "daily_errors.csv", index=False)
    scores.to_csv(OUT_DIR / "scores.csv", index=False)
    match.to_csv(OUT_DIR / "samples_check.csv", index=False)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    published.to_csv(OUT_DIR / "published_window_errors.csv", index=False)
    tables()

    def table(region: str, subset: str) -> dict:
        t = summary[(summary["region"] == region) & (summary["subset"] == subset)]
        return {
            r["model"]: {
                "mae": round(r["mae"], 4),
                "vs_esdf_pct": round(r["mae_vs_esdf_pct"], 1),
                "series_better_than_esdf": int(r["beats_esdf"]),
                "of": int(r["series"]),
            }
            for _, r in t.iterrows()
        }

    out = {
        "samples_match_paper": {
            k: {
                "series": int(len(g)),
                "same_sample_count": int((g["samples"] == g["paper_samples"]).sum()),
                "largest_difference_in_true": float(g["max_abs_diff_true"].max()),
                "window": f"{g['window_first_day'].iloc[0]} to {g['window_last_day'].iloc[0]}",
                "window_days": int(REGIONS[k].days),
                "window_days_hot_or_cold_mean": float(g["window_hot_or_cold_days"].mean()),
                "test_days": float(g["samples"].mean()),
                "hot_days_mean": float(g["hot_days"].mean()),
                "cold_days_mean": float(g["cold_days"].mean()),
            }
            for k, g in match.groupby("region")
        },
        **{
            f"{region}: {subset}": table(region, subset) for region in REGIONS for subset in SUBSETS
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2, default=float) + "\n")
    return out
