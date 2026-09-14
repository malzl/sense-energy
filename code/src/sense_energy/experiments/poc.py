"""Day-ahead proof of concept: data assembly, origins, baselines, scoring.

The experiment is a set of forecast *origins* (one per test day, at the
configured local issue time on D-1); at each origin every model forecasts the
48 half-hours of the target day for every selected site. Forecasts are stored
long - one row per (model, site, origin, target) with a quantile grid - and
scored only against observed (non-imputed) values.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger

logger = get_logger(__name__)

QUANTILE_COLS = {
    q: f"q{int(round(q * 100)):02d}" for q in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


@dataclass
class Panel:
    """The demand panel plus its local-time calendar, aligned on one UTC grid."""

    y_raw: pd.DataFrame  # observed only, NaN elsewhere
    y_imp: pd.DataFrame  # short gaps imputed
    index: pd.DatetimeIndex
    local: pd.DatetimeIndex
    local_date: np.ndarray  # datetime64[D] per row (local calendar day)
    tod: np.ndarray  # half-hour slot 0..47 (local)
    dow: np.ndarray  # 0=Mon (local)
    sites: list[str] = field(default_factory=list)
    site_meta: pd.DataFrame | None = None
    level: str = "site"
    #: series -> {site_code: demand weight}; the sites whose price/weather stand for a series
    members: dict[str, dict[str, float]] | None = None


AGGREGATE_LEVELS = ("trust", "region", "total")


def load_panel(config: dict[str, Any]) -> Panel:
    level = str(config.get("level", "site"))
    if level in AGGREGATE_LEVELS:
        from .hierarchy import aggregate_panel

        return aggregate_panel(load_panel({**config, "level": "site"}), level, config)
    base = PROCESSED_DIR / config["energy"] / level
    y_raw = pd.read_parquet(base / "wide_raw.parquet")
    y_imp = pd.read_parquet(base / "wide_imputed.parquet")
    y_raw.index = pd.to_datetime(y_raw.index, utc=True)
    y_imp.index = pd.to_datetime(y_imp.index, utc=True)
    y_imp = y_imp.reindex(columns=y_raw.columns)
    index = y_raw.index
    local = index.tz_convert(config["timezone"])
    panel = Panel(
        y_raw=y_raw,
        y_imp=y_imp,
        index=index,
        local=local,
        local_date=local.tz_localize(None).normalize().to_numpy().astype("datetime64[D]"),
        tod=(local.hour * 2 + local.minute // 30).to_numpy(),
        dow=local.dayofweek.to_numpy(),
    )
    panel.sites = select_sites(panel, config)
    idx = pd.read_parquet(base / "series_index.parquet").set_index("series_id")
    panel.site_meta = idx.reindex(panel.sites)
    panel.level = level
    if level == "meter":
        panel.members = {m: {str(idx.loc[m, "site_code"]): 1.0} for m in panel.sites}
    else:
        panel.members = {s: {s: 1.0} for s in panel.sites}
    return panel


def coverage_ok(col: pd.Series, index: pd.DatetimeIndex, config: dict[str, Any]) -> bool:
    """The PoC's selection rule for one series: observed share in the train and test
    windows and enough history before the test window."""
    tr = (index >= pd.Timestamp(config["train_start"], tz="UTC")) & (
        index <= pd.Timestamp(config["train_end"], tz="UTC") + pd.Timedelta(days=1)
    )
    te = (index >= pd.Timestamp(config["test_start"], tz="UTC")) & (
        index <= pd.Timestamp(config["test_end"], tz="UTC") + pd.Timedelta(days=1)
    )
    if (
        col[tr].notna().mean() < config["min_coverage_train"]
        or col[te].notna().mean() < config["min_coverage_test"]
    ):
        return False
    first = col.first_valid_index()
    return (
        first is not None
        and (pd.Timestamp(config["test_start"], tz="UTC") - first).days
        >= config["min_history_days"]
    )


def select_sites(panel: Panel, config: dict[str, Any]) -> list[str]:
    """Sites with enough observed history and enough observed test data."""
    idx = pd.read_parquet(
        PROCESSED_DIR / config["energy"] / config.get("level", "site") / "series_index.parquet"
    ).set_index("series_id")
    keep = []
    for s in panel.y_raw.columns:
        if not coverage_ok(panel.y_raw[s], panel.index, config):
            continue
        if (
            config.get("exclude_heavily_imputed", True)
            and s in idx.index
            and bool(idx.loc[s].get("imputed_heavily", False))
        ):
            continue
        keep.append(s)
    limit = config.get("site_limit")
    if limit:
        keep = keep[: int(limit)]
    logger.info("%d of %d sites selected", len(keep), panel.y_raw.shape[1])
    return keep


# --------------------------------------------------------------------------- #
# Origins and targets
# --------------------------------------------------------------------------- #


@dataclass
class Origin:
    target_day: np.datetime64  # local calendar day D
    origin_time: pd.Timestamp  # UTC, issue time on D-1
    origin_pos: int  # index of the last period at or before the issue time
    target_pos: np.ndarray  # positions of the target day's periods
    lead_hours: np.ndarray  # hours from origin to each target period


def make_origins(
    panel: Panel, config: dict[str, Any], start: str, end: str, stride_days: int
) -> list[Origin]:
    tz = config["timezone"]
    hh, mm = (int(x) for x in config["issue_time_local"].split(":"))
    lag = int(config.get("data_lag_periods", 0))
    days = pd.date_range(start, end, freq=f"{stride_days}D")
    out = []
    for d in days:
        target_day = np.datetime64(d.date(), "D")
        issue_local = (d - pd.Timedelta(days=1)).replace(hour=hh, minute=mm).tz_localize(tz)
        origin_time = issue_local.tz_convert("UTC")
        pos = int(panel.index.searchsorted(origin_time, side="right") - 1) - lag
        tpos = np.flatnonzero(panel.local_date == target_day)
        if pos < 0 or len(tpos) == 0 or tpos[0] <= pos:
            continue
        lead = (panel.index[tpos] - panel.index[pos]) / pd.Timedelta(hours=1)
        out.append(Origin(target_day, origin_time, pos, tpos, lead.to_numpy()))
    return out


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def _fill(series: np.ndarray) -> np.ndarray:
    """Fill remaining NaN in a context with the last value, then the median."""
    s = pd.Series(series).ffill()
    return s.fillna(s.median()).to_numpy()


def seasonal_naive(panel: Panel, o: Origin, site: str, weeks_back: int = 1) -> np.ndarray:
    """Same half-hour one week earlier (falls back a further week when missing)."""
    y = panel.y_imp[site].to_numpy()
    out = np.full(len(o.target_pos), np.nan)
    for k in range(weeks_back, weeks_back + 4):
        src = o.target_pos - 336 * k
        ok = np.isnan(out) & (src >= 0) & (src <= o.origin_pos)
        out[ok] = y[src[ok]]
        if not np.isnan(out).any():
            break
    return out


def profile_quantiles(
    panel: Panel, o: Origin, site: str, weeks: int, quantiles: list[float]
) -> np.ndarray:
    """Quantiles of the last ``weeks`` same-weekday, same-slot values (recent climatology)."""
    y = panel.y_imp[site].to_numpy()
    hist = np.stack(
        [y[np.clip(o.target_pos - 336 * k, 0, None)] for k in range(1, weeks + 1)]
    )  # weeks x targets
    valid_rows = (o.target_pos[None, :] - 336 * np.arange(1, weeks + 1)[:, None]) >= 0
    hist = np.where(valid_rows, hist, np.nan)
    with np.errstate(all="ignore"):
        q = np.nanquantile(hist, quantiles, axis=0)  # Q x targets
    return q.T


# --------------------------------------------------------------------------- #
# Forecast store and scoring
# --------------------------------------------------------------------------- #


def forecast_frame(
    model: str,
    site: str,
    o: Origin,
    panel: Panel,
    quantiles: list[float],
    q: np.ndarray,
    point: np.ndarray | None = None,
) -> pd.DataFrame:
    """One origin's forecast as long rows with the quantile grid."""
    n = len(o.target_pos)
    q = np.asarray(q, dtype="float32")
    if q.ndim == 1:
        q = np.repeat(q[:, None], len(quantiles), axis=1)
    frame = pd.DataFrame(
        {
            "model": model,
            "site_code": site,
            "origin": o.origin_time,
            "target": panel.index[o.target_pos],
            "lead_hours": o.lead_hours.astype("float32"),
            "point": (point if point is not None else q[:, quantiles.index(0.5)]).astype("float32"),
        }
    )
    for i, ql in enumerate(quantiles):
        frame[QUANTILE_COLS[ql]] = q[:, i]
    assert len(frame) == n
    return frame


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    d = y - q
    return np.maximum(tau * d, (tau - 1) * d)


def score(
    forecasts: pd.DataFrame,
    panel: Panel,
    quantiles: list[float],
    naive_model: str = "seasonal_naive",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-(model, site) and pooled scores against observed values only."""
    y_long = panel.y_raw[panel.sites].stack(future_stack=True).rename("y").reset_index()
    y_long.columns = ["target", "site_code", "y"]
    f = forecasts.merge(y_long, on=["target", "site_code"], how="left").dropna(subset=["y"])

    f["ae"] = (f["y"] - f["point"]).abs()
    f["se"] = (f["y"] - f["point"]) ** 2
    qcols = [QUANTILE_COLS[q] for q in quantiles]
    for q, c in zip(quantiles, qcols, strict=True):
        f[f"pb_{c}"] = pinball(f["y"].to_numpy(), f[c].to_numpy(), q)
    f["crps_q"] = 2 * f[[f"pb_{c}" for c in qcols]].mean(
        axis=1
    )  # CRPS approximated on the quantile grid
    lo, hi = QUANTILE_COLS[0.1], QUANTILE_COLS[0.9]
    f["in80"] = ((f["y"] >= f[lo]) & (f["y"] <= f[hi])).astype(float)
    f["width80"] = f[hi] - f[lo]

    per = (
        f.groupby(["model", "site_code"])
        .agg(
            n=("y", "size"),
            mae=("ae", "mean"),
            rmse=("se", lambda s: float(np.sqrt(s.mean()))),
            crps_q=("crps_q", "mean"),
            pinball_10=("pb_q10", "mean"),
            pinball_50=("pb_q50", "mean"),
            pinball_90=("pb_q90", "mean"),
            coverage_80=("in80", "mean"),
            width_80=("width80", "mean"),
            y_mean=("y", "mean"),
        )
        .reset_index()
    )
    per["nmae"] = per["mae"] / per["y_mean"]
    # skill vs the naive model, per site
    base = per[per["model"] == naive_model].set_index("site_code")
    per["mae_skill_vs_naive"] = 1 - per["mae"] / per["site_code"].map(base["mae"])
    per["crps_skill_vs_naive"] = 1 - per["crps_q"] / per["site_code"].map(base["crps_q"])

    pooled = (
        per.groupby("model")
        .agg(
            sites=("site_code", "nunique"),
            mae=("mae", "mean"),
            nmae=("nmae", "mean"),
            rmse=("rmse", "mean"),
            crps_q=("crps_q", "mean"),
            pinball_10=("pinball_10", "mean"),
            pinball_50=("pinball_50", "mean"),
            pinball_90=("pinball_90", "mean"),
            coverage_80=("coverage_80", "mean"),
            mae_skill_vs_naive_median=("mae_skill_vs_naive", "median"),
            crps_skill_vs_naive_median=("crps_skill_vs_naive", "median"),
        )
        .reset_index()
    )
    by_lead = (
        f.assign(lead_bucket=(f["lead_hours"] // 6 * 6).astype(int))
        .groupby(["model", "lead_bucket"])
        .agg(mae=("ae", "mean"), crps_q=("crps_q", "mean"), coverage_80=("in80", "mean"))
        .reset_index()
    )
    return per, pooled, by_lead


def outputs_dir(config: dict[str, Any], level: str | None = None) -> Path:
    """``outputs_dir`` for the site level; ``<outputs_dir>_<level>`` for any other level."""
    d = PROJECT_ROOT / config.get("outputs_dir", "code/reports/poc")
    level = level or str(config.get("level", "site"))
    if level != "site":
        d = d.with_name(f"{d.name}_{level}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_baselines(panel: Panel, origins: list[Origin], config: dict[str, Any]) -> pd.DataFrame:
    quantiles = list(config["quantiles"])
    frames = []
    for o in origins:
        for s in panel.sites:
            frames.append(
                forecast_frame(
                    "seasonal_naive", s, o, panel, quantiles, seasonal_naive(panel, o, s)
                )
            )
            frames.append(
                forecast_frame(
                    "profile_quantiles",
                    s,
                    o,
                    panel,
                    quantiles,
                    profile_quantiles(panel, o, s, int(config["profile_weeks"]), quantiles),
                )
            )
    out = pd.concat(frames, ignore_index=True)
    logger.info("baselines: %s forecast rows", f"{len(out):,}")
    return out
