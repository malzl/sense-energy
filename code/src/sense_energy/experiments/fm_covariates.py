"""Known-ahead covariates for the foundation models, as past/future windows.

For every (origin, site) the *past* window is the model's context (the last
``context_periods`` half-hours up to the issue time) and the *future* window
runs from the period after the issue time to the end of the target day, i.e.
the horizon the zero-shot runs already predict. Every covariate is known at
16:30 on D-1:

* local-clock calendar terms (half-hour of day and day of week as sine/cosine,
  weekend, bank holiday),
* the site's Octopus Agile price (published ~16:00 D-1 for D),
* NESO's day-ahead forecasts - the same ``_da`` columns LightGBM uses,
* weather: ERA5 over the past in both variants; over the future either ERA5
  again (``era5``: perfect weather, an upper bound) or the ensemble mean of
  the 00z IFS ENS run of D-1 (``ifs``: forecast weather). Where that run is
  missing the future weather falls back to persistence of the last known
  day. Radiation is left out: the IFS field is accumulated over the run and
  ERA5's is hourly, so the two are not the same covariate.

Chronos-2 takes the windows as named past/future covariates; TimesFM 3.0 as
one standardised (n_covariates, past + future) array; TabPFN-TS as extra
columns of its feature table.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..logging_utils import get_logger
from .features import Covariates, _holiday_set
from .poc import Origin, Panel

logger = get_logger(__name__)

CALENDAR = ("hod_sin", "hod_cos", "dow_sin", "dow_cos", "is_weekend", "is_holiday")
WEATHER_TO_IFS = {"t2m": "f_t2m_mean", "d2m": "f_d2m_mean", "wind": "f_wind_mean"}


def clean(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over gaps (edges held); all-missing becomes zeros."""
    x = np.asarray(x, dtype="float32").copy()
    ok = np.isfinite(x)
    if ok.all():
        return x
    if not ok.any():
        return np.zeros_like(x)
    idx = np.arange(len(x))
    x[~ok] = np.interp(idx[~ok], idx[ok], x[ok])
    return x


def standardise(past: dict, future: dict) -> tuple[dict, dict]:
    """z-score each covariate with the past window's mean and spread."""
    p, f = {}, {}
    for name, x in past.items():
        mu, sd = float(np.mean(x)), float(np.std(x))
        sd = sd if sd > 1e-6 else 1.0
        p[name] = ((x - mu) / sd).astype("float32")
        f[name] = ((future[name] - mu) / sd).astype("float32")
    return p, f


class CovariateWindows:
    """Past/future covariate windows on the panel grid for one weather variant."""

    def __init__(self, config: dict[str, Any], panel: Panel, weather: str):
        if weather not in ("era5", "ifs"):
            raise ValueError(f"weather must be 'era5' or 'ifs', got {weather!r}")
        self.panel, self.weather = panel, weather
        self.cov = Covariates(config, panel)
        c = config.get("covariates", {})
        self.weather_vars = [str(v) for v in c.get("weather_vars", ["t2m", "d2m", "wind"])]
        unknown = [v for v in self.weather_vars if v not in WEATHER_TO_IFS]
        if unknown:
            raise KeyError(f"no IFS counterpart for weather covariates {unknown}")
        self.shared: dict[str, np.ndarray] = {}  # (T,)
        self.per_site: dict[str, np.ndarray] = {}  # (T, S)
        if c.get("calendar", True):
            tod, dow = panel.tod.astype(float), panel.dow.astype(float)
            hol = _holiday_set(range(2020, 2028))
            self.shared["hod_sin"] = np.sin(2 * np.pi * tod / 48)
            self.shared["hod_cos"] = np.cos(2 * np.pi * tod / 48)
            self.shared["dow_sin"] = np.sin(2 * np.pi * dow / 7)
            self.shared["dow_cos"] = np.cos(2 * np.pi * dow / 7)
            self.shared["is_weekend"] = (dow >= 5).astype(float)
            self.shared["is_holiday"] = np.isin(panel.local_date, list(hol)).astype(float)
        if c.get("price", True):
            self.per_site["price"] = self.cov.price
        if c.get("neso", True):
            for i, col in enumerate(Covariates.NESO_COLS):
                self.shared[col] = self.cov.neso[:, i]
        for v in self.weather_vars:
            self.per_site[v] = self.cov.era5[v]
        self.names = list(self.shared) + list(self.per_site)
        self.n_ifs_missing = 0
        self.n_windows = 0

    def window(self, o: Origin, j: int, site: str, L: int, H: int) -> tuple[dict, dict]:
        """Past (length L) and future (length H) covariates for site ``j`` at origin ``o``."""
        T = len(self.panel.index)
        ctx = np.clip(np.arange(o.origin_pos - L + 1, o.origin_pos + 1), 0, T - 1)
        fut = np.clip(np.arange(o.origin_pos + 1, o.origin_pos + 1 + H), 0, T - 1)
        past = {n: a[ctx] for n, a in self.shared.items()}
        past |= {n: a[ctx, j] for n, a in self.per_site.items()}
        future = {n: a[fut] for n, a in self.shared.items()}
        future |= {n: a[fut, j] for n, a in self.per_site.items()}
        if self.weather == "ifs":
            block = self.cov.ifs_block(site, o, self.panel.index[fut].asi8)
            # persistence fallback: the same half-hour on the last day fully known at the origin
            back = np.clip(fut - 48 * np.ceil((fut - o.origin_pos) / 48).astype(int), 0, T - 1)
            for v in self.weather_vars:
                f = block[WEATHER_TO_IFS[v]].astype("float64")
                bad = ~np.isfinite(f)
                if bad.all():
                    self.n_ifs_missing += 1
                if bad.any():
                    f[bad] = self.per_site[v][back, j][bad]
                future[v] = f
        self.n_windows += 1
        return {n: clean(x) for n, x in past.items()}, {n: clean(x) for n, x in future.items()}

    def member_weather(self, o: Origin, j: int, site: str, H: int) -> dict[str, np.ndarray]:
        """Future weather per IFS member: var -> (members x H), persistence where missing."""
        T = len(self.panel.index)
        fut = np.clip(np.arange(o.origin_pos + 1, o.origin_pos + 1 + H), 0, T - 1)
        back = np.clip(fut - 48 * np.ceil((fut - o.origin_pos) / 48).astype(int), 0, T - 1)
        raw = self.cov.ifs_members(site, o, self.panel.index[fut].asi8)
        out = {}
        for v in self.weather_vars:
            arr = raw.get(v)
            if arr is None:
                arr = np.full((1, H), np.nan)
            fallback = self.per_site[v][back, j]
            arr = np.where(np.isfinite(arr), arr, fallback[None, :])
            out[v] = np.stack([clean(row) for row in arr])
        return out


def covariate_inputs(
    panel: Panel,
    keys: list[tuple[Origin, str]],
    config: dict[str, Any],
    weather: str,
    L: int,
    H: int,
    standardised: bool = False,
) -> tuple[list[tuple[dict, dict]], list[str]]:
    """Windows for every (origin, site) key in order, plus the covariate names."""
    cw = CovariateWindows(config, panel, weather)
    site_index = {s: j for j, s in enumerate(panel.sites)}
    out = []
    for o, s in keys:
        past, future = cw.window(o, site_index[s], s, L, H)
        out.append(standardise(past, future) if standardised else (past, future))
    logger.info(
        "%s covariates: %s for %d windows (%d without an IFS run, persistence used)",
        weather,
        cw.names,
        cw.n_windows,
        cw.n_ifs_missing,
    )
    return out, cw.names
