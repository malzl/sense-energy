"""Direct multi-horizon feature table for the gradient-boosting model.

One row per (site, origin, target period). Every feature is something known
at the origin: calendar, demand lags and aggregates ending at the origin,
Agile price and NESO day-ahead covariates for the target period, and
weather at the target period from one of three sources:

* ``none``  - no weather
* ``era5``  - reanalysis at the target time: *perfect* weather, an upper bound
* ``ifs``   - the 00z IFS ENS run of D-1 (control, ensemble mean and spread)

Targets and lag features are scaled by the site's recent mean so one model
serves sites from 1 kW to 3 MW.
"""

from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd

from ..config import GEO_DIR, INTERIM_DIR
from ..logging_utils import get_logger
from .poc import Origin, Panel

logger = get_logger(__name__)

CATEGORICAL = ["site_code", "organisation_type", "gsp_group"]


def _holiday_set(years: range) -> set:
    return {np.datetime64(d, "D") for d in holidays.UK(subdiv="ENG", years=years)}


def calendar_block(panel: Panel, o: Origin, hol: set) -> pd.DataFrame:
    tp = o.target_pos
    dates = panel.local_date[tp]
    frame = pd.DataFrame(
        {
            "tod": panel.tod[tp].astype("int16"),
            "dow": panel.dow[tp].astype("int8"),
            "is_weekend": (panel.dow[tp] >= 5).astype("int8"),
            "is_holiday": np.isin(dates, list(hol)).astype("int8"),
            "month": panel.local[tp].month.to_numpy().astype("int8"),
            "doy_sin": np.sin(2 * np.pi * panel.local[tp].dayofyear.to_numpy() / 365.25).astype(
                "float32"
            ),
            "doy_cos": np.cos(2 * np.pi * panel.local[tp].dayofyear.to_numpy() / 365.25).astype(
                "float32"
            ),
            "lead_hours": o.lead_hours.astype("float32"),
        }
    )
    frame["is_holiday_prev"] = np.isin(dates - np.timedelta64(1, "D"), list(hol)).astype("int8")
    return frame


def demand_block(panel: Panel, o: Origin, site: str) -> tuple[pd.DataFrame, float]:
    """Lags and aggregates available at the origin, scaled by the site's recent mean."""
    y = panel.y_imp[site].to_numpy()
    tp, op = o.target_pos, o.origin_pos
    recent = y[max(0, op - 48 * 28 + 1) : op + 1]
    scale = (
        float(np.nanmean(recent))
        if np.isfinite(np.nanmean(recent)) and np.nanmean(recent) > 0
        else 1.0
    )
    lags = {f"lag_{k}w": y[tp - 336 * k] / scale for k in (1, 2, 3, 4)}
    slot_hist = np.stack([y[tp - 336 * k] for k in range(1, 5)])
    with np.errstate(all="ignore"):
        frame = pd.DataFrame(
            {
                **lags,
                "slot_mean_4w": np.nanmean(slot_hist, axis=0) / scale,
                "slot_min_4w": np.nanmin(slot_hist, axis=0) / scale,
                "slot_max_4w": np.nanmax(slot_hist, axis=0) / scale,
            }
        )
    last = y[op]
    frame["last_value"] = np.float32(last / scale if np.isfinite(last) else np.nan)
    frame["mean_24h"] = np.float32(np.nanmean(y[max(0, op - 47) : op + 1]) / scale)
    frame["mean_7d"] = np.float32(np.nanmean(y[max(0, op - 335) : op + 1]) / scale)
    frame["std_7d"] = np.float32(np.nanstd(y[max(0, op - 335) : op + 1]) / scale)
    frame["scale"] = np.float32(scale)
    return frame.astype("float32"), scale


# --------------------------------------------------------------------------- #
# Covariates at the target period
# --------------------------------------------------------------------------- #


class Covariates:
    """Target-time covariates precomputed on the panel grid, so a block is an array slice.

    A series' price and weather are those of its member sites (``panel.members``):
    the site itself at site level, the meter's site at meter level, and the
    demand-weighted mean over the sites of a trust / region / the nation above.
    Categorical statics come from the largest member, floor area is summed.
    """

    def __init__(self, config: dict[str, Any], panel: Panel):
        self.config = config
        self.panel = panel
        self.sites = panel.sites  # series ids at this level
        self.members = panel.members or {s: {s: 1.0} for s in self.sites}
        self.member_sites = sorted({m for d in self.members.values() for m in d})
        geo = pd.read_parquet(GEO_DIR / "sites_geo.parquet").set_index("site_code")
        # weight matrix (member sites x series), columns sum to one
        W = pd.DataFrame(0.0, index=self.member_sites, columns=self.sites)
        for s, d in self.members.items():
            for m, w in d.items():
                W.loc[m, s] = w
        self.W = W.to_numpy(dtype="float64") / np.maximum(W.to_numpy().sum(axis=0), 1e-12)
        largest = {s: max(d, key=d.get) for s, d in self.members.items()}
        self.member_region = geo["gsp_group"].reindex(self.member_sites)
        self.site_region = pd.Series(
            [geo["gsp_group"].get(largest[s]) for s in self.sites], index=self.sites
        )
        gia = geo["site_gross_internal_area"]
        self.site_static = pd.DataFrame(
            {
                "organisation_type": [geo["organisation_type"].get(largest[s]) for s in self.sites],
                "site_gross_internal_area": [
                    float(np.nansum([gia.get(m, np.nan) for m in d])) if d else np.nan
                    for d in (self.members[s] for s in self.sites)
                ],
            },
            index=self.sites,
        )
        self._price = None
        self._neso = None
        self._era5 = None
        self._ifs: dict[str, dict] = {}

    def _weighted(self, X: np.ndarray) -> np.ndarray:
        """(T x member sites) -> (T x series) weighted means, ignoring missing members."""
        ok = np.isfinite(X)
        num = np.where(ok, X, 0.0) @ self.W
        den = ok.astype("float64") @ self.W
        with np.errstate(invalid="ignore", divide="ignore"):
            out = num / den
        return np.where(den > 0, out, np.nan).astype("float32")

    # -- prices: (T x S) matrix on the panel grid --------------------------
    @property
    def price(self) -> np.ndarray:
        if self._price is None:
            a = pd.read_parquet(
                INTERIM_DIR / "prices_agile.parquet", columns=["datetime", "region", "rate_exc_vat"]
            )
            wide = a.pivot(index="datetime", columns="region", values="rate_exc_vat").reindex(
                self.panel.index
            )
            cols = [
                wide[r].to_numpy() if r in wide.columns else np.full(len(wide), np.nan)
                for r in self.member_region
            ]
            self._price = self._weighted(np.stack(cols, axis=1).astype("float64"))
        return self._price

    # -- NESO day-ahead-known: (T x k) on the panel grid ---------------------
    NESO_COLS = [
        "nd_forecast_da_mw",
        "embedded_solar_forecast_da_mw",
        "embedded_wind_forecast_da_mw",
        "nd_forecast_1d_peak_da_mw",
    ]

    @property
    def neso(self) -> np.ndarray:
        if self._neso is None:
            n = pd.read_parquet(
                INTERIM_DIR / "neso_covariates_halfhourly.parquet",
                columns=["datetime", *self.NESO_COLS],
            ).set_index("datetime")
            self._neso = n.reindex(self.panel.index)[self.NESO_COLS].to_numpy(dtype="float32")
        return self._neso

    # -- ERA5 perfect weather: dict var -> (T x S) on the panel grid ----------
    ERA5_VARS = ("t2m", "d2m", "ssrd", "wind")

    @property
    def era5(self) -> dict[str, np.ndarray]:
        if self._era5 is None:
            w = pd.read_parquet(
                INTERIM_DIR / "weather_reanalysis.parquet",
                columns=["site_code", "datetime", "t2m", "ssrd", "u10", "v10", "d2m"],
            )
            w = w[w["site_code"].isin(self.member_sites)]
            w["wind"] = np.hypot(w["u10"], w["v10"])
            w["t2m"] -= 273.15
            w["d2m"] -= 273.15
            grid = self.panel.index.asi8
            per_site = {
                v: np.full((len(grid), len(self.member_sites)), np.nan, dtype="float64")
                for v in self.ERA5_VARS
            }
            for j, s in enumerate(self.member_sites):
                g = w[w["site_code"] == s].sort_values("datetime")
                if g.empty:
                    continue
                x = g["datetime"].to_numpy().astype("datetime64[ns]").astype("int64")
                for v in self.ERA5_VARS:
                    per_site[v][:, j] = np.interp(grid, x, g[v].to_numpy())
            self._era5 = {v: self._weighted(per_site[v]) for v in self.ERA5_VARS}
        return self._era5

    # -- IFS ENS forecast weather: per (site, run) 6-hourly aggregates ----------
    IFS_COLS = ("t2m_ctrl", "t2m_mean", "t2m_std", "wind_mean", "ssrd_mean", "d2m_mean")

    def _ifs_month(self, ym: str) -> dict:
        if ym not in self._ifs:
            p = INTERIM_DIR / "forecast_ifs_ens" / f"{ym}.parquet"
            table: dict = {}
            if p.exists():
                import pyarrow.parquet as pq

                wanted = [
                    "site_code",
                    "run_time",
                    "valid_time",
                    "member",
                    "t2m",
                    "u10",
                    "v10",
                    "ssrd",
                    "d2m",
                ]
                present = set(pq.read_schema(p).names)
                f = pd.read_parquet(p, columns=[c for c in wanted if c in present])
                for c in wanted:  # the 0.4-degree era lacks dewpoint and radiation
                    if c not in f.columns:
                        f[c] = np.nan
                f = f[f["site_code"].isin(self.member_sites)]
                f["wind"] = np.hypot(f["u10"], f["v10"])
                g = f.groupby(["site_code", "run_time", "valid_time"], observed=True)
                agg = pd.DataFrame(
                    {
                        "t2m_mean": g["t2m"].mean(),
                        "t2m_std": g["t2m"].std(),
                        "wind_mean": g["wind"].mean(),
                        "ssrd_mean": g["ssrd"].mean(),
                        "d2m_mean": g["d2m"].mean(),
                    }
                )
                ctrl = (
                    f[f["member"] == 0]
                    .set_index(["site_code", "run_time", "valid_time"])["t2m"]
                    .rename("t2m_ctrl")
                )
                agg = agg.join(ctrl, how="left").reset_index()
                for c in ("t2m_ctrl", "t2m_mean", "d2m_mean"):
                    agg[c] -= 273.15
                for (s, r), block in agg.groupby(["site_code", "run_time"], observed=True):
                    block = block.sort_values("valid_time")
                    table[(s, r)] = (
                        block["valid_time"].to_numpy().astype("datetime64[ns]").astype("int64"),
                        {c: block[c].to_numpy() for c in self.IFS_COLS},
                    )
            self._ifs[ym] = table
        return self._ifs[ym]

    def _ifs_site_block(
        self, site: str, o: Origin, targets_ns: np.ndarray
    ) -> dict[str, np.ndarray]:
        run_day = pd.Timestamp(o.target_day) - pd.Timedelta(days=1)
        run_time = pd.Timestamp(
            f"{run_day:%Y-%m-%d}T{self.config['weather']['ifs_run_time']}:00", tz="UTC"
        )
        entry = self._ifs_month(f"{run_day:%Y-%m}").get((site, run_time))
        n = len(targets_ns)
        if entry is None:
            return {f"f_{c}": np.full(n, np.nan, dtype="float32") for c in self.IFS_COLS} | {
                "f_t2m_daymean": np.full(n, np.nan, dtype="float32")
            }
        x, cols = entry
        out = {}
        for c in self.IFS_COLS:
            v = cols[c]
            ok = np.isfinite(v)
            out[f"f_{c}"] = (
                np.interp(targets_ns, x[ok], v[ok]).astype("float32")
                if ok.sum() >= 2
                else np.full(n, np.nan, dtype="float32")
            )
        out["f_t2m_daymean"] = np.full(n, np.float32(np.nanmean(out["f_t2m_mean"])))
        return out

    # -- IFS ENS per member: (site, run) -> valid times, member ids, var -> (members x steps)
    MEMBER_VARS = ("t2m", "d2m", "wind")

    def _ifs_members_month(self, ym: str) -> dict:
        if not hasattr(self, "_ifs_members"):
            self._ifs_members: dict[str, dict] = {}
        if ym not in self._ifs_members:
            p = INTERIM_DIR / "forecast_ifs_ens" / f"{ym}.parquet"
            table: dict = {}
            if p.exists():
                import pyarrow.parquet as pq

                wanted = [
                    "site_code",
                    "run_time",
                    "valid_time",
                    "member",
                    "t2m",
                    "u10",
                    "v10",
                    "d2m",
                ]
                present = set(pq.read_schema(p).names)
                f = pd.read_parquet(p, columns=[c for c in wanted if c in present])
                for c in wanted:
                    if c not in f.columns:
                        f[c] = np.nan
                f = f[f["site_code"].isin(self.member_sites)]
                f["wind"] = np.hypot(f["u10"], f["v10"])
                f["t2m"] -= 273.15
                f["d2m"] -= 273.15
                for (s, r), block in f.groupby(["site_code", "run_time"], observed=True):
                    piv = {
                        v: block.pivot_table(index="member", columns="valid_time", values=v)
                        for v in self.MEMBER_VARS
                    }
                    valid = piv["t2m"].columns
                    table[(s, r)] = (
                        valid.to_numpy().astype("datetime64[ns]").astype("int64"),
                        piv["t2m"].index.to_numpy(),
                        {
                            v: piv[v].reindex(columns=valid).to_numpy(dtype="float64")
                            for v in self.MEMBER_VARS
                        },
                    )
            self._ifs_members[ym] = table
        return self._ifs_members[ym]

    def ifs_members(self, series: str, o: Origin, targets_ns: np.ndarray) -> dict[str, np.ndarray]:
        """Per-member forecast weather at the targets for a series: var -> (members x targets),
        demand-weighted over member sites; NaN where the run is missing."""
        members = self.members.get(series, {series: 1.0})
        run_day = pd.Timestamp(o.target_day) - pd.Timedelta(days=1)
        run_time = pd.Timestamp(
            f"{run_day:%Y-%m-%d}T{self.config['weather']['ifs_run_time']}:00", tz="UTC"
        )
        table = self._ifs_members_month(f"{run_day:%Y-%m}")
        acc: dict[str, list] = {v: [] for v in self.MEMBER_VARS}
        ws = []
        for site, w in members.items():
            entry = table.get((site, run_time))
            if entry is None:
                continue
            x, _, cols = entry
            ws.append(w)
            for v in self.MEMBER_VARS:
                arr = cols[v]
                out = np.full((arr.shape[0], len(targets_ns)), np.nan)
                for m in range(arr.shape[0]):
                    ok = np.isfinite(arr[m])
                    if ok.sum() >= 2:
                        out[m] = np.interp(targets_ns, x[ok], arr[m][ok])
                acc[v].append(out)
        if not ws:
            return {}
        wv = np.array(ws)[:, None, None]
        result = {}
        for v in self.MEMBER_VARS:
            stack = np.stack(acc[v])  # sites x members x targets
            ok = np.isfinite(stack)
            den = (ok * wv).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                num = np.where(ok, stack * wv, 0.0).sum(axis=0) / den
            result[v] = np.where(den > 0, num, np.nan)
        return result

    def ifs_block(self, series: str, o: Origin, targets_ns: np.ndarray) -> dict[str, np.ndarray]:
        """IFS ENS features for a series: its site's block, or the demand-weighted mean
        over its member sites (missing members ignored)."""
        members = self.members.get(series, {series: 1.0})
        if len(members) == 1:
            return self._ifs_site_block(next(iter(members)), o, targets_ns)
        blocks = [(w, self._ifs_site_block(m, o, targets_ns)) for m, w in members.items()]
        ws = np.array([w for w, _ in blocks], dtype="float64")[:, None]
        out = {}
        for key in blocks[0][1]:
            vals = np.stack([b[key].astype("float64") for _, b in blocks])
            ok = np.isfinite(vals)
            den = (ok * ws).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                num = np.where(ok, vals * ws, 0.0).sum(axis=0) / den
            out[key] = np.where(den > 0, num, np.nan).astype("float32")
        return out


def build_rows(
    panel: Panel,
    origins: list[Origin],
    cov: Covariates,
    weather: str,
    hol: set,
    with_target: bool = True,
) -> pd.DataFrame:
    """The full feature table for a set of origins, vectorised across sites per origin."""
    sites = panel.sites
    S = len(sites)
    Y = panel.y_imp[sites].to_numpy(dtype="float32")  # T x S
    Yraw = panel.y_raw[sites].to_numpy(dtype="float32")
    site_idx = np.arange(S)
    frames = []
    for o in origins:
        tp, op = o.target_pos, o.origin_pos
        n = len(tp)
        cal = calendar_block(panel, o, hol)
        # site-level scale from the last 28 days
        with np.errstate(all="ignore"):
            scale = np.nanmean(Y[max(0, op - 48 * 28 + 1) : op + 1], axis=0)
        scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0).astype("float32")
        lag = {f"lag_{k}w": Y[tp - 336 * k] / scale for k in (1, 2, 3, 4)}  # each n x S
        hist = np.stack([Y[tp - 336 * k] for k in range(1, 5)])  # 4 x n x S
        with np.errstate(all="ignore"):
            slot_mean = np.nanmean(hist, axis=0) / scale
            slot_min = np.nanmin(hist, axis=0) / scale
            slot_max = np.nanmax(hist, axis=0) / scale
            last = Y[op] / scale
            mean_24h = np.nanmean(Y[max(0, op - 47) : op + 1], axis=0) / scale
            mean_7d = np.nanmean(Y[max(0, op - 335) : op + 1], axis=0) / scale
            std_7d = np.nanstd(Y[max(0, op - 335) : op + 1], axis=0) / scale
        block = {
            **{
                c: np.repeat(cal[c].to_numpy()[:, None], S, axis=1).ravel(order="F")
                for c in cal.columns
            },
            **{k: v.ravel(order="F") for k, v in lag.items()},
            "slot_mean_4w": slot_mean.ravel(order="F"),
            "slot_min_4w": slot_min.ravel(order="F"),
            "slot_max_4w": slot_max.ravel(order="F"),
            "last_value": np.repeat(last, n),
            "mean_24h": np.repeat(mean_24h, n),
            "mean_7d": np.repeat(mean_7d, n),
            "std_7d": np.repeat(std_7d, n),
            "scale": np.repeat(scale, n),
            "price": cov.price[tp].ravel(order="F"),
        }
        for i, c in enumerate(cov.NESO_COLS):
            block[c] = np.tile(cov.neso[tp, i], S)
        if weather == "era5":
            e = cov.era5
            for v in cov.ERA5_VARS:
                block[f"w_{v}"] = e[v][tp].ravel(order="F")
            day = e["t2m"][tp]  # n x S
            with np.errstate(all="ignore"):
                block["w_t2m_daymean"] = np.repeat(np.nanmean(day, axis=0), n)
                block["w_t2m_daymin"] = np.repeat(np.nanmin(day, axis=0), n)
                block["w_t2m_daymax"] = np.repeat(np.nanmax(day, axis=0), n)
        elif weather == "ifs":
            targets_ns = panel.index[tp].asi8
            parts = [cov.ifs_block(s, o, targets_ns) for s in sites]
            for c in parts[0]:
                block[c] = np.concatenate([p[c] for p in parts])
        frame = pd.DataFrame(block)
        frame["site_code"] = np.repeat(np.array(sites), n)
        frame["organisation_type"] = cov.site_static["organisation_type"].to_numpy()[
            np.repeat(site_idx, n)
        ]
        frame["gsp_group"] = cov.site_region.to_numpy()[np.repeat(site_idx, n)]
        frame["gia"] = cov.site_static["site_gross_internal_area"].to_numpy(dtype="float32")[
            np.repeat(site_idx, n)
        ]
        frame["origin"] = o.origin_time
        frame["target"] = np.tile(panel.index[tp].to_numpy(), S)
        if with_target:
            frame["y"] = (Yraw[tp] / scale).ravel(order="F")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    out["target"] = pd.to_datetime(out["target"], utc=True)
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out
