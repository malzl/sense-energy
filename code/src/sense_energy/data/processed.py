"""interim -> processed: clean, analysis-ready demand datasets.

For each energy type (electricity, gas) and level of the hierarchy (meter,
site, trust) this writes a long table, a wide matrix (time x series) and a
series index, in two flavours: ``raw`` (missing left as NaN) and ``imputed``
(short and medium gaps filled by profile-KNN, see below). Everything the build
does - parameters, counts, exclusions, file hashes - is written to
``doc/data_manifest.md`` and ``processed/manifest.json`` by the same run.

Imputation follows the smart-meter literature's standard recipe (Peppanen et
al. 2016, *Handling bad or missing smart meter data through advanced data
imputation*): each day of a series is a 48-slot profile; a day with gaps is
matched to its *k* nearest complete days of the same series and day type within
a window, and the gaps are filled from those neighbours, scaled to the day's
observed level. Guard-rails keep it honest: only days that are at least
partly observed, no gap longer than a day, and any series that would need more
than a set share imputed is flagged and left out of the imputed matrix.

Totals at site and trust level are formed only when every constituent has a
value; a partial sum is not a total.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import GEO_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..logging_utils import get_logger
from .make_dataset import load_interim, load_sites

logger = get_logger(__name__)

PERIODS_PER_DAY = 48
SERIES_KEY = {"meter": "mpxn", "site": "site_code", "trust": "trust_id"}


# --------------------------------------------------------------------------- #
# Profile-KNN imputation
# --------------------------------------------------------------------------- #


@dataclass
class ImputationStats:
    n_values: int = 0
    n_missing_before: int = 0
    n_imputed: int = 0
    n_days_imputed: int = 0
    n_days_skipped_too_sparse: int = 0
    n_days_skipped_long_gap: int = 0
    n_days_skipped_no_neighbours: int = 0

    @property
    def imputed_fraction(self) -> float:
        return self.n_imputed / self.n_values if self.n_values else 0.0


def _longest_run(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


def impute_profile_knn(
    values: np.ndarray,
    day_type: np.ndarray,
    k: int = 5,
    window_days: int = 56,
    min_observed_fraction: float = 0.25,
    max_gap_periods: int = 48,
    level_scaling: bool = True,
) -> tuple[np.ndarray, np.ndarray, ImputationStats]:
    """Fill gaps in a (n_days, 48) profile matrix from nearest complete days.

    Returns the filled matrix, a boolean matrix of which cells were imputed,
    and the bookkeeping. Whole missing days are never filled.
    """
    m = values.astype(float).copy()
    n_days = m.shape[0]
    missing = np.isnan(m)
    imputed = np.zeros_like(missing)
    stats = ImputationStats(n_values=m.size, n_missing_before=int(missing.sum()))
    complete = ~missing.any(axis=1)
    flat_missing = missing.ravel()

    for d in np.flatnonzero(missing.any(axis=1)):
        obs = ~missing[d]
        if obs.mean() < min_observed_fraction:
            stats.n_days_skipped_too_sparse += 1
            continue
        lo, hi = (
            max(0, d * PERIODS_PER_DAY - max_gap_periods),
            min(flat_missing.size, (d + 1) * PERIODS_PER_DAY + max_gap_periods),
        )
        if _longest_run(flat_missing[lo:hi]) > max_gap_periods:
            stats.n_days_skipped_long_gap += 1
            continue
        lo_d, hi_d = max(0, d - window_days), min(n_days, d + window_days + 1)
        cand = np.flatnonzero(complete[lo_d:hi_d] & (day_type[lo_d:hi_d] == day_type[d])) + lo_d
        cand = cand[cand != d]
        if len(cand) < k:
            stats.n_days_skipped_no_neighbours += 1
            continue
        dist = np.sqrt(np.mean((values[cand][:, obs] - values[d, obs]) ** 2, axis=1))
        nn = cand[np.argsort(dist, kind="stable")[:k]]
        neighbour_profile = values[nn].mean(axis=0)
        scale = 1.0
        if level_scaling:
            denom = neighbour_profile[obs].mean()
            if np.isfinite(denom) and denom > 0:
                scale = float(np.clip(values[d, obs].mean() / denom, 0.25, 4.0))
        fill = ~obs
        m[d, fill] = neighbour_profile[fill] * scale
        imputed[d, fill] = True
        stats.n_imputed += int(fill.sum())
        stats.n_days_imputed += 1
    return m, imputed, stats


def impute_series(
    series: pd.Series, cfg: dict[str, Any]
) -> tuple[pd.Series, pd.Series, ImputationStats]:
    """Profile-KNN on one series indexed by a complete 30-min UTC grid."""
    idx = series.index
    n_pad = (-len(idx)) % PERIODS_PER_DAY
    vals = np.concatenate([series.to_numpy(dtype=float), np.full(n_pad, np.nan)])
    days = vals.reshape(-1, PERIODS_PER_DAY)
    day_starts = idx[::PERIODS_PER_DAY]
    weekday = day_starts.tz_convert("Europe/London").dayofweek.to_numpy()
    day_type = (
        (weekday >= 5).astype(int)
        if cfg.get("same_day_type", True)
        else np.zeros(len(day_starts), dtype=int)
    )

    filled, imputed, stats = impute_profile_knn(
        days,
        day_type,
        k=int(cfg.get("k", 5)),
        window_days=int(cfg.get("window_weeks", 8)) * 7,
        min_observed_fraction=float(cfg.get("min_observed_fraction_of_day", 0.25)),
        max_gap_periods=int(cfg.get("max_gap_periods", 48)),
        level_scaling=bool(cfg.get("level_scaling", True)),
    )
    n = len(idx)
    return (
        pd.Series(filled.ravel()[:n], index=idx, name=series.name),
        pd.Series(imputed.ravel()[:n], index=idx, name="is_imputed"),
        stats,
    )


# --------------------------------------------------------------------------- #
# Levels
# --------------------------------------------------------------------------- #


def _trust_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_").upper()[:60]


def meter_frame(interim: pd.DataFrame, energy_type: str, cfg: dict[str, Any]) -> pd.DataFrame:
    """Meter-level long frame on a common grid: series_id, datetime, kwh, quality columns."""
    df = interim[interim["energy_type"] == energy_type].copy()
    if cfg.get("outliers_as_missing", True):
        df.loc[df["is_outlier"], "consumption_kwh"] = np.nan
    df = df.rename(columns={"mpxn": "series_id", "consumption_kwh": "kwh"})
    df["is_observed"] = df["kwh"].notna() & ~df["is_interpolated"]
    return df[
        [
            "series_id",
            "site_code",
            "datetime",
            "kwh",
            "is_observed",
            "is_interpolated",
            "is_estimated",
        ]
    ]


def to_wide(
    long: pd.DataFrame, value: str = "kwh", grid: pd.DatetimeIndex | None = None
) -> pd.DataFrame:
    wide = long.pivot(index="datetime", columns="series_id", values=value).sort_index()
    if grid is not None:
        wide = wide.reindex(grid)
    wide.index.name = "datetime"
    wide.columns.name = None
    return wide


def aggregate_wide(
    wide: pd.DataFrame, mapping: pd.Series, min_member_coverage: float = 0.1
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum columns by group; a total is NaN unless every *active* member is present.

    A member is active when at least ``min_member_coverage`` of its values are
    present in the window. Without that floor, a meter that reported for a few
    weeks and then died would block its site's total for the whole window (the
    Royal Devon & Exeter site lost all of 2024 to a 1 %-coverage meter).
    Returns the totals and, per group, the number of active members.
    """
    groups = mapping.reindex(wide.columns)
    totals = {}
    n_members = {}
    for gid, cols in groups.groupby(groups).groups.items():
        block = wide[list(cols)]
        active = block.columns[block.notna().mean() >= min_member_coverage]
        if len(active) == 0:
            active = block.columns[block.notna().any()]
        block = block[active]
        complete = block.notna().all(axis=1)
        s = block.sum(axis=1, min_count=1).where(complete)
        totals[gid] = s
        n_members[gid] = len(active)
    out = pd.DataFrame(totals).sort_index()
    out.index.name = "datetime"
    return out, pd.Series(n_members, name="n_members")


def series_index(
    wide_raw: pd.DataFrame,
    wide_imp: pd.DataFrame | None,
    meta: pd.DataFrame,
    stats: dict[str, ImputationStats] | None,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    max_frac = float(cfg["imputation"].get("max_imputed_fraction_per_series", 0.1))
    for sid in wide_raw.columns:
        s = wide_raw[sid]
        present = s.notna()
        first, last = (
            (present.idxmax(), present[::-1].idxmax()) if present.any() else (pd.NaT, pd.NaT)
        )
        span = int(((last - first) / pd.Timedelta("30min")) + 1) if present.any() else 0
        row = {
            "series_id": sid,
            "first": first,
            "last": last,
            "n_periods_span": span,
            "n_observed": int(present.sum()),
            "coverage_in_span": float(present.sum() / span) if span else 0.0,
            "coverage_in_grid": float(present.mean()),
            "mean_kwh": float(s.mean()) if present.any() else np.nan,
            "max_kwh": float(s.max()) if present.any() else np.nan,
        }
        if stats is not None and sid in stats:
            st = stats[sid]
            row.update(
                {
                    "n_imputed": st.n_imputed,
                    "imputed_fraction": st.imputed_fraction,
                    "days_imputed": st.n_days_imputed,
                    "days_skipped_sparse": st.n_days_skipped_too_sparse,
                    "days_skipped_long_gap": st.n_days_skipped_long_gap,
                    "days_skipped_no_neighbours": st.n_days_skipped_no_neighbours,
                    "imputed_heavily": st.imputed_fraction > max_frac,
                }
            )
        if wide_imp is not None and sid in wide_imp.columns:
            row["coverage_in_grid_imputed"] = float(wide_imp[sid].notna().mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.merge(meta, on="series_id", how="left")


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


@dataclass
class BuildRecord:
    steps: list[dict[str, Any]] = field(default_factory=list)
    files: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, Any] = field(default_factory=dict)

    def step(self, title: str, **detail: Any) -> None:
        self.steps.append({"n": len(self.steps) + 1, "title": title, **detail})
        logger.info(
            "[%d] %s %s",
            len(self.steps),
            title,
            {k: v for k, v in detail.items() if not isinstance(v, (list, dict))},
        )


def _sha(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _write(
    frame: pd.DataFrame, path: Path, record: BuildRecord, csv: bool = False, index: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=index, compression="zstd")
    record.files.append(
        {
            "path": str(path.relative_to(PROCESSED_DIR)),
            "rows": int(len(frame)),
            "cols": int(frame.shape[1]),
            "mb": round(path.stat().st_size / 1e6, 2),
            "sha256_12": _sha(path),
        }
    )
    if csv:
        cpath = path.with_suffix(".csv")
        frame.to_csv(cpath, index=index, float_format="%.4f")
        record.files.append(
            {
                "path": str(cpath.relative_to(PROCESSED_DIR)),
                "rows": int(len(frame)),
                "cols": int(frame.shape[1]),
                "mb": round(cpath.stat().st_size / 1e6, 2),
                "sha256_12": _sha(cpath),
            }
        )


def build(config: dict[str, Any]) -> BuildRecord:
    record = BuildRecord()
    imp_cfg = config["imputation"]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    interim = load_interim()
    sites = load_sites()
    record.step(
        "Load interim",
        rows=int(len(interim)),
        meters=int(interim["mpxn"].nunique()),
        sites=int(interim["site_code"].nunique()),
        window=[str(interim["datetime"].min()), str(interim["datetime"].max())],
        source="interim/consumption_halfhourly.parquet",
    )

    pattern = config.get("exclude_site_code_regex")
    if pattern:
        drop = interim["site_code"].astype(str).str.contains(pattern, regex=True)
        dropped_sites = sorted(interim.loc[drop, "site_code"].unique())
        interim = interim[~drop]
        record.step(
            "Exclude placeholder sites",
            regex=pattern,
            sites_dropped=dropped_sites,
            rows_dropped=int(drop.sum()),
        )

    grid = pd.date_range(
        interim["datetime"].min(), interim["datetime"].max(), freq="30min", tz="UTC"
    )
    record.step(
        "Common time grid",
        start=str(grid[0]),
        end=str(grid[-1]),
        periods=int(len(grid)),
        timezone="UTC, period start",
    )

    site_meta = sites.set_index("site_code")
    site_to_trust = site_meta["organisation_name"].map(_trust_id)

    # site attributes for clustering
    geo = (
        pd.read_parquet(GEO_DIR / "sites_geo.parquet")
        if (GEO_DIR / "sites_geo.parquet").exists()
        else sites
    )
    attrs = geo[~geo["site_code"].astype(str).str.contains(pattern or r"$^", regex=True)].copy()
    attrs["trust_id"] = attrs["organisation_name"].map(_trust_id)
    _write(
        attrs, PROCESSED_DIR / "site_attributes.parquet", record, csv=config.get("write_csv", True)
    )
    record.step("Site attributes", rows=int(len(attrs)), columns=list(attrs.columns))

    for energy in config["energy_types"]:
        long = meter_frame(interim, energy, config)
        if long.empty:
            continue
        n_outliers = int((interim["energy_type"].eq(energy) & interim["is_outlier"]).sum())
        record.step(
            f"{energy}: meter frame",
            meters=int(long["series_id"].nunique()),
            outliers_set_missing=n_outliers if config.get("outliers_as_missing", True) else 0,
        )

        wide_raw = to_wide(long, grid=grid)
        meta = long.groupby("series_id").agg(site_code=("site_code", "first")).reset_index()
        meta["trust_id"] = meta["site_code"].map(site_to_trust)
        meta = meta.merge(
            site_meta[
                ["site_name", "organisation_name", "organisation_type", "commissioning_region"]
            ],
            left_on="site_code",
            right_index=True,
            how="left",
        )

        # ---- impute at meter level
        wide_imp = wide_raw.copy()
        flags = pd.DataFrame(False, index=grid, columns=wide_raw.columns)
        stats: dict[str, ImputationStats] = {}
        for sid in wide_raw.columns:
            filled, imputed, st = impute_series(wide_raw[sid], imp_cfg)
            wide_imp[sid] = filled
            flags[sid] = imputed
            stats[sid] = st
        heavy = [
            s
            for s, st in stats.items()
            if st.imputed_fraction > float(imp_cfg["max_imputed_fraction_per_series"])
        ]
        total_missing = sum(st.n_missing_before for st in stats.values())
        total_imp = sum(st.n_imputed for st in stats.values())
        record.step(
            f"{energy}: profile-KNN imputation at meter level",
            **{
                k: imp_cfg[k]
                for k in (
                    "method",
                    "k",
                    "window_weeks",
                    "same_day_type",
                    "min_observed_fraction_of_day",
                    "max_gap_periods",
                    "level_scaling",
                    "max_imputed_fraction_per_series",
                )
            },
            cells_missing_before=int(total_missing),
            cells_imputed=int(total_imp),
            share_of_missing_filled=round(total_imp / total_missing, 4) if total_missing else 0.0,
            days_skipped={
                "too_sparse": sum(s.n_days_skipped_too_sparse for s in stats.values()),
                "gap_over_24h": sum(s.n_days_skipped_long_gap for s in stats.values()),
                "no_neighbours": sum(s.n_days_skipped_no_neighbours for s in stats.values()),
            },
            series_flagged_heavily_imputed=heavy,
        )

        idx_meter = series_index(wide_raw, wide_imp, meta, stats, config)
        min_cov = float(config.get("min_coverage_for_wide", 0.0))
        keep_raw = idx_meter.loc[idx_meter["coverage_in_grid"] >= min_cov, "series_id"].tolist()
        keep_imp = idx_meter.loc[
            (idx_meter["coverage_in_grid_imputed"] >= min_cov) & ~idx_meter["imputed_heavily"],
            "series_id",
        ].tolist()

        levels: dict[
            str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        ] = {}
        levels["meter"] = (wide_raw[keep_raw], wide_imp[keep_imp], flags, idx_meter, meta)

        # ---- site and trust totals: only when every active member has a value
        if "site" in config["levels"] or "trust" in config["levels"]:
            mapping_site = meta.set_index("series_id")["site_code"]
            min_cov = float(config.get("min_member_coverage", 0.1))
            inactive = [c for c in wide_raw.columns if wide_raw[c].notna().mean() < min_cov]
            site_raw, n_site = aggregate_wide(wide_raw, mapping_site, min_cov)
            site_imp, _ = aggregate_wide(wide_imp[keep_imp], mapping_site, min_cov)
            record.step(
                f"{energy}: inactive meters excluded from totals",
                min_member_coverage=min_cov,
                meters=inactive,
            )
            site_flag_frac = (flags.T.groupby(mapping_site).mean().T).reindex(
                columns=site_imp.columns
            )
            site_meta_frame = (
                pd.DataFrame({"series_id": site_raw.columns})
                .merge(
                    site_meta.reset_index()[
                        [
                            "site_code",
                            "site_name",
                            "organisation_name",
                            "organisation_type",
                            "commissioning_region",
                            "site_gross_internal_area",
                            "site_use_type",
                        ]
                    ],
                    left_on="series_id",
                    right_on="site_code",
                    how="left",
                )
                .drop(columns="site_code")
            )
            site_meta_frame["trust_id"] = site_meta_frame["organisation_name"].map(_trust_id)
            site_meta_frame["n_meters"] = site_meta_frame["series_id"].map(n_site)
            idx_site = series_index(site_raw, site_imp, site_meta_frame, None, config)
            idx_site["imputed_fraction"] = idx_site["series_id"].map(site_flag_frac.mean())
            levels["site"] = (site_raw, site_imp, site_flag_frac, idx_site, site_meta_frame)
            record.step(
                f"{energy}: site totals",
                sites=int(site_raw.shape[1]),
                rule="total = sum of meters only where every active meter has a value (observed or imputed); else NaN",
                raw_complete_share=round(float(site_raw.notna().mean().mean()), 4),
                imputed_complete_share=round(float(site_imp.notna().mean().mean()), 4),
            )

            mapping_trust = site_meta_frame.set_index("series_id")["trust_id"]
            trust_raw, n_trust = aggregate_wide(site_raw, mapping_trust, min_cov)
            trust_imp, _ = aggregate_wide(site_imp, mapping_trust, min_cov)
            trust_flag_frac = (
                site_flag_frac.T.groupby(mapping_trust.reindex(site_flag_frac.columns)).mean().T
            ).reindex(columns=trust_imp.columns)
            trust_meta_frame = (
                site_meta_frame.groupby("trust_id")
                .agg(
                    organisation_name=("organisation_name", "first"),
                    organisation_type=("organisation_type", "first"),
                    commissioning_region=("commissioning_region", "first"),
                    n_sites=("series_id", "nunique"),
                    n_meters=("n_meters", "sum"),
                )
                .reset_index()
                .rename(columns={"trust_id": "series_id"})
            )
            idx_trust = series_index(trust_raw, trust_imp, trust_meta_frame, None, config)
            idx_trust["imputed_fraction"] = idx_trust["series_id"].map(trust_flag_frac.mean())
            levels["trust"] = (trust_raw, trust_imp, trust_flag_frac, idx_trust, trust_meta_frame)
            record.step(
                f"{energy}: trust totals",
                trusts=int(trust_raw.shape[1]),
                rule="total = sum of sites only where every active site has a value; else NaN",
                raw_complete_share=round(float(trust_raw.notna().mean().mean()), 4),
                imputed_complete_share=round(float(trust_imp.notna().mean().mean()), 4),
            )

        # ---- write
        for level in config["levels"]:
            if level not in levels:
                continue
            w_raw, w_imp, fl, idx, mt = levels[level]
            out = PROCESSED_DIR / energy / level
            csv = bool(config.get("write_csv", True))
            _write(w_raw, out / "wide_raw.parquet", record, csv=csv, index=True)
            _write(w_imp, out / "wide_imputed.parquet", record, csv=csv, index=True)
            long_raw = (
                w_raw.stack(future_stack=True)
                .rename("kwh")
                .reset_index()
                .rename(columns={"level_1": "series_id"})
            )
            long_raw = long_raw.dropna(subset=["kwh"])
            _write(long_raw, out / "long_raw.parquet", record)
            long_imp = (
                w_imp.stack(future_stack=True)
                .rename("kwh")
                .reset_index()
                .rename(columns={"level_1": "series_id"})
            )
            fl_long = (
                fl.reindex(columns=w_imp.columns)
                .stack(future_stack=True)
                .rename("imputed")
                .reset_index()
                .rename(columns={"level_1": "series_id"})
            )
            long_imp = long_imp.merge(fl_long, on=["datetime", "series_id"], how="left").dropna(
                subset=["kwh"]
            )
            long_imp["imputed"] = long_imp["imputed"].fillna(0.0)
            if level == "meter":
                long_imp["imputed"] = long_imp["imputed"].astype(bool)
            _write(long_imp, out / "long_imputed.parquet", record)
            _write(idx, out / "series_index.parquet", record, csv=csv)
            record.tables[f"{energy}/{level}"] = {
                "series_raw": int(w_raw.shape[1]),
                "series_imputed": int(w_imp.shape[1]),
                "periods": int(len(w_raw)),
                "raw_cells_present_share": round(float(w_raw.notna().mean().mean()), 4),
                "imputed_cells_present_share": round(float(w_imp.notna().mean().mean()), 4),
                "median_coverage_in_span": round(float(idx["coverage_in_span"].median()), 4),
                "imputed_fraction_mean": round(float(idx["imputed_fraction"].mean()), 4)
                if "imputed_fraction" in idx
                else None,
            }

    _write_manifest(config, record)
    return record


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _write_manifest(config: dict[str, Any], record: BuildRecord) -> None:
    payload = {
        "generated": pd.Timestamp.now(tz="UTC").isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": config,
        "steps": record.steps,
        "tables": record.tables,
        "files": record.files,
    }
    (PROCESSED_DIR / "manifest.json").write_text(json.dumps(payload, indent=1, default=str))

    lines = [
        "# Data manifest - processed demand datasets",
        "",
        f"Generated {payload['generated']} by `sense-energy build-processed` at commit `{payload['git_commit']}`.",
        "Regenerate with `make processed`; this file and `code/data/processed/manifest.json` are written by the build and should not be edited by hand.",
        "",
        "## Lineage",
        "",
        "```",
        "Energy Systems Catapult extract (raw)",
        "  -> sense-energy build-interim   : reactive/cumulative rows dropped, meter grain, zeros->NaN,",
        "                                     30-min grid, outliers flagged, gaps <= 2 h interpolated",
        "  -> sense-energy build-processed : this manifest",
        "```",
        "",
        "## Layout",
        "",
        "`code/data/processed/<energy>/<level>/` for energy in {elec, gas} and level in {meter, site, trust}:",
        "",
        "| File | Content |",
        "|---|---|",
        "| `wide_raw.parquet` / `.csv` | time x series matrix, kWh per half hour, missing = NaN |",
        "| `wide_imputed.parquet` / `.csv` | same, gaps filled by profile-KNN; heavily-imputed series omitted |",
        "| `long_raw.parquet` | `series_id, datetime, kwh` - present values only |",
        "| `long_imputed.parquet` | `series_id, datetime, kwh, imputed` (bool at meter level; imputed share of members above) |",
        "| `series_index.parquet` / `.csv` | one row per series: window, coverage, imputation share, flags, site/trust attributes |",
        "",
        "Plus `site_attributes.parquet` / `.csv` (site register with coordinates, GSP group/region) for clustering.",
        "",
        "Conventions: timestamps are UTC period *starts* on a common 30-minute grid; units are kWh per half hour; `series_id` is the MPxN at meter level, `site_code` at site level and a normalised trust name at trust level.",
        "",
        "## Steps",
        "",
    ]
    for s in record.steps:
        detail = {k: v for k, v in s.items() if k not in ("n", "title")}
        lines.append(f"{s['n']}. **{s['title']}**")
        for k, v in detail.items():
            if isinstance(v, list) and len(v) > 12:
                v = f"{len(v)} items: {v[:6]} ..."
            lines.append(f"   - {k}: `{v}`")
        lines.append("")
    lines += [
        "## Imputation method",
        "",
        "Profile-KNN within each series (Peppanen et al. 2016). Each UTC day is a 48-slot vector. For a day with gaps that is at least "
        f"{config['imputation']['min_observed_fraction_of_day']:.0%} observed and contains no gap longer than {config['imputation']['max_gap_periods']} periods, "
        f"the k={config['imputation']['k']} nearest *complete* days of the same series and day type (weekday/weekend, local time) within ±{config['imputation']['window_weeks']} weeks are found by RMSE over the observed slots; "
        "the missing slots take the neighbours' mean profile, scaled to the day's observed level (scale clipped to [0.25, 4]). Whole missing days are never filled. "
        f"Series needing more than {config['imputation']['max_imputed_fraction_per_series']:.0%} imputation are flagged `imputed_heavily` and excluded from `wide_imputed`. "
        "Site and trust totals are formed only where every active member has a value.",
        "",
        "## Tables",
        "",
        "| energy/level | series (raw) | series (imputed) | periods | raw cells present | imputed cells present | median coverage in span | mean imputed share |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for key, t in record.tables.items():
        imp_share = (
            "-" if t["imputed_fraction_mean"] is None else f"{t['imputed_fraction_mean']:.2%}"
        )
        lines.append(
            f"| {key} | {t['series_raw']} | {t['series_imputed']} | {t['periods']:,} | {t['raw_cells_present_share']:.1%} | {t['imputed_cells_present_share']:.1%} | {t['median_coverage_in_span']:.1%} | {imp_share} |"
        )
    lines += [
        "",
        "## Files",
        "",
        "| path | rows | cols | MB | sha256[:12] |",
        "|---|---|---|---|---|",
    ]
    for f in record.files:
        lines.append(
            f"| `{f['path']}` | {f['rows']:,} | {f['cols']} | {f['mb']} | `{f['sha256_12']}` |"
        )
    lines += [
        "",
        "## Known limitations",
        "",
        "- Gaps longer than a day, and days observed less than the threshold, remain missing in the imputed tables by design.",
        "- Site/trust `imputed` in the long tables is the share of members whose value was imputed at that time, not a boolean.",
        "- Gas has far fewer meters and a much weaker daily cycle; profile-KNN is less well suited to it.",
        "- Coverage differs across series: check `series_index` before forming rectangular blocks for clustering or foundation models.",
        "",
    ]
    (PROJECT_ROOT / "doc" / "data_manifest.md").write_text("\n".join(lines))
    logger.info("Manifest written: doc/data_manifest.md, processed/manifest.json")
