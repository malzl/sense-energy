"""Structure of the demand data: shape-based clustering of meters, sites and trusts.

Each series is described by its *shape* - the mean daily profile on weekdays
and on weekends in local time, normalised by the series mean (96 values,
series mean = 1) - plus scalar descriptors that are easier to read: size,
night/day ratio, weekend/weekday ratio, winter/summer ratio, temperature
sensitivity (slope of the daily demand index on daily mean 2 m temperature),
day-to-day volatility and the imputed share. Shapes are clustered with
k-means (k chosen by silhouette inside a configured range, all k kept) and,
for the dendrogram, Ward linkage on the same vectors. Descriptors and
metadata (organisation type, site use, region) are summarised per cluster to
interpret it. Outputs go to ``code/reports/clustering``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import INTERIM_DIR, PROCESSED_DIR, PROJECT_ROOT
from ..eda import common as C
from ..logging_utils import get_logger

logger = get_logger(__name__)

OUT_DIR = PROJECT_ROOT / "code" / "reports" / "clustering"
PROFILE_COLS = [f"wd_{h:02d}" for h in range(48)] + [f"we_{h:02d}" for h in range(48)]
DESCRIPTORS = [
    "mean_kw",
    "night_day_ratio",
    "weekend_ratio",
    "winter_summer_ratio",
    "temp_slope_per_c",
    "daily_cv",
    "imputed_fraction",
    "n_days",
]


def _local(wide: pd.DataFrame) -> pd.DataFrame:
    out = wide.copy()
    out.index = out.index.tz_convert(C.LOCAL_TZ)
    return out


def shape_features(wide: pd.DataFrame, min_days: int = 120) -> pd.DataFrame:
    """Normalised weekday/weekend daily profiles (96 columns) per series."""
    loc = _local(wide)
    norm = loc / loc.mean()
    tod = loc.index.hour * 2 + loc.index.minute // 30
    weekend = loc.index.dayofweek >= 5
    days = loc.notna().resample("D").sum().ge(40).sum()  # days with most half-hours
    keep = days[days >= min_days].index
    wd = norm.loc[~weekend, keep].groupby(tod[~weekend]).mean().T
    we = norm.loc[weekend, keep].groupby(tod[weekend]).mean().T
    wd.columns = [f"wd_{h:02d}" for h in wd.columns]
    we.columns = [f"we_{h:02d}" for h in we.columns]
    out = pd.concat([wd, we], axis=1).reindex(columns=PROFILE_COLS)
    out.index.name = "series_id"
    out["n_days"] = days[keep]
    return out.dropna(subset=PROFILE_COLS[:1])


def _temperature_slope(wide: pd.DataFrame, site_of: dict[str, list[str]]) -> pd.Series:
    """Slope of the daily demand index on daily mean 2 m temperature (ERA5, site mean)."""
    w = pd.read_parquet(
        INTERIM_DIR / "weather_reanalysis.parquet", columns=["site_code", "datetime", "t2m"]
    )
    w["date"] = w["datetime"].dt.tz_convert(C.LOCAL_TZ).dt.date
    temp = w.groupby(["site_code", "date"])["t2m"].mean().sub(273.15).unstack("site_code")
    loc = _local(wide / wide.mean())
    daily = loc.groupby(loc.index.date).mean()
    count = loc.groupby(loc.index.date).count()
    daily = daily.where(count >= 40)
    slopes = {}
    for sid in wide.columns:
        sites = [s for s in site_of.get(sid, []) if s in temp.columns]
        if not sites:
            slopes[sid] = np.nan
            continue
        t = temp[sites].mean(axis=1)
        d = daily[sid]
        both = pd.concat([d, t], axis=1, keys=["d", "t"]).dropna()
        if len(both) < 60:
            slopes[sid] = np.nan
            continue
        x = both["t"].to_numpy()
        y = both["d"].to_numpy()
        slopes[sid] = float(np.polyfit(x, y, 1)[0])
    return pd.Series(slopes, name="temp_slope_per_c")


def descriptors(wide: pd.DataFrame, index: pd.DataFrame, level: str) -> pd.DataFrame:
    """Scalar descriptors per series (see module docstring)."""
    loc = _local(wide)
    tod = loc.index.hour * 2 + loc.index.minute // 30
    mean = loc.mean()
    night = loc[(tod >= 4) & (tod < 10)].mean()  # 02:00-05:00
    day = loc[(tod >= 20) & (tod < 32)].mean()  # 10:00-16:00
    weekend = loc.index.dayofweek >= 5
    month = loc.index.month
    winter = loc[np.isin(month, [12, 1, 2])].mean()
    summer = loc[np.isin(month, [6, 7, 8])].mean()
    daily = loc.resample("D").sum(min_count=40)
    out = pd.DataFrame(
        {
            "mean_kw": mean * 2,
            "night_day_ratio": night / day,
            "weekend_ratio": loc[weekend].mean() / loc[~weekend].mean(),
            "winter_summer_ratio": winter / summer,
            "daily_cv": daily.std() / daily.mean(),
        }
    )
    idx = index.set_index("series_id")
    out["imputed_fraction"] = (
        idx["imputed_fraction"].reindex(out.index) if "imputed_fraction" in idx else np.nan
    )
    site_of = _site_of(index, level)
    out["temp_slope_per_c"] = _temperature_slope(wide, site_of).reindex(out.index)
    out.index.name = "series_id"
    return out


def _site_of(index: pd.DataFrame, level: str) -> dict[str, list[str]]:
    idx = index.set_index("series_id")
    if level == "site":
        return {s: [s] for s in idx.index}
    if level == "meter":
        return {m: [str(idx.loc[m, "site_code"])] for m in idx.index}
    site_idx = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "series_index.parquet")
    by_trust = (
        site_idx.groupby("trust_id")["series_id"].apply(list) if "trust_id" in site_idx else {}
    )
    return {t: list(by_trust.get(t, [])) for t in idx.index}


def metadata(index: pd.DataFrame, level: str) -> pd.DataFrame:
    idx = index.set_index("series_id")
    cols = [
        c
        for c in (
            "organisation_type",
            "commissioning_region",
            "site_use_type",
            "organisation_name",
            "site_code",
            "n_sites",
            "n_meters",
        )
        if c in idx.columns
    ]
    meta = idx[cols].copy()
    if level == "site":
        geo = pd.read_parquet(PROCESSED_DIR.parent / "geo" / "sites_geo.parquet").set_index(
            "site_code"
        )
        for c in ("site_use_type", "site_construction_year_band", "latitude", "longitude"):
            if c in geo.columns:
                meta[c] = geo[c].reindex(meta.index)
    return meta


def cluster(
    X: pd.DataFrame, k_range: range, seed: int = 0
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """k-means for every k; silhouette per k; the best k's labels first."""
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    labels = pd.DataFrame(index=X.index)
    rows = []
    centroids = {}
    for k in k_range:
        if k >= len(X):
            break
        km = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(X.to_numpy())
        labels[f"k{k}"] = km.labels_
        rows.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(X.to_numpy(), km.labels_)),
                "inertia": float(km.inertia_),
            }
        )
        centroids[k] = pd.DataFrame(km.cluster_centers_, columns=X.columns)
    sil = pd.DataFrame(rows)
    return labels, sil, centroids


def _relabel_by_size(labels: pd.Series, X: pd.DataFrame, size: pd.Series) -> pd.Series:
    """Cluster ids ordered by the members' mean size (0 = largest), fixed across runs."""
    order = size.groupby(labels).mean().sort_values(ascending=False).index
    remap = {old: new for new, old in enumerate(order)}
    return labels.map(remap)


def run(energy: str, level: str, config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = {"k_min": 2, "k_max": 8, "min_days": 120, **(config or {})}
    wide, index = C.series_for(energy, level)
    wide = wide[[c for c in wide.columns if c in set(index["series_id"])]]
    feats = shape_features(wide, int(cfg["min_days"]))
    if len(feats) < cfg["k_min"] + 2:
        logger.warning(
            "%s/%s: only %d series with %d+ days; skipping",
            energy,
            level,
            len(feats),
            cfg["min_days"],
        )
        return {}
    desc = descriptors(wide[feats.index], index, level).reindex(feats.index)
    meta = metadata(index, level).reindex(feats.index)
    X = feats[PROFILE_COLS]
    labels, sil, centroids = cluster(X, range(int(cfg["k_min"]), int(cfg["k_max"]) + 1))
    best_k = int(sil.loc[sil["silhouette"].idxmax(), "k"]) if len(sil) else int(cfg["k_min"])
    best = _relabel_by_size(labels[f"k{best_k}"], X, desc["mean_kw"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{energy}_{level}"
    table = pd.concat([desc, meta], axis=1)
    table.insert(0, "cluster", best)
    table.to_csv(OUT_DIR / f"{stem}_clusters.csv")
    feats.to_parquet(OUT_DIR / f"{stem}_profiles.parquet")
    labels.to_csv(OUT_DIR / f"{stem}_labels_all_k.csv")
    sil.to_csv(OUT_DIR / f"{stem}_silhouette.csv", index=False)
    # centroids of the chosen k in the size-ordered labelling
    cen = X.groupby(best).mean()
    cen.index.name = "cluster"
    cen.to_csv(OUT_DIR / f"{stem}_centroids.csv")
    from scipy.cluster.hierarchy import linkage

    Z = linkage(X.to_numpy(), method="ward")
    np.save(OUT_DIR / f"{stem}_ward_linkage.npy", Z)
    pd.Series(X.index, name="series_id").to_csv(OUT_DIR / f"{stem}_ward_order.csv", index=False)
    summary = table.groupby("cluster")[[c for c in DESCRIPTORS if c in table]].median().round(3)
    summary["n"] = table.groupby("cluster").size()
    logger.info(
        "%s/%s: %d series, best k=%d (silhouette %.3f)\n%s",
        energy,
        level,
        len(X),
        best_k,
        sil["silhouette"].max(),
        summary.to_string(),
    )
    return {"table": table, "silhouette": sil, "best_k": best_k, "centroids": cen}
