"""Levels of the demand hierarchy for the PoC: meter -> site -> trust -> region -> national.

The site level is the PoC's 91 selected sites. Every level above it is a sum
over those sites (grouped by trust, NHS commissioning region, or all of them);
a total exists only where every member has a value - strictly observed for the
scored series (``y_raw``), observed-or-short-gap-imputed for the model contexts
(``y_imp``). Aggregate series are kept when their *imputed* series meets the
PoC coverage rule (the strictly observed national total is complete for 85 % of
the test window, the imputed one for 98 %). The meter level comes from the
processed meter files with the usual selection.

Bottom-up forecasts sum the members' point forecasts from the level below for
the same origin and target periods; they exist only where every member was
forecast, and carry no interval (their quantile columns equal the point).
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..config import PROCESSED_DIR
from ..logging_utils import get_logger
from .poc import Panel, coverage_ok, load_panel, outputs_dir, score

logger = get_logger(__name__)

LEVELS = ["meter", "site", "trust", "region", "total"]
LOWER = {"site": "meter", "trust": "site", "region": "site", "total": "site"}
NATIONAL = "national"


def region_id(name: str) -> str:
    """'EAST OF ENGLAND COMMISSIONING REGION' -> 'east_of_england'."""
    name = re.sub(r"\s+COMMISSIONING REGION$", "", str(name).strip().upper())
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def group_of(site_meta: pd.DataFrame, level: str) -> pd.Series:
    """site_code -> aggregate id for ``level``."""
    if level == "trust":
        return site_meta["trust_id"].astype(str)
    if level == "region":
        return site_meta["commissioning_region"].map(region_id)
    if level == "total":
        return pd.Series(NATIONAL, index=site_meta.index)
    raise KeyError(level)


def aggregate_panel(site_panel: Panel, level: str, config: dict[str, Any]) -> Panel:
    """Sum the selected sites into ``level`` series (all members present, else NaN)."""
    sites = site_panel.sites
    groups = group_of(site_panel.site_meta, level).reindex(sites)
    y_raw_s, y_imp_s = site_panel.y_raw[sites], site_panel.y_imp[sites]
    mean = y_imp_s.mean()
    raw, imp, members, rows = {}, {}, {}, []
    meta = site_panel.site_meta
    for gid, cols in groups.groupby(groups).groups.items():
        cols = list(cols)
        raw[gid] = y_raw_s[cols].sum(axis=1, min_count=1).where(y_raw_s[cols].notna().all(axis=1))
        imp[gid] = y_imp_s[cols].sum(axis=1, min_count=1).where(y_imp_s[cols].notna().all(axis=1))
        w = mean[cols] / mean[cols].sum()
        members[gid] = {s: float(w[s]) for s in cols}
        largest = w.idxmax()
        rows.append(
            {
                "series_id": gid,
                "organisation_name": (
                    meta.loc[largest, "organisation_name"]
                    if level == "trust"
                    else (
                        meta.loc[largest, "commissioning_region"]
                        if level == "region"
                        else "All selected sites"
                    )
                ),
                "organisation_type": meta.loc[largest, "organisation_type"],
                "commissioning_region": meta.loc[largest, "commissioning_region"],
                "n_sites": len(cols),
                "n_meters": int(
                    pd.to_numeric(meta.loc[cols, "n_meters"], errors="coerce").fillna(0).sum()
                )
                if "n_meters" in meta.columns
                else np.nan,
                "mean_kwh": float(mean[cols].sum()),
                "members": ",".join(cols),
            }
        )
    y_raw, y_imp = pd.DataFrame(raw), pd.DataFrame(imp)
    keep = [g for g in y_raw.columns if coverage_ok(y_imp[g], site_panel.index, config)]
    limit = config.get("site_limit")
    if limit:
        keep = keep[: int(limit)]
    panel = Panel(
        y_raw=y_raw,
        y_imp=y_imp,
        index=site_panel.index,
        local=site_panel.local,
        local_date=site_panel.local_date,
        tod=site_panel.tod,
        dow=site_panel.dow,
        sites=keep,
        site_meta=pd.DataFrame(rows).set_index("series_id").reindex(keep),
        level=level,
        members={g: members[g] for g in keep},
    )
    logger.info(
        "%s level: %d of %d aggregates kept (members: %d sites); observed share in grid %.3f",
        level,
        len(keep),
        y_raw.shape[1],
        len(sites),
        float(y_raw[keep].notna().mean().mean()) if keep else float("nan"),
    )
    return panel


def lower_mapping(
    config: dict[str, Any], level: str, upper: Panel
) -> tuple[pd.Series, dict[str, set]]:
    """lower series -> upper series, and the members every upper series needs."""
    if level == "site":
        idx = pd.read_parquet(
            PROCESSED_DIR / config["energy"] / "meter" / "series_index.parquet"
        ).set_index("series_id")
        wide = pd.read_parquet(PROCESSED_DIR / config["energy"] / "meter" / "wide_raw.parquet")
        mapping = idx["site_code"].astype(str)
        min_cov = float(config.get("min_member_coverage", 0.1))
        active = wide.columns[wide.notna().mean() >= min_cov]  # the processed site rule
        mapping = mapping.reindex(active).dropna()
        need = {s: set(mapping.index[mapping == s]) for s in upper.sites}
        return mapping, need
    mapping = pd.Series({s: g for g, d in (upper.members or {}).items() for s in d})
    need = {g: set(d) for g, d in (upper.members or {}).items()}
    return mapping, need


def bottom_up(config: dict[str, Any], level: str) -> pd.DataFrame | None:
    """Sum the lower level's point forecasts into ``level`` series; None when there are none."""
    lower = LOWER[level]
    lower_dir = outputs_dir(config, lower)
    files = [
        p
        for p in sorted(lower_dir.glob("forecasts_*.parquet"))
        if not p.name.endswith(".partial.parquet")
    ]
    if not files:
        logger.warning("no %s-level forecasts under %s", lower, lower_dir)
        return None
    upper = load_panel({**config, "level": level})
    mapping, need = lower_mapping(config, level, upper)
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    f["group"] = f["site_code"].map(mapping)
    f = f.dropna(subset=["group"])
    f = f[f["group"].isin(upper.sites)]
    keys = ["model", "group", "origin", "target"]
    g = f.groupby(keys, sort=False)
    agg = g.agg(
        point=("point", "sum"), n=("site_code", "nunique"), lead_hours=("lead_hours", "first")
    )
    agg = agg.reset_index()
    agg = agg[agg["n"] == agg["group"].map({k: len(v) for k, v in need.items()})]
    qcols = [c for c in f.columns if re.fullmatch(r"q\d\d", c)]
    out = agg.rename(columns={"group": "site_code"}).drop(columns="n")
    out["model"] = "bu_" + out["model"].astype(str)
    for c in qcols:
        out[c] = out["point"].astype("float32")
    out["point"] = out["point"].astype("float32")
    complete = agg.groupby("model")["group"].nunique()
    logger.info(
        "bottom-up %s <- %s: %s rows; aggregates covered per model: %s",
        level,
        lower,
        f"{len(out):,}",
        complete.to_dict(),
    )
    path = outputs_dir(config, level) / f"bottom_up_from_{lower}.parquet"
    out.to_parquet(path, index=False)
    return out


def summary(config: dict[str, Any]) -> pd.DataFrame:
    """Direct and bottom-up scores of every level in one table."""
    rows = []
    for level in LEVELS:
        d = outputs_dir(config, level)
        pooled_path = d / "scores_pooled.csv"
        if pooled_path.exists():
            pooled = pd.read_csv(pooled_path)
            for r in pooled.itertuples(index=False):
                rows.append({"level": level, "kind": "direct", **r._asdict()})
        bu = sorted(d.glob("bottom_up_from_*.parquet"))
        if bu:
            panel = load_panel({**config, "level": level})
            fc = pd.concat([pd.read_parquet(p) for p in bu], ignore_index=True)
            fc["origin"] = pd.to_datetime(fc["origin"], utc=True)
            fc["target"] = pd.to_datetime(fc["target"], utc=True)
            naive = (
                pd.read_parquet(d / "forecasts_baselines.parquet")
                if (d / "forecasts_baselines.parquet").exists()
                else None
            )
            both = pd.concat([fc, naive], ignore_index=True) if naive is not None else fc
            per, pooled, _ = score(both, panel, list(config["quantiles"]))
            per.to_csv(d / "scores_per_site_bottom_up.csv", index=False)
            for r in pooled[pooled["model"].str.startswith("bu_")].itertuples(index=False):
                row = r._asdict()
                row["model"] = row["model"][3:]
                rows.append({"level": level, "kind": "bottom_up", **row})
    out = pd.DataFrame(rows)
    hdir = outputs_dir(config, "hierarchy")
    out.to_csv(hdir / "summary.csv", index=False)
    return out
