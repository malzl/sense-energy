"""Shared loading for the PoC figure scripts (one plot per script).

Every script reads frozen result files under ``code/reports/poc`` (scores CSVs
written by ``sense-energy run-poc`` and the long forecast parquet files) and
never trains anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sense_energy.config import PROCESSED_DIR, PROJECT_ROOT
from sense_energy.visualization import style

RESULTS = PROJECT_ROOT / "code" / "reports" / "poc"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "poc"
NAIVE = "seasonal_naive"
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
QCOLS = [f"q{int(round(q * 100)):02d}" for q in QUANTILES]
TZ = "Europe/London"


def display(model_key: str) -> str:
    return style.METHOD_NAMES.get(model_key, model_key)


def ordered_models(keys) -> list[str]:
    names = [display(k) for k in keys]
    order = {m: i for i, m in enumerate(style.METHOD_ORDER)}
    return [k for _, k in sorted(zip([order.get(n, 99) for n in names], keys, strict=True))]


def plain(keys) -> list[str]:
    """Methods without covariate variants suffix (the family's plain version), fixed order."""
    return [k for k in ordered_models(keys) if style.method_variant(display(k)) == "none"]


def per_site() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scores_per_site.csv")


def by_lead() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "scores_by_lead.csv")


def forecasts(model_key: str) -> pd.DataFrame:
    name = "baselines" if model_key in ("seasonal_naive", "profile_quantiles") else model_key
    f = pd.read_parquet(RESULTS / f"forecasts_{name}.parquet")
    f = f[f["model"] == model_key]
    f["target"] = pd.to_datetime(f["target"], utc=True)
    return f


def all_forecasts() -> pd.DataFrame:
    files = [
        p
        for p in sorted(RESULTS.glob("forecasts_*.parquet"))
        if not p.name.endswith(".partial.parquet")
    ]
    f = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    f["target"] = pd.to_datetime(f["target"], utc=True)
    return f


def observed() -> pd.DataFrame:
    y = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet")
    y.index = pd.to_datetime(y.index, utc=True)
    long = y.stack(future_stack=True).rename("y").reset_index()
    long.columns = ["target", "site_code", "y"]
    return long.dropna(subset=["y"])


def joined() -> pd.DataFrame:
    """Every forecast row with its observed value, the site's mean demand, the
    normalised error of the point forecast and local calendar terms."""
    f = all_forecasts().merge(observed(), on=["target", "site_code"], how="inner")
    site_mean = per_site().groupby("site_code")["y_mean"].first()
    f["site_mean"] = f["site_code"].map(site_mean)
    f["err"] = (f["point"] - f["y"]) / f["site_mean"]  # signed, as a share of mean demand
    local = f["target"].dt.tz_convert(TZ)
    f["tod"] = (local.dt.hour * 2 + local.dt.minute // 30).astype(int)
    f["dow"] = local.dt.dayofweek.astype(int)
    return f


def iqr_rows(per: pd.DataFrame, column: str, models: list[str], scale: float = 1.0):
    rows = []
    for m in models:
        s = per.loc[per["model"] == m, column].dropna() * scale
        rows.append(
            {
                "model": m,
                "method": display(m),
                "median": s.median(),
                "q25": s.quantile(0.25),
                "q75": s.quantile(0.75),
                "n_sites": len(s),
            }
        )
    return pd.DataFrame(rows)


def dot_iqr(ax, frame: pd.DataFrame):
    """Median as a marker (fixed per method) and the IQR as a line, one row per method."""
    y = np.arange(len(frame))[::-1]
    for yi, r in zip(y, frame.itertuples(), strict=True):
        c = style.method_color(r.method)
        ax.plot([r.q25, r.q75], [yi, yi], color=c, linewidth=style.LINEWIDTH["secondary"], zorder=2)
        ax.plot(
            [r.median],
            [yi],
            marker=style.method_marker(r.method),
            markersize=style.MARKER_SIZE + 1,
            color=c,
            markeredgecolor="white",
            markeredgewidth=style.MARKER_EDGE,
            linestyle="none",
            zorder=3,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(frame["method"])
    ax.set_ylim(-0.7, len(frame) - 0.3)
    return y


def boxes(ax, groups: list[np.ndarray], methods: list[str], whis=(5, 95)):
    """Thin, family-coloured box plots (whiskers at the 5th/95th percentiles, no fliers)."""
    bp = ax.boxplot(
        groups,
        positions=np.arange(len(groups)),
        widths=0.55,
        whis=whis,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "#000000", "linewidth": style.LINEWIDTH["secondary"]},
        whiskerprops={"linewidth": style.AXIS_LINEWIDTH},
        capprops={"linewidth": style.AXIS_LINEWIDTH},
        boxprops={"linewidth": style.AXIS_LINEWIDTH},
    )
    for i, (patch, m) in enumerate(zip(bp["boxes"], methods, strict=True)):
        c = style.method_color(m)
        patch.set_facecolor(c)
        patch.set_alpha(0.35)
        patch.set_edgecolor(c)
        for artist in (
            bp["whiskers"][2 * i],
            bp["whiskers"][2 * i + 1],
            bp["caps"][2 * i],
            bp["caps"][2 * i + 1],
        ):
            artist.set_color(c)
    ax.set_xticks(np.arange(len(methods)))
    ax.set_xticklabels(methods, rotation=30, ha="right")
    return bp
