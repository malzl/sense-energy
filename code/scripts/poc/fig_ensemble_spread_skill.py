"""Spread-skill of the weather-driven demand ensemble: absolute error of the member-mean
forecast against the spread of the 51 member medians (both as a share of the site's mean
demand), pooled over sites and target periods and binned by spread decile. One plot."""

import pandas as pd
from _common import OUT, RESULTS, observed, per_site

from sense_energy.visualization import style

m = pd.read_parquet(RESULTS / "ensemble_members_chronos2.parquet")
m["target"] = pd.to_datetime(m["target"], utc=True)
g = m.groupby(["site_code", "origin", "target"])["median"]
agg = g.agg(mean="mean", spread="std").reset_index()
agg = agg.merge(observed(), on=["target", "site_code"], how="inner")
site_mean = per_site().groupby("site_code")["y_mean"].first()
agg["site_mean"] = agg["site_code"].map(site_mean)
agg["ae"] = (agg["mean"] - agg["y"]).abs() / agg["site_mean"] * 100
agg["spread_n"] = agg["spread"] / agg["site_mean"] * 100
agg["bin"] = pd.qcut(agg["spread_n"], 10, labels=False, duplicates="drop")
frame = (
    agg.groupby("bin")
    .agg(spread=("spread_n", "mean"), mae=("ae", "mean"), n=("ae", "size"))
    .reset_index()
)

fig, ax = style.figure(style.SINGLE_COLUMN)
ax.plot(
    frame["spread"],
    frame["mae"],
    color=style.METHOD_COLORS["Chronos-2 + IFS members"],
    linewidth=style.LINEWIDTH["primary"],
    marker=style.method_marker("Chronos-2 + IFS members"),
    markersize=style.MARKER_SIZE,
    markeredgecolor="white",
    markeredgewidth=style.MARKER_EDGE,
)
ax.set_xlabel("Spread of member medians (% of mean demand)")
ax.set_ylabel("MAE of the member mean (% of mean demand)")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "ensemble_spread_skill", data=frame)
