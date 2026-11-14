"""Forecast error against site size. Sites without on-site generation: normalised error
against mean demand, with the fitted line. Sites with on-site CHP twice: the error over the grid
import at the mean import (filled), and the same error over the site's total demand (import
plus the plant's reported share) at that total (open). Best model, 69 test days; both axes log.

analysis/forecast_anatomy.py (error_by_site.csv, chp_size_adjusted.csv).
"""

import numpy as np
import pandas as pd
from _common import FA, OUT
from matplotlib.ticker import FuncFormatter, NullFormatter

from sense_energy.visualization import style

s = pd.read_csv(FA / "error_by_site.csv")
c = pd.read_csv(FA / "chp_size_adjusted.csv")
plain = s[s["group"] != "on-site CHP"]
slope, intercept = np.polyfit(np.log(plain["mean_kw"]), np.log(plain["nmae_pct"]), 1)
fig, ax = style.figure(style.WIDE_SINGLE)
ax.plot(
    plain["mean_kw"],
    plain["nmae_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE - 1,
    markeredgewidth=0,
    color=style.PLANT_SERIES_COLORS["no on-site CHP"],
    alpha=0.6,
    label="no on-site CHP",
)
xs = np.array([plain["mean_kw"].min(), c["total_kw"].max()])
ax.plot(
    xs,
    np.exp(intercept + slope * np.log(xs)),
    color=style.PLANT_SERIES_COLORS["no on-site CHP"],
    linewidth=style.LINEWIDTH["secondary"],
)
k = style.PLANT_SERIES_COLORS["on-site CHP, against import"]
for a, b in zip(
    c[["mean_kw", "nmae_pct"]].to_numpy(),
    c[["total_kw", "error_of_total_pct"]].to_numpy(),
    strict=True,
):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=k, linewidth=0.6, alpha=0.6, zorder=2)
ax.plot(
    c["mean_kw"],
    c["nmae_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE + 1,
    color=k,
    label="on-site CHP, against import",
    zorder=3,
)
ax.plot(
    c["total_kw"],
    c["error_of_total_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE + 1,
    markerfacecolor="white",
    markeredgecolor=k,
    markeredgewidth=1.0,
    label="on-site CHP, against total demand",
    zorder=3,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xticks([1, 10, 100, 1000])
ax.set_yticks([2, 5, 10, 20])
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Mean demand (kW)")
ax.set_ylabel("Normalised absolute error (%)")
style.style_axis(ax, grid="both")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.24, bottom=0.3)
data = pd.concat(
    [
        plain[["site_code", "group", "mean_kw", "nmae_pct"]],
        c[["site_code", "group", "mean_kw", "nmae_pct", "total_kw", "error_of_total_pct"]],
    ]
)
style.save_figure(fig, OUT / "chp_size_adjusted", data=data)
