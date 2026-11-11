"""Rooftop solar against the real load of the sites without CHP, financial year 2024/25: the
share of the array's generation the site consumes itself and the share of the site's demand it
covers, by array size relative to the site's mean demand. Lines are medians over sites, bands the
interquartile range.

analysis/flexibility.py (solar_sites.csv).
"""

import pandas as pd
from _common import FLEX, OUT

from sense_energy.visualization import style

s = pd.read_csv(FLEX / "solar_sites.csv")
series = {
    "self_consumption": ("generation consumed on site", style.COLORS["black"], "-"),
    "demand_covered": ("demand covered", style.COLORS["blue"], "--"),
}
fig, ax = style.figure(style.SINGLE_COLUMN)
rows = []
for col, (label, color, ls) in series.items():
    q = 100 * s.groupby("size")[col].quantile([0.25, 0.5, 0.75]).unstack()
    ax.fill_between(q.index, q[0.25], q[0.75], color=color, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(
        q.index,
        q[0.5],
        color=color,
        linestyle=ls,
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE,
        label=label,
    )
    rows.append(q.assign(series=label))
ax.set_xscale("log", base=2)
ax.set_xticks(sorted(s["size"].unique()))
ax.set_xticklabels([f"{v:g}" for v in sorted(s["size"].unique())])
ax.minorticks_off()
ax.set_ylim(0, 105)
ax.set_xlabel("Array size (kWp per kW of mean demand)")
ax.set_ylabel("Share (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "solar_self_consumption", data=pd.concat(rows).reset_index())
