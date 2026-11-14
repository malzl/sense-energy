"""Share of the achievable peak cut that a battery captures when its threshold is planned the
day before and planned again on the day with the forecast issued at 00:00 or at 06:00, by site
group. 91 sites, 69 test days, Chronos-2 without covariates.

analysis/forecast_anatomy.py (issue_value.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

v = pd.read_csv(FA / "issue_value.csv")
fig, ax = style.figure(style.SINGLE_COLUMN)
for group in style.SITE_GROUP_ORDER:
    d = v[v["group"] == group].set_index("issue").reindex(style.ISSUE_ORDER)
    ax.plot(
        range(len(d)),
        d["captured_pct"],
        color=style.SITE_GROUP_COLORS[group],
        marker="o",
        markersize=style.MARKER_SIZE,
        linewidth=style.LINEWIDTH["primary"],
        label=group,
    )
ax.set_xticks(range(len(style.ISSUE_ORDER)))
ax.set_xticklabels(["16:30\nday before", "00:00\non the day", "06:00\non the day"])
ax.set_ylim(0, 100)
ax.set_xlabel("Time of the last plan")
ax.set_ylabel("Share of the achievable peak cut (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, anchor_y=-0.34, bottom=0.36)
style.save_figure(fig, OUT / "issue_value", data=v)
