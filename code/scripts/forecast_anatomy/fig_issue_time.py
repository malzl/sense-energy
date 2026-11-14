"""Error of Chronos-2 without covariates by issue time (16:30 the day before, 00:00 and 06:00
on the target day), on the periods that all three issues forecast (the target day from 06:30),
by site group. 91 sites, 69 test days.

analysis/forecast_anatomy.py (issue_time.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

t = pd.read_csv(FA / "issue_time.csv")
fig, ax = style.figure(style.SINGLE_COLUMN)
for group in style.SITE_GROUP_ORDER:
    d = t[t["group"] == group].set_index("issue").reindex(style.ISSUE_ORDER)
    ax.plot(
        range(len(d)),
        d["nmae_pct"],
        color=style.SITE_GROUP_COLORS[group],
        marker="o",
        markersize=style.MARKER_SIZE,
        linewidth=style.LINEWIDTH["primary"],
        label=group,
    )
ax.set_xticks(range(len(style.ISSUE_ORDER)))
ax.set_xticklabels(["16:30\nday before", "00:00\non the day", "06:00\non the day"])
ax.set_ylim(0, None)
ax.set_xlabel("Issue time")
ax.set_ylabel("Normalised absolute error (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, anchor_y=-0.34, bottom=0.36)
style.save_figure(fig, OUT / "issue_time_error", data=t)
