"""Day-ahead error of Chronos-2 without covariates by the length of history that it gets,
from one day to six weeks, by site group (91 sites, 69 test days).

analysis/forecast_anatomy.py (context.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

c = pd.read_csv(FA / "context.csv")
c["days"] = c["context_periods"] / 48.0
fig, ax = style.figure(style.SINGLE_COLUMN)
for group in style.SITE_GROUP_ORDER:
    d = c[c["group"] == group].sort_values("days")
    ax.plot(
        d["days"],
        d["nmae_pct"],
        color=style.SITE_GROUP_COLORS[group],
        marker="o",
        markersize=style.MARKER_SIZE,
        linewidth=style.LINEWIDTH["primary"],
        label=group,
    )
ax.set_xscale("log")
ax.set_xticks([1, 3, 7, 14, 42])
ax.set_xticklabels(["1", "3", "7", "14", "42"])
ax.minorticks_off()
ax.set_ylim(0, None)
ax.set_xlabel("History given to the model (days)")
ax.set_ylabel("Normalised absolute error (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False)
style.save_figure(
    fig, OUT / "context_length", data=c[["group", "days", "nmae_pct", "captured_point_pct"]]
)
