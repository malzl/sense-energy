"""Share of the half-hours inside the 80 % range of the day-ahead forecast, by season, site
group, day type and event days: the model's range and the range with its two bounds tracked
online (conformal quantile tracking with delayed feedback). The dashed line is the nominal
80 %. Whole record after 90 days; mean over sites, 95 % interval from resampling sites.

analysis/forecast_range.py (coverage.csv).
"""

import numpy as np
from _common import MODEL, OUT, table

from sense_energy.visualization import style

c = table("coverage")
c = c[(c["model"] == MODEL) & c["grouping"].isin(["season", "site group", "day type", "event"])]
keys = list(dict.fromkeys(zip(c["grouping"], c["group"], strict=True)))
fig, ax = style.figure(style.DOUBLE_COLUMN)
style.reference_line(ax, 80.0, axis="x")
methods = ("model", "quantile tracking")
for j, method in enumerate(methods):
    d = c[c["method"] == method].set_index(["grouping", "group"]).reindex(keys)
    ax.errorbar(
        d["inside_pct"],
        np.arange(len(keys)) + (j - 0.5) * 0.3,
        xerr=[d["inside_pct"] - d["inside_lo"], d["inside_hi"] - d["inside_pct"]],
        fmt=style.RANGE_METHOD_MARKERS[method],
        markersize=style.MARKER_SIZE + 1,
        color=style.RANGE_METHOD_COLORS[method],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.RANGE_METHOD_NAMES[method],
    )
ax.set_yticks(range(len(keys)))
ax.set_yticklabels([g for _, g in keys])
ax.invert_yaxis()
ax.set_xlabel("Half-hours inside the 80 % range (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.16, bottom=0.2)
style.save_figure(fig, OUT / "range_groups", data=c)
