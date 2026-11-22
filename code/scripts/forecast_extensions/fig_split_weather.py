"""Change in the day-ahead error when the weather inputs of the zero-shot model are given in
another way, against the run with one channel for each weather variable (measured over the
past, forecast over the future). Every third day of the record; paired on the same site-days,
95 % intervals from resampling weeks.

analysis/forecast_extensions.py (split_differences.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

d = table("split_differences")
d = d[d["b"] == "chronos2_ifs"]
groups = [g for g in dict.fromkeys(d["group"])]
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0, axis="x")
for j, model in enumerate(style.SPLIT_WEATHER_NAMES):
    x = d[d["a"] == model].set_index("group").reindex(groups)
    ax.errorbar(
        x["diff_pts"],
        np.arange(len(groups)) + (j - 0.5) * 0.24,
        xerr=[x["diff_pts"] - x["lo"], x["hi"] - x["diff_pts"]],
        fmt=style.SPLIT_WEATHER_MARKERS[model],
        markersize=style.MARKER_SIZE + 1,
        color=style.SPLIT_WEATHER_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.SPLIT_WEATHER_NAMES[model],
    )
ax.set_yticks(range(len(groups)))
ax.set_yticklabels(["all sites and days" if g == "all" else g for g in groups])
ax.invert_yaxis()
ax.set_xlabel("Change in the day-ahead error (points)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.22, bottom=0.36)
style.save_figure(fig, OUT / "split_weather", data=d)
