"""What footfall adds to the forecast when the model gets it in context: error of Chronos-2 with
the footfall minus its error without it (both with calendar, price and weather), at the two
footfall sites without CHP, on all days and on the days that are not typical. Paired differences
with 95 % intervals from resampling weeks.

analysis/forecast_anatomy.py (mobility_differences.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

d = pd.read_csv(FA / "mobility_differences.csv")
d = d[
    (d["sites"] == "two sites without CHP")
    & (d["metric"] == "nMAE")
    & (d["against"] == "mob_base")
    & d["model"].isin(style.FOOTFALL_VARIANT_NAMES)
]
kinds = ["all days", "not typical"]
variants = [m for m in style.FOOTFALL_VARIANT_NAMES if m in set(d["model"])]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for j, model in enumerate(variants):
    x = d[d["model"] == model].set_index("kind").reindex(kinds)
    pos = [k + (j - (len(variants) - 1) / 2) * 0.22 for k in range(len(kinds))]
    ax.errorbar(
        pos,
        x["diff_pts"],
        yerr=[x["diff_pts"] - x["lo"], x["hi"] - x["diff_pts"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        color=style.FOOTFALL_VARIANT_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.FOOTFALL_VARIANT_NAMES[model],
    )
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels(["all days", "days that are not typical"])
ax.set_xlim(-0.5, len(kinds) - 0.5)
ax.set_ylim(-0.5, 0.5)
ax.set_ylabel("Change in error (percentage points)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, bottom=0.36)
style.save_figure(fig, OUT / "mobility_forecast_difference", data=d)
