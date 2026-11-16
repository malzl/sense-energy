"""Share of the achievable peak cut that a battery gets when it plans on the forecast, in each
year of the record: the zero-shot model and the seasonal naive forecast, with the 95 % interval
from resampling sites. Battery: 20 % of the site's mean demand, two hours.

analysis/forecast_record.py (by_period.csv).
"""

import numpy as np
from _common import OUT, YEARS, table

from sense_energy.visualization import style

p = table("by_period")
p = p[p["window"].isin(YEARS)]
models = [
    m
    for m in style.RECORD_MODEL_NAMES
    if (p[p["model"] == m]["days"] > 100).all() and m in set(p["model"])
]
fig, ax = style.figure(style.SINGLE_COLUMN)
for j, model in enumerate(models):
    d = p[p["model"] == model].set_index("window").reindex(YEARS)
    ax.errorbar(
        np.arange(len(YEARS)) + (j - (len(models) - 1) / 2) * 0.16,
        d["captured_point_pct"],
        yerr=[
            d["captured_point_pct"] - d["captured_lo"],
            d["captured_hi"] - d["captured_point_pct"],
        ],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.RECORD_MODEL_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.RECORD_MODEL_NAMES[model],
    )
ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels(YEARS)
ax.set_ylim(0, 100)
ax.set_ylabel("Share of the achievable peak cut (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, bottom=0.34)
style.save_figure(fig, OUT / "record_value_years", data=p)
