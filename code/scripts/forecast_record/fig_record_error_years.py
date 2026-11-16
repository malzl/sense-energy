"""Day-ahead error in each year of the record (2026: to 8 May): the zero-shot model and the
seasonal naive forecast, mean over sites with the 95 % interval from resampling sites. Every
day of the year, 91 sites.

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
        d["nmae_pct"],
        yerr=[d["nmae_pct"] - d["nmae_lo"], d["nmae_hi"] - d["nmae_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.RECORD_MODEL_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.RECORD_MODEL_NAMES[model],
    )
ax.set_xticks(range(len(YEARS)))
ax.set_xticklabels(YEARS)
ax.set_ylim(0, None)
ax.set_ylabel("Day-ahead error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, bottom=0.34)
style.save_figure(fig, OUT / "record_error_years", data=p)
