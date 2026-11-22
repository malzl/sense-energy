"""Error of the next two hours when the forecast is made again every two hours on the day:
the day-ahead forecast, the day-ahead forecast plus its last known error, the day-ahead
forecast plus a zero-shot forecast of its error, a new zero-shot forecast of the demand, and
the last metered value. Every third day of the record; mean over sites with the 95 % interval
from resampling sites. The meter is taken as read in real time.

analysis/forecast_extensions.py (correction_errors.csv).
"""

import numpy as np
from _common import OUT, PLAIN, PLANT, table

from sense_energy.visualization import style

e = table("correction_errors")
groups = [PLAIN, PLANT, f"{PLANT}, days with a large step"]
labels = [
    "no on-site\ngeneration",
    "on-site\ngeneration",
    "on-site generation,\ndays with a large step",
]
models = list(style.CORRECTION_NAMES)
fig, ax = style.figure(style.WIDE_SINGLE)
for j, model in enumerate(models):
    d = e[e["model"] == model].set_index("group").reindex(groups)
    ax.errorbar(
        np.arange(len(groups)) + (j - (len(models) - 1) / 2) * 0.15,
        d["nmae_pct"],
        yerr=[d["nmae_pct"] - d["nmae_lo"], d["nmae_hi"] - d["nmae_pct"]],
        fmt=style.CORRECTION_MARKERS[model],
        markersize=style.MARKER_SIZE + 1,
        color=style.CORRECTION_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.CORRECTION_NAMES[model],
    )
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(labels)
ax.set_ylim(0, None)
ax.set_ylabel("Error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.4)
style.save_figure(fig, OUT / "correction_groups", data=e)
