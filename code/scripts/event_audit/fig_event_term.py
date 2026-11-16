"""Is an event more than its temperatures? The extra reaction from the third day of an event
on, after the model has the temperature of the day alone, and after it has the temperature of
the day and of the three days before. Heat: degrees of daily maximum above 24 degC. Cold:
degrees of daily mean. 95 % intervals clustered by site and date. Zero means the temperatures
explain the event.

analysis/event_audit.py (lag_models.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

lag = table("lag_models")
lag = lag[lag["event_terms"] & (lag["term"] == "late")]
models = ["the day's temperature", "the day and three days before"]
labels = ["temperature\nof the day", "temperature of the day\nand three days before"]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for j, (season, kind) in enumerate((("cold", "cold spell"), ("heat", "heatwave"))):
    d = lag[lag["season"] == season].set_index("model").reindex(models)
    ax.errorbar(
        np.arange(len(models)) + (j - 0.5) * 0.2,
        d["coef"],
        yerr=[d["coef"] - d["lo"], d["hi"] - d["coef"]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.WEATHER_EVENT_COLORS[kind],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.EVENT_NAMES[kind],
    )
ax.set_xticks(range(len(models)))
ax.set_xticklabels(labels)
ax.set_xlim(-0.5, len(models) - 0.5)
ax.set_xlabel("What the model knows")
ax.set_ylabel("Extra reaction from day 3 of an event (points)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.36, bottom=0.4)
style.save_figure(fig, OUT / "event_term", data=lag)
