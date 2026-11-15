"""Does the reaction grow inside an event? Reaction per degree on the first day of an event and
from its third day on, sites without on-site generation that have both.

analysis/extreme_events.py (build_up.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

b = table("build_up").set_index("type").reindex(["cold spell", "heatwave"])
fig, ax = style.figure(style.SINGLE_COLUMN)
x = np.arange(len(b))
for j, (col, label, alpha) in enumerate(
    (
        ("first_day_pct_per_degree", "first day", 0.45),
        ("third_day_on_pct_per_degree", "third day on", 1.0),
    )
):
    ax.bar(
        x + (j - 0.5) * 0.36,
        b[col],
        width=0.34,
        color=[style.WEATHER_EVENT_COLORS[k] for k in b.index],
        alpha=alpha,
        label=label,
    )
ax.set_xticks(x)
ax.set_xticklabels([style.EVENT_NAMES[k] for k in b.index])
ax.set_ylabel("Reaction (% per °C)")
style.style_axis(ax, grid="y")
leg = style.add_bottom_legend(ax, ncol=2)
for h in leg.legend_handles:
    h.set_facecolor("#7A7A7A")
style.save_figure(fig, OUT / "build_up", data=b.reset_index())
