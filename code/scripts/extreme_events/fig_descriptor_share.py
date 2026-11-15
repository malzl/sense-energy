"""What the site descriptors explain: share of the between-site variance of the reaction that
each descriptor adds to the size of the event (adjusted R2, one descriptor at a time), sites
without on-site generation. Negative shares (a descriptor that costs more than it explains) are
drawn at zero.

analysis/extreme_events.py (alignment_fits.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

f = table("alignment_fits")
f = f[f["descriptor"].isin(style.EVENT_DESCRIPTOR_ORDER)]
order = list(style.EVENT_DESCRIPTOR_ORDER)[::-1]
fig, ax = style.figure(style.WIDE_SINGLE)
for j, kind in enumerate(("heatwave", "cold spell")):
    d = f[f["type"] == kind].set_index("descriptor").reindex(order)
    ax.barh(
        np.arange(len(order)) + (0.5 - j) * 0.36,
        100 * d["adj_r2_gain"].clip(lower=0),
        height=0.34,
        color=style.WEATHER_EVENT_COLORS[kind],
        label=style.EVENT_NAMES[kind],
    )
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlim(0, 40)
ax.set_xlabel("Variance of the site reactions explained (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.18, bottom=0.24)
style.save_figure(fig, OUT / "descriptor_share", data=f)
