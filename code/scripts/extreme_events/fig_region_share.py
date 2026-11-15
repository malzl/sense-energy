"""The weather is regional, the reaction is not: share of the between-site variance that the
seven NHS regions account for, for the temperature of the event at the site and for the
reaction of the site. Ticks: the share that regions get by chance (permutation mean).

analysis/extreme_events.py (spatial.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

s = table("spatial")
s = s[s["sites"] == "all sites"]
kinds = ["cold spell", "heatwave"]
fig, ax = style.figure(style.SINGLE_COLUMN)
for j, q in enumerate(("size of the event", "reaction")):
    d = s[s["quantity"] == q].set_index("type").reindex(kinds)
    pos = np.arange(len(kinds)) + (j - 0.5) * 0.36
    ax.bar(
        pos,
        100 * d["region_share"],
        width=0.34,
        color=style.EVENT_QUANTITY_COLORS[q],
        label=style.EVENT_QUANTITY_NAMES[q],
    )
    ax.scatter(pos, 100 * d["region_share_chance"], marker="_", s=120, color="black", zorder=4)
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels([style.EVENT_NAMES[k] for k in kinds])
ax.set_ylabel("Variance between sites explained by region (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / "region_share", data=s)
