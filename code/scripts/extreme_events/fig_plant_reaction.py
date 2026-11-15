"""On-site generation and the reaction: mean site reaction in heat waves and cold spells at the
sites with and without on-site generation (grid import where there is a plant), with 95 %
intervals across sites.

analysis/extreme_events.py (group_means.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

g = table("group_means")
g = g[g["column"] == "chp"]
kinds = ["cold spell", "heatwave"]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for j, group in enumerate(style.PLANT_GROUP_ORDER):
    d = g[g["level"] == group].set_index("type").reindex(kinds)
    ax.errorbar(
        np.arange(len(kinds)) + (j - 0.5) * 0.2,
        d["reaction_pct"],
        yerr=[d["reaction_pct"] - d["lo"], d["hi"] - d["reaction_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.PLANT_GROUP_COLORS[group],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=group,
    )
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels([style.EVENT_NAMES[k] for k in kinds])
ax.set_xlim(-0.5, len(kinds) - 0.5)
ax.set_ylabel("Reaction of demand (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / "plant_reaction", data=g)
