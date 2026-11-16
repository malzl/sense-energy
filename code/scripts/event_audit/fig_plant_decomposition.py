"""What the meter of a site with a plant shows in an event: the observed reaction of the summed
grid import, and the reaction that the import would show if the site's demand reacted like
that of the sites without a plant and the plant did not change. The gap is a change in plant
output. Bars: the assumption 'like all sites without a plant'; ticks: 'like the acute sites
without a plant'.

analysis/event_audit.py (plant_decomposition.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

p = table("plant_decomposition")
p = p[p["sites"] == "all sites with a plant"]
a = p[p["assumption"] == "like all sites without a plant"].set_index("type")
b = p[p["assumption"] == "like the acute sites without a plant"].set_index("type")
kinds = ["cold spell", "heatwave"]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
x = np.arange(len(kinds))
ax.bar(
    x - 0.19,
    a["import_reaction_summed_pct"].reindex(kinds),
    width=0.36,
    color="#CC79A7",
    label="observed at the meter",
)
ax.bar(
    x + 0.19,
    a["import_reaction_if_plant_unchanged_pct"].reindex(kinds),
    width=0.36,
    color="#B8B8B8",
    label="expected with the plant unchanged",
)
ax.scatter(
    x + 0.19,
    b["import_reaction_if_plant_unchanged_pct"].reindex(kinds),
    marker="_",
    s=160,
    color="black",
    zorder=4,
)
lo = a["import_reaction_without_one_site_min_pct"].reindex(kinds)
hi = a["import_reaction_without_one_site_max_pct"].reindex(kinds)
mid = a["import_reaction_summed_pct"].reindex(kinds)
ax.errorbar(
    x - 0.19, mid, yerr=[mid - lo, hi - mid], fmt="none", ecolor="black", elinewidth=0.8, capsize=0
)
ax.set_xticks(x)
ax.set_xticklabels([style.EVENT_NAMES[k] for k in kinds])
ax.set_ylabel("Reaction of the summed grid import (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, bottom=0.34)
style.save_figure(fig, OUT / "plant_decomposition", data=p)
