"""Finding 2, chart: share of the variance between sites that the seven NHS regions explain, for
the temperature of the event at the site and for the reaction of the site. Black ticks: the
share that regions get by chance."""

import numpy as np
import pandas as pd
from _deck import BLUE, CHARCOAL, MID_GREY, REPORTS, figure, legend_below, save

s = pd.read_csv(REPORTS / "extreme_events" / "spatial.csv")
s = s[s["sites"] == "all sites"]
kinds = ["heatwave", "cold spell"]
fig, ax = figure(25, 41)
for j, (q, colour, label) in enumerate(
    (("size of the event", MID_GREY, "temperature"), ("reaction", BLUE, "reaction"))
):
    d = s[s["quantity"] == q].set_index("type").reindex(kinds)
    pos = np.arange(len(kinds)) + (j - 0.5) * 0.38
    ax.bar(pos, 100 * d["region_share"], width=0.36, color=colour, label=label)
    ax.scatter(pos, 100 * d["region_share_chance"], marker="_", s=38, color=CHARCOAL, zorder=4)
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels(["heat\nwaves", "cold\nspells"])
ax.set_ylim(0, 60)
ax.set_yticks([0, 20, 40, 60])
ax.set_ylabel("Explained by region (%)")
ax.grid(axis="y")
ax.tick_params(axis="x", length=0)
legend_below(fig, ax, ncol=1)
save(fig, "region_share", data=s)
print(
    s[["type", "quantity", "morans_i", "p", "region_share", "region_share_chance", "region_p"]]
    .round(3)
    .to_string(index=False)
)
