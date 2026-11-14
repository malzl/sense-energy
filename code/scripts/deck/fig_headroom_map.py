"""Finding 1, map: gas-heated English sites without CHP (estates return 2023/24) by peak demand
with heat pumps over the capacity of the grid connection."""

import numpy as np
import pandas as pd
from _deck import LIGHT_BLUE, ORANGE, REPORTS, YELLOW, england_map, figure, legend_below, save

e = pd.read_csv(REPORTS / "estates" / "heat_pump_headroom_england.csv").dropna(subset=["latitude"])
e["band"] = np.select([e["util_hp"] > 1.0, e["util_hp"] > 0.8], ["above", "near"], "below")
bands = {
    "below": ("below 80 %", LIGHT_BLUE),
    "near": ("80 to 100 %", YELLOW),
    "above": ("above capacity", ORANGE),
}
fig, ax = figure(48, 56)
england_map(ax)
for key, (label, colour) in bands.items():
    d = e[e["band"] == key]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=5.5,
        color=colour,
        edgecolor="white",
        linewidth=0.2,
        zorder=3 if key != "above" else 4,
        label=label,
    )
legend_below(fig, ax, ncol=3, markerscale=1.6, handletextpad=0.1)
save(fig, "headroom_map", data=e[["site_code", "latitude", "longitude", "util_hp", "band"]])
print(e["band"].value_counts().to_dict(), round((e["util_hp"] > 1).mean(), 3))
