"""English NHS sites (ERIC 2023/24, gas-heated, no CHP) coloured by peak demand over available
grid capacity with all gas heating from heat pumps. One plot."""

import numpy as np
from _common import OUT, headroom_class, headroom_england

from sense_energy.eda import common as C
from sense_energy.visualization import style

e = headroom_england().dropna(subset=["latitude", "longitude"])
e["headroom"] = headroom_class(e["util_hp"])

fig, ax = style.figure(style.SQUARE)
C.england().plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
for name, colour in style.HEADROOM_CLASSES.items():
    d = e[e["headroom"] == name]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=7,
        color=colour,
        edgecolor="white",
        linewidth=0.3,
        alpha=0.9,
        zorder=3,
        label=f"{name} (n = {len(d)})",
    )
minx, miny, maxx, maxy = C.england().total_bounds
ax.set_xlim(minx - 0.2, maxx + 0.2)
ax.set_ylim(miny - 0.1, maxy + 0.1)
ax.set_aspect(1 / np.cos(np.deg2rad(53)))
ax.set_axis_off()
style.add_bottom_legend(ax, anchor_y=-0.02, bottom=0.12, ncol=3)
style.save_figure(
    fig,
    OUT / "headroom_map",
    data=e[["site_code", "site_type", "latitude", "longitude", "util_now", "util_hp", "headroom"]],
)
