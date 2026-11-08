"""Where the types are: each site coloured by its heat type (rises with heat / also falls with
sunshine / falls with sunshine only / falls with heat / neither), marker area by mean demand.

analysis/heat_drivers.py.
"""

import numpy as np
from _common import OUT, england_axes, heat_features, site_points

from sense_energy.visualization import style

f = heat_features().join(site_points().set_index("site_code")[["latitude", "longitude"]])
f = f.dropna(subset=["latitude", "type"])
size = 8 + 30 * np.sqrt(f["mean_kw"] / f["mean_kw"].median())

fig, ax = england_axes()
for kind in style.HEAT_TYPE_ORDER:
    d = f[f["type"] == kind]
    if d.empty:
        continue
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=size[d.index],
        color=style.HEAT_TYPE_COLORS[kind],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.9,
        label=f"{kind} ({len(d)})",
        zorder=3,
    )
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.02, bottom=0.3)
style.save_figure(fig, OUT / "map_heat_types", data=f)
