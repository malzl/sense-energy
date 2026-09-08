"""Sites with gas data on an England outline, marker area proportional to mean demand. One plot."""

import numpy as np

from sense_energy.eda import common as C
from sense_energy.visualization import style

ENERGY = "gas"
_, index = C.series_for(ENERGY, "site")
pts = C.site_points().merge(
    index[["series_id", "mean_kwh"]], left_on="site_code", right_on="series_id"
)
pts["mean_kw"] = pts["mean_kwh"] * 2
top = pts["mean_kw"].max()
area = 6 + 240 * pts["mean_kw"] / top

fig, ax = style.figure(style.SQUARE)
C.england().plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
ax.scatter(
    pts["longitude"],
    pts["latitude"],
    s=area,
    facecolor=style.ENERGY_COLORS[ENERGY],
    edgecolor="white",
    linewidth=style.MARKER_EDGE,
    alpha=0.85,
    zorder=3,
)
for v in (top / 10, top / 2, top):
    ax.scatter(
        [],
        [],
        s=6 + 240 * v / top,
        facecolor=style.ENERGY_COLORS[ENERGY],
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        label=f"{v:,.0f} kW",
    )
minx, miny, maxx, maxy = C.england().total_bounds
ax.set_xlim(minx - 0.2, maxx + 0.2)
ax.set_ylim(miny - 0.1, maxy + 0.1)
ax.set_aspect(1 / np.cos(np.deg2rad(53)))
ax.set_axis_off()
style.add_bottom_legend(ax, anchor_y=-0.02, bottom=0.12, ncol=3)
style.save_figure(
    fig,
    C.OUT / f"map_sites_{ENERGY}",
    data=pts[["site_code", "site_name", "latitude", "longitude", "mean_kw"]],
)
