"""Electricity sites on the England map coloured by profile cluster, marker area proportional
to mean demand. One plot."""

import numpy as np
from _common import OUT, clusters

from sense_energy.eda import common as C
from sense_energy.visualization import style

table = clusters("elec", "site")
pts = C.site_points().merge(table[["cluster", "mean_kw"]], left_on="site_code", right_index=True)
area = 8 + 160 * pts["mean_kw"] / pts["mean_kw"].max()

fig, ax = style.figure(style.SQUARE)
C.england().plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
for i in sorted(pts["cluster"].unique()):
    d = pts[pts["cluster"] == i]
    ax.scatter(
        d["longitude"],
        d["latitude"],
        s=area[d.index],
        color=style.cluster_color(i),
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        alpha=0.85,
        zorder=3,
        label=f"cluster {i} (n = {len(d)})",
    )
minx, miny, maxx, maxy = C.england().total_bounds
ax.set_xlim(minx - 0.2, maxx + 0.2)
ax.set_ylim(miny - 0.1, maxy + 0.1)
ax.set_aspect(1 / np.cos(np.deg2rad(53)))
ax.set_axis_off()
style.add_bottom_legend(ax, anchor_y=-0.02, bottom=0.12, ncol=3)
style.save_figure(
    fig,
    OUT / "cluster_map_elec_site",
    data=pts[["site_code", "latitude", "longitude", "cluster", "mean_kw"]],
)
