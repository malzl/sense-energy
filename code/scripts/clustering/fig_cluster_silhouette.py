"""Silhouette score of the k-means solution against k for electricity meters, sites and trusts. One plot."""

import pandas as pd
from _common import LEVEL_NAMES, OUT, silhouette

from sense_energy.visualization import style

frames = []
fig, ax = style.figure(style.SINGLE_COLUMN)
for i, level in enumerate(("meter", "site", "trust")):
    s = silhouette("elec", level).assign(level=level)
    frames.append(s)
    ax.plot(
        s["k"],
        s["silhouette"],
        color=style.cluster_color(i),
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=LEVEL_NAMES[level],
    )
ax.set_xlabel("Number of clusters k")
ax.set_ylabel("Silhouette score")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "cluster_silhouette_elec", data=pd.concat(frames))
