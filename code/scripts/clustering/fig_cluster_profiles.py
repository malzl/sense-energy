"""Mean daily demand profile of each cluster (series mean = 1): weekdays solid, weekends
dashed, one colour per cluster. One plot. Arguments: energy level (default elec site)."""

import sys

import numpy as np
import pandas as pd
from _common import OUT, RESULTS, args, centroids, cluster_label, clusters, hours, profiles

from sense_energy.analysis.clustering import PROFILE_COLS
from sense_energy.visualization import style

energy, level = args()
K = (
    int(sys.argv[3]) if len(sys.argv) > 3 else None
)  # optional: a k other than the silhouette choice
table = clusters(energy, level)
if K is None:
    cen = centroids(energy, level)
    suffix = ""
else:  # centroids of the k-means solution with K clusters, ids ordered by members' mean size
    lab = pd.read_csv(RESULTS / f"{energy}_{level}_labels_all_k.csv").set_index("series_id")[
        f"k{K}"
    ]
    prof = profiles(energy, level)[PROFILE_COLS].reindex(lab.index)
    order = (
        table["mean_kw"].reindex(lab.index).groupby(lab).mean().sort_values(ascending=False).index
    )
    lab = lab.map({old: new for new, old in enumerate(order)})
    table = table.assign(cluster=lab.reindex(table.index))
    cen = prof.groupby(lab).mean()
    cen.index.name = "cluster"
    suffix = f"_k{K}"

fig, ax = style.figure(style.SINGLE_COLUMN)
for i, row in cen.iterrows():
    c = style.cluster_color(i)
    wd = row[[f"wd_{h:02d}" for h in range(48)]].to_numpy(dtype=float)
    we = row[[f"we_{h:02d}" for h in range(48)]].to_numpy(dtype=float)
    ax.plot(
        hours(), wd, color=c, linewidth=style.LINEWIDTH["primary"], label=cluster_label(i, table)
    )
    ax.plot(hours(), we, color=c, linewidth=style.LINEWIDTH["secondary"], linestyle="--")
ax.set_xlabel("Local time of day (h)")
ax.set_ylabel("Demand index (series mean = 1)")
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 25, 6))
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.26, bottom=0.34)
style.save_figure(fig, OUT / f"cluster_profiles_{energy}_{level}{suffix}", data=cen.reset_index())
