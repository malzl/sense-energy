"""Ward dendrogram of the electricity trusts on their daily-profile shape, leaves coloured by
the k-means cluster. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, RESULTS, clusters
from scipy.cluster.hierarchy import dendrogram

from sense_energy.eda import common as C
from sense_energy.visualization import style

table = clusters("elec", "trust")
Z = np.load(RESULTS / "elec_trust_ward_linkage.npy")
order = pd.read_csv(RESULTS / "elec_trust_ward_order.csv")["series_id"].tolist()
names = [
    C.short_trust_name(table.loc[s, "organisation_name"]) if s in table.index else s for s in order
]

fig, ax = style.figure(style.WIDE_SINGLE)
d = dendrogram(
    Z,
    labels=names,
    orientation="left",
    ax=ax,
    color_threshold=0,
    above_threshold_color="#7A7A7A",
    leaf_font_size=style.FONT_SIZE["tick"],
)
for lbl in ax.get_yticklabels():
    sid = order[names.index(lbl.get_text())] if lbl.get_text() in names else None
    if sid in table.index:
        lbl.set_color(style.cluster_color(table.loc[sid, "cluster"]))
for line in ax.get_lines():
    line.set_linewidth(style.LINEWIDTH["secondary"])
for coll in ax.collections:
    coll.set_linewidth(style.LINEWIDTH["secondary"])
ax.set_xlabel("Ward distance")
style.style_axis(ax)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.subplots_adjust(left=0.45)
style.save_figure(
    fig,
    OUT / "cluster_dendrogram_elec_trust",
    data=pd.DataFrame({"series_id": order, "name": names}).merge(
        table[["cluster"]], left_on="series_id", right_index=True, how="left"
    ),
)
