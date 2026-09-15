"""Electricity sites in the first two principal components of their daily-profile shape,
coloured by cluster, marker by organisation group. One plot."""

import pandas as pd
from _common import GROUP_MARKERS, OUT, clusters, profiles
from sklearn.decomposition import PCA

from sense_energy.analysis.clustering import PROFILE_COLS
from sense_energy.visualization import style

table = clusters("elec", "site")
X = profiles("elec", "site")[PROFILE_COLS].reindex(table.index)
pca = PCA(n_components=2, random_state=0).fit(X.to_numpy())
Z = pca.transform(X.to_numpy())
frame = pd.DataFrame(Z, index=X.index, columns=["pc1", "pc2"]).join(
    table[["cluster", "org_group", "mean_kw"]]
)

fig, ax = style.figure(style.SQUARE)
for i in sorted(frame["cluster"].unique()):
    for g, m in GROUP_MARKERS.items():
        d = frame[(frame["cluster"] == i) & (frame["org_group"] == g)]
        if d.empty:
            continue
        ax.plot(
            d["pc1"],
            d["pc2"],
            linestyle="none",
            marker=m,
            markersize=style.MARKER_SIZE,
            color=style.cluster_color(i),
            markeredgecolor="white",
            markeredgewidth=style.MARKER_EDGE,
            alpha=0.9,
        )
for i in sorted(frame["cluster"].unique()):
    ax.plot(
        [], [], linestyle="none", marker="o", color=style.cluster_color(i), label=f"cluster {i}"
    )
for g, m in GROUP_MARKERS.items():
    ax.plot([], [], linestyle="none", marker=m, color="#7A7A7A", label=g)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.0f} % of variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.0f} % of variance)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=4, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "cluster_pca_elec_site", data=frame.reset_index())
