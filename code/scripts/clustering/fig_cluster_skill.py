"""Day-ahead CRPS skill of Chronos-2 + IFS ENS against seasonal naive per electricity-site
cluster (boxes: IQR, whiskers 5-95%). One plot."""

import numpy as np
import pandas as pd
from _common import OUT, POC, clusters

from sense_energy.visualization import style

MODEL = "chronos2_ifs"
table = clusters("elec", "site")
per = pd.read_csv(POC / "scores_per_site.csv")
per = per[per["model"] == MODEL].set_index("site_code")["crps_skill_vs_naive"]
frame = table[["cluster"]].join(per.rename("skill"), how="inner")
groups = [
    frame.loc[frame["cluster"] == i, "skill"].to_numpy() for i in sorted(frame["cluster"].unique())
]
labels = [
    f"cluster {i} (n = {len(g)})"
    for i, g in zip(sorted(frame["cluster"].unique()), groups, strict=True)
]

fig, ax = style.figure(style.SINGLE_COLUMN)
bp = ax.boxplot(
    groups,
    positions=np.arange(len(groups)),
    widths=0.55,
    whis=(5, 95),
    showfliers=False,
    patch_artist=True,
    medianprops={"color": "#000000", "linewidth": style.LINEWIDTH["secondary"]},
    whiskerprops={"linewidth": style.AXIS_LINEWIDTH},
    capprops={"linewidth": style.AXIS_LINEWIDTH},
    boxprops={"linewidth": style.AXIS_LINEWIDTH},
)
for i, patch in enumerate(bp["boxes"]):
    c = style.cluster_color(sorted(frame["cluster"].unique())[i])
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
    patch.set_edgecolor(c)
    for artist in (
        bp["whiskers"][2 * i],
        bp["whiskers"][2 * i + 1],
        bp["caps"][2 * i],
        bp["caps"][2 * i + 1],
    ):
        artist.set_color(c)
style.reference_line(ax, 0.0)
ax.set_xticks(np.arange(len(groups)))
ax.set_xticklabels(labels)
ax.set_ylabel("CRPS skill vs seasonal naive")
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "cluster_skill_elec_site", data=frame.reset_index())
