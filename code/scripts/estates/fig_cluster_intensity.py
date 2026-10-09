"""Electricity per square metre (ERIC 2023/24) of the panel's electricity sites by demand-shape
cluster (four-group solution; boxes: IQR, whiskers 5-95 %). One plot."""

import numpy as np
import pandas as pd
from _common import CLUSTERING, OUT, eric

from sense_energy.visualization import style

t = pd.read_csv(CLUSTERING / "elec_site_clusters.csv").set_index("series_id")
lab = pd.read_csv(CLUSTERING / "elec_site_labels_all_k.csv").set_index("series_id")["k4"]
order = t["mean_kw"].reindex(lab.index).groupby(lab).mean().sort_values(ascending=False).index
t["cluster"] = lab.map({o: n for n, o in enumerate(order)}).reindex(t.index)
e = eric().drop_duplicates("site_code").set_index("site_code")
t = t.join(e[["elec_kwh_per_m2", "mean_build_year", "chp_units"]], how="inner").dropna(
    subset=["elec_kwh_per_m2"]
)
t = t[(t["elec_kwh_per_m2"] > 1) & (t["elec_kwh_per_m2"] < 2000)]
t.index.name = "series_id"
ids = sorted(t["cluster"].dropna().unique())
groups = [t.loc[t["cluster"] == i, "elec_kwh_per_m2"].to_numpy() for i in ids]

fig, ax = style.figure(style.SINGLE_COLUMN)
bp = ax.boxplot(
    groups,
    positions=np.arange(len(ids)),
    widths=0.55,
    whis=(5, 95),
    showfliers=False,
    patch_artist=True,
    medianprops={"color": "#000000", "linewidth": style.LINEWIDTH["secondary"]},
    whiskerprops={"linewidth": style.AXIS_LINEWIDTH},
    capprops={"linewidth": style.AXIS_LINEWIDTH},
    boxprops={"linewidth": style.AXIS_LINEWIDTH},
)
for k, (patch, i) in enumerate(zip(bp["boxes"], ids, strict=True)):
    c = style.cluster_color(int(i))
    patch.set_facecolor(c)
    patch.set_alpha(0.35)
    patch.set_edgecolor(c)
    for a in (
        bp["whiskers"][2 * k],
        bp["whiskers"][2 * k + 1],
        bp["caps"][2 * k],
        bp["caps"][2 * k + 1],
    ):
        a.set_color(c)
ax.set_xticks(np.arange(len(ids)))
ax.set_xticklabels([f"cluster {int(i)}\n(n = {len(g)})" for i, g in zip(ids, groups, strict=True)])
ax.set_ylabel("Electricity (kWh per m² per year)")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.save_figure(
    fig,
    OUT / "cluster_intensity",
    data=t.reset_index()[
        ["series_id", "cluster", "elec_kwh_per_m2", "mean_build_year", "chp_units"]
    ],
)
