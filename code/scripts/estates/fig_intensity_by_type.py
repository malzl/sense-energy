"""Electricity use per square metre of floor area by site type, all English NHS sites in ERIC
2023/24 (types with at least 20 sites; boxes: IQR, whiskers 5-95 %). One plot."""

import numpy as np
from _common import OUT, eric, short_type

from sense_energy.visualization import style

e = eric()
e = e[(e["elec_kwh_per_m2"] > 1) & (e["elec_kwh_per_m2"] < 2000)]
e["type"] = e["site_type"].map(short_type)
counts = e["type"].value_counts()
types = counts[counts >= 20].index
order = (
    e[e["type"].isin(types)]
    .groupby("type")["elec_kwh_per_m2"]
    .median()
    .sort_values()
    .index.tolist()
)
groups = [e.loc[e["type"] == t, "elec_kwh_per_m2"].to_numpy() for t in order]

fig, ax = style.figure(style.WIDE_SHORT)
bp = ax.boxplot(
    groups,
    positions=np.arange(len(groups)),
    widths=0.55,
    whis=(5, 95),
    showfliers=False,
    patch_artist=True,
    medianprops={"color": "#000000", "linewidth": style.LINEWIDTH["secondary"]},
    whiskerprops={"linewidth": style.AXIS_LINEWIDTH, "color": "#7A7A7A"},
    capprops={"linewidth": style.AXIS_LINEWIDTH, "color": "#7A7A7A"},
    boxprops={"linewidth": style.AXIS_LINEWIDTH, "edgecolor": "#7A7A7A", "facecolor": "#B8B8B8"},
)
ax.set_xticks(np.arange(len(order)))
ax.set_xticklabels([f"{t} ({counts[t]})" for t in order], rotation=25, ha="right")
ax.set_ylabel("Electricity (kWh per m² per year)")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y", wide_short=True)
fig.subplots_adjust(bottom=0.42)
summary = e[e["type"].isin(types)].groupby("type")["elec_kwh_per_m2"].describe()
style.save_figure(fig, OUT / "intensity_by_type", data=summary.reset_index())
