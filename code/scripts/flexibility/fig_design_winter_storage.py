"""The storage that would hold each over-capacity site under its connection in the one-in-twenty
winter with heat pumps: the store's size in hours of the connection's capacity against that
winter's peak as a share of capacity. Dashed lines at 2 and 24 hours separate battery scale, thermal-store
scale and beyond storage.

analysis/design_winter.py (design_sites.csv).
"""

import pandas as pd
from _common import DESIGN, OUT

from sense_energy.visualization import style

d = pd.read_csv(DESIGN / "design_sites.csv")
d = d[d["util_hp_design"] > 1]
fig, ax = style.figure(style.SINGLE_COLUMN)
for v in (2.0, 24.0):
    style.reference_line(ax, v)
for c in style.STORE_CLASS_ORDER:
    g = d[d["store_class_design"] == c]
    ax.plot(
        g["util_hp_design"],
        g["store_hours_design"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.STORE_CLASS_COLORS[c],
        label=c,
        zorder=3,
    )
ax.set_yscale("log")
ax.set_xlabel("Peak with heat pumps / capacity")
ax.set_ylabel("Store needed (hours of capacity)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, bottom=0.46, anchor_y=-0.3)
style.save_figure(fig, OUT / "design_winter_storage", data=d)
