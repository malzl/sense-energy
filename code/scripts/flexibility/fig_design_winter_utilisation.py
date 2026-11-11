"""Peak demand with heat pumps as a share of the connection's capacity, site by site: in the
reference year, in the one-in-twenty winter of the last 30 years and in the worst winter since
1960. Sites ordered by the one-in-twenty value; the dashed line is the capacity.

analysis/design_winter.py (design_sites.csv).
"""

import numpy as np
import pandas as pd
from _common import DESIGN, OUT

from sense_energy.visualization import style

d = pd.read_csv(DESIGN / "design_sites.csv").sort_values("util_hp_design").reset_index(drop=True)
cols = {
    "reference year": "util_hp_ref",
    "one-in-twenty winter": "util_hp_design",
    "worst winter on record": "util_hp_worst",
}
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 1.0)
x = np.arange(len(d))
ax.vlines(x, d["util_hp_ref"], d["util_hp_worst"], color=style.SPAGHETTI, linewidth=0.8, zorder=2)
for name, col in cols.items():
    ax.plot(
        x,
        d[col],
        linestyle="none",
        marker=style.DESIGN_CASE_MARKERS[name],
        markersize=style.MARKER_SIZE,
        color=style.DESIGN_CASE_COLORS[name],
        label=name,
        zorder=3,
    )
ax.set_xticks([])
ax.set_xlabel("Sites, ordered by the one-in-twenty winter")
ax.set_ylabel("Peak with heat pumps / capacity")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, bottom=0.26, anchor_y=-0.14)
style.save_figure(fig, OUT / "design_winter_utilisation", data=d)
