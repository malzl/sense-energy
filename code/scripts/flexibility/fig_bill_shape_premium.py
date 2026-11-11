"""What each site's load shape costs at the wholesale index: the price it pays per kWh against the
time-average price of the same half-hours, financial year 2024/25, one point per site by group;
bars mark the group medians.

analysis/flexibility.py (bill_anatomy.csv).
"""

import numpy as np
import pandas as pd
from _common import FLEX, OUT, YEAR

from sense_energy.visualization import style

b = pd.read_csv(FLEX / "bill_anatomy.csv")
b = b[b["fy"] == YEAR]
rng = np.random.default_rng(1)
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for k, g in enumerate(style.SITE_GROUP_ORDER):
    d = b.loc[b["group"] == g, "shape_premium_pct"]
    ax.plot(
        k + rng.uniform(-0.2, 0.2, len(d)),
        d,
        linestyle="none",
        marker="o",
        markersize=3,
        markeredgewidth=0,
        color=style.SITE_GROUP_COLORS[g],
        alpha=0.8,
        zorder=3,
    )
    ax.hlines(d.median(), k - 0.32, k + 0.32, color="#000000", linewidth=1.2, zorder=4)
ax.set_xticks(range(len(style.SITE_GROUP_ORDER)))
ax.set_xticklabels([f"{g}\n({int((b['group'] == g).sum())})" for g in style.SITE_GROUP_ORDER])
ax.set_ylabel("Price paid above the time-average price (%)")
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "bill_shape_premium", data=b)
