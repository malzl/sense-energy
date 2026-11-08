"""Day-to-day variability of weekday grid import per site, by group (on-site CHP, acute hospitals
without CHP, other sites). One plot.

Coefficient of variation of daily mean import on weekdays over the whole record (analysis/chp.py).
Points are spread horizontally by rank within the group (no random jitter); the black bar is the
group median.
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

p = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "chp" / "site_profile.csv").dropna(
    subset=["daily_cv"]
)

fig, ax = style.figure(style.SINGLE_COLUMN)
for i, g in enumerate(style.CHP_GROUP_ORDER):
    v = np.sort(100 * p.loc[p["group"] == g, "daily_cv"].to_numpy())
    x = i + (np.linspace(-0.28, 0.28, len(v)) if len(v) > 1 else np.zeros(1))
    ax.plot(
        x,
        v,
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.CHP_GROUP_COLORS[g],
    )
    ax.hlines(
        np.median(v), i - 0.33, i + 0.33, color="#000000", linewidth=style.LINEWIDTH["primary"]
    )
ax.set_xticks(range(len(style.CHP_GROUP_ORDER)))
ax.set_xticklabels([f"{g}\n({int((p['group'] == g).sum())} sites)" for g in style.CHP_GROUP_ORDER])
ax.set_ylim(bottom=0)
ax.set_ylabel("Day-to-day variability of import (CV, %)")
style.style_axis(ax, grid="y")
print(p.groupby("group")["daily_cv"].median().round(3).to_string())
style.save_figure(fig, OUT / "chp_variability", data=p)
