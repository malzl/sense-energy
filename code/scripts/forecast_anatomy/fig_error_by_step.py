"""Share of the panel's forecast error by the size of the day's largest step that the last four
weeks do not explain (two hours after against two hours before, as a share of the site's mean
demand), per site group. Best model, 91 sites, 69 test days.

analysis/forecast_anatomy.py (error_by_group_step.csv).
"""

import numpy as np
import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

t = pd.read_csv(FA / "error_by_group_step.csv")
fig, ax = style.figure(style.WIDE_SINGLE)
x = np.arange(len(style.STEP_ORDER))
width = 0.26
for k, group in enumerate(style.SITE_GROUP_ORDER):
    d = t[t["group"] == group].set_index("step_bin").reindex(style.STEP_ORDER)
    ax.bar(
        x + (k - 1) * width,
        d["error_share_pct"],
        width=width,
        color=style.SITE_GROUP_COLORS[group],
        label=group,
    )
ax.set_xticks(x)
ax.set_xticklabels(list(style.STEP_ORDER))
ax.set_xlabel("Largest unexplained step of the day (share of mean demand)")
ax.set_ylabel("Share of the panel's error (%)")
ax.set_ylim(0, 50)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, anchor_y=-0.26, bottom=0.32)
style.save_figure(fig, OUT / "error_by_step", data=t)
