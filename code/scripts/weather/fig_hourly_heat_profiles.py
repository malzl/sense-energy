"""Every site's heat response by hour of the day: the empirical-Bayes posterior of the hour's
demand change per degC of daily mean above 20 degC (smoothed over neighbouring hours), one row
per site, sites ordered by the hour at which their response peaks; the red band (16:00-19:00)
outlined.

analysis/hourly_response.py, analysis/estate_counterfactuals.py (smoothed_profiles).
"""

import matplotlib.patches as mpatches
import numpy as np
from _common import OUT

from sense_energy.analysis.estate_counterfactuals import smoothed_profiles
from sense_energy.visualization import style

p = smoothed_profiles()["heat"] * 100
order = p.idxmax(axis=1).sort_values(kind="stable").index
p = p.loc[order]
lim = float(np.nanpercentile(np.abs(p.to_numpy()), 97))

fig, ax = style.figure(style.WIDE_SINGLE)
im = ax.imshow(
    p.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest"
)
ax.add_patch(
    mpatches.Rectangle((15.5, -0.5), 3, len(p), fill=False, edgecolor="#222222", linewidth=0.8)
)
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
ax.set_yticks([])
ax.set_xlabel("Local hour of the day")
ax.set_ylabel(f"Sites, ordered by peak hour ({len(p)})")
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
cb.set_label("Heat response (% per °C above 20 °C)", fontsize=style.FONT_SIZE["label"])
cb.ax.tick_params(labelsize=style.FONT_SIZE["tick"])
style.style_axis(ax)
style.save_figure(fig, OUT / "hourly_heat_profiles", data=p.reset_index())
