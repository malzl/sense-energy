"""Does a hot day move a site's demand peak? The local hour of the highest mean weekday demand
on summer days at or above 25 degC maximum (y) against other summer weekdays (x), one point per
site, jittered; above the diagonal the hot day moves the peak later.

analysis/hourly_response.py (site_profiles.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

p = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "hourly_response" / "site_profiles.csv").dropna(
    subset=["peak_hour_hot", "peak_hour_other"]
)
rng = np.random.default_rng(0)
jx, jy = rng.uniform(-0.25, 0.25, len(p)), rng.uniform(-0.25, 0.25, len(p))
later = p["peak_hour_hot"] > p["peak_hour_other"]
earlier = p["peak_hour_hot"] < p["peak_hour_other"]

fig, ax = style.figure(style.SQUARE)
ax.plot([0, 23], [0, 23], color="#B8B8B8", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
for mask, color, label in (
    (later, style.TERM_COLORS["cdd20"], f"later on hot days ({int(later.sum())})"),
    (earlier, style.TERM_COLORS["hdd14"], f"earlier ({int(earlier.sum())})"),
    (~later & ~earlier, "#7A7A7A", f"unchanged ({int((~later & ~earlier).sum())})"),
):
    ax.plot(
        p.loc[mask, "peak_hour_other"] + jx[mask.to_numpy()],
        p.loc[mask, "peak_hour_hot"] + jy[mask.to_numpy()],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=color,
        label=label,
        zorder=3,
    )
ax.set_xlim(5, 22)
ax.set_ylim(5, 22)
ax.set_xlabel("Peak hour, other summer weekdays")
ax.set_ylabel("Peak hour, summer weekdays ≥ 25 °C")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.22, bottom=0.3)
style.save_figure(fig, OUT / "hot_day_peak", data=p)
