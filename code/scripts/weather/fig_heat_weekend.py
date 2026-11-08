"""Weekday against weekend heat slope, one point per site, net of sunshine; colour by building
use. Sites below the diagonal respond to heat on working days more than at weekends, the mark of
plant that runs with the working day; a ring marks sites whose weekend response is
significantly smaller.

analysis/heat_drivers.py (heat x weekend x site term in the specification-B regression).
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features().dropna(subset=["heat", "heat_weekend"])
f["use"] = f["use"].fillna("non-inpatient")
lim = float(np.percentile(np.abs(np.r_[f["heat_pct"], f["heat_weekend_pct"]]), 98)) * 1.05

fig, ax = style.figure(style.SQUARE)
ax.plot([-lim, lim], [-lim, lim], color="#B8B8B8", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
for use in style.USE_ORDER:
    d = f[f["use"] == use]
    if d.empty:
        continue
    ax.plot(
        d["heat_pct"],
        d["heat_weekend_pct"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.USE_COLORS[use],
        label=f"{use} ({len(d)})",
        zorder=3,
    )
wo = f[f["weekday_only"].astype(bool)]
ax.plot(
    wo["heat_pct"],
    wo["heat_weekend_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE + 3,
    markerfacecolor="none",
    markeredgewidth=0.7,
    markeredgecolor="#222222",
    label=f"weekend response significantly smaller ({len(wo)})",
    zorder=4,
)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Weekday heat slope (% per °C above 20 °C)")
ax.set_ylabel("Weekend heat slope (% per °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.32)
print(
    f"corr weekday-weekend {f[['heat_pct', 'heat_weekend_pct']].corr().iloc[0, 1]:.3f}; weekday-only {len(wo)}"
)
style.save_figure(fig, OUT / "heat_weekend", data=f)
