"""Each site's heat slope against temperature alone (x) and net of a site-specific sunshine term
(y). Sites on the diagonal owe nothing of their heat slope to sunshine; sites above it had a
negative sunshine response (panels, daylighting) read as a heat effect.

analysis/heat_drivers.py, specification B, sites without CHP. Colour by building use.
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features().dropna(subset=["heat_plain", "heat"])
f["use"] = f["use"].fillna("non-inpatient")
lim = float(np.percentile(np.abs(np.r_[f["heat_plain_pct"], f["heat_pct"]]), 98)) * 1.05

fig, ax = style.figure(style.SQUARE)
ax.plot([-lim, lim], [-lim, lim], color="#B8B8B8", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
for use in style.USE_ORDER:
    d = f[f["use"] == use]
    if d.empty:
        continue
    ax.plot(
        d["heat_plain_pct"],
        d["heat_pct"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.USE_COLORS[use],
        label=f"{use} ({len(d)})",
        zorder=3,
    )
neg_plain = ((f["heat_plain"] + 1.96 * f["heat_plain_se"]) < 0).sum()
neg_adj = ((f["heat"] + 1.96 * f["heat_se"]) < 0).sum()
ax.annotate(
    f"sites falling on hot days (p < 0.05):\n{neg_plain} against temperature alone,\n{neg_adj} net of sunshine",
    xy=(0.03, 0.97),
    xycoords="axes fraction",
    va="top",
    fontsize=style.FONT_SIZE["annotation"],
)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Heat slope, temperature only (% per °C above 20 °C)")
ax.set_ylabel("Heat slope, net of sunshine (% per °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.3)
print(
    f"corr {f[['heat_plain_pct', 'heat_pct']].corr().iloc[0, 1]:.3f}; falling {neg_plain} -> {neg_adj}"
)
style.save_figure(fig, OUT / "heat_sunshine_control", data=f)
