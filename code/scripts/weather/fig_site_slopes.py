"""Every site's cold and heat slope: the raw estimate with its 95 % interval, and the empirical-Bayes
posterior after shrinkage towards the pool, sites ordered by posterior. One plot per dose
(argument: hdd14 or cdd20).

Dose x site interactions in one specification-B regression (analysis/weather_heterogeneity.py);
the dashed line is the pooled slope. Colour by building use.
"""

import sys

import numpy as np
from _common import OUT, heterogeneity

from sense_energy.visualization import style

DOSE = sys.argv[1] if len(sys.argv) > 1 else "hdd14"
s = heterogeneity("site_slopes")
s = s[s["dose"] == DOSE].sort_values("posterior_pct").reset_index(drop=True)
s["use"] = s["use"].fillna("non-inpatient")
pooled = 100 * np.expm1(s["pooled"].iloc[0])
y = np.arange(len(s))

fig, ax = style.figure(style.WIDE_SHORT)
ax.errorbar(
    s["slope_pct"],
    y,
    xerr=1.96 * s["se_pct"],
    fmt="none",
    ecolor="#B8B8B8",
    elinewidth=0.5,
    zorder=2,
)
for use in style.USE_ORDER:
    d = s[s["use"] == use]
    ax.plot(
        d["posterior_pct"],
        y[d.index],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.USE_COLORS[use],
        label=f"{use} ({len(d)})",
        zorder=3,
    )
style.reference_line(ax, pooled, axis="x")
style.reference_line(ax, 0.0, axis="x")
ax.set_yticks([])
ax.set_ylabel(f"Sites, ordered ({len(s)})")
ax.set_xlabel(
    f"{style.DOSE_NAMES[DOSE][:1].upper() + style.DOSE_NAMES[DOSE][1:]}: demand change (% per °C)"
)
lim = np.percentile(np.abs(s["slope_pct"]) + 1.96 * s["se_pct"], 95)
ax.set_xlim(-lim, lim)
style.style_axis(ax, wide_short=True, grid="x")
style.add_bottom_legend(ax, anchor_y=-0.3, bottom=0.34)
print(f"{DOSE}: pooled {pooled:+.2f}, tau {100 * s['tau'].iloc[0]:.2f}, I2 {s['i2'].iloc[0]:.2f}")
style.save_figure(fig, OUT / f"site_slopes_{DOSE}", data=s)
