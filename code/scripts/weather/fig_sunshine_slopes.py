"""Every site's sunshine slope: the change in daily electricity per kWh/m2 of surface solar
radiation, net of temperature, with its 95 % interval; sites ordered by slope and coloured by
building use. Small grey crosses: the same site's night-block sunshine slope (00-06), which
should sit at zero if the daytime slope is sunshine and not something that follows hot days.

analysis/heat_drivers.py, specification B, sites without CHP.
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features().dropna(subset=["sun", "sun_se"])
f = f[f["sun_se"] > 0].sort_values("sun_pct").reset_index()
f["use"] = f["use"].fillna("non-inpatient")
y = np.arange(len(f))
w = 1 / f["sun_se"] ** 2
pooled = float(100 * (w * f["sun"]).sum() / w.sum())

fig, ax = style.figure(style.WIDE_SHORT)
ax.errorbar(
    f["sun_pct"],
    y,
    xerr=1.96 * f["sun_se_pct"],
    fmt="none",
    ecolor="#B8B8B8",
    elinewidth=0.5,
    zorder=2,
)
for use in style.USE_ORDER:
    d = f[f["use"] == use]
    if d.empty:
        continue
    ax.plot(
        d["sun_pct"],
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
ax.plot(
    f["sun_night_pct"],
    y,
    linestyle="none",
    marker="x",
    markersize=2.5,
    markeredgewidth=0.5,
    color="#7A7A7A",
    label="night block (00–06), placebo",
    zorder=4,
)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, pooled, axis="x")
ax.set_yticks([])
ax.set_ylabel(f"Sites, ordered ({len(f)})")
ax.set_xlabel("Sunshine slope: demand change per kWh/m² of daily solar radiation (%)")
lim = float(np.percentile(np.abs(f["sun_pct"]) + 1.96 * f["sun_se_pct"], 95))
ax.set_xlim(-lim, lim)
style.style_axis(ax, wide_short=True, grid="x")
style.add_bottom_legend(ax, anchor_y=-0.3, bottom=0.34)
neg = ((f["sun"] + 1.96 * f["sun_se"]) < 0).sum()
pos = ((f["sun"] - 1.96 * f["sun_se"]) > 0).sum()
print(f"pooled {pooled:+.2f} % per kWh/m2; {neg} sites < 0, {pos} > 0")
style.save_figure(fig, OUT / "sunshine_slopes", data=f)
