"""The two signatures side by side, one point per site: the daytime sunshine slope (06-18, net
of temperature; x) and the afternoon heat slope (12-18, net of sunshine; y). Colour by the
site's type, marker area by mean demand. Cooling plant lives in the upper half; panels and
daylighting in the left half.

analysis/heat_drivers.py.
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features().dropna(subset=["sun_day", "heat_b2"])
size = np.clip(2.5 + 1.2 * np.sqrt(f["mean_kw"] / f["mean_kw"].median()), 2.5, 7.0)

fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
for kind in style.HEAT_TYPE_ORDER:
    d = f[f["type"] == kind]
    if d.empty:
        continue
    for (_, r), s in zip(d.iterrows(), size[d.index], strict=True):
        ax.plot(
            r["sun_day_pct"],
            r["heat_b2_pct"],
            linestyle="none",
            marker="o",
            markersize=s,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=style.HEAT_TYPE_COLORS[kind],
            alpha=0.9,
            zorder=3,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.HEAT_TYPE_COLORS[kind],
        label=f"{kind} ({len(d)})",
    )
xl = float(np.percentile(np.abs(f["sun_day_pct"]), 98)) * 1.1
yl = float(np.percentile(np.abs(f["heat_b2_pct"]), 98)) * 1.1
ax.set_xlim(-xl, xl)
ax.set_ylim(-yl, yl)
ax.set_xlabel("Daytime sunshine slope, 06–18 (% per kWh/m²)")
ax.set_ylabel("Afternoon heat slope, 12–18 (% per °C above 20 °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.36)
print(f.groupby("type").size().reindex(style.HEAT_TYPE_ORDER).to_string())
style.save_figure(fig, OUT / "heat_signature", data=f)
