"""The heat dose split in two, one point per site: the warm-night slope (per degC of daily
minimum above 12 degC; x) and the hot-afternoon slope (per degC of daily maximum above 24 degC; y),
both net of cold and sunshine. Cooling plant lives in the upper half; heating that still runs in
the warm season, and switches off on warm nights, in the left half. Colour by the sign of the
site's daily heat slope.

analysis/heat_drivers.py (night_day_terms).
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features().dropna(subset=["warm_night", "hot_afternoon"])
size = np.clip(2.5 + 1.2 * np.sqrt(f["mean_kw"] / f["mean_kw"].median()), 2.5, 7.0)

fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
for sign in style.HEAT_SIGN_ORDER:
    d = f[f["sign"] == sign]
    if d.empty:
        continue
    ax.errorbar(
        d["warm_night_pct"],
        d["hot_afternoon_pct"],
        xerr=1.96 * d["warm_night_se_pct"],
        yerr=1.96 * d["hot_afternoon_se_pct"],
        fmt="none",
        ecolor="#D8D8D8",
        elinewidth=0.4,
        zorder=2,
    )
    for (_, r), s in zip(d.iterrows(), size[d.index], strict=True):
        ax.plot(
            r["warm_night_pct"],
            r["hot_afternoon_pct"],
            linestyle="none",
            marker="o",
            markersize=s,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=style.HEAT_SIGN_COLORS[sign],
            alpha=0.9,
            zorder=3,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.HEAT_SIGN_COLORS[sign],
        label=f"{sign} on the daily slope ({len(d)})",
    )
xl = float(np.percentile(np.abs(f["warm_night_pct"]), 97)) * 1.15
yl = float(np.percentile(np.abs(f["hot_afternoon_pct"]), 97)) * 1.15
ax.set_xlim(-xl, xl)
ax.set_ylim(-yl, yl)
ax.set_xlabel("Warm-night slope (% per °C of daily minimum above 12 °C)")
ax.set_ylabel("Hot-afternoon slope (% per °C of daily maximum above 24 °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.32)
print(
    "fallers with a heating tail:",
    int((f["heating_tail"] & (f["sign"] == "falls with heat")).sum()),
    "of",
    int((f["sign"] == "falls with heat").sum()),
    "; risers with afternoon cooling:",
    int((f["cooling_afternoon"] & (f["sign"] == "rises with heat")).sum()),
    "of",
    int((f["sign"] == "rises with heat").sum()),
)
style.save_figure(fig, OUT / "heat_night_day", data=f)
