"""Day-ahead error on heat-wave and cold-wave days against the other days of the same site and
calendar month, by site group (every day of the test window, Chronos-2 without covariates).
A wave day has a site temperature beyond the 5th or 95th percentile of the days before the test
window, the label of Lu et al. (2026).

analysis/forecast_anatomy.py (daily_error_wave_matched.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

w = pd.read_csv(FA / "daily_error_wave_matched.csv")
w = w[w["group"].isin(style.SITE_GROUP_ORDER)]
cells = [(m, g) for m in ("cold wave", "heat wave") for g in style.SITE_GROUP_ORDER]
fig, ax = style.figure(style.WIDE_SINGLE)
for j, (matched, group) in enumerate(cells):
    d = w[(w["matched_to"] == matched) & (w["group"] == group)].set_index("wave")
    for wave, fill in ((matched, True), ("other days", False)):
        c = style.WAVE_COLORS[matched]
        ax.plot(
            j,
            d.loc[wave, "nmae_pct"],
            linestyle="none",
            marker="o",
            markersize=style.MARKER_SIZE + 1,
            markerfacecolor=c if fill else "white",
            markeredgecolor=c,
            markeredgewidth=1.0,
            zorder=3,
        )
handles = [
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE + 1,
        markerfacecolor=style.WAVE_COLORS[k] if fill else "white",
        markeredgecolor=style.WAVE_COLORS[k],
    )[0]
    for k, fill in (("cold wave", True), ("heat wave", True), ("other days", False))
]
ax.set_xticks(range(len(cells)))
ax.set_xticklabels([g for _, g in cells])
ax.set_ylim(0, None)
ax.set_ylabel("Normalised absolute error (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=["cold-wave days", "heat-wave days", "other days, same site and month"],
)
style.save_figure(fig, OUT / "wave_error", data=w)
