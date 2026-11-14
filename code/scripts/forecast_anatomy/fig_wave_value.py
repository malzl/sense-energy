"""Share of the achievable peak cut that a battery captures when it plans on the day-ahead
forecast, on heat-wave and cold-wave days against the other days of the same site and calendar
month (all sites, every day of the test window, Chronos-2 without covariates), by plan.

analysis/forecast_anatomy.py (daily_value_wave_matched.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

v = pd.read_csv(FA / "daily_value_wave_matched.csv")
v = v[v["group"] == "all sites"]
cells = [
    ("cold wave", "other days"),
    ("cold wave", "cold wave"),
    ("heat wave", "other days"),
    ("heat wave", "heat wave"),
]
labels = [
    "other days\n(cold months)",
    "cold-wave days",
    "other days\n(hot months)",
    "heat-wave days",
]
fig, ax = style.figure(style.WIDE_SINGLE)
for j, (matched, wave) in enumerate(cells):
    r = v[(v["matched_to"] == matched) & (v["wave"] == wave)].iloc[0]
    c = style.WAVE_COLORS[wave]
    for plan, fill in (("point", True), ("q70", False), ("level", True)):
        ax.plot(
            j,
            r[f"captured_{plan}_pct"],
            linestyle="none",
            marker="_" if plan == "level" else style.PLAN_MARKERS[plan],
            markersize=style.MARKER_SIZE + (5 if plan == "level" else 1),
            markerfacecolor=c if fill else "white",
            markeredgecolor=c,
            markeredgewidth=1.4 if plan == "level" else 1.0,
            zorder=3,
        )
handles = [
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="_" if plan == "level" else style.PLAN_MARKERS[plan],
        markersize=style.MARKER_SIZE + (5 if plan == "level" else 1),
        markerfacecolor="#000000" if fill else "white",
        markeredgecolor="#000000",
        markeredgewidth=1.4 if plan == "level" else 1.0,
    )[0]
    for plan, fill in (("point", True), ("q70", False), ("level", True))
]
ax.set_xticks(range(len(cells)))
ax.set_xticklabels(labels)
ax.set_xlim(-0.5, len(cells) - 0.5)
ax.set_ylim(0, 100)
ax.set_ylabel("Share of the achievable peak cut (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=[style.PLAN_NAMES[p] for p in ("point", "q70", "level")],
    ncol=1,
    anchor_y=-0.2,
    bottom=0.42,
)
style.save_figure(fig, OUT / "wave_value", data=v)
