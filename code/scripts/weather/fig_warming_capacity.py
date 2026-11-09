"""Warming against connection capacity: for each whole-site-metered site, the summer peak
half-hour as a share of its available capacity today (open) and with every day 2 and 4 degC
warmer (filled), against the same site's winter peak with heat pumps replacing gas (diamond, from
the headroom analysis). Sites ordered by the +4 degC summer peak.

analysis/estate_counterfactuals.py (warming_sites.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

w = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "estate_counterfactuals" / "warming_sites.csv")
w = w[w["capacity_kw"].notna()]
u = w.pivot(index="site_code", columns="shift_c", values="summer_util")
u["winter_hp"] = w.groupby("site_code")["winter_util_heat_pumps"].first()
u = u.sort_values(4.0)
y = np.arange(len(u))

fig, ax = style.figure(style.DOUBLE_COLUMN)
ax.axvline(1.0, color="#222222", linewidth=0.8, zorder=1)
ax.axvline(0.9, color="#888888", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
ax.hlines(y, u[0.0], u[4.0], color="#CFCFCF", linewidth=0.8, zorder=2)
ax.plot(
    u[0.0],
    y,
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    markerfacecolor="white",
    markeredgecolor="#7A7A7A",
    markeredgewidth=0.8,
    label="summer peak today",
    zorder=3,
)
ax.plot(
    u[2.0],
    y,
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    color="#E69F00",
    markeredgecolor="white",
    markeredgewidth=style.MARKER_EDGE,
    label="+2 °C",
    zorder=4,
)
ax.plot(
    u[4.0],
    y,
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    color="#D55E00",
    markeredgecolor="white",
    markeredgewidth=style.MARKER_EDGE,
    label="+4 °C",
    zorder=5,
)
ax.plot(
    u["winter_hp"].clip(upper=3.0),
    y,
    linestyle="none",
    marker="D",
    markersize=style.MARKER_SIZE - 0.5,
    color="#0072B2",
    label="winter peak with heat pumps",
    zorder=4,
)
ax.set_yticks(y)
ax.set_yticklabels(u.index, fontsize=style.FONT_SIZE["tick"] - 3)
ax.set_xlabel("Peak half-hour as a share of available capacity (winter values capped at 3)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=4, anchor_y=-0.16, bottom=0.18)
style.save_figure(fig, OUT / "warming_capacity", data=u.reset_index())
