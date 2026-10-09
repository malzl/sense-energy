"""Share of English NHS sites (ERIC 2023/24, gas-heated, no CHP) whose peak would exceed
available grid capacity with all gas heating from heat pumps, by site type (types with at
least 20 sites). One plot."""

import numpy as np
from _common import OUT, headroom_england, short_type

from sense_energy.visualization import style

e = headroom_england()
e["type"] = e["site_type"].map(short_type)
g = e.groupby("type").agg(
    n=("util_hp", "size"),
    today=("util_now", lambda x: (x > 1).mean() * 100),
    hp=("util_hp", lambda x: (x > 1).mean() * 100),
)
g = g[g["n"] >= 20].sort_values("hp")
y = np.arange(len(g))

fig, ax = style.figure(style.SINGLE_COLUMN)
ax.barh(
    y,
    g["hp"],
    color=style.SCENARIO_COLORS["with heat pumps"],
    height=0.6,
    label="with heat pumps",
    zorder=2,
)
ax.barh(y, g["today"], color=style.SCENARIO_COLORS["today"], height=0.6, label="today", zorder=3)
ax.set_yticks(y)
ax.set_yticklabels([f"{t} ({n})" for t, n in zip(g.index, g["n"], strict=True)])
ax.set_xlim(0, 100)
ax.set_xlabel("Sites above available capacity (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "headroom_by_type", data=g.reset_index())
