"""Share of English NHS sites (ERIC 2023/24, gas-heated, no CHP) whose peak demand stays below
a given share of available grid capacity, today and with all gas heating from heat pumps. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, headroom_england

from sense_energy.visualization import style

e = headroom_england()
grid = np.logspace(-2, 1, 400)
frame = pd.DataFrame({"util": grid})

fig, ax = style.figure(style.SINGLE_COLUMN)
for col, name in (("util_now", "today"), ("util_hp", "with heat pumps")):
    v = np.sort(e[col].dropna().to_numpy())
    share = np.searchsorted(v, grid, side="right") / len(v) * 100
    frame[name] = share
    ax.plot(
        grid,
        share,
        color=style.SCENARIO_COLORS[name],
        linewidth=style.LINEWIDTH["primary"],
        label=name,
    )
style.reference_line(ax, 1.0, axis="x")
ax.set_xscale("log")
ax.set_xlim(0.02, 10)
ax.xaxis.set_major_formatter(
    __import__("matplotlib.ticker", fromlist=["FuncFormatter"]).FuncFormatter(lambda v, _: f"{v:g}")
)
ax.set_ylim(0, 100)
ax.set_xlabel("Peak demand / available capacity")
ax.set_ylabel(f"Sites at or below (%, n = {len(e)})")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "headroom_england", data=frame)
