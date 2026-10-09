"""Fit of the heat-demand model on the project's gas meters: median R² of daily gas use
against heating degree-days, by the base temperature of the degree-days. One plot."""

import pandas as pd
from _common import OUT, RESULTS

from sense_energy.visualization import style

f = pd.read_csv(RESULTS / "heat_model_fits.csv")
g = (
    f.groupby("base_temp_c")["r2"]
    .agg(median="median", q25=lambda x: x.quantile(0.25), q75=lambda x: x.quantile(0.75), n="size")
    .reset_index()
)

fig, ax = style.figure(style.SINGLE_COLUMN)
c = style.SCENARIO_COLORS["with heat pumps"]
ax.fill_between(g["base_temp_c"], g["q25"], g["q75"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
ax.plot(
    g["base_temp_c"],
    g["median"],
    color=c,
    linewidth=style.LINEWIDTH["primary"],
    marker="o",
    markersize=style.MARKER_SIZE,
    markeredgecolor="white",
    markeredgewidth=style.MARKER_EDGE,
)
ax.set_xlabel("Base temperature of the degree-days (°C)")
ax.set_ylabel(f"R² of daily gas use (n = {int(g['n'].max())} meters)")
ax.set_ylim(0, 1)
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "heat_model", data=g)
