"""Inpatient sites without CHP: demand in each local hour against daily mean temperature, relative to
a 14-16 degC day of the same month. One plot (heatmap).

One binned fixed-effect regression per hour (analysis/weather_response.py). Diverging colours centred
on zero with symmetric limits; tail bins labelled < -2 and >= 24.
"""

import numpy as np
from _common import OUT, response

from sense_energy.visualization import style

GROUP = "inpatient"
h = response("hourly")
h = h[(h["group"] == GROUP) & (h["variable"] == "daily mean")]
m = h.pivot_table(index="hour", columns="bin_mid", values="coef_pct").sort_index()
labels = h.drop_duplicates("bin_mid").set_index("bin_mid")["bin"].reindex(m.columns)
lim = float(np.nanmax(np.abs(m.to_numpy())))

fig, ax = style.figure(style.WIDE_SINGLE)
im = ax.imshow(m.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, origin="lower")
ax.set_xticks(range(len(m.columns)))
ax.set_xticklabels(labels, rotation=0)
ax.set_yticks(range(0, 24, 3))
ax.set_yticklabels([f"{x:02d}:00" for x in range(0, 24, 3)])
ax.set_xlabel("Daily mean temperature, bin lower edge (°C)")
ax.set_ylabel("Hour of day (local time)")
cb = fig.colorbar(im, ax=ax, pad=0.02)
cb.set_label("Demand vs a 15 °C day (%)", fontsize=style.FONT_SIZE["label"])
cb.ax.tick_params(
    labelsize=style.FONT_SIZE["tick"], width=style.TICK_WIDTH, length=style.TICK_LENGTH
)
cb.outline.set_linewidth(style.AXIS_LINEWIDTH)
style.style_axis(ax, all_spines=True)
print(f"{GROUP}: |max| {lim:.1f} %")
style.save_figure(fig, OUT / "temperature_response_hourly", data=h)
