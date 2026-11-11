"""Weather risk in the annual electricity budget: today's electricity of the median site and the
electricity of the heat pumps (31 sites), under the weather of each year since 1960, against the
reference year 2023/24.

analysis/design_winter.py (years.csv).
"""

import pandas as pd
from _common import DESIGN, OUT

from sense_energy.visualization import style

y = pd.read_csv(DESIGN / "years.csv")
series = {
    "today": ("energy_change_median_pct", "today's electricity, median site"),
    "with heat pumps": ("hp_energy_vs_ref_pct", "heat-pump electricity"),
}
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for k, (col, label) in series.items():
    ax.plot(
        y["weather_year"],
        y[col],
        color=style.SCENARIO_COLORS[k],
        linewidth=style.LINEWIDTH["primary"],
        label=label,
    )
ax.set_xlabel("Weather year (April to March, by its first year)")
ax.set_ylabel("Annual electricity against 2023/24 (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, bottom=0.3)
style.save_figure(fig, OUT / "design_winter_budget", data=y)
