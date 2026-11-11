"""Solar arrays sized from the meter against the arrays implied by the generation the 2024/25
estates return reports, at the four panel sites that report solar. The meter sees only what the
site consumes itself; the dashed line marks equality.

analysis/solar_programme.py (summer_measured.csv).
"""

import pandas as pd
from _common import OUT, SOLAR

from sense_energy.visualization import style

m = pd.read_csv(SOLAR / "summer_measured.csv")
m = m[m["eric_total_kwp"] > 0]
lim = (5, 1000)
fig, ax = style.figure(style.SQUARE)
ax.plot(lim, lim, **style.REFERENCE_LINE, zorder=1)
ax.plot(
    m["eric_total_kwp"],
    m["implied_kwp_level"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE + 1,
    color=style.COLORS["blue"],
    zorder=3,
)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(*lim)
ax.set_ylim(*lim)
ax.set_xlabel("Array implied by reported generation (kWp)")
ax.set_ylabel("Array sized from the meter (kWp)")
style.style_axis(ax)
style.save_figure(fig, OUT / "solar_metered_vs_reported", data=m)
