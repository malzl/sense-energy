"""Quarterly mean electricity demand against occupied beds within each panel trust, both
standardised per trust (Dec 2022 - Jun 2024). One plot."""

import numpy as np
import pandas as pd
from _common import OUT, RESULTS

from sense_energy.visualization import style

b = pd.read_csv(RESULTS / "beds_vs_demand_quarterly.csv")
z = b.groupby("trust")[["demand_kw", "occupied_beds"]].transform(
    lambda x: (x - x.mean()) / x.std(ddof=0) if x.std(ddof=0) > 0 else x * 0
)
b = b.assign(demand_z=z["demand_kw"], beds_z=z["occupied_beds"])
r = np.corrcoef(b["beds_z"], b["demand_z"])[0, 1]

fig, ax = style.figure(style.SQUARE)
ax.plot(
    b["beds_z"],
    b["demand_z"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    color="#0072B2",
    markeredgecolor="white",
    markeredgewidth=style.MARKER_EDGE,
    alpha=0.85,
)
style.reference_line(ax, 0.0)
style.reference_line(ax, 0.0, axis="x")
ax.set_xlabel(f"Occupied beds (standardised within trust; r = {r:.2f})")
ax.set_ylabel("Mean demand (standardised within trust)")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
style.style_axis(ax)
style.save_figure(fig, OUT / "beds_vs_demand", data=b)
