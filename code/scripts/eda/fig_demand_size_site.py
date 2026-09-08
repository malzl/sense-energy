"""Distribution of mean site demand, electricity and gas, log scale. One plot."""

import numpy as np
import pandas as pd

from sense_energy.eda import common as C
from sense_energy.visualization import style

rows = []
for energy in ("elec", "gas"):
    _, index = C.series_for(energy, "site")
    for _, r in index.iterrows():
        rows.append(
            {"energy": energy, "series_id": r["series_id"], "mean_kw": r["mean_kwh"] * 2}
        )  # kWh per half hour -> kW
frame = pd.DataFrame(rows).dropna()

fig, ax = style.figure(style.WIDE_SHORT)
rng = np.random.default_rng(0)
for i, energy in enumerate(("elec", "gas")):
    vals = frame.loc[frame["energy"] == energy, "mean_kw"]
    y = i + rng.uniform(-0.18, 0.18, len(vals))
    ax.scatter(
        vals,
        y,
        s=style.MARKER_SIZE**2,
        facecolor=style.ENERGY_COLORS[energy],
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        alpha=0.85,
        label=f"{style.ENERGY_LABELS[energy]} (n = {len(vals)})",
        zorder=3,
    )
    q = vals.quantile([0.25, 0.5, 0.75])
    ax.plot(
        [q[0.25], q[0.75]],
        [i, i],
        color="#000000",
        linewidth=style.LINEWIDTH["secondary"],
        zorder=4,
    )
    ax.plot([q[0.5]], [i], marker="|", markersize=10, color="#000000", zorder=5)
ax.set_xscale("log")
ax.set_yticks([0, 1])
ax.set_yticklabels(["electricity", "gas"])
ax.set_xlabel("Mean demand (kW, site total)")
style.style_axis(ax, grid="x", wide_short=True)
style.add_bottom_legend(ax, anchor_y=-0.30, bottom=0.34, fontsize=8)
style.save_figure(fig, C.OUT / "demand_size_site", data=frame)
