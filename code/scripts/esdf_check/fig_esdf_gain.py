"""The paper's method against its own backbone: error of ESDF (LSTM) relative to the LSTM without
synthetic samples, one point per series, on the window that the paper scores and on the other
days of the test period. Below zero the method is better.

experiments/esdf_check.py (scores.csv).
"""

import numpy as np
import pandas as pd
from _common import ESDF, OUT

from sense_energy.visualization import style

s = pd.read_csv(ESDF / "scores.csv")
p = s[s["model"].isin(["ESDF (LSTM)", "LSTM"])].pivot_table(
    index=["region", "series", "subset"], columns="model", values="mae"
)
p["change_pct"] = 100 * (p["ESDF (LSTM)"] / p["LSTM"] - 1)
p = p.reset_index()
subsets = ["paper window", "hot and cold days", "other days", "all test days"]
colors = {"europe": style.COLORS["blue"], "pjm": style.COLORS["orange"]}
rng = np.random.default_rng(0)
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for j, region in enumerate(("europe", "pjm")):
    for k, subset in enumerate(subsets):
        d = p[(p["region"] == region) & (p["subset"] == subset)]
        x = k + (j - 0.5) * 0.3 + rng.uniform(-0.08, 0.08, len(d))
        ax.plot(
            x,
            d["change_pct"].clip(-80, 80),
            linestyle="none",
            marker="o",
            markersize=style.MARKER_SIZE - 1,
            markeredgewidth=0,
            alpha=0.7,
            color=colors[region],
            label=style.ESDF_REGION_NAMES[region] if k == 0 else None,
        )
        ax.plot(
            [k + (j - 0.5) * 0.3 - 0.12, k + (j - 0.5) * 0.3 + 0.12],
            [d["change_pct"].median()] * 2,
            color=colors[region],
            linewidth=style.LINEWIDTH["primary"],
        )
ax.set_xticks(range(len(subsets)))
ax.set_xticklabels([style.ESDF_SUBSET_NAMES[k] for k in subsets])
ax.set_ylim(-80, 80)
ax.set_ylabel("Error of ESDF against its LSTM (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, anchor_y=-0.16, bottom=0.24)
style.save_figure(fig, OUT / "esdf_gain", data=p)
