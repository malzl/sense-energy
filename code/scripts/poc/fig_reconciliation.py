"""Normalised MAE at every level of the hierarchy after each reconciliation approach, for the
primary model's base forecasts (Chronos-2 + IFS ENS), scored on the same origins. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, RESULTS
from _hierarchy import LEVEL_LABELS, LEVEL_ORDER

from sense_energy.visualization import style

MODEL = "chronos2_ifs"

r = pd.read_csv(RESULTS.with_name(RESULTS.name + "_hierarchy") / "reconciliation.csv")
r = r[r["model"] == MODEL]
r["approach"] = r["method"].map(style.RECONCILIATION_NAMES)
r["level_pos"] = r["level"].map({lv: i for i, lv in enumerate(LEVEL_ORDER)})
approaches = [a for a in style.RECONCILIATION_ORDER if a in set(r["approach"])]

fig, ax = style.figure(style.SINGLE_COLUMN)
for a in approaches:
    d = r[r["approach"] == a].sort_values("level_pos")
    ax.plot(
        d["level_pos"],
        d["nmae"] * 100,
        color=style.RECONCILIATION_COLORS[a],
        linestyle=style.RECONCILIATION_LINESTYLES[a],
        linewidth=style.LINEWIDTH["reference"]
        if a == "Base (direct)"
        else style.LINEWIDTH["secondary"],
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=a,
    )
ax.set_xticks(np.arange(len(LEVEL_ORDER)))
ax.set_xticklabels([LEVEL_LABELS[lv] for lv in LEVEL_ORDER])
ax.set_xlabel("Level of the hierarchy")
ax.set_ylabel("MAE / mean demand (%)")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.40)  # approach names
style.save_figure(
    fig,
    OUT / "reconciliation",
    data=r[["model", "method", "approach", "level", "n_series", "n_origins", "mae", "nmae"]],
)
