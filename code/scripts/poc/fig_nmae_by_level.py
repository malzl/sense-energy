"""Normalised MAE of direct forecasts at every level of the hierarchy (meter -> site -> trust ->
region -> national) for the plain variant of each family, SARIMA and the baselines. One plot."""

import numpy as np
from _common import OUT, display, plain
from _hierarchy import LEVEL_LABELS, LEVEL_ORDER, summary

from sense_energy.visualization import style

s = summary()
s = s[s["kind"] == "direct"]
models = plain(s["model"].unique())

fig, ax = style.figure(style.SINGLE_COLUMN)
for m in models:
    d = s[s["model"] == m].sort_values("level_pos")
    name = display(m)
    baseline = style.METHOD_ROLES[name] == "simple_baseline"
    ax.plot(
        d["level_pos"],
        d["nmae"] * 100,
        color=style.method_color(name),
        linestyle=style.method_linestyle(name),
        linewidth=style.LINEWIDTH["reference"] if baseline else style.LINEWIDTH["primary"],
        marker=style.method_marker(name),
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=name,
    )
ax.set_xticks(np.arange(len(LEVEL_ORDER)))
ax.set_xticklabels([LEVEL_LABELS[lv] for lv in LEVEL_ORDER])
ax.set_xlabel("Level of the hierarchy")
ax.set_ylabel("MAE / mean demand (%)")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36)  # product names
style.save_figure(
    fig,
    OUT / "nmae_by_level",
    data=s[["level", "model", "method", "sites", "nmae", "crps_q", "coverage_80"]],
)
