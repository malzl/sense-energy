"""Bottom-up minus direct normalised MAE at each aggregate level (site from meters; trust, region
and national from sites), per method: negative means summing the members' forecasts beats
forecasting the total directly. One plot."""

import numpy as np
from _common import OUT, display, ordered_models
from _hierarchy import LEVEL_LABELS, summary

from sense_energy.visualization import style

s = summary()
direct = s[s["kind"] == "direct"].set_index(["level", "model"])["nmae"]
bu = s[s["kind"] == "bottom_up"].set_index(["level", "model"])["nmae"]
both = (bu - direct.reindex(bu.index)).dropna().rename("gain").reset_index()
both["gain"] *= 100
levels = [lv for lv in ("site", "trust", "region", "total") if lv in set(both["level"])]
models = ordered_models(both["model"].unique())

fig, ax = style.figure(style.SINGLE_COLUMN)
x = {lv: i for i, lv in enumerate(levels)}
k = len(models)
for i, m in enumerate(models):
    d = both[both["model"] == m]
    name = display(m)
    ax.plot(
        d["level"].map(x) + (i - (k - 1) / 2) * 0.12,
        d["gain"],
        linestyle="none",
        marker=style.method_marker(name),
        markersize=style.MARKER_SIZE + 1,
        color=style.method_color(name),
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=name,
    )
style.reference_line(ax, 0.0)
ax.set_xticks(np.arange(len(levels)))
ax.set_xticklabels([LEVEL_LABELS[lv] for lv in levels])
ax.set_xlabel("Aggregate level")
ax.set_ylabel("Bottom-up − direct nMAE (percentage points)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36)  # product names
style.save_figure(fig, OUT / "bottom_up_gain", data=both.assign(method=both["model"].map(display)))
