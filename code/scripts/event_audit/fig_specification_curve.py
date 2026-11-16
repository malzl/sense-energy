"""One headline number of the event findings under every specification (argument: reaction,
build_up, evening, plant, region or own_events). Each small point is one of the 90 combinations
of reference weeks, event definition, site sample and treatment of holidays, in order of size;
the large point is the baseline with its 95 % interval; the band is the range of the placebo
events (the catalogue's events moved in time), which should hold zero.

analysis/event_audit.py (specification_grid.csv, placebo.csv).
"""

import numpy as np
from _common import OUT, metric_arg, table

from sense_energy.visualization import style

key, metric, label = metric_arg()
g, p = table("specification_grid"), table("placebo")
g, p = g[g["metric"] == metric], p[p["metric"] == metric]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
offset = 0
for kind in ("cold spell", "heatwave"):
    d = g[g["type"] == kind].sort_values("value").reset_index(drop=True)
    if d.empty:
        continue
    color = style.WEATHER_EVENT_COLORS[kind]
    x = offset + np.arange(len(d))
    q = p[p["type"] == kind]["value"]
    if len(q):
        ax.fill_between([x[0], x[-1]], q.min(), q.max(), color="#7A7A7A", alpha=0.18, linewidth=0)
    ax.scatter(
        x, d["value"], s=5, color=color, alpha=0.6, linewidth=0, label=style.EVENT_NAMES[kind]
    )
    b = d[d["baseline"]]
    yerr = None
    if b["lo"].notna().all():
        yerr = [b["value"] - b["lo"], b["hi"] - b["value"]]
    ax.errorbar(
        offset + b.index,
        b["value"],
        yerr=yerr,
        fmt="o",
        markersize=style.MARKER_SIZE + 2,
        color=color,
        markeredgecolor="black",
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
    )
    offset += len(d) + 8
ax.set_xticks([])
ax.set_xlabel("Specifications, in order of size")
ax.set_ylabel(label)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.14, bottom=0.22)
style.save_figure(fig, OUT / f"specification_curve_{key}", data=g)
