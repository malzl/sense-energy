"""Share of the achievable peak cut that a battery captures when its threshold is planned on
each model's forecast: on the median, on the 70 % quantile, and on the median rescaled to the
day's true mean (the level oracle). 91 sites, 69 test days.

analysis/forecast_anatomy.py (value_overall.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

v = pd.read_csv(FA / "value_overall.csv")
v["method"] = v["model"].map(style.METHOD_NAMES)
methods = style.ordered(v["method"].unique())[::-1]
d = v.set_index("method").reindex(methods)
fig, ax = style.figure(style.WIDE_SINGLE)
for plan, fill in (("point", True), ("q70", False), ("level", True)):
    for k, m in enumerate(methods):
        val = d.loc[m, f"captured_{plan}_pct"]
        if pd.isna(val):
            continue
        c = style.method_color(m)
        ax.plot(
            val,
            k,
            linestyle="none",
            marker=style.PLAN_MARKERS[plan],
            markersize=style.MARKER_SIZE + (5 if plan == "level" else 1),
            markerfacecolor=c if fill else "white",
            markeredgecolor=c,
            markeredgewidth=1.4 if plan == "level" else 1.0,
            zorder=3,
        )
handles = [
    ax.plot(
        [],
        [],
        linestyle="none",
        marker=style.PLAN_MARKERS[plan],
        markersize=style.MARKER_SIZE + (5 if plan == "level" else 1),
        markerfacecolor="#000000" if fill else "white",
        markeredgecolor="#000000",
        markeredgewidth=1.4 if plan == "level" else 1.0,
    )[0]
    for plan, fill in (("point", True), ("q70", False), ("level", True))
]
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of the achievable peak cut captured (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=[style.PLAN_NAMES[p] for p in ("point", "q70", "level")],
    ncol=1,
    bottom=0.36,
)
fig.subplots_adjust(left=0.3)
style.save_figure(fig, OUT / "value_plans", data=v)
