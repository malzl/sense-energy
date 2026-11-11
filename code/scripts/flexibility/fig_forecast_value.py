"""What a day-ahead forecast is worth for peak shaving: the share of the peak cut achievable with
perfect foresight that a battery captures when its threshold is planned on each model's
forecast, on the median and on the 70 % quantile (91 sites, 69 days).

analysis/flexibility.py (forecast_value.csv).
"""

import pandas as pd
from _common import FLEX, OUT

from sense_energy.visualization import style

f = pd.read_csv(FLEX / "forecast_value.csv")
f["method"] = f["model"].map(style.METHOD_NAMES)
methods = style.ordered(f["method"].unique())[::-1]
fig, ax = style.figure(style.WIDE_SINGLE)
for plan, marker, fill in (("point", "o", True), ("q70", "D", False)):
    d = f[f["plan"] == plan].set_index("method").reindex(methods)
    for k, m in enumerate(methods):
        v = d.loc[m, "captured_share"]
        if pd.isna(v):
            continue
        c = style.method_color(m)
        ax.plot(
            100 * v,
            k,
            linestyle="none",
            marker=marker,
            markersize=style.MARKER_SIZE + 1,
            markerfacecolor=c if fill else "white",
            markeredgecolor=c,
            markeredgewidth=1.0,
            zorder=3,
        )
handles = [
    ax.plot(
        [],
        [],
        linestyle="none",
        marker=mk,
        markersize=style.MARKER_SIZE + 1,
        markerfacecolor="#000000" if fill else "white",
        markeredgecolor="#000000",
    )[0]
    for mk, fill in (("o", True), ("D", False))
]
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_xlim(0, 100)
ax.set_xlabel("Share of the achievable peak cut captured (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(
    ax, handles=handles, labels=[style.PLAN_NAMES["point"], style.PLAN_NAMES["q70"]], bottom=0.28
)
fig.subplots_adjust(left=0.3)
style.save_figure(fig, OUT / "forecast_value", data=f)
