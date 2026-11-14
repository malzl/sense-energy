"""Error of the paper's models (from its saved forecasts) and of four references on the paper's
samples: on the window that the paper scores (filled) and on every day of its test period
(open). Mean over the series of the region; error in per cent of the load range of the training
period, the paper's unit. Argument: europe or pjm.

experiments/esdf_check.py (summary.csv).
"""

import sys

import pandas as pd
from _common import ESDF, OUT

from sense_energy.visualization import style

region = sys.argv[1] if len(sys.argv) > 1 else "europe"
s = pd.read_csv(ESDF / "summary.csv")
s = s[s["region"] == region]
methods = [m for m in style.ESDF_METHOD_ORDER if m in set(s["model"])][::-1]
fig, ax = style.figure(style.WIDE_SINGLE)
for subset, fill in (("paper window", True), ("all test days", False)):
    d = s[s["subset"] == subset].set_index("model").reindex(methods)
    for k, m in enumerate(methods):
        c = style.ESDF_METHOD_COLORS[m]
        ax.plot(
            100 * d.loc[m, "mae"],
            k,
            linestyle="none",
            marker="o",
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
        marker="o",
        markersize=style.MARKER_SIZE + 1,
        markerfacecolor="#000000" if fill else "white",
        markeredgecolor="#000000",
    )[0]
    for fill in (True, False)
]
ax.set_yticks(range(len(methods)))
ax.set_yticklabels(methods)
ax.set_xlim(0, None)
ax.set_xlabel("Mean absolute error (% of the training load range)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=[style.ESDF_SUBSET_NAMES["paper window"], style.ESDF_SUBSET_NAMES["all test days"]],
    bottom=0.26,
)
fig.subplots_adjust(left=0.3)
style.save_figure(fig, OUT / f"esdf_subsets_{region}", data=s)
