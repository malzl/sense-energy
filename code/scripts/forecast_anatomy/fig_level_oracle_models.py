"""Day-ahead error of each model and the error that is left when the forecast is rescaled to
the day's true mean (the level oracle): the most that a perfect daily signal could remove.
91 sites, 69 test days, site mean of the normalised absolute error.

analysis/forecast_anatomy.py (error_overall.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

e = pd.read_csv(FA / "error_overall.csv")
e["method"] = e["model"].map(style.METHOD_NAMES)
methods = style.ordered(e["method"].unique())[::-1]
d = e.set_index("method").reindex(methods)
fig, ax = style.figure(style.WIDE_SINGLE)
for k, m in enumerate(methods):
    c = style.method_color(m)
    ax.plot(
        [d.loc[m, "nmae_level_oracle_pct"], d.loc[m, "nmae_pct"]],
        [k, k],
        color=c,
        linewidth=style.LINEWIDTH["secondary"],
        zorder=2,
    )
    for col, fill in (("nmae_pct", True), ("nmae_level_oracle_pct", False)):
        ax.plot(
            d.loc[m, col],
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
ax.set_xlim(0, 20)
ax.set_xlabel("Normalised absolute error (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=[style.ERROR_PART_NAMES["forecast"], style.ERROR_PART_NAMES["level"]],
    bottom=0.28,
)
fig.subplots_adjust(left=0.3)
style.save_figure(
    fig,
    OUT / "level_oracle_models",
    data=e[["model", "method", "nmae_pct", "nmae_level_oracle_pct", "level_share_pct"]],
)
