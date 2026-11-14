"""Day-ahead error by kind of day, every day of the test window (91 sites, 342 days, Chronos-2
without covariates): the error of the forecast and of the forecast rescaled to the day's true
mean. A day has one kind, in this order: holiday or Christmas, doctor strike, hot or cold,
level shift (the day's mean differs by more than 15 % from the same weekday of the last four
weeks), typical.

analysis/forecast_anatomy.py (daily_error_by_kind.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

k = pd.read_csv(FA / "daily_error_by_kind.csv").set_index("kind")
kinds = [x for x in style.DAY_KIND_ORDER if x in k.index][::-1]
c = style.method_color("Chronos-2")
fig, ax = style.figure(style.WIDE_SINGLE)
for j, kind in enumerate(kinds):
    r = k.loc[kind]
    ax.plot(
        [r["nmae_level_oracle_pct"], r["nmae_pct"]],
        [j, j],
        color=c,
        linewidth=style.LINEWIDTH["secondary"],
    )
    for col, fill in (("nmae_pct", True), ("nmae_level_oracle_pct", False)):
        ax.plot(
            r[col],
            j,
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
ax.set_yticks(range(len(kinds)))
ax.set_yticklabels(kinds)
ax.set_xlim(0, None)
ax.set_xlabel("Normalised absolute error (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(
    ax,
    handles=handles,
    labels=[style.ERROR_PART_NAMES["forecast"], style.ERROR_PART_NAMES["level"]],
    bottom=0.28,
)
fig.subplots_adjust(left=0.3)
style.save_figure(fig, OUT / "daily_error_by_kind", data=k.reset_index())
