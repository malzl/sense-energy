"""The strike effect hour by hour: demand in each local hour of a strike weekday away from
holidays against the same site, month and weekday (34 strike weekdays, 132 sites), with the
95 % interval; the pseudo-strikes one week before and after as lines.

analysis/forecast_anatomy.py (strike_profile.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

p = pd.read_csv(FA / "strike_profile.csv")
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for term in (
    "one week before (placebo)",
    "one week after (placebo)",
    "strike weekday, away from holidays",
):
    d = p[p["term"] == term].sort_values("hour")
    c = style.STRIKE_TERM_COLORS[term]
    main = term.startswith("strike")
    if main:
        ax.fill_between(
            d["hour"] + 0.5, d["lo_pct"], d["hi_pct"], color=c, alpha=style.BAND_ALPHA, linewidth=0
        )
    ax.plot(
        d["hour"] + 0.5,
        d["effect_pct"],
        color=c,
        linewidth=style.LINEWIDTH["primary" if main else "secondary"],
        label=term,
    )
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_ylim(-6, 6)
ax.set_xlabel("Hour of the day (local time)")
ax.set_ylabel("Change in demand (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.42)
style.save_figure(fig, OUT / "strike_profile", data=p)
