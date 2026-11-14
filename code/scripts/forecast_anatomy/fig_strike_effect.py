"""Demand on doctor strike days against the same site, month and weekday, with weather and
calendar controls: the working hours (08:00 to 18:00), with 95 % intervals (standard errors
clustered by site and date). Pseudo-strikes one week before and after are the placebos.

analysis/forecast_anatomy.py (strike_effect.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

s = pd.read_csv(FA / "strike_effect.csv")
s = s[(s["site_type"] == "all sites") & (s["outcome"] == "08:00 to 18:00")]
d = s.set_index("term").reindex(style.STRIKE_TERM_ORDER[::-1])
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0, axis="x")
for k, term in enumerate(d.index):
    r = d.loc[term]
    ax.errorbar(
        r["effect_pct"],
        k,
        xerr=[[r["effect_pct"] - r["lo_pct"]], [r["hi_pct"] - r["effect_pct"]]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.STRIKE_TERM_COLORS[term],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
    )
ax.set_yticks(range(len(d)))
ax.set_yticklabels(list(d.index))
ax.set_xlim(-6, 6)
ax.set_xlabel("Change in demand, 08:00 to 18:00 (%)")
style.style_axis(ax, grid="x")
fig.subplots_adjust(left=0.48, bottom=0.18)
style.save_figure(fig, OUT / "strike_effect", data=s)
