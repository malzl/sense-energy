"""Does a site's past predict its next event? Mean absolute error of four predictions of a
site's reaction in an event that the prediction has not seen: no reaction, the mean of the
other sites in the event, the mean of the other sites of the region, and the site's own
reaction in its other events. Sites without on-site generation.

analysis/event_audit.py (prediction_scores.csv).
"""

import numpy as np
from _common import OUT, PLAIN, table

from sense_energy.visualization import style

s = table("prediction_scores")
s = s[(s["sites"] == PLAIN) & s["predictor"].isin(style.PREDICTOR_ORDER)]
order = list(style.PREDICTOR_ORDER)[::-1]
fig, ax = style.figure(style.WIDE_SINGLE)
for j, kind in enumerate(("heatwave", "cold spell")):
    d = s[s["type"] == kind].set_index("predictor").reindex(order)
    ax.barh(
        np.arange(len(order)) + (0.5 - j) * 0.36,
        d["mae_pts"],
        height=0.34,
        color=style.WEATHER_EVENT_COLORS[kind],
        label=style.EVENT_NAMES[kind],
    )
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_xlabel("Error of the predicted reaction (points)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.26)
style.save_figure(fig, OUT / "prediction", data=s)
