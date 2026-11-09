"""What the 2025 heatwaves added: electricity attributed to the heat on each heatwave's site-days
(observed demand minus demand at the site's normal temperature for the time of year), for sites
without CHP (demand) and with CHP (grid import, the plant turning down), with the share of the
group's demand on those days.

analysis/estate_counterfactuals.py (heatwave_events.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

e = pd.read_csv(
    PROJECT_ROOT / "code" / "reports" / "estate_counterfactuals" / "heatwave_events.csv"
)
events = list(dict.fromkeys(e["event_id"]))
labels = [
    f"{pd.Timestamp(r.start):%d %b}–{pd.Timestamp(r.end):%d %b}"
    for r in e.drop_duplicates("event_id").itertuples()
]
groups = ["no CHP", "on-site CHP (import)"]
colors = {
    "no CHP": style.CHP_GROUP_COLORS["acute, no CHP"],
    "on-site CHP (import)": style.CHP_COLORS["on-site CHP"],
}
x = np.arange(len(events))
fig, ax = style.figure(style.SINGLE_COLUMN)
for k, g in enumerate(groups):
    d = e[e["group"] == g].set_index("event_id").reindex(events)
    bars = ax.bar(
        x + (k - 0.5) * 0.38, d["kwh_attr"] / 1000, width=0.38, color=colors[g], label=g, zorder=3
    )
    for b, s in zip(bars, d["share_pct"], strict=True):
        if np.isfinite(s):
            ax.annotate(
                f"{s:.0f} %",
                xy=(b.get_x() + b.get_width() / 2, b.get_height()),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                fontsize=style.FONT_SIZE["annotation"] - 1,
            )
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Electricity attributed to the heat (MWh)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.22, bottom=0.28)
style.save_figure(fig, OUT / "heatwave_attribution", data=e)
