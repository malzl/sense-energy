"""What the value result owes to the battery: captured share of the achievable peak cut for
five batteries (power as a share of the site's mean demand, and hours of storage), whole record,
with the 95 % interval from resampling sites.

analysis/forecast_record.py (decision.csv).
"""

import numpy as np
from _common import OUT, best_model, table

from sense_energy.visualization import style

d = table("decision")
best = best_model(table("by_period"))
order = [(0.1, 2.0), (0.2, 2.0), (0.4, 2.0), (0.2, 1.0), (0.2, 4.0)]
labels = [f"{int(100 * s)} %\n{int(h)} h" for s, h in order]
fig, ax = style.figure(style.SINGLE_COLUMN)
for j, model in enumerate(("seasonal_naive", best)):
    x = (
        d[d["model"] == model]
        .set_index(["battery_share_of_mean_demand", "battery_hours"])
        .reindex(order)
    )
    ax.errorbar(
        np.arange(len(order)) + (j - 0.5) * 0.18,
        x["captured_point_pct"],
        yerr=[x["captured_point_pct"] - x["lo"], x["hi"] - x["captured_point_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.RECORD_MODEL_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.RECORD_MODEL_NAMES[model],
    )
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels)
ax.set_ylim(0, 100)
ax.set_xlabel("Battery: power (share of mean demand) and storage")
ax.set_ylabel("Share of the achievable peak cut (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.34, bottom=0.42)
style.save_figure(fig, OUT / "record_decision", data=d)
