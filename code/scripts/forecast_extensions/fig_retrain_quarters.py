"""Day-ahead error in each quarter from January 2024 to May 2026: the trained model (LightGBM)
trained once on 2023, the same model trained again at the start of each quarter, the zero-shot
model and the seasonal naive forecast. The same site-days for every model; mean over sites.

analysis/forecast_extensions.py (retrain_quarters.csv).
"""

import numpy as np
from _common import OUT, table

from sense_energy.visualization import style

q = table("retrain_quarters")
quarters = sorted(q["quarter"].unique())
fig, ax = style.figure(style.WIDE_SINGLE)
for model, name in style.RETRAIN_MODEL_NAMES.items():
    d = q[q["model"] == model].set_index("quarter").reindex(quarters)
    if d["nmae_pct"].notna().any():
        ax.plot(
            np.arange(len(quarters)),
            d["nmae_pct"],
            color=style.RETRAIN_MODEL_COLORS[model],
            linestyle=style.RETRAIN_MODEL_LINESTYLES[model],
            marker=style.RETRAIN_MODEL_MARKERS[model],
            markersize=style.MARKER_SIZE,
            linewidth=style.LINEWIDTH["primary"],
            label=name,
        )
ax.set_xticks(range(len(quarters)))
ax.set_xticklabels([x.replace(" ", "\n") for x in quarters])
ax.set_ylim(0, None)
ax.set_ylabel("Day-ahead error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.34)
style.save_figure(fig, OUT / "retrain_quarters", data=q)
