"""Day-ahead error by the kind of day: site-mean nMAE of each model on days at or above 28 degC,
on 25-28 degC days, on other June-September days and on the rest of the year (test window June
2025 to May 2026). Models ordered by their error on the hottest days.

analysis/forecast_heat_days.py (scores_by_day_class.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT, PROJECT_ROOT

from sense_energy.analysis.forecast_heat_days import CLASS_ORDER
from sense_energy.visualization import style

s = pd.read_csv(
    PROJECT_ROOT / "code" / "reports" / "forecast_heat_days" / "scores_by_day_class.csv"
)
classes = [CLASS_ORDER[0], CLASS_ORDER[1], CLASS_ORDER[2], CLASS_ORDER[4]]
t = s.pivot(index="model", columns="day_class", values="nmae")[classes] * 100
t = t.drop(index=[m for m in ("profile_quantiles",) if m in t.index])
t = t.sort_values(classes[0], ascending=False)
colors = {
    classes[0]: "#D55E00",
    classes[1]: "#E69F00",
    classes[2]: "#56B4E9",
    classes[3]: "#7A7A7A",
}
y = np.arange(len(t))
h = 0.2

NAMES = {**style.METHOD_NAMES, "lightgbm_ifs_heat": "LightGBM + IFS ENS + heat features"}
fig, ax = style.figure(style.WIDE_SINGLE)
for k, c in enumerate(classes):
    ax.barh(y + (1.5 - k) * h, t[c], height=h, color=colors[c], label=c, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(
    [NAMES.get(m, m) for m in t.index],
    fontsize=style.FONT_SIZE["tick"] - 1,
)
ax.set_xlabel("nMAE, site mean (%)")
ax.set_xlim(0, float(t.max().max()) * 1.08)
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.16, bottom=0.2)
print(t.round(2).to_string())
style.save_figure(fig, OUT / "forecast_heat_days", data=t.reset_index())
