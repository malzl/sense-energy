"""The rule for special days scored on all days, normal days, special days (with their two
parts, bank holidays and the Christmas period) and proximity days (the working day before and
after a bank holiday): day-ahead error of the zero-shot forecast and of the forecast with the
rule under three policies. A day outside the policy keeps the forecast. Whole-record forecasts
with calendar and weather inputs, 2024 to May 2026, 91 sites; mean over sites.

analysis/special_days.py (day_classes.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

c = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "special_days" / "day_classes.csv")
c = c[c["model"] == "chronos2_ifs"]
classes = [
    "all days",
    "normal days",
    "special days",
    "bank holidays",
    "Christmas period",
    "proximity days",
]
labels = [
    "all days",
    "normal\ndays",
    "special\ndays",
    "of these:\nbank holidays",
    "of these:\nChristmas period",
    "proximity\ndays",
]
policies = list(dict.fromkeys(c["policy"]))
colors = ["#E69F00", "#D55E00", "#CC79A7"]
hatches = ["", "//", ".."]
fig, ax = style.figure(style.DOUBLE_COLUMN)
x = np.arange(len(classes))
first = c[c["policy"] == policies[0]].set_index("day_class").reindex(classes)
ax.bar(x - 0.3, first["nmae_forecast_pct"], width=0.19, color="#0072B2", label="forecast")
for j, policy in enumerate(policies):
    d = c[c["policy"] == policy].set_index("day_class").reindex(classes)
    ax.bar(
        x - 0.1 + j * 0.2,
        d["nmae_with_rule_pct"],
        width=0.19,
        color=colors[j],
        hatch=hatches[j],
        edgecolor="white",
        linewidth=0.3,
        label="with the " + policy,
    )
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("Day-ahead error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "special_day_classes", data=c)
