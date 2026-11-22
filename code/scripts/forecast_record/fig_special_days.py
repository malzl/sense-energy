"""The rule for special days on top of the zero-shot forecast: error by kind of special day for
the forecast, the forecast corrected with the same special day in the past (Arora and Taylor),
and the load of the most recent Sunday. Whole-record forecasts with calendar and weather
inputs, 2024 to May 2026, 91 sites; site-days that have a corresponding past special day.

analysis/special_days.py (scores.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

s = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "special_days" / "scores.csv")
s = s[(s["model"] == "chronos2_ifs") & (s["sites_group"] == "all sites")].set_index("kind")
kinds = [
    "bank holiday",
    "Christmas period",
    "day before a bank holiday",
    "day after a bank holiday",
]
series = (
    ("nmae_forecast_pct", "forecast", "#0072B2"),
    ("nmae_rule_half_hour_pct", "forecast with the rule", "#D55E00"),
    ("nmae_sunday_pct", "most recent Sunday", "#7A7A7A"),
)
fig, ax = style.figure(style.WIDE_SINGLE)
for j, (col, label, color) in enumerate(series):
    ax.bar(
        np.arange(len(kinds)) + (j - 1) * 0.27,
        s.loc[kinds, col],
        width=0.25,
        color=color,
        label=label,
    )
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels(
    [k.replace(" a bank", "\na bank").replace("Christmas ", "Christmas\n") for k in kinds]
)
ax.set_ylabel("Day-ahead error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.24, bottom=0.3)
style.save_figure(fig, OUT / "special_days", data=s.reset_index())
