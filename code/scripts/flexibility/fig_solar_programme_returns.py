"""The solar programme in the estates returns: solar generation reported for 2024/25 and 2025/26
by the English sites at postcodes on the Great British Energy list and by all other sites.

analysis/solar_programme.py (returns_check.csv).
"""

import pandas as pd
from _common import OUT, SOLAR

from sense_energy.visualization import style

r = pd.read_csv(SOLAR / "returns_check.csv")
g = r.groupby("funded")[["solar_generated_kwh_a", "solar_generated_kwh_b"]].sum() / 1e6
d = pd.DataFrame(
    {
        "group": ["funded postcodes", "funded postcodes", "other sites", "other sites"],
        "year": ["2024/25", "2025/26"] * 2,
        "gwh": [
            g.loc[True, "solar_generated_kwh_a"],
            g.loc[True, "solar_generated_kwh_b"],
            g.loc[False, "solar_generated_kwh_a"],
            g.loc[False, "solar_generated_kwh_b"],
        ],
    }
)
colors = {"2024/25": style.COLORS["grey"], "2025/26": style.COLORS["orange"]}
fig, ax = style.figure(style.SINGLE_COLUMN)
for k, (year, c) in enumerate(colors.items()):
    v = d[d["year"] == year]
    ax.bar(
        [i + (k - 0.5) * 0.36 for i in range(len(v))],
        v["gwh"],
        width=0.34,
        color=c,
        label=year,
        zorder=3,
    )
ax.set_xticks([0, 1])
ax.set_xticklabels(["funded postcodes", "other sites"])
ax.set_ylim(0, None)
ax.set_ylabel("Reported solar generation (GWh per year)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "solar_programme_returns", data=d)
