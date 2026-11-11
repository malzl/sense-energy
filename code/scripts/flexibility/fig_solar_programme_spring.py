"""Sunshine slope of spring daytime demand (1 March to 8 May) by year at the panel sites on the
Great British Energy solar list and at the sites that break downwards from 2025 to 2026.

analysis/solar_programme.py (spring_slopes.csv, spring_breaks.csv).
"""

import pandas as pd
from _common import OUT, SOLAR

from sense_energy.visualization import style

s = pd.read_csv(SOLAR / "spring_slopes.csv")
b = pd.read_csv(SOLAR / "spring_breaks.csv").set_index("site_code")
b["role"] = None
b.loc[b["funded"] & b["break_down"], "role"] = "funded, break"
b.loc[b["funded"] & ~b["break_down"], "role"] = "funded, no break"
b.loc[~b["funded"] & b["break_down"], "role"] = "not funded, break"
s["role"] = s["site_code"].map(b["role"])
s = s.dropna(subset=["role"])
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
seen = set()
for role in ("funded, no break", "not funded, break", "funded, break"):
    for _, g in s[s["role"] == role].groupby("site_code"):
        g = g.sort_values("spring")
        ax.plot(
            g["spring"],
            g["slope_pct"],
            color=style.SOLAR_PROGRAMME_COLORS[role],
            linewidth=style.LINEWIDTH["secondary" if role == "funded, no break" else "primary"],
            marker="o",
            markersize=style.MARKER_SIZE - 1,
            label=None if role in seen else role,
            zorder=2 if role == "funded, no break" else 3,
        )
        seen.add(role)
ax.set_xticks(sorted(s["spring"].unique()))
ax.set_xlabel("Spring (1 March to 8 May)")
ax.set_ylabel("Sunshine slope (% per kWh/m²)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, bottom=0.34)
style.save_figure(fig, OUT / "solar_programme_spring", data=s)
