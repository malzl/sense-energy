"""Finding 4, chart: how each site's daytime demand follows sunshine, by summer. Sites with a
downward break are the sites where solar panels started to operate."""

import pandas as pd
from _deck import BLUE, ORANGE, REPORTS, STONE, figure, legend_below, save

s = pd.read_csv(REPORTS / "solar_watch" / "slopes.csv")
b = pd.read_csv(REPORTS / "solar_watch" / "breaks.csv").set_index("site_code")
reported = b["eric_solar_new_or_grown"].fillna(False).astype(bool)
s["role"] = "other"
s.loc[s["site_code"].isin(b.index[b["break_down"] & ~reported]), "role"] = "break"
s.loc[s["site_code"].isin(b.index[reported]), "role"] = "reported"
roles = {
    "other": ("other sites", STONE, 0.5, 1),
    "break": ("break, no solar in the return", BLUE, 1.1, 3),
    "reported": ("new solar in the return", ORANGE, 1.1, 4),
}
fig, ax = figure(43, 56)
for role, (label, colour, lw, z) in roles.items():
    first = True
    for _, g in s[s["role"] == role].groupby("site_code"):
        g = g.sort_values("summer")
        ax.plot(
            g["summer"],
            g["slope_pct"],
            color=colour,
            linewidth=lw,
            zorder=z,
            marker="o" if role != "other" else None,
            markersize=2.2,
            label=label if first else None,
        )
        first = False
ax.set_xticks([2023, 2024, 2025])
ax.set_xlim(2022.85, 2025.15)
ax.set_ylim(-27, 6)
ax.set_xlabel("Summer")
ax.set_ylabel("Demand change (%)\nper kWh/m² of sunshine")
ax.grid(axis="y")
legend_below(fig, ax, ncol=1, reverse=True, labelspacing=0.25, handlelength=1.4, fontsize=6.3)
save(fig, "solar_lines", data=s)
print(s.drop_duplicates("site_code")["role"].value_counts().to_dict())
