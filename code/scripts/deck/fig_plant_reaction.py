"""Finding 3, left: mean reaction of the sites with and without on-site generation in heat
waves and cold spells, with the 95 % interval across sites. With a plant on site the meter shows
grid import."""

import pandas as pd
from _deck import ASH, BLUE, ORANGE, REPORTS, figure, legend_below, save

g = pd.read_csv(REPORTS / "extreme_events" / "group_means.csv")
g = g[g["column"] == "chp"]
kinds = ["heatwave", "cold spell"]
fig, ax = figure(46, 39)
ax.axhline(0, color=ASH, linewidth=0.5)
for j, (level, colour, label) in enumerate(
    (
        ("no on-site generation", BLUE, "no on-site generation"),
        ("on-site generation", ORANGE, "on-site generation"),
    )
):
    d = g[g["level"] == level].set_index("type").reindex(kinds)
    pos = [k + (j - 0.5) * 0.36 for k in range(len(kinds))]
    ax.bar(pos, d["reaction_pct"], width=0.34, color=colour, label=label)
    ax.errorbar(
        pos,
        d["reaction_pct"],
        yerr=[d["reaction_pct"] - d["lo"], d["hi"] - d["reaction_pct"]],
        fmt="none",
        ecolor=ASH,
        elinewidth=0.6,
        capsize=0,
    )
ax.set_xticks(range(len(kinds)))
ax.set_xticklabels(["heat waves", "cold spells"])
ax.set_ylim(-5, 45)
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_ylabel("Change in demand (%)")
ax.grid(axis="y")
ax.tick_params(axis="x", length=0)
legend_below(fig, ax, ncol=1)
save(fig, "plant_reaction", data=g)
cols = ["type", "level", "sites", "reaction_pct", "lo", "hi", "median_pct", "share_rising"]
print(g[cols].round(1).to_string(index=False))
