"""Finding 2, maps: the reaction of each site in heat waves (left) and in cold spells (right):
demand on the event days against the reference days of the same site. One colour scale, cut at
+/- 25 %. Sites with on-site generation have a dark edge."""

import matplotlib as mpl
import pandas as pd
from _deck import BLUE, CHARCOAL, ORANGE, REPORTS, SHELL, england_map, figure, save

s = pd.read_csv(REPORTS / "extreme_events" / "site_reactions.csv")
s = s[s["clean"]].dropna(subset=["latitude", "longitude"])
cmap = mpl.colors.LinearSegmentedColormap.from_list("deck", [BLUE, "#FFFFFF", ORANGE])
norm = mpl.colors.Normalize(-25, 25)
fig, axes = figure(65, 41, ncols=2)
for ax, kind, label in zip(
    axes, ("heatwave", "cold spell"), ("Heat waves", "Cold spells"), strict=True
):
    england_map(ax, face=SHELL)
    d = s[s["type"] == kind].sort_values("reaction_pct", key=abs)
    plant = d["chp"] == "on-site generation"
    ax.scatter(
        d["longitude"],
        d["latitude"],
        c=d["reaction_pct"],
        cmap=cmap,
        norm=norm,
        s=13,
        edgecolor=[CHARCOAL if p else "#8A8A88" for p in plant],
        linewidth=[0.7 if p else 0.25 for p in plant],
        zorder=3,
    )
    ax.set_title(label, fontsize=6.5, pad=1.5, color=CHARCOAL)
cb = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    orientation="vertical",
    fraction=0.04,
    pad=0.02,
    aspect=28,
    extend="both",
)
cb.set_label("Change in demand (%)", fontsize=6.3)
cb.ax.tick_params(labelsize=6.0, length=1.5)
cb.outline.set_linewidth(0.3)
cols = ["site_code", "type", "latitude", "longitude", "reaction_pct", "chp"]
save(fig, "reaction_maps", data=s[cols])
print(s.groupby("type")["reaction_pct"].describe().round(1).to_string())
