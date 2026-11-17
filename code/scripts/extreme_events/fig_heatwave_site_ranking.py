"""Which sites react most and least to heat waves: the ten highest and the ten lowest site
reactions (demand on the site's heat-wave days against its reference days), with the 95 %
interval from the site's events where it has two or more (number of events in brackets).
Sites with complete meter data. For sites with on-site generation the meter records grid
import. The figure names sites: for internal use, not for slides.

analysis/extreme_events.py (site_reactions.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT, PLAIN, table

from sense_energy.config import INTERIM_DIR
from sense_energy.visualization import style

N = 10
s = table("site_reactions")
s = s[(s["type"] == "heatwave") & s["clean"]].sort_values("reaction_pct", ascending=False)
names = pd.read_parquet(INTERIM_DIR / "sites.parquet").drop_duplicates("site_code")
s = s.merge(names[["site_code", "site_name"]], on="site_code", how="left")
d = pd.concat([s.head(N), s.tail(N)]).reset_index(drop=True)
pos = np.r_[np.arange(N), np.arange(N) + N + 1.2][::-1]
fig, ax = style.figure(style.DOUBLE_COLUMN)
style.reference_line(ax, 0.0, axis="x")
for group in style.PLANT_GROUP_ORDER:
    m = (d["chp"] == group).to_numpy()
    half = 1.96 * d.loc[m, "se_pct"].where(d.loc[m, "events"] >= 2)
    ax.barh(
        pos[m],
        d.loc[m, "reaction_pct"],
        height=0.7,
        color=style.PLANT_GROUP_COLORS[group],
        label=group,
    )
    ax.errorbar(
        d.loc[m, "reaction_pct"],
        pos[m],
        xerr=half,
        fmt="none",
        ecolor="black",
        elinewidth=0.6,
        capsize=0,
    )
labels = [
    f"{str(n).title()[:44]} ({int(e)})" for n, e in zip(d["site_name"], d["events"], strict=True)
]
ax.set_yticks(pos)
ax.set_yticklabels(labels, fontsize=6)
ax.set_xlabel("Change in demand on heat-wave days (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.14, bottom=0.2)
cols = [
    "site_code",
    "site_name",
    "chp",
    "use",
    "mean_kw",
    "events",
    "event_days",
    "reaction_pct",
    "se_pct",
    "extra_kw",
]
style.save_figure(fig, OUT / "heatwave_site_ranking", data=s[cols])
print(len(s), "sites;", int((s["chp"] != PLAIN).sum()), "with on-site generation")
