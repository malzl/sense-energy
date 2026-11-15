"""Finding 3, right: share of the variance of the site reactions that each source explains, at
the sites without on-site generation: the fields of the Display Energy Certificate, the estates
return, and the reaction of the same site in its other events."""

import numpy as np
import pandas as pd
from _deck import BLUE, ORANGE, REPORTS, figure, legend_below, save

EE = REPORTS / "extreme_events"
f = pd.read_csv(EE / "alignment_fits.csv")
r = pd.read_csv(EE / "reliability.csv")
rows = [
    ("certificate: building environment", "certificate: environment"),
    ("certificate: operational rating", "certificate: rating"),
    ("certificate: floor share with any air conditioning", "certificate: air conditioning"),
    ("building use", "building use"),
    ("estates return: gas over electricity", "gas-to-electricity ratio"),
]
d = f[f["descriptor"].isin([a for a, _ in rows])].copy()
d["share_pct"] = 100 * d["adj_r2_gain"].clip(lower=0)
own = r.assign(descriptor="own", share_pct=100 * r["other_events_r2"])
d = pd.concat([d[["type", "descriptor", "share_pct"]], own[["type", "descriptor", "share_pct"]]])
order = [a for a, _ in rows] + ["own"]
labels = [b for _, b in rows] + ["the site's other events"]
fig, ax = figure(46, 39)
for j, (kind, colour, label) in enumerate(
    (("heatwave", ORANGE, "heat waves"), ("cold spell", BLUE, "cold spells"))
):
    x = d[d["type"] == kind].set_index("descriptor").reindex(order)
    pos = np.arange(len(order))[::-1] + (0.5 - j) * 0.38
    ax.barh(pos, x["share_pct"], height=0.36, color=colour, label=label)
ax.set_yticks(np.arange(len(order))[::-1])
ax.set_yticklabels(labels, fontsize=5.8)
ax.set_xlim(0, 60)
ax.set_xticks([0, 20, 40, 60])
ax.set_xlabel("Share explained (%)")
ax.grid(axis="x")
ax.tick_params(axis="y", length=0)
legend_below(fig, ax, ncol=2)
save(fig, "descriptor_share", data=d)
print(d.pivot(index="descriptor", columns="type", values="share_pct").round(1).to_string())
