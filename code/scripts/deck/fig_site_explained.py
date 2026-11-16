"""Question 3, chart: how much of the difference between sites each source explains (sites
with no on-site generation): the NHS region (share above chance), the best field of the energy
certificate, the gas-to-electricity ratio of the estates return, and the reaction of the same
site in its other events."""

import numpy as np
import pandas as pd
from _deck import BLUE, ORANGE, REPORTS, figure, legend_below, save

g = pd.read_csv(REPORTS / "event_audit" / "grid_summary.csv").set_index(["metric", "type"])
f = pd.read_csv(REPORTS / "extreme_events" / "alignment_fits.csv")
cert = f[f["descriptor"].str.startswith("certificate")].groupby("type")["adj_r2_gain"].max()
gas = f[f["descriptor"] == "estates return: gas over electricity"].set_index("type")["adj_r2_gain"]
rows = []
for kind in ("heatwave", "cold spell"):
    rows += [
        (kind, "region", max(g.loc[("region share minus chance", kind), "baseline"], 0.0)),
        (kind, "certificate", 100 * max(cert[kind], 0.0)),
        (kind, "gas share", 100 * max(gas[kind], 0.0)),
        (kind, "own events", g.loc[("other events r2", kind), "baseline"]),
    ]
d = pd.DataFrame(rows, columns=["type", "source", "share_pct"])
order = ["region", "certificate", "gas share", "own events"]
fig, ax = figure(29, 41)
for j, (kind, colour, label) in enumerate(
    (("heatwave", ORANGE, "heat waves"), ("cold spell", BLUE, "cold spells"))
):
    x = d[d["type"] == kind].set_index("source").reindex(order)
    pos = np.arange(len(order))[::-1] + (0.5 - j) * 0.38
    ax.barh(pos, x["share_pct"], height=0.36, color=colour, label=label)
ax.set_yticks(np.arange(len(order))[::-1])
ax.set_yticklabels(order, fontsize=5.8)
ax.set_xlim(0, 80)
ax.set_xticks([0, 40, 80])
ax.set_xlabel("Explained (%)")
ax.grid(axis="x")
ax.tick_params(axis="y", length=0)
legend_below(fig, ax, ncol=1)
save(fig, "site_explained", data=d)
print(d.pivot(index="source", columns="type", values="share_pct").round(1).to_string())
