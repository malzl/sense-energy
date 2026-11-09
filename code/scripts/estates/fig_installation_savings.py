"""Weather-normalised change in annual electricity from 2023/24 to 2025/26 at the sites without
CHP, by what their estates returns report changed between the two returns, one point per site;
bars mark the group medians, and the label gives the difference in medians from the sites with
no change with its bootstrap 95 % interval.

analysis/installation_savings.py (site_changes.csv, group_effects.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

D = PROJECT_ROOT / "code" / "reports" / "installation_savings"
s = pd.read_csv(D / "site_changes.csv")
s = s[~s["chp_site"]]
ge = pd.read_csv(D / "group_effects.csv").set_index("group")
groups = ["no change", "LED only", "solar new or grown", "heat pumps installed"]
colors = {
    "no change": "#7A7A7A",
    "LED only": "#E69F00",
    "solar new or grown": "#D55E00",
    "heat pumps installed": "#0072B2",
}
rng = np.random.default_rng(2)
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for k, g in enumerate(groups):
    d = s[s["group"] == g]["nac_change_pct"]
    ax.plot(
        k + rng.uniform(-0.18, 0.18, len(d)),
        d,
        linestyle="none",
        marker="o",
        markersize=3,
        color=colors[g],
        alpha=0.85,
        zorder=3,
    )
    ax.hlines(d.median(), k - 0.3, k + 0.3, color="#222222", linewidth=1.2, zorder=4)
    if g != "no change" and g in ge.index and pd.notna(ge.loc[g, "did_median_pct"]):
        r = ge.loc[g]
        ax.annotate(
            f"{r['did_median_pct']:+.1f}\n[{r['did_lo']:+.0f}, {r['did_hi']:+.0f}]",
            xy=(k, 58),
            ha="center",
            va="top",
            fontsize=style.FONT_SIZE["annotation"] - 1,
        )
short = {
    "no change": "no change",
    "LED only": "LED only",
    "solar new or grown": "solar new\nor grown",
    "heat pumps installed": "heat pumps\ninstalled",
}
ax.set_xticks(range(len(groups)))
ax.set_xticklabels(
    [f"{short[g]}\n({int((s['group'] == g).sum())})" for g in groups],
    fontsize=style.FONT_SIZE["tick"] - 1,
)
hidden = int((s["nac_change_pct"] < -50).sum())
ax.set_ylim(-50, 62)
if hidden:
    ax.annotate(
        f"{hidden} site{'s' if hidden > 1 else ''} below −50 % not shown",
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=style.FONT_SIZE["annotation"] - 1,
        color="#555555",
    )
ax.set_ylabel("Weather-normalised change in electricity (%)")
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "installation_savings", data=s)
