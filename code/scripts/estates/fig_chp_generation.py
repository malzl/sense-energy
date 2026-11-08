"""What the CHP sites really use: for each site with fossil-fuel CHP in the ERIC 2024/25 return,
the grid import the return reports (what our meters record) and the CHP's own electrical output
net of exports, stacked to the implied site demand, with our metered import over the same
reporting year as a marker. Sites with an implausible capacity factor (a unit mismatch in the
return) are shown hollow and excluded from the summary.

analysis/chp.py (generation_2024_25.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

g = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "chp" / "generation_2024_25.csv")
g = g[g["chp_elec_output_kwh"].notna() & (g["implied_mean_kw"] > 0)].copy()
g = g.sort_values("implied_mean_kw", ascending=True).reset_index(drop=True)
y = np.arange(len(g))
imp = g["eric_import_mean_kw"] / 1000
chp = (g["chp_elec_output_kwh"] - g["exported_elec_kwh"].fillna(0)) / 8760 / 1000
ok = g["plausible"].to_numpy()


def short(name: str) -> str:
    n = (
        name.title()
        .replace(" Hospital", "")
        .replace(" University", " Univ.")
        .replace(" District General", " DGH")
    )
    n = n.replace(" (Wonford)", "").replace(" And ", " & ").replace(" General", "")
    return "University Hospital (Coventry)" if n == "University" else n[:30]


fig, ax = style.figure(style.DOUBLE_COLUMN)
ax.barh(
    y,
    imp,
    color=style.CHP_GROUP_COLORS["acute, no CHP"],
    height=0.7,
    label="grid import (ERIC 2024/25)",
    zorder=3,
)
ax.barh(
    y,
    chp,
    left=imp,
    color=style.CHP_COLORS["on-site CHP"],
    height=0.7,
    label="CHP output net of exports",
    zorder=3,
)
for yy, p in zip(y, ok, strict=True):
    if not p:
        ax.barh(
            yy,
            imp[yy] + chp[yy],
            color="none",
            edgecolor="#444444",
            linewidth=0.6,
            height=0.7,
            zorder=4,
        )
ax.plot(
    g["metered_mean_kw"] / 1000,
    y,
    linestyle="none",
    marker="|",
    markersize=7,
    markeredgewidth=1.2,
    color="#222222",
    label="our metered import, same year",
    zorder=5,
)
ax.set_yticks(y)
ax.set_yticklabels([short(n) for n in g["site_name"]], fontsize=style.FONT_SIZE["tick"] - 1)
ax.set_xlabel("Mean electrical power over April 2024 – March 2025 (MW)")
ax.set_xlim(0, float((imp + chp).max()) * 1.08)
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.14, bottom=0.16)
okg = g[ok]
print(
    f"{len(g)} CHP sites with output, {len(okg)} plausible: CHP share median {okg['chp_share'].median():.2f}, implied {okg['implied_mean_kw'].sum() / 1000:.1f} MW vs import {okg['eric_import_mean_kw'].sum() / 1000:.1f} MW"
)
style.save_figure(fig, OUT / "chp_generation", data=g)
