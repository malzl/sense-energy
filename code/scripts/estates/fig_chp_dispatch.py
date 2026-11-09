"""How each CHP plant is dispatched, read from its site's grid import against the sites without
CHP: the import's response to cold (per degC of the cold dose; negative = the plant runs more on
cold days, heat-led), to the power price within the day (per GBP 100/MWh; negative = runs more
when power is dear) and to the working-day red band (16:00-19:00; negative = runs to avoid the
red-band charges), each minus the same coefficient at sites without CHP, with 95 % intervals.
Argument: cold, price or red.

analysis/chp_dispatch.py (classification.csv).
"""

import sys

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

KIND = sys.argv[1] if len(sys.argv) > 1 else "cold"
col, label, flag = {
    "cold": ("cold", "Import response to cold, minus sites without CHP (% per °C)", "heat_led"),
    "price": (
        "price_hourly",
        "Import response to the power price, minus sites without CHP (% per £100/MWh)",
        "price_following",
    ),
    "red": (
        "red_band",
        "Working-day red-band import, minus sites without CHP (%)",
        "red_band_avoiding",
    ),
}[KIND]
c = pd.read_csv(
    PROJECT_ROOT / "code" / "reports" / "chp_dispatch" / "classification.csv"
).set_index("site_code")
c = c.dropna(subset=[f"{col}_vs_no_chp_pct"]).sort_values(f"{col}_vs_no_chp_pct")
y = np.arange(len(c))
m, se = c[f"{col}_vs_no_chp_pct"], c[f"{col}_vs_no_chp_se"]
colors = np.where(c[flag], style.CHP_COLORS["on-site CHP"], "#9A9A9A")
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0, axis="x")
ax.errorbar(m, y, xerr=1.96 * se, fmt="none", ecolor="#CFCFCF", elinewidth=0.7, zorder=2)
ax.scatter(m, y, c=colors, s=14, edgecolor="white", linewidth=0.4, zorder=3)
ax.set_yticks(y)
ax.set_yticklabels(
    [str(n).title()[:30] for n in c["site_name"]], fontsize=style.FONT_SIZE["tick"] - 2
)
lim = float(np.nanpercentile(np.abs(m) + 1.96 * se, 90)) * 1.1
ax.set_xlim(-lim, lim)
ax.set_xlabel(label, fontsize=style.FONT_SIZE["label"] - 1)
style.style_axis(ax, grid="x")
print(f"{KIND}: {int(c[flag].sum())} of {len(c)} sites flagged")
style.save_figure(fig, OUT / f"chp_dispatch_{KIND}", data=c.reset_index())
