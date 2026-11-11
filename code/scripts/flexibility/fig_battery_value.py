"""Annual value of a two-hour battery per kWh of capacity under each price case, financial year
2024/25, with perfect knowledge of the day's prices. The Agile bar is the median of the regional
tariffs with their range.

analysis/flexibility.py (battery_unit_annual.csv).
"""

import pandas as pd
from _common import FLEX, OUT, YEAR

from sense_energy.visualization import style

a = pd.read_csv(FLEX / "battery_unit_annual.csv")
a = a[a["fy"] == YEAR]
agile = a[a["case"].str.startswith("Agile")]["value_gbp_kwh_year"]
rows = []
for c in style.PRICE_CASE_ORDER:
    if c == "Agile":
        rows.append({"case": c, "value": agile.median(), "lo": agile.min(), "hi": agile.max()})
    else:
        v = float(a.loc[a["case"] == c, "value_gbp_kwh_year"].iloc[0])
        rows.append({"case": c, "value": v, "lo": v, "hi": v})
d = pd.DataFrame(rows)
fig, ax = style.figure(style.SINGLE_COLUMN)
ax.bar(
    range(len(d)),
    d["value"],
    color=[style.PRICE_CASE_COLORS[c] for c in d["case"]],
    width=0.65,
    zorder=3,
)
k = len(d) - 1
ax.errorbar(
    [k],
    [d["value"].iloc[k]],
    yerr=[[d["value"].iloc[k] - d["lo"].iloc[k]], [d["hi"].iloc[k] - d["value"].iloc[k]]],
    color="#000000",
    linewidth=style.LINEWIDTH["secondary"],
    capsize=2,
    linestyle="none",
    zorder=4,
)
ax.set_xticks(range(len(d)))
ax.set_xticklabels(
    [
        style.PRICE_CASE_NAMES[c]
        .replace(" red band", "\nred band")
        .replace("wholesale index", "wholesale\nindex")
        .replace("Agile tariff", "Agile\ntariff")
        for c in d["case"]
    ],
)
ax.set_ylim(0, None)
ax.set_ylabel("Battery value (£ per kWh of capacity per year)")
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "battery_value", data=d)
