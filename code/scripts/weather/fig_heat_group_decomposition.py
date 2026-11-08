"""The heat response of the three site groups (rising, falling, neither on the daily heat slope)
decomposed: per degC of daily maximum above 24 degC (hot afternoons) and per degC of daily
minimum above 12 degC (warm nights), with the sunshine slope per kWh/m2 alongside; pooled
group x dose interactions in one specification-B regression, 95 % intervals.

analysis/heat_drivers.py (group_decomposition.csv, model "night and day").
"""

import numpy as np
from _common import OUT, heat_drivers

from sense_energy.visualization import style

g = heat_drivers("group_decomposition")
g = g[(g["model"] == "night and day") & (g["term"] != "hdd14")]
terms = ["hot_afternoon", "warm_night", "solar_kwh_m2"]
labels = {
    "hot_afternoon": "hot afternoon\n(% per °C of Tmax > 24 °C)",
    "warm_night": "warm night\n(% per °C of Tmin > 12 °C)",
    "solar_kwh_m2": "sunshine\n(% per kWh/m²)",
}
x = np.arange(len(terms))
width = 0.26
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for k, sign in enumerate(style.HEAT_SIGN_ORDER):
    d = g[g["sign"] == sign].set_index("term").reindex(terms)
    if d["coef_pct"].isna().all():
        continue
    ax.bar(
        x + (k - 1) * width,
        d["coef_pct"],
        width=width,
        color=style.HEAT_SIGN_COLORS[sign],
        label=f"{sign} ({int(d['sites'].dropna().iloc[0])} sites)",
        zorder=3,
    )
    ax.errorbar(
        x + (k - 1) * width,
        d["coef_pct"],
        yerr=[d["coef_pct"] - d["lo_pct"], d["hi_pct"] - d["coef_pct"]],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.6,
        capsize=1.5,
        zorder=4,
    )
ax.set_xticks(x)
ax.set_xticklabels([labels[t] for t in terms])
ax.set_ylabel("Demand change (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.3, bottom=0.3)
style.save_figure(fig, OUT / "heat_group_decomposition", data=g)
