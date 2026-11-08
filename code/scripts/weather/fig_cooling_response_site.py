"""Per-site electricity response to cooling degree days (base 15.5 degC) against site size. One plot.

Slope from the month-fixed-effect regression in analysis/cooling.py, as % of the site's mean
demand per degC day, with 95 % HAC confidence intervals; colour = fossil-fuel CHP on site.
"""

import matplotlib.ticker as mticker
import pandas as pd
from _common import COOLING, OUT

from sense_energy.visualization import style

r = pd.read_csv(COOLING / "cooling_response_site.csv")
r["group"] = r["chp"].map(style.CHP_NAMES)

fig, ax = style.figure(style.SINGLE_COLUMN)
for chp in (False, True):
    name = style.CHP_NAMES[chp]
    g = r[r["group"] == name]
    ax.errorbar(
        g["mean_kw"],
        g["cdd_pct"],
        yerr=[g["cdd_pct"] - g["cdd_pct_lo"], g["cdd_pct_hi"] - g["cdd_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=style.LINEWIDTH["reference"],
        color=style.CHP_COLORS[name],
        label=f"{name} ({len(g)} sites)",
    )
style.reference_line(ax, 0.0)
ax.set_xscale("log")
ax.xaxis.set_major_locator(mticker.FixedLocator([1, 10, 100, 1000]))
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlabel("Site mean electricity demand (kW)")
ax.set_ylabel("Cooling response (% per °C day)")
style.style_axis(ax)
style.add_bottom_legend(ax)
print(r.groupby("group")["cdd_pct"].describe().round(2).to_string())
style.save_figure(fig, OUT / "cooling_response_site", data=r)
