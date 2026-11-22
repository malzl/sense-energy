"""Share of the half-hours inside the 80 % range of the day-ahead forecast over the record,
in windows of 30 days, at the sites with on-site generation: the model's range, a fixed rule
on the last 90 days and the range tracked online. The dashed line is the nominal 80 %.

analysis/forecast_range.py (over_time.csv).
"""

import pandas as pd
from _common import OUT, table

from sense_energy.visualization import style

t = table("over_time")
t = t[t["group"] == "on-site generation"]
t["day"] = pd.to_datetime(t["day"])
fig, ax = style.figure(style.WIDE_SHORT)
style.reference_line(ax, 80.0)
for method in ("model", "rolling quantile", "quantile tracking"):
    d = t[t["method"] == method]
    ax.plot(
        d["day"],
        d["inside_pct"],
        color=style.RANGE_METHOD_COLORS[method],
        linestyle=style.RANGE_METHOD_LINESTYLES[method],
        linewidth=style.LINEWIDTH["primary"],
        label=style.RANGE_METHOD_NAMES[method],
    )
ax.set_ylabel("Inside the 80 % range (%)")
style.style_axis(ax, grid="y", wide_short=True)
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.3, bottom=0.34)
style.save_figure(fig, OUT / "range_over_time", data=t)
