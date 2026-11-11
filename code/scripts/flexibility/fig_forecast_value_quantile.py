"""Which quantile of the forecast to plan a peak-shaving threshold on: the share of the
achievable peak cut captured, by planning quantile and model.

analysis/flexibility.py (forecast_value.csv).
"""

import pandas as pd
from _common import FLEX, OUT

from sense_energy.visualization import style

f = pd.read_csv(FLEX / "forecast_value.csv")
f["method"] = f["model"].map(style.METHOD_NAMES)
f["q"] = f["plan"].map({"point": 50, "q60": 60, "q70": 70, "q80": 80, "q90": 90})
f = f.dropna(subset=["q"])
keep = [m for m in style.ordered(f["method"].unique()) if (f["method"] == m).sum() >= 4]
fig, ax = style.figure(style.WIDE_SINGLE)
for m in keep:
    d = f[f["method"] == m].sort_values("q")
    ax.plot(
        d["q"],
        100 * d["captured_share"],
        color=style.method_color(m),
        linestyle=style.method_linestyle(m),
        marker=style.method_marker(m),
        markersize=style.MARKER_SIZE,
        linewidth=style.LINEWIDTH["primary"],
        label=m,
    )
ax.set_xticks([50, 60, 70, 80, 90])
ax.set_xlabel("Forecast quantile the threshold is planned on (%)")
ax.set_ylabel("Share of the achievable peak cut captured (%)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, bottom=0.32)
style.save_figure(fig, OUT / "forecast_value_quantile", data=f[f["method"].isin(keep)])
