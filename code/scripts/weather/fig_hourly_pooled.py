"""The pooled response by hour of the day: random-effects mean across 83 sites of the hour's
demand change per degC of the heat dose, per degC of the cold dose and per kWh/m2 of solar
radiation, with 95 % intervals.

analysis/hourly_response.py (pooled.csv).
"""

import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

p = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "hourly_response" / "pooled.csv")
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
ax.axvspan(15.5, 18.5, color="#EDEDED", zorder=0, linewidth=0)
for term in ("cdd20", "hdd14", "solar_kwh_m2"):
    d = p[p["term"] == term].sort_values("hour")
    m, se = 100 * d["pooled"].to_numpy(), 100 * d["pooled_se"].to_numpy()
    ax.fill_between(
        d["hour"],
        m - 1.96 * se,
        m + 1.96 * se,
        color=style.TERM_COLORS[term],
        alpha=0.18,
        linewidth=0,
        zorder=2,
    )
    ax.plot(
        d["hour"],
        m,
        color=style.TERM_COLORS[term],
        linewidth=1.2,
        marker="o",
        markersize=2.5,
        label=style.TERM_NAMES[term],
        zorder=3,
    )
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 3)])
ax.set_xlabel("Local hour of the day (red band shaded)")
ax.set_ylabel("Demand change (% per °C or per kWh/m²)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.3, bottom=0.3)
style.save_figure(fig, OUT / "hourly_pooled", data=p)
