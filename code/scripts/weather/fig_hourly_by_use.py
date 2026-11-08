"""Hour-of-day demand response on cold (< 2 degC) and hot (>= 22 degC daily mean) days, relative to a
14-16 degC day, by building use. One plot per band (argument: cold or hot).

Site x month and site x weekday effects, 95 % intervals clustered by site and date
(analysis/weather_heterogeneity.py); groups with fewer than 150 site-days in the band are absent.
"""

import sys

from _common import OUT, heterogeneity

from sense_energy.visualization import style

BAND = sys.argv[1] if len(sys.argv) > 1 else "cold"
h = heterogeneity("hourly_by_use")
h = h[h["band"] == BAND]

fig, ax = style.figure(style.WIDE_SINGLE)
for use in style.USE_ORDER:
    d = h[h["use"] == use].sort_values("hour")
    if d.empty:
        continue
    c = style.USE_COLORS[use]
    ax.fill_between(
        d["hour"], d["lo_pct"], d["hi_pct"], color=c, alpha=style.BAND_ALPHA, linewidth=0
    )
    ax.plot(
        d["hour"],
        d["coef_pct"],
        color=c,
        linewidth=style.LINEWIDTH["primary"],
        label=f"{use} ({int(d['sites'].iloc[0])} sites)",
    )
style.reference_line(ax, 0.0)
ax.set_xticks(range(0, 24, 3))
ax.set_xticklabels([f"{x:02d}:00" for x in range(0, 24, 3)])
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel(f"Demand on {'< 2' if BAND == 'cold' else '≥ 22'} °C days vs a 15 °C day (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.22, bottom=0.32)
print(h.pivot_table(index="hour", columns="use", values="coef_pct").round(1).iloc[::3].to_string())
style.save_figure(fig, OUT / f"hourly_by_use_{BAND}", data=h)
