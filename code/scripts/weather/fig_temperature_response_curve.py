"""Daily electricity demand against daily mean temperature, relative to a 14-16 degC day of the same month. One plot.

Binned fixed-effect regression (analysis/weather_response.py): 2 degC bins, site x year-month and
site x weekday effects, 95 % intervals clustered by week. Open-ended tail bins (< -2 and >= 24 degC)
are drawn at -3 and 25 degC.
"""

from _common import OUT, response

from sense_energy.visualization import style

c = response("curve")
c = c[(c["outcome"] == "daily mean") & (c["variable"] == "daily mean")]

fig, ax = style.figure(style.SINGLE_COLUMN)
for g in style.SITE_GROUP_ORDER:
    d = c[c["group"] == g].sort_values("bin_mid")
    color = style.SITE_GROUP_COLORS[g]
    ax.fill_between(
        d["bin_mid"], d["lo_pct"], d["hi_pct"], color=color, alpha=style.BAND_ALPHA, linewidth=0
    )
    ax.plot(
        d["bin_mid"],
        d["coef_pct"],
        color=color,
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        label=f"{g} ({int(d['sites'].iloc[0])} sites)",
    )
style.reference_line(ax, 0.0)
ax.set_xlabel("Daily mean temperature (°C)")
ax.set_ylabel("Demand vs a 15 °C day (%)")
style.style_axis(ax)
style.add_bottom_legend(ax)
print(c.pivot_table(index="bin", columns="group", values="coef_pct").round(1).to_string())
style.save_figure(fig, OUT / "temperature_response_curve", data=c)
