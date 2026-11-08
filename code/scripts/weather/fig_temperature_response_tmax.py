"""Daily electricity demand against the daily maximum temperature, relative to an 18-20 degC day of
the same month, by site group. One plot.

Same fixed-effect regression as the daily-mean curve (analysis/weather_response.py) with bands
on the daily maximum up to 30-32 degC and 32 degC and above; the open-ended bands are drawn at 0
and 33 degC. 95 % intervals clustered by week.
"""

from _common import OUT, response

from sense_energy.visualization import style

c = response("curve")
c = c[(c["outcome"] == "daily mean") & (c["variable"] == "daily maximum")]

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
ax.set_xlabel("Daily maximum temperature (°C)")
ax.set_ylabel("Demand vs an 18–20 °C day (%)")
style.style_axis(ax)
style.add_bottom_legend(ax)
print(c.pivot_table(index="bin_lo", columns="group", values="coef_pct").round(1).to_string())
style.save_figure(fig, OUT / "temperature_response_tmax", data=c)
