"""Treatment-response curve on the daily mean under specification B (date effects) with each
thermometer, sites without CHP, common site-days. One plot.

Reference 14-16 degC; open-ended bands drawn at -3 and 25 degC (analysis/thermometers.py).
"""

from _common import OUT, thermometers

from sense_energy.visualization import style

c = thermometers("response_curve")
c = c[(c["variable"] == "daily mean") & (c["spec"] == "B")]

fig, ax = style.figure(style.SINGLE_COLUMN)
for name in style.THERMOMETER_ORDER:
    d = c[c["thermometer"] == name].sort_values("bin_mid")
    if d.empty:
        continue
    color = style.THERMOMETER_COLORS[name]
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
        label=style.THERMOMETER_NAMES[name],
    )
style.reference_line(ax, 0.0)
ax.set_xlabel("Daily mean temperature (°C)")
ax.set_ylabel("Demand vs a 15 °C day (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False)
print(c.pivot_table(index="bin", columns="thermometer", values="coef_pct").round(1).to_string())
style.save_figure(fig, OUT / "thermometer_response", data=c)
