"""Annual electricity 2023/24: our half-hourly meters (grossed up for gaps) against the site's
ERIC return, one marker per site coloured by the coverage check; dashed line = equality. One plot."""

from _common import OUT, flags, loglog

from sense_energy.visualization import style

f = flags()
f = f[f["elec_kwh"] > 0].dropna(subset=["energy_ratio"])
f = f[f["coverage_flag"] != "no_eric"]

fig, ax = style.figure(style.SQUARE)
for key, name in style.COVERAGE_FLAG_NAMES.items():
    d = f[f["coverage_flag"] == key]
    if d.empty:
        continue
    ax.plot(
        d["elec_kwh"] / 1e3,
        d["meter_kwh_fy"] / 1e3,
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.COVERAGE_FLAG_COLORS[name],
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=f"{name} (n = {len(d)})",
        zorder=3,
    )
loglog(ax, 5, 6e4)
ax.set_xlabel("ERIC electricity 2023/24 (MWh)")
ax.set_ylabel("Metered electricity 2023/24 (MWh)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "meter_vs_eric", data=f)
