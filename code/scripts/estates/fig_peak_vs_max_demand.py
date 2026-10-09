"""Highest metered half-hour of 2023/24 against the site's ERIC maximum electrical demand,
one marker per site; dashed line = equality. One plot."""

from _common import OUT, flags, loglog

from sense_energy.visualization import style

f = flags().dropna(subset=["peak_ratio"])
f = f[f["coverage_flag"].isin(style.COVERAGE_FLAG_NAMES)]

fig, ax = style.figure(style.SQUARE)
for key, name in style.COVERAGE_FLAG_NAMES.items():
    d = f[f["coverage_flag"] == key]
    if d.empty:
        continue
    ax.plot(
        d["max_demand_kw"],
        d["meter_peak_kw"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.COVERAGE_FLAG_COLORS[name],
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=f"{name} (n = {len(d)})",
        zorder=3,
    )
loglog(ax, 3, 2e4)
ax.set_xlabel("ERIC maximum electrical demand (kW)")
ax.set_ylabel("Metered peak half-hour (kW)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "peak_vs_max_demand", data=f)
