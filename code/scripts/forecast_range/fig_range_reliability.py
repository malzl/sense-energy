"""Share of the half-hours with the demand below each quantile of the day-ahead forecast
against the nominal share, by site group. On the diagonal the quantiles are right. Whole
record, zero-shot model with the weather forecast; mean over sites, 95 % interval from
resampling sites.

analysis/forecast_range.py (reliability.csv).
"""

from _common import OUT, table

from sense_energy.visualization import style

r = table("reliability")
fig, ax = style.figure(style.SQUARE)
ax.plot([0, 100], [0, 100], **style.REFERENCE_LINE, zorder=1)
for group, marker in zip(style.PLANT_GROUP_ORDER, ("o", "D"), strict=True):
    d = r[r["group"] == group].sort_values("nominal_pct")
    ax.errorbar(
        d["nominal_pct"],
        d["below_pct"],
        yerr=[d["below_pct"] - d["lo"], d["hi"] - d["below_pct"]],
        fmt=marker + "-",
        markersize=style.MARKER_SIZE,
        color=style.PLANT_GROUP_COLORS[group],
        linewidth=style.LINEWIDTH["secondary"],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=group,
    )
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks([10, 30, 50, 70, 90])
ax.set_yticks([10, 30, 50, 70, 90])
ax.set_xlabel("Quantile of the forecast (%)")
ax.set_ylabel("Half-hours with the demand below it (%)")
style.style_axis(ax, grid="both")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "range_reliability", data=r)
