"""Error by half-hour after the issue at the sites with on-site generation, for the forecasts
that are made again every two hours on the day. The day-ahead forecast does not depend on the
half-hour after the issue and is the reference. Every third day of the record; mean over sites.

analysis/forecast_extensions.py (correction_steps.csv).
"""

from _common import OUT, PLANT, table

from sense_energy.visualization import style

s = table("correction_steps")
s = s[s["group"] == PLANT]
fig, ax = style.figure(style.SINGLE_COLUMN)
for model, name in style.CORRECTION_NAMES.items():
    d = s[s["model"] == model].sort_values("step")
    ax.plot(
        d["step"] * 30,
        d["nmae_pct"],
        color=style.CORRECTION_COLORS[model],
        marker=style.CORRECTION_MARKERS[model],
        markersize=style.MARKER_SIZE,
        linewidth=style.LINEWIDTH["primary"],
        linestyle="--" if model == "day_ahead" else "-",
        label=name,
    )
ax.set_xticks([30, 60, 90, 120])
ax.set_xlabel("Minutes after the issue")
ax.set_ylim(0, None)
ax.set_ylabel("Error (% of mean demand)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.26, bottom=0.5)
style.save_figure(fig, OUT / "correction_steps", data=s)
