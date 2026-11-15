"""The reaction per degree in each summer and winter of the record (sites without on-site
generation; mean site reaction over the mean size of the events, 95 % interval from resampling
sites).

analysis/extreme_events.py (seasons.csv).
"""

from _common import OUT, table

from sense_energy.visualization import style

s = table("seasons").sort_values(["type", "season"]).reset_index(drop=True)
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for kind, d in s.groupby("type"):
    ax.errorbar(
        d.index,
        d["pct_per_degree"],
        yerr=[d["pct_per_degree"] - d["lo"], d["hi"] - d["pct_per_degree"]],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.WEATHER_EVENT_COLORS[kind],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.EVENT_NAMES[kind],
    )
ax.set_xticks(s.index)
ax.set_xticklabels(s["season"].str.replace(" ", "\n"))
ax.set_ylabel("Reaction (% per °C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.3)
style.save_figure(fig, OUT / "seasons", data=s)
