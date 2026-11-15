"""Every national heat wave and cold spell in the electricity record: the mean reaction of the
sites without on-site generation (demand on the event days against the reference days of the
same site), with the 95 % interval across sites, at the date of the event.

analysis/extreme_events.py (events.csv).
"""

from _common import OUT, PLAIN, table

from sense_energy.visualization import style

e = table("events", parse_dates=["start", "end"])
e = e[(e["group"] == PLAIN) & (e["sites"] >= 10)]
fig, ax = style.figure(style.WIDE_SHORT)
style.reference_line(ax, 0.0)
for kind, d in e.groupby("type"):
    ax.errorbar(
        d["start"],
        d["reaction_pct"],
        yerr=[d["reaction_pct"] - d["lo"], d["hi"] - d["reaction_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        color=style.WEATHER_EVENT_COLORS[kind],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=style.EVENT_NAMES[kind],
    )
ax.set_ylabel("Reaction of demand (%)")
style.style_axis(ax, grid="y", wide_short=True)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.26)
style.save_figure(fig, OUT / "event_timeline", data=e)
