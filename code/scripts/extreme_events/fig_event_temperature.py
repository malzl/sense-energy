"""The size of the event by day from its onset at the site: degrees warmer (heat wave, daily
maximum) or colder (cold spell, daily mean) than the reference days. The companion of
``event_time``: the temperature steps up on day 0, the reaction to heat follows later.

analysis/extreme_events.py (event_time.csv).
"""

from _common import OUT, PLAIN, table

from sense_energy.visualization import style

t = table("event_time")
t = t[t["group"] == PLAIN]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
style.reference_line(ax, 0.0, axis="x")
for kind in ("cold spell", "heatwave"):
    d = t[t["type"] == kind]
    ax.plot(
        d["day"],
        d["degrees"],
        color=style.WEATHER_EVENT_COLORS[kind],
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=style.EVENT_NAMES[kind],
    )
ax.set_xlabel("Days from the first day of the event")
ax.set_ylabel("Degrees beyond the reference days (°C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / "event_temperature", data=t)
