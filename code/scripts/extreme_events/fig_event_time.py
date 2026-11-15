"""The reaction by day from the onset of the event at the site (day 0 is the first day): mean
over the sites without on-site generation, 95 % interval across sites. Heat waves and cold
spells that are part of a national event.

analysis/extreme_events.py (event_time.csv).
"""

from _common import OUT, PLAIN, band_line, table

from sense_energy.visualization import style

t = table("event_time")
t = t[t["group"] == PLAIN]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
style.reference_line(ax, 0.0, axis="x")
for kind in ("cold spell", "heatwave"):
    d = t[t["type"] == kind]
    band_line(ax, d, "day", style.WEATHER_EVENT_COLORS[kind], style.EVENT_NAMES[kind], "o")
ax.set_xlabel("Days from the first day of the event")
ax.set_ylabel("Reaction of demand (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / "event_time", data=t)
