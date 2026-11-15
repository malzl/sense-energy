"""The reaction by half-hour of the day (argument: heatwave or cold): demand of the half-hour on
the event days against the same half-hour of the reference days, mean over sites with the 95 %
interval, for sites with and without on-site generation (whose meters record grid import).

analysis/extreme_events.py (hour_profile.csv).
"""

from _common import OUT, band_line, kind_arg, table

from sense_energy.visualization import style

kind, tag = kind_arg()
h = table("hour_profile")
h = h[h["type"] == kind]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for group in style.PLANT_GROUP_ORDER:
    band_line(ax, h[h["group"] == group], "hour", style.PLANT_GROUP_COLORS[group], group)
ax.set_xlim(0, 23.5)
ax.set_xticks(range(0, 24, 4))
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Reaction of demand (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / f"hour_profile_{tag}", data=h)
