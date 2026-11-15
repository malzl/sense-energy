"""Finding 1, left: the reaction of demand by day from the first day of the event at the site.
Mean over the sites without on-site generation with the 95 % interval; heat waves and cold
spells that are part of a national event (7 and 13 events)."""

import pandas as pd
from _deck import ASH, BLUE, ORANGE, REPORTS, figure, legend_below, save

t = pd.read_csv(REPORTS / "extreme_events" / "event_time.csv")
t = t[(t["group"] == "no on-site generation") & t["day"].between(-5, 10)]
fig, ax = figure(46, 39)
ax.axhline(0, color=ASH, linewidth=0.5)
ax.axvline(0, color=ASH, linewidth=0.5, linestyle=(0, (2, 2)))
for kind, colour, label in (("cold spell", BLUE, "cold spell"), ("heatwave", ORANGE, "heat wave")):
    d = t[t["type"] == kind]
    ax.fill_between(d["day"], d["lo"], d["hi"], color=colour, alpha=0.2, linewidth=0)
    ax.plot(d["day"], d["reaction_pct"], color=colour, linewidth=1.3, label=label)
ax.set_xlim(-5, 10)
ax.set_xticks([-5, 0, 5, 10])
ax.set_ylim(-5, 20)
ax.set_yticks([-5, 0, 5, 10, 15, 20])
ax.set_xlabel("Days from the first day of the event")
ax.set_ylabel("Change in demand (%)")
ax.grid(axis="y")
legend_below(fig, ax, ncol=2, handlelength=1.2)
save(fig, "event_days", data=t)
b = pd.read_csv(REPORTS / "extreme_events" / "build_up.csv").round(2)
print(b.to_string(index=False))
