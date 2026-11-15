"""Finding 1, right: the reaction of demand by hour of the day on the event days. Mean over the
sites without on-site generation with the 95 % interval."""

import pandas as pd
from _deck import ASH, BLUE, ORANGE, REPORTS, figure, legend_below, save

h = pd.read_csv(REPORTS / "extreme_events" / "hour_profile.csv")
h = h[h["group"] == "no on-site generation"]
fig, ax = figure(46, 39)
ax.axhline(0, color=ASH, linewidth=0.5)
for kind, colour, label in (("cold spell", BLUE, "cold spell"), ("heatwave", ORANGE, "heat wave")):
    d = h[h["type"] == kind]
    x = d["hour"] + 0.25
    ax.fill_between(x, d["lo"], d["hi"], color=colour, alpha=0.2, linewidth=0)
    ax.plot(x, d["reaction_pct"], color=colour, linewidth=1.3, label=label)
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_ylim(-5, 20)
ax.set_yticks([-5, 0, 5, 10, 15, 20])
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Change in demand (%)")
ax.grid(axis="y")
legend_below(fig, ax, ncol=2, handlelength=1.2)
save(fig, "event_hours", data=h)
for kind, d in h.groupby("type"):
    eve = d[(d["hour"] >= 16) & (d["hour"] < 22)]["reaction_pct"].mean()
    morn = d[(d["hour"] >= 8) & (d["hour"] < 12)]["reaction_pct"].mean()
    print(kind, "16:00-22:00", round(eve, 1), "08:00-12:00", round(morn, 1))
