"""Finding 3, left: demand on doctor strike weekdays against the same site, month and weekday,
hour by hour, with the 95 % interval (34 strike weekdays away from holidays, 132 sites). Grey:
the same days one week later, when there was no strike."""

import pandas as pd
from _deck import ASH, BLUE, MID_GREY, REPORTS, figure, legend_below, save

p = pd.read_csv(REPORTS / "forecast_anatomy" / "strike_profile.csv")
s = p[p["term"] == "strike weekday, away from holidays"].sort_values("hour")
q = p[p["term"] == "one week after (placebo)"].sort_values("hour")
fig, ax = figure(46, 39)
ax.axhline(0, color=ASH, linewidth=0.5)
ax.fill_between(s["hour"] + 0.5, s["lo_pct"], s["hi_pct"], color=BLUE, alpha=0.2, linewidth=0)
ax.plot(q["hour"] + 0.5, q["effect_pct"], color=MID_GREY, linewidth=1.0, label="one week later")
ax.plot(s["hour"] + 0.5, s["effect_pct"], color=BLUE, linewidth=1.3, label="strike days")
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_ylim(-10, 10)
ax.set_yticks([-10, -5, 0, 5, 10])
ax.set_xlabel("Hour of the day")
ax.set_ylabel("Change in demand (%)")
ax.grid(axis="y")
legend_below(fig, ax, ncol=2, handlelength=1.2, reverse=True)
save(fig, "strike_profile", data=p)
print(s["effect_pct"].round(2).min(), s["effect_pct"].round(2).max())
