"""What a heat wave does to the electricity demand of a site, in two views: by day from the
first day of the heat wave, and by hour of the day on heat-wave days. Sites with complete meter
data and no on-site generation, 7 heat waves from 2023 to 2025 (Met Office definition). Change
against the same weekdays one to three weeks before and after; band: 95 % interval across
sites (analysis/extreme_events.py)."""

import pandas as pd
from _deck import ASH, ORANGE, REPORTS, SHELL, TEXT_ORANGE, figure, save

EE = REPORTS / "extreme_events"
PLAIN = "no on-site generation"
t = pd.read_csv(EE / "event_time.csv")
t = t[(t["type"] == "heatwave") & (t["group"] == PLAIN) & t["day"].between(-5, 10)]
h = pd.read_csv(EE / "hour_profile.csv")
h = h[(h["type"] == "heatwave") & (h["group"] == PLAIN)]

fig, (a, b) = figure(145.6, 50, ncols=2, sharey=True)
for ax, d, x in ((a, t, t["day"]), (b, h, h["hour"] + 0.25)):
    ax.axhline(0, color=ASH, linewidth=0.5)
    ax.fill_between(x, d["lo"], d["hi"], color=ORANGE, alpha=0.22, linewidth=0)
    ax.plot(x, d["reaction_pct"], color=ORANGE, linewidth=1.6)
    ax.grid(axis="y")
    ax.set_ylim(-8, 18.5)
    ax.set_yticks([-5, 0, 5, 10, 15])
a.axvline(0, color=ASH, linewidth=0.6, linestyle=(0, (2, 2)))
a.set_xlim(-5, 10)
a.set_xticks([-5, 0, 5, 10])
a.set_xlabel("Days from the first day of the heat wave")
a.set_title("Change in demand, day by day (%)", loc="left", fontsize=6.6, pad=3, color=ASH)
peak = t.loc[t["reaction_pct"].idxmax()]
a.annotate(
    f"day {int(peak['day']) + 1}: +{peak['reaction_pct']:.0f} %",
    (peak["day"], peak["reaction_pct"]),
    xytext=(4.6, 13.4),
    fontsize=6.6,
    color=TEXT_ORANGE,
    fontweight="bold",
    arrowprops={"arrowstyle": "-", "color": TEXT_ORANGE, "linewidth": 0.5},
)
first = t[t["day"] == 0].iloc[0]
a.text(0.3, -6.3, f"first day: +{first['reaction_pct']:.0f} %", fontsize=6.2, color=TEXT_ORANGE)
b.axvspan(16, 22, color=SHELL, zorder=0, linewidth=0)
b.set_xlim(0, 24)
b.set_xticks([0, 6, 12, 18, 24])
b.set_xlabel("Hour of the day")
b.set_title("Change in demand, hour by hour (%)", loc="left", fontsize=6.6, pad=3, color=ASH)
evening = h[(h["hour"] >= 16) & (h["hour"] < 22)]["reaction_pct"].mean()
morning = h[(h["hour"] >= 8) & (h["hour"] < 12)]["reaction_pct"].mean()
b.text(
    19,
    16.6,
    f"16:00 to 22:00: +{evening:.0f} %",
    fontsize=6.6,
    color=TEXT_ORANGE,
    fontweight="bold",
    ha="center",
)
b.text(5.2, -6.3, f"08:00 to 12:00: +{morning:.0f} %", fontsize=6.2, color=TEXT_ORANGE, ha="center")
save(fig, "heatwave_effects", data=pd.concat([t.assign(view="day"), h.assign(view="hour")]))
print(
    "sites:",
    int(t["sites"].max()),
    "| first day",
    round(first["reaction_pct"], 1),
    "| peak day",
    int(peak["day"]),
    round(peak["reaction_pct"], 1),
    "| evening",
    round(evening, 1),
    "| morning",
    round(morning, 1),
)
