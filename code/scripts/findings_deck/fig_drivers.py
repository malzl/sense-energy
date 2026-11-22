"""Slide 4, question 2 (what changes the demand of a site?), three panels.

A: the change in daily demand by day from the first day of the event at the site; B: by hour
   of the day on the event days. Sites with complete meter data and no on-site generation;
   7 heat waves and 13 cold spells (Met Office and UKHSA criteria); change against the same
   weekdays one to three weeks before and after; band: 95 % interval across sites.
C: change in demand in a cold spell and in a heat wave (the same sites), on placebo events
   (the events moved by five to eight weeks and by one and two years: the range of 12 runs) and
   on doctor-strike weekdays (08:00 to 18:00, 34 weekdays away from holidays, 132 sites).

analysis/extreme_events.py (event_time.csv, hour_profile.csv), analysis/event_audit.py
(grid_summary.csv), analysis/forecast_anatomy.py (strike_effect.csv).
"""

from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from _fd import (
    ASH,
    CHARCOAL,
    COLD,
    HEAT,
    HEIGHT,
    MID_GREY,
    REPORTS,
    TEXT_ORANGE,
    WIDTH,
    figure,
    save,
    title,
)

EE = REPORTS / "extreme_events"
t = pd.read_csv(EE / "event_time.csv")
t = t[(t["group"] == "no on-site generation") & t["day"].between(-4, 10)]
h = pd.read_csv(EE / "hour_profile.csv")
h = h[h["group"] == "no on-site generation"]
g = pd.read_csv(REPORTS / "event_audit" / "grid_summary.csv")
g = g[g["metric"] == "reaction, no plant"].set_index("type")
s = pd.read_csv(REPORTS / "forecast_anatomy" / "strike_effect.csv")
s = s[
    (s["outcome"] == "08:00 to 18:00")
    & (s["term"] == "strike weekday, away from holidays")
    & (s["site_type"] == "all sites")
].iloc[0]
KINDS = (("cold spell", COLD, "cold spell"), ("heatwave", HEAT, "heat wave"))

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [1, 1, 1.02]})
for ax in (a, b):
    ax.axhline(0, color=ASH, linewidth=0.5)
    ax.set_ylim(-4, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(axis="y")
for kind, colour, label in KINDS:
    d = t[t["type"] == kind]
    a.fill_between(d["day"], d["lo"], d["hi"], color=colour, alpha=0.18, linewidth=0)
    a.plot(d["day"], d["reaction_pct"], color=colour, linewidth=1.3, label=label)
    d = h[h["type"] == kind]
    b.fill_between(d["hour"] + 0.25, d["lo"], d["hi"], color=colour, alpha=0.18, linewidth=0)
    b.plot(d["hour"] + 0.25, d["reaction_pct"], color=colour, linewidth=1.3)
a.axvline(0, color=ASH, linewidth=0.5, linestyle=(0, (2, 2)))
a.set_xlim(-4, 10)
a.set_xticks([-4, 0, 2, 4, 6, 8, 10])
a.set_xlabel("Days from the first day of the event")
a.set_ylabel("Change in demand (%)")
title(a, "Day by day")
b.set_xlim(0, 24)
b.set_xticks([0, 6, 12, 18, 24])
b.set_xticklabels(["00:00", "06:00", "12:00", "18:00", "24:00"])
b.set_xlabel("Hour of the day, event days")
b.tick_params(axis="y", labelleft=False)
title(b, "Hour by hour")

rows = [
    ("Cold spell", *g.loc["cold spell", ["baseline", "baseline_lo", "baseline_hi"]], COLD),
    (
        "Cold spell, placebo",
        *g.loc["cold spell", ["placebo_mean", "placebo_min", "placebo_max"]],
        MID_GREY,
    ),
    ("Heat wave", *g.loc["heatwave", ["baseline", "baseline_lo", "baseline_hi"]], HEAT),
    (
        "Heat wave, placebo",
        *g.loc["heatwave", ["placebo_mean", "placebo_min", "placebo_max"]],
        MID_GREY,
    ),
    ("Doctor strike", s["effect_pct"], s["lo_pct"], s["hi_pct"], CHARCOAL),
]
d = pd.DataFrame(rows, columns=["label", "value", "lo", "hi", "colour"])
pos = [4.4, 3.6, 2.4, 1.6, 0.4]
c.axvline(0, color=ASH, linewidth=0.5)
for y, r in zip(pos, d.itertuples(), strict=True):
    c.plot([r.lo, r.hi], [y, y], color=r.colour, linewidth=2.4, solid_capstyle="butt")
    c.plot(r.value, y, "o", color=r.colour, ms=3.8, mec="white", mew=0.5)
    v = Decimal(f"{r.value:.2f}").quantize(Decimal("0.1"), ROUND_HALF_UP)
    c.text(r.hi + 0.6, y, f"{v:+} %".replace("-", "−"), va="center", fontsize=5.9, color=CHARCOAL)
c.set_yticks(pos)
c.set_yticklabels(d["label"])
c.set_ylim(-0.2, 5.0)
c.set_xlim(-5, 18)
c.set_xticks([-5, 0, 5, 10, 15])
c.set_xlabel("Change in demand (%)")
c.grid(axis="x")
c.tick_params(axis="y", length=0)
title(c, "Weather against patient activity")
a.text(1.2, 11.6, "cold spell", fontsize=6.0, color=COLD)
a.text(4.3, 8.3, "heat wave", fontsize=6.0, color=TEXT_ORANGE)
save(fig, "drivers", data=d.drop(columns="colour"))
print(d.drop(columns="colour").round(2).to_string(index=False))
for kind, x in h.groupby("type"):
    eve = x[(x["hour"] >= 16) & (x["hour"] < 22)]["reaction_pct"].mean()
    morn = x[(x["hour"] >= 8) & (x["hour"] < 12)]["reaction_pct"].mean()
    print(kind, "16:00-22:00", round(eve, 1), "08:00-12:00", round(morn, 1))
