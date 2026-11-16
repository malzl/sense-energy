"""Question 2, left: what changes the demand of a site. Change in daily demand in a cold spell
and in a heat wave (sites with no on-site generation, 95 % interval across sites), on placebo
days (the same events moved in time: the range of 12 placebo runs), and on doctor strike
weekdays (08:00 to 18:00, 132 sites, 95 % interval)."""

import pandas as pd
from _deck import ASH, BLUE, MID_GREY, ORANGE, REPORTS, TEAL, figure, save

g = pd.read_csv(REPORTS / "event_audit" / "grid_summary.csv")
g = g[g["metric"] == "reaction, no plant"].set_index("type")
s = pd.read_csv(REPORTS / "forecast_anatomy" / "strike_effect.csv")
s = s[
    (s["outcome"] == "08:00 to 18:00")
    & (s["term"] == "strike weekday, away from holidays")
    & (s["site_type"] == "all sites")
].iloc[0]
rows = [
    (
        "Cold spell",
        g.loc["cold spell", "baseline"],
        g.loc["cold spell", "baseline_lo"],
        g.loc["cold spell", "baseline_hi"],
        BLUE,
    ),
    (
        "Cold spell, placebo",
        g.loc["cold spell", "placebo_mean"],
        g.loc["cold spell", "placebo_min"],
        g.loc["cold spell", "placebo_max"],
        MID_GREY,
    ),
    (
        "Heat wave",
        g.loc["heatwave", "baseline"],
        g.loc["heatwave", "baseline_lo"],
        g.loc["heatwave", "baseline_hi"],
        ORANGE,
    ),
    (
        "Heat wave, placebo",
        g.loc["heatwave", "placebo_mean"],
        g.loc["heatwave", "placebo_min"],
        g.loc["heatwave", "placebo_max"],
        MID_GREY,
    ),
    ("Doctor strike", s["effect_pct"], s["lo_pct"], s["hi_pct"], TEAL),
]
d = pd.DataFrame(rows, columns=["label", "value", "lo", "hi", "colour"])
fig, ax = figure(46, 39)
ax.axvline(0, color=ASH, linewidth=0.5)
pos = [4.3, 3.5, 2.3, 1.5, 0.3]
for y, r in zip(pos, d.itertuples(), strict=True):
    ax.plot(
        [r.lo, r.hi], [y, y], color=r.colour, linewidth=2.2, solid_capstyle="butt", marker="none"
    )
    ax.plot(
        r.value,
        y,
        "o",
        color=r.colour,
        markersize=3.6,
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
ax.set_yticks(pos)
ax.set_yticklabels(d["label"])
ax.set_ylim(-0.3, 4.9)
ax.set_xlim(-5, 15)
ax.set_xticks([-5, 0, 5, 10, 15])
ax.set_xlabel("Change in demand (%)")
ax.grid(axis="x")
ax.tick_params(axis="y", length=0)
save(fig, "drivers", data=d.drop(columns="colour"))
print(d.drop(columns="colour").round(2).to_string(index=False))
