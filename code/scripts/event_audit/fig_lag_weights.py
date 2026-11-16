"""How long the temperature acts: the effect on today's demand of a degree today and of a
degree one, two and three days before (per cent per degree; cold: a degree colder in the daily
mean; heat: a degree of daily maximum above 24 degC). 95 % intervals clustered by site and date.

analysis/event_audit.py (lag_models.csv).
"""

from _common import OUT, table

from sense_energy.visualization import style

lag = table("lag_models")
lag = lag[(~lag["event_terms"]) & (lag["model"] == "the day and three days before")].copy()
lag["lag"] = lag["term"].str[-1].astype(int)
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for j, (season, kind, sign) in enumerate((("cold", "cold spell", -1.0), ("heat", "heatwave", 1.0))):
    d = lag[lag["season"] == season].sort_values("lag")
    coef = sign * d["coef"]
    lo, hi = (sign * d["lo"], sign * d["hi"]) if sign > 0 else (sign * d["hi"], sign * d["lo"])
    ax.errorbar(
        d["lag"] + (j - 0.5) * 0.14,
        coef,
        yerr=[coef - lo, hi - coef],
        fmt="o",
        markersize=style.MARKER_SIZE + 1,
        color=style.WEATHER_EVENT_COLORS[kind],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label="colder" if kind == "cold spell" else "hotter",
    )
ax.set_xticks(range(4))
ax.set_xticklabels(["today", "1 day\nbefore", "2 days\nbefore", "3 days\nbefore"])
ax.set_ylabel("Effect on today's demand (% per °C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.26, bottom=0.32)
style.save_figure(fig, OUT / "lag_weights", data=lag)
