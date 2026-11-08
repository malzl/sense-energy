"""Stacked event-study difference-in-differences: demand at sites inside a national heatwave or cold
spell against clean-control sites, by day from the national onset. One plot.

Sites without CHP. Reference period: 5 or more days before onset. Bands are 95 % intervals clustered
by site and date within the stacked sample (analysis/weather_causal.py). The dashed vertical line
marks the onset.
"""

from _common import OUT, causal

from sense_energy.visualization import style

e = causal("event_study")
e = e[(e["group"] == "no CHP") & (e["outcome"] == "demand")]

fig, ax = style.figure(style.WIDE_SINGLE)
for kind in ("heatwave", "cold spell"):
    d = e[e["type"] == kind].sort_values("k")
    c = style.WEATHER_EVENT_COLORS[kind]
    ax.fill_between(d["k"], d["lo_pct"], d["hi_pct"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(
        d["k"],
        d["coef_pct"],
        color=c,
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE - 1,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        label=f"{kind} (pre-trend p = {d['p_leads'].iloc[0]:.2f})",
    )
style.reference_line(ax, 0.0)
style.reference_line(ax, -0.5, axis="x")
ax.set_xlabel("Days from national event onset")
ax.set_ylabel("Demand, treated vs control sites (%)")
style.style_axis(ax)
style.add_bottom_legend(ax)
print(e.pivot_table(index="k", columns="type", values="coef_pct").round(2).to_string())
style.save_figure(fig, OUT / "causal_event_study", data=e)
