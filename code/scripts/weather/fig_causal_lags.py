"""Distributed-lag and placebo-lead coefficients of the cold and heat doses on daily demand at sites
without CHP (specification B). One plot.

Negative lags are leads: tomorrow's and the next day's temperature, which cannot cause today's
demand, given today's and the past week's. 95 % intervals clustered by site and date
(analysis/weather_causal.py). Doses offset horizontally.
"""

from _common import OUT, causal

from sense_energy.visualization import style

t = causal("leads_lags")
t = t[t["kind"] == "coefficient"]

fig, ax = style.figure(style.SINGLE_COLUMN)
offsets = {"hdd14": -0.12, "cdd20": 0.12}
for dose, off in offsets.items():
    d = t[t["dose"] == dose].sort_values("lag")
    ax.errorbar(
        d["lag"] + off,
        d["coef_pct"],
        yerr=[d["coef_pct"] - d["lo_pct"], d["hi_pct"] - d["coef_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=style.LINEWIDTH["reference"],
        color=style.DOSE_COLORS[dose],
        label=style.DOSE_NAMES[dose],
    )
style.reference_line(ax, 0.0)
style.reference_line(ax, -0.5, axis="x")
ax.set_xticks(range(int(t["lag"].min()), int(t["lag"].max()) + 1))
ax.set_xlabel("Lag of temperature behind demand (days)")
ax.set_ylabel("Demand change (% per °C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
print(t.pivot_table(index="lag", columns="dose", values="coef_pct").round(3).to_string())
print(f"joint p-value of the leads: {t['p_leads_joint'].iloc[0]:.3f}")
style.save_figure(fig, OUT / "causal_lags", data=t)
