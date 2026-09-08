"""Mean daily demand profile, weekday vs weekend, median and IQR across sites (gas). One plot."""

from sense_energy.eda import common as C
from sense_energy.visualization import style

ENERGY, LEVEL = "gas", "site"
wide, _ = C.series_for(ENERGY, LEVEL)
prof = C.daily_profile(C.normalise(wide))

fig, ax = style.figure(style.SINGLE_COLUMN)
for day_type in ("weekday", "weekend"):
    p = prof[prof["day_type"] == day_type]
    c = style.DAYTYPE_COLORS[day_type]
    ax.fill_between(p["hour"], p["q25"], p["q75"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(p["hour"], p["median"], color=c, linewidth=style.LINEWIDTH["primary"], label=day_type)
ax.set_xlim(0, 24)
ax.set_xticks([0, 6, 12, 18, 24])
ax.set_xlabel("Hour of day (local time)")
ax.set_ylabel(f"Demand index ({LEVEL} mean = 1)")
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(fig, C.OUT / f"daily_profile_{ENERGY}_{LEVEL}", data=prof)
