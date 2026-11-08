"""Monthly cooling degree days (base 15.5 degC, integration method) per site: median and IQR across sites. One plot."""

from _common import OUT, weather_daily

from sense_energy.visualization import style

d = weather_daily()
d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
monthly = d.groupby(["site_code", "month"])["cdd_15_5"].sum()
# Only complete months (the ERA5 window starts on 1 December 2022 and ends 31 May 2026).
q = monthly.groupby("month").quantile([0.25, 0.5, 0.75]).unstack()
q.columns = ["q25", "median", "q75"]
q = q.reset_index()

fig, ax = style.figure(style.WIDE_SINGLE)
ax.fill_between(
    q["month"], q["q25"], q["q75"], step="mid", color="#000000", alpha=style.BAND_ALPHA, linewidth=0
)
ax.step(q["month"], q["median"], where="mid", color="#000000", linewidth=style.LINEWIDTH["primary"])
ax.set_ylim(bottom=0)
ax.set_xlabel("Month")
ax.set_ylabel("Cooling degree days, base 15.5 °C (°C day)")
style.style_axis(ax)
print(f"{d['site_code'].nunique()} sites, {d['date'].min().date()} -> {d['date'].max().date()}")
style.save_figure(fig, OUT / "cooling_need_monthly", data=q)
