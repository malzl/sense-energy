"""Slide 1: the forecasting problem in one picture. One hospital site, eight days in 2024, as
indices of the site's own mean (no site is named): the inputs that are known (outdoor
temperature, visitors per day, past demand) and the demand of the next day, which the model
forecasts at 16:30 on the day before for each half-hour, with an 80 % interval.

Demand: half-hourly meter. Temperature: ERA5 at the site. Visitors: daily counts (The Oxford
Partnership). Forecast: Chronos-2 with calendar, price, weather and the visitor count of the
last full day (experiments/forecast_probe.py, run "mobility").
"""

import pandas as pd
import pyarrow.parquet as pq
from _deck import (
    ASH,
    BLUE,
    MID_GREY,
    ORANGE,
    REPORTS,
    SHELL,
    TEXT_GOLD,
    TEXT_TEAL,
    YELLOW,
    figure,
    save,
)

from sense_energy.config import INTERIM_DIR, PROCESSED_DIR

SITE, TARGET, HISTORY_DAYS, MODEL, TZ = "RH8K6", "2024-07-25", 7, "mob_prev_day", "Europe/London"

day = pd.Timestamp(TARGET, tz=TZ)
start, end = day - pd.Timedelta(days=HISTORY_DAYS), day + pd.Timedelta(days=1)
issue = day - pd.Timedelta(hours=7.5)

y = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet", columns=[SITE])[SITE]
y.index = pd.to_datetime(y.index, utc=True).tz_convert(TZ)
level = y.loc[start - pd.Timedelta(days=28) : issue].mean()
y = 100 * y.loc[start:end] / level

f = pd.read_parquet(REPORTS / "forecast_anatomy" / "forecasts_mobility.parquet")
f = f[(f["model"] == MODEL) & (f["site_code"] == SITE)].copy()
f["target"] = pd.to_datetime(f["target"], utc=True).dt.tz_convert(TZ)
f = f[(f["target"] >= day) & (f["target"] < end)].set_index("target").sort_index()
f = 100 * f[["point", "q10", "q90"]] / level

t = pq.read_table(INTERIM_DIR / "weather_reanalysis.parquet", filters=[("site_code", "=", SITE)])
t = t.to_pandas().drop_duplicates("datetime").set_index("datetime")["t2m"].sort_index()
t.index = pd.to_datetime(t.index, utc=True).tz_convert(TZ)
t = t.loc[start:end]
t = t - 273.15 if t.mean() > 100 else t

v = pd.read_parquet(INTERIM_DIR / "mobility_visits_daily.parquet")
v = v[v["id"] == SITE].assign(date=lambda d: pd.to_datetime(d["date"]))
mean_visits = v.loc[
    v["date"].between(str((start - pd.Timedelta(days=28)).date()), TARGET), "visits"
].mean()
v = v.set_index("date")["visits"].loc[str(start.date()) : str((day - pd.Timedelta(days=1)).date())]
v = 100 * v / mean_visits
v.index = pd.DatetimeIndex(v.index).tz_localize(TZ) + pd.Timedelta(hours=12)

fig, axes = figure(
    145.6, 41.5, nrows=3, sharex=True, gridspec_kw={"height_ratios": [0.9, 0.75, 2.6]}
)
a, b, c = axes
for ax in axes:
    ax.axvspan(day, end, color=SHELL, zorder=0, linewidth=0)
    ax.axvline(issue, color=ASH, linewidth=0.6, linestyle=(0, (2, 2)), zorder=2)
a.plot(t.index, t, color=TEXT_TEAL, linewidth=1.0)
a.set_title("Weather: air temperature (°C)", loc="left", fontsize=6.0, pad=2, color=ASH)
a.set_yticks([10, 20] if t.max() > 20 else [0, 10])
b.bar(v.index, v, width=0.62, color=YELLOW, edgecolor=TEXT_GOLD, linewidth=0.35)
b.set_title("Mobility: visitors each day (index)", loc="left", fontsize=6.0, pad=2, color=ASH)
b.set_ylim(0, 1.25 * v.max())
b.set_yticks([0, 100])
past = y.loc[:issue]
later = y.loc[issue:]
c.plot(past.index, past, color=BLUE, linewidth=0.9, label="known demand")
c.plot(later.index, later, color=MID_GREY, linewidth=0.8, label="demand of the next day")
c.fill_between(
    f.index, f["q10"], f["q90"], color=ORANGE, alpha=0.28, linewidth=0, label="80 % range"
)
c.plot(
    f.index, f["point"], color=ORANGE, linewidth=1.4, linestyle=(0, (3.0, 1.1)), label="forecast"
)
c.set_title(
    "Electricity demand for each half-hour (index)", loc="left", fontsize=6.0, pad=2, color=ASH
)
c.set_xlim(start, end)
ticks = pd.date_range(start, end, freq="D")[:-1]
c.set_xticks(list(ticks + pd.Timedelta(hours=12)))
c.set_xticklabels([d.strftime("%a") for d in ticks])
c.tick_params(axis="x", length=0)
for ax in axes:
    ax.grid(axis="y")
    for d in ticks[1:]:
        ax.axvline(d, color="#E6E4E0", linewidth=0.35, zorder=1)
h, lab = c.get_legend_handles_labels()
fig.legend(
    h, lab, loc="outside lower center", ncol=4, handlelength=1.8, columnspacing=1.6, fontsize=6.2
)
out = pd.concat([y.rename("demand_index"), f.add_prefix("forecast_")], axis=1)
save(fig, "problem_panel", data=out.reset_index())
print("level kW", round(level, 1), "| forecast rows", len(f), "| visitors", v.round(0).tolist())
print(
    "temperature",
    round(t.min(), 1),
    round(t.max(), 1),
    "| demand index",
    round(y.min()),
    round(y.max()),
)
