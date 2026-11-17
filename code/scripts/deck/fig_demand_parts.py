"""Slide 2, strip above the maps: the parts of the demand of one site above its base load in
one year. Heating follows cold days, cooling follows hot days, lighting follows dark days. The
parts come from the same site slopes as the maps (analysis/heat_drivers.py: cold below 14 degC,
heat above 20 degC, daily sunshine), applied to the weather of 2025 at the site and smoothed
over three weeks. An example with initial data; no site is named."""

import numpy as np
import pandas as pd
from _deck import ASH, BLUE, CHARCOAL, ORANGE, REPORTS, TEXT_GOLD, TEXT_ORANGE, YELLOW, figure, save

from sense_energy.config import INTERIM_DIR

SITE, YEAR, SMOOTH_DAYS = "RYVF1", 2025, 9

b = pd.read_csv(REPORTS / "heat_drivers" / "site_features.csv").set_index("site_code").loc[SITE]
w = pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")
w = w[(w["site_code"] == SITE) & (w["date"].dt.year == YEAR)].set_index("date").sort_index()
terms = pd.DataFrame(
    {
        "heating": b["cold"] * (14.0 - w["t_mean"]).clip(lower=0),
        "cooling": b["heat"] * (w["t_mean"] - 20.0).clip(lower=0),
        "lighting": -b["sun"]
        * (w["solar_kwh_m2"].quantile(0.98) - w["solar_kwh_m2"]).clip(lower=0),
    }
)
total = terms.sum(axis=1)
above = 100 * (np.exp(total) - 1)  # per cent of the base load
parts = terms.div(total.where(total > 0), axis=0).fillna(0).mul(above, axis=0)
parts = parts.rolling(SMOOTH_DAYS, center=True, min_periods=7).mean()

fig, (label, ax) = figure(145.6, 13.6, ncols=2, gridspec_kw={"width_ratios": [0.2, 1]})
label.set_axis_off()
label.text(
    0.0,
    0.98,
    "Demand above\nthe base load",
    ha="left",
    va="top",
    fontsize=6.6,
    color=CHARCOAL,
    fontweight="bold",
    linespacing=1.15,
)
label.text(0.0, 0.2, f"one site in {YEAR}", ha="left", va="top", fontsize=5.8, color=ASH)
x = parts.index
low = np.zeros(len(parts))
for name, colour in (("heating", BLUE), ("lighting", YELLOW), ("cooling", ORANGE)):
    ax.fill_between(x, low, low + parts[name], color=colour, linewidth=0, alpha=0.92)
    low = low + parts[name].to_numpy()
ax.plot(x, low, color=CHARCOAL, linewidth=0.35)
top = float(np.nanmax(low))
ax.set_ylim(0, 1.08 * top)
ax.set_yticks([0, 10 * int(top // 10)])
ax.set_yticklabels(["0", f"+{10 * int(top // 10)} %"])
ax.set_xlim(x[0], x[-1])
months = pd.date_range(f"{YEAR}-01-01", f"{YEAR}-12-01", freq="MS")
ax.set_xticks(list(months + pd.Timedelta(days=14)))
ax.set_xticklabels([m.strftime("%b") for m in months])
ax.tick_params(axis="x", length=0, pad=1.5)
ax.tick_params(axis="y", length=0, pad=1.5)
for side in ("left",):
    ax.spines[side].set_visible(False)
top_line = pd.Series(low, index=x)
winter = pd.Timestamp(f"{YEAR}-01-24")
ax.text(
    winter,
    0.45 * parts.loc[winter, "heating"],
    "heating",
    color="white",
    fontsize=6.0,
    va="center",
    fontweight="bold",
)
spring = pd.Timestamp(f"{YEAR}-03-22")
ax.text(
    spring,
    top_line.loc[spring] + 0.16 * top,
    "lighting",
    color=TEXT_GOLD,
    fontsize=6.0,
    va="center",
    fontweight="bold",
)
peak = parts["cooling"].idxmax()
ax.text(
    peak,
    top_line.loc[peak] + 0.16 * top,
    "cooling",
    color=TEXT_ORANGE,
    fontsize=6.0,
    va="center",
    ha="center",
    fontweight="bold",
)
save(fig, "demand_parts", data=parts.reset_index())
print(parts.resample("MS").mean().round(1).to_string())
