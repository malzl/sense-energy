"""Question 4, left: day-ahead error in each year of the record (2026 to 8 May), every day, 91
sites: the zero-shot model and the seasonal naive forecast, with the 95 % interval from
resampling sites."""

import numpy as np
import pandas as pd
from _deck import BLUE, MID_GREY, REPORTS, figure, legend_below, save

p = pd.read_csv(REPORTS / "forecast_record" / "by_period.csv")
whole = p[p["window"] == "whole record"].set_index("model")["days"]
best = "chronos2_ifs" if whole.get("chronos2_ifs", 0) > 1000 else "chronos2"
years = ["2023", "2024", "2025", "2026"]
p = p[p["window"].isin(years)]
fig, ax = figure(46, 39)
for j, (model, colour, label) in enumerate(
    (("seasonal_naive", MID_GREY, "same day last week"), (best, BLUE, "model with no training"))
):
    d = p[p["model"] == model].set_index("window").reindex(years)
    x = np.arange(len(years)) + (j - 0.5) * 0.38
    ax.bar(x, d["nmae_pct"], width=0.36, color=colour, label=label)
    ax.errorbar(
        x,
        d["nmae_pct"],
        yerr=[d["nmae_pct"] - d["nmae_lo"], d["nmae_hi"] - d["nmae_pct"]],
        fmt="none",
        ecolor="#3A3A3A",
        elinewidth=0.5,
        capsize=0,
    )
ax.set_xticks(range(len(years)))
ax.set_xticklabels(years)
ax.set_ylim(0, 25)
ax.set_yticks([0, 5, 10, 15, 20, 25])
ax.set_ylabel("Day-ahead error (%)")
ax.grid(axis="y")
ax.tick_params(axis="x", length=0)
legend_below(fig, ax, ncol=2)
save(fig, "record_years", data=p[p["model"].isin(["seasonal_naive", best])])
cols = [
    "window",
    "model",
    "site_days",
    "nmae_pct",
    "nmae_lo",
    "nmae_hi",
    "captured_point_pct",
    "plant_error_share_pct",
]
print(best)
print(p[p["model"].isin(["seasonal_naive", best])][cols].round(1).to_string(index=False))
