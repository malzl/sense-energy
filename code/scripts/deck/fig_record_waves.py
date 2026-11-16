"""Question 4, right: share of the possible peak cut that a battery gets when it plans on the
forecast, on heat-wave days and on the other days of the same site and month, in the summers
of 2023 and 2025 (2024 had one small heat wave). 91 sites, events by the Met Office criterion."""

import numpy as np
import pandas as pd
from _deck import MID_GREY, ORANGE, REPORTS, figure, legend_below, save

FR = REPORTS / "forecast_record"
p = pd.read_csv(FR / "by_period.csv")
whole = p[p["window"] == "whole record"].set_index("model")["days"]
best = "chronos2_ifs" if whole.get("chronos2_ifs", 0) > 1000 else "chronos2"
e = pd.read_csv(FR / "event_days.csv")
e = e[
    (e["model"] == best)
    & (e["definition"] == "catalogue")
    & (e["type"] == "heatwave")
    & (e["sites_group"] == "all sites")
].set_index("season")
seasons = ["summer 2023", "summer 2025", "all"]
d = e.reindex(seasons)
fig, ax = figure(46, 39)
x = np.arange(len(seasons))
ax.bar(
    x - 0.19, d["captured_other_pct"], width=0.36, color=MID_GREY, label="other days, same month"
)
ax.bar(x + 0.19, d["captured_event_pct"], width=0.36, color=ORANGE, label="heat-wave days")
ax.set_xticks(x)
ax.set_xticklabels(["2023", "2025", "all\nsummers"])
ax.set_ylim(0, 60)
ax.set_yticks([0, 20, 40, 60])
ax.set_ylabel("Share of peak cut (%)")
ax.grid(axis="y")
ax.tick_params(axis="x", length=0)
legend_below(fig, ax, ncol=1)
save(fig, "record_waves", data=d.reset_index())
cols = [
    "sites",
    "event_site_days",
    "captured_event_pct",
    "captured_other_pct",
    "captured_diff_pts",
    "captured_diff_lo",
    "captured_diff_hi",
    "nmae_event_pct",
    "nmae_other_pct",
]
print(best)
print(d[cols].round(1).to_string())
