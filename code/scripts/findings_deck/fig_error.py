"""Slide 9, question 4, where the error is and what removes it: three panels.

A: the 91 forecast sites, circle area by the summed absolute error of the day-ahead forecast
   over the whole record (zero-shot model with the weather forecast); sites with on-site
   generation in gold. No site is named.
B: the share of the metered demand and the share of the forecast error at the 12 sites with
   a plant, whole record.
C: error of the next two hours when the forecast is made again every two hours with the meter
   data of the same day (every third day of the record): the day-ahead forecast, the
   day-ahead forecast plus a zero-shot forecast of its error, and a new zero-shot forecast of
   the demand. All sites, the sites with a plant, and their days with a large step (the
   import moves by more than its mean). 95 % intervals from resampling sites.

analysis/forecast_record.py (site_days.parquet, by_period.csv),
analysis/forecast_extensions.py (correction_errors.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    CHARCOAL,
    HEIGHT,
    LIGHT_BLUE,
    MID_GREY,
    MODEL,
    PLANT,
    PLANT_EDGE,
    REPORTS,
    SHELL,
    TEAL,
    TEXT_TEAL,
    WIDTH,
    england_map,
    figure,
    legend_below,
    save,
    title,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from sense_energy.eda import common as C

FR = REPORTS / "forecast_record"
sd = pd.read_parquet(FR / "site_days.parquet", columns=["site_code", "model", "a", "ae"])
sd = sd[sd["model"] == "chronos2_ifs"]
per = sd.groupby("site_code")[["a", "ae"]].sum().reset_index()
plant_sites = set(pd.read_csv(REPORTS / "chp_dispatch" / "classification.csv")["site_code"])
per["plant"] = per["site_code"].isin(plant_sites)
per = per.merge(C.site_points()[["site_code", "latitude", "longitude"]], on="site_code")
per = per.dropna(subset=["latitude"]).sort_values("ae", ascending=False)
bp = pd.read_csv(FR / "by_period.csv")
whole = bp[(bp["window"] == "whole record") & (bp["model"] == "chronos2_ifs")].iloc[0]
cor = pd.read_csv(REPORTS / "forecast_extensions" / "correction_errors.csv")

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [0.9, 0.42, 1.25]})

# A: map of the error
england_map(a, face=SHELL)
mwh = per["ae"] / 1000
scale = 30.0 / np.sqrt(mwh.max())
for is_plant, face, edge in ((False, LIGHT_BLUE, "white"), (True, PLANT, PLANT_EDGE)):
    d = per[per["plant"] == is_plant]
    a.scatter(
        d["longitude"],
        d["latitude"],
        s=(1.2 + scale * np.sqrt(d["ae"] / 1000)) ** 2 / 4,
        color=face,
        edgecolor=edge,
        linewidth=0.4,
        alpha=0.9,
        zorder=4 if is_plant else 3,
    )
title(a, "Forecast error of each site", pad=2)
a.legend(
    handles=[
        Line2D(
            [], [], marker="o", ls="none", ms=3.2, mfc=PLANT, mec=PLANT_EDGE, mew=0.4, label="plant"
        ),
        Line2D(
            [],
            [],
            marker="o",
            ls="none",
            ms=3.2,
            mfc=LIGHT_BLUE,
            mec="white",
            mew=0.4,
            label="no plant",
        ),
    ],
    loc="upper left",
    bbox_to_anchor=(-0.06, 0.93),
    fontsize=5.8,
    handletextpad=0.2,
    borderaxespad=0.1,
)

# B: share of demand and of error
shares = [("demand", whole["plant_demand_share_pct"]), ("error", whole["plant_error_share_pct"])]
for x, (_label, v) in enumerate(shares):
    b.bar(x, v, width=0.66, color=PLANT, edgecolor=PLANT_EDGE, linewidth=0.4)
    b.bar(x, 100 - v, bottom=v, width=0.66, color=LIGHT_BLUE)
    b.text(x, v / 2, f"{v:.0f} %", ha="center", va="center", fontsize=6.0, color=CHARCOAL)
    b.text(
        x,
        v + (100 - v) / 2,
        f"{100 - v:.0f} %",
        ha="center",
        va="center",
        fontsize=6.0,
        color=CHARCOAL,
    )
b.set_xticks([0, 1])
b.set_xticklabels([s[0] for s in shares])
b.set_ylim(0, 100)
b.set_yticks([0, 50, 100])
b.set_ylabel("Share (%)")
b.tick_params(axis="x", length=0)
title(b, "12 sites with a plant")

# C: the next two hours with the meter of the day
groups = ["all", "on-site generation", "on-site generation, days with a large step"]
labels = ["all sites", "sites with\na plant", "sites with a plant,\ndays with a large step"]
methods = (
    ("day_ahead", "day-ahead forecast", MODEL),
    ("corrected", "day-ahead plus a forecast of its error", TEAL),
    ("direct", "new forecast every two hours", TEXT_TEAL),
)
for j, (model, label, colour) in enumerate(methods):
    d = cor[cor["model"] == model].set_index("group").reindex(groups)
    x = np.arange(len(groups)) + (j - 1) * 0.27
    c.bar(x, d["nmae_pct"], width=0.25, color=colour, label=label)
    c.errorbar(
        x,
        d["nmae_pct"],
        yerr=[d["nmae_pct"] - d["nmae_lo"], d["nmae_hi"] - d["nmae_pct"]],
        fmt="none",
        ecolor=CHARCOAL,
        elinewidth=0.5,
        capsize=0,
    )
    for xx, hi, v in zip(x, d["nmae_hi"], d["nmae_pct"], strict=True):
        c.text(xx, hi + 0.8, f"{v:.0f}", ha="center", fontsize=5.6, color=CHARCOAL)
c.set_xticks(range(len(groups)))
c.set_xticklabels(labels)
c.set_ylim(0, 45)
c.set_yticks([0, 10, 20, 30, 40])
c.set_ylabel("Error of the next two hours (%)")
c.tick_params(axis="x", length=0)
c.grid(axis="y")
legend_below(fig, c, ncol=3, handlelength=1.0)
title(c, "With the meter data of the same day")
save(fig, "error", data=per[["site_code", "plant", "a", "ae", "latitude", "longitude"]])
print(
    whole[["plant_demand_share_pct", "plant_error_share_pct", "top3_error_share_pct"]]
    .astype(float)
    .round(1)
    .to_dict()
)
print(
    cor[cor["group"].isin(groups)]
    .pivot(index="group", columns="model", values="nmae_pct")
    .round(1)
    .to_string()
)
_ = (MID_GREY, Patch)
