"""Slide 8, question 4, what the forecast is worth: three panels.

A: share of the possible peak cut that a battery gets when it plans on the day-ahead forecast
   (power 20 % of the site's mean demand; one, two and four hours of storage), with the
   zero-shot model and with the same half-hour last week. Whole record (January 2023 to May
   2026), 91 sites; 95 % interval from resampling sites.
B: the same share on heat-wave days and on the other days of the same site and calendar month
   (two-hour battery), in the summers of 2023 and 2025 and over all summers. Heat-wave days by
   the Met Office criterion at the site.
C: heat-wave days minus the other days: the day-ahead error and the share of the peak cut, in
   points, with 95 % intervals from resampling sites.

analysis/forecast_record.py (decision.csv, event_days.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    CHARCOAL,
    HEAT,
    HEIGHT,
    MID_GREY,
    MODEL,
    NAIVE,
    REPORTS,
    WIDTH,
    figure,
    save,
    title,
)
from matplotlib.patches import Patch

FR = REPORTS / "forecast_record"
dec = pd.read_csv(FR / "decision.csv")
dec = dec[dec["battery_share_of_mean_demand"] == 0.2]
ev = pd.read_csv(FR / "event_days.csv")
ev = ev[
    (ev["model"] == "chronos2_ifs")
    & (ev["definition"] == "catalogue")
    & (ev["type"] == "heatwave")
    & (ev["sites_group"] == "all sites")
].set_index("season")

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [1, 1, 0.95]})

# A: battery size
hours = [1.0, 2.0, 4.0]
for j, (model, colour, label) in enumerate(
    (
        ("seasonal_naive", NAIVE, "same half-hour last week"),
        ("chronos2_ifs", MODEL, "model with no training"),
    )
):
    d = dec[dec["model"] == model].set_index("battery_hours").reindex(hours)
    x = np.arange(len(hours)) + (j - 0.5) * 0.38
    a.bar(x, d["captured_point_pct"], width=0.36, color=colour, label=label)
    a.errorbar(
        x,
        d["captured_point_pct"],
        yerr=[d["captured_point_pct"] - d["lo"], d["hi"] - d["captured_point_pct"]],
        fmt="none",
        ecolor=CHARCOAL,
        elinewidth=0.5,
        capsize=0,
    )
    if model == "chronos2_ifs":
        for xx, v in zip(x, d["captured_point_pct"], strict=True):
            a.text(xx, 2.0, f"{v:.0f}", ha="center", fontsize=5.8, color="white")
a.set_xticks(range(len(hours)))
a.set_xticklabels(["1 hour", "2 hours", "4 hours"])
a.set_xlabel("Storage of the battery")
a.set_ylim(0, 80)
a.set_yticks([0, 20, 40, 60, 80])
a.set_ylabel("Share of the possible peak cut (%)")
a.grid(axis="y")
a.tick_params(axis="x", length=0)
a.legend(loc="upper left", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
title(a, "A battery that plans on the forecast")

# B: heat-wave days against other days
seasons = ["summer 2023", "summer 2025", "all"]
d = ev.reindex(seasons)
x = np.arange(len(seasons))
b.bar(x - 0.19, d["captured_other_pct"], width=0.36, color=MID_GREY)
b.bar(x + 0.19, d["captured_event_pct"], width=0.36, color=HEAT)
for xx, v, w in zip(x, d["captured_other_pct"], d["captured_event_pct"], strict=True):
    b.text(xx - 0.19, 2.0, f"{v:.0f}", ha="center", fontsize=5.8, color=CHARCOAL)
    b.text(xx + 0.19, 2.0, f"{w:.0f}", ha="center", fontsize=5.8, color=CHARCOAL)
b.set_xticks(x)
b.set_xticklabels(["2023", "2025", "all summers"])
b.set_ylim(0, 80)
b.set_yticks([0, 20, 40, 60, 80])
b.tick_params(axis="y", labelleft=False)
b.tick_params(axis="x", length=0)
b.grid(axis="y")
b.legend(
    handles=[
        Patch(color=MID_GREY, label="other days, same month"),
        Patch(color=HEAT, label="heat-wave days"),
    ],
    loc="upper left",
    fontsize=5.8,
    handlelength=1.0,
    borderaxespad=0.1,
)
b.set_xlabel("Summer")
title(b, "Heat-wave days")

# C: error against value on heat-wave days
w = ev.loc["all"]
rows = [
    ("error", w["nmae_diff_pts"], w["nmae_diff_lo"], w["nmae_diff_hi"], MODEL),
    ("peak cut", w["captured_diff_pts"], w["captured_diff_lo"], w["captured_diff_hi"], HEAT),
]
c.axvline(0, color=ASH, linewidth=0.5)
for y, (_label, v, lo, hi, colour) in zip([1.0, 0.0], rows, strict=True):
    c.plot([lo, hi], [y, y], color=colour, linewidth=2.6, solid_capstyle="butt")
    c.plot(v, y, "o", ms=4.0, color=colour, mec="white", mew=0.5)
    sign = "+" if v > 0 else "−"
    c.annotate(
        f"{sign}{abs(v):.1f} points",
        (v, y),
        xytext=(-3, 7) if v > 0 else (0, 7),
        textcoords="offset points",
        ha="right" if v > 0 else "center",
        fontsize=5.9,
        color=CHARCOAL,
    )
c.set_yticks([1.0, 0.0])
c.set_yticklabels([r[0] for r in rows])
c.set_ylim(-0.6, 1.6)
c.set_xlim(-32, 6)
c.set_xticks([-30, -20, -10, 0])
c.set_xlabel("Difference (points)")
c.tick_params(axis="y", length=0)
c.grid(axis="x")
title(c, "Heat-wave days minus other days")

save(fig, "value", data=d.reset_index())
print(
    dec[["model", "battery_hours", "captured_point_pct", "lo", "hi"]]
    .round(1)
    .to_string(index=False)
)
cols = [
    "event_site_days",
    "captured_event_pct",
    "captured_other_pct",
    "captured_diff_pts",
    "captured_diff_lo",
    "captured_diff_hi",
    "nmae_diff_pts",
    "nmae_diff_lo",
    "nmae_diff_hi",
]
print(d[cols].round(1).to_string())
