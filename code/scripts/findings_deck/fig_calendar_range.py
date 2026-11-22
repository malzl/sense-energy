"""Slide 10, question 4, two systematic faults of the forecast and their corrections.

A: day-ahead error on all days, normal days, special days (bank holidays and the Christmas
   period, also shown apart) and proximity days (the working day before and after a bank
   holiday): the forecast, the forecast with the rule for special days (Arora and Taylor) on
   bank holidays only, and the rule on all special and proximity days. A day outside the
   policy keeps the forecast. Whole-record forecasts with weather and calendar inputs, 2024 to
   May 2026, 91 sites.
B: where the 80 % range of the forecast misses: the share of the half-hours below its 10 %
   quantile (left) and above its 90 % quantile (right), nominal 10 % on each side, for the
   range of the model and for the range with its two bounds tracked online (conformal
   quantile tracking, Angelopoulos et al. 2023, with the delay of operation). Whole record
   after a start of 90 days; mean over sites with the 95 % interval from resampling sites.

analysis/special_days.py (day_classes.csv), analysis/forecast_range.py (coverage.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    CHARCOAL,
    HEIGHT,
    MODEL,
    REPORTS,
    TEAL,
    TEXT_TEAL,
    WIDTH,
    figure,
    save,
    title,
)
from matplotlib.patches import Patch

dc = pd.read_csv(REPORTS / "special_days" / "day_classes.csv")
dc = dc[dc["model"] == "chronos2_ifs"]
cov = pd.read_csv(REPORTS / "forecast_range" / "coverage.csv")
cov = cov[cov["model"] == "chronos2_ifs"]

fig, (a, b) = figure(WIDTH, HEIGHT, ncols=2, gridspec_kw={"width_ratios": [1.35, 1]})

# A: the rule for special days
classes = [
    "all days",
    "normal days",
    "special days",
    "bank holidays",
    "Christmas period",
    "proximity days",
]
labels = [
    "all\ndays",
    "normal\ndays",
    "special\ndays",
    "bank\nholidays",
    "Christmas\nperiod",
    "proximity\ndays",
]
pol = dc.set_index(["policy", "day_class"])
series = (
    ("forecast", pol.loc["rule on bank holidays"]["nmae_forecast_pct"], MODEL, None, MODEL),
    (
        "rule on bank holidays",
        pol.loc["rule on bank holidays"]["nmae_with_rule_pct"],
        TEAL,
        None,
        TEAL,
    ),
    (
        "rule on all special and proximity days",
        pol.loc["rule on special and proximity days"]["nmae_with_rule_pct"],
        "white",
        "/////",
        TEXT_TEAL,
    ),
)
x = np.arange(len(classes))
for j, (label, v, face, hatch, edge) in enumerate(series):
    v = v.reindex(classes)
    a.bar(
        x + (j - 1) * 0.27,
        v,
        width=0.25,
        color=face,
        hatch=hatch,
        edgecolor=edge,
        linewidth=0.5,
        label=label,
    )
    if label != "rule on all special and proximity days":
        for xx, vv in zip(x + (j - 1) * 0.27, v, strict=True):
            a.text(xx, vv + 0.5, f"{vv:.0f}", ha="center", fontsize=5.5, color=CHARCOAL)
a.axvspan(2.5, 4.5, color="#F1EEE9", zorder=0, linewidth=0)
a.set_xticks(x)
a.set_xticklabels(labels)
a.set_ylim(0, 27.5)
a.set_yticks([0, 5, 10, 15, 20, 25])
a.set_ylabel("Day-ahead error (%)")
a.tick_params(axis="x", length=0)
a.grid(axis="y")
a.text(3.5, 25.6, "parts of the special days", ha="center", fontsize=5.7, color=ASH)
title(a, "Special days: a rule from the same day of the last year")

# B: where the range misses
groups = [
    ("all", "all days"),
    ("heatwave day", "heat-wave days"),
    ("cold spell day", "cold-spell days"),
    ("bank holiday", "bank holidays"),
]
methods = (
    ("model", "range of the model", MODEL, 0.36),
    ("quantile tracking", "range tracked online", TEAL, 0.2),
)
y = np.arange(len(groups))[::-1]
for method, label, colour, height in methods:
    d = cov[cov["method"] == method].set_index("group").reindex([g for g, _ in groups])
    b.barh(y, -d["below_pct"], height=height, color=colour, label=label)
    b.barh(y, d["above_pct"], height=height, color=colour)
    if method == "model":
        for yy, lo, hi in zip(y, d["below_pct"], d["above_pct"], strict=True):
            b.text(
                -lo - 0.8, yy, f"{lo:.0f}", ha="right", va="center", fontsize=5.6, color=CHARCOAL
            )
            b.text(hi + 0.8, yy, f"{hi:.0f}", ha="left", va="center", fontsize=5.6, color=CHARCOAL)
for v in (-10, 10):
    b.axvline(v, color=ASH, linewidth=0.6, linestyle=(0, (2, 2)))
b.axvline(0, color=ASH, linewidth=0.5)
b.set_yticks(y)
b.set_yticklabels([n for _, n in groups])
b.set_xlim(-28, 22)
b.set_xticks([-20, -10, 0, 10, 20])
b.set_xticklabels(["20", "10", "0", "10", "20"])
b.set_xlabel("Half-hours outside the range (%)")
b.text(-14, len(groups) - 0.35, "below", ha="center", fontsize=6.0, color=CHARCOAL)
b.text(14, len(groups) - 0.35, "above", ha="center", fontsize=6.0, color=CHARCOAL)
b.set_ylim(-0.6, len(groups) - 0.1)
b.tick_params(axis="y", length=0)
title(b, "Where the 80 % range misses")
fig.legend(
    handles=[
        Patch(color=MODEL, label="forecast"),
        Patch(color=TEAL, label="with the correction"),
        Patch(
            facecolor="white",
            edgecolor=TEXT_TEAL,
            hatch="/////",
            linewidth=0.5,
            label="rule also on the Christmas period and proximity days",
        ),
    ],
    loc="outside lower center",
    ncol=3,
    fontsize=6.0,
    handlelength=1.2,
    columnspacing=1.4,
)
save(fig, "calendar_range", data=dc)
print(
    dc.pivot(index="day_class", columns="policy", values="nmae_with_rule_pct").round(2).to_string()
)
