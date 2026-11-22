"""Slide 7, question 4 (how good is the forecast?), two panels.

A: day-ahead error in each quarter from January 2024 to May 2026 (2026 Q2: to 8 May): the
   zero-shot model (Chronos-2 with calendar, price and the weather forecast of the day before),
   the gradient-boosted model (LightGBM, the same inputs) trained once on 2023 and trained again
   at the start of each quarter, and the same half-hour one week before. The same site-days
   for every model, 91 sites; error is the site's absolute error over its demand, mean over
   sites.
B: error of the zero-shot model without inputs against the length of its history (the 69
   test days of the proof of concept, June 2025 to May 2026). Six weeks are the standard.

analysis/forecast_extensions.py (retrain_quarters.csv), forecast_probe.run_context_long
(context_long.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    CHARCOAL,
    HEIGHT,
    MODEL,
    NAIVE,
    REPORTS,
    TEAL,
    TRAINED,
    WIDTH,
    figure,
    legend_below,
    save,
    title,
)

q = pd.read_csv(REPORTS / "forecast_extensions" / "retrain_quarters.csv")
h = pd.read_csv(REPORTS / "forecast_anatomy" / "context_long.csv")
quarters = sorted(q["quarter"].unique())
SERIES = (
    ("seasonal_naive", "same half-hour last week", NAIVE, (0, (1.2, 1.2)), "x", 3.0),
    ("lightgbm_once", "trained model, trained once", TEAL, (0, (3, 1.5)), "s", 2.6),
    ("lightgbm_quarterly", "trained model, trained each quarter", TRAINED, "-", "D", 2.6),
    ("chronos2_ifs", "model with no training", MODEL, "-", "o", 3.0),
)

fig, (a, b) = figure(WIDTH, HEIGHT, ncols=2, gridspec_kw={"width_ratios": [1.75, 1]})
x = np.arange(len(quarters))
for model, label, colour, ls, marker, ms in SERIES:
    d = q[q["model"] == model].set_index("quarter").reindex(quarters)
    a.plot(
        x,
        d["nmae_pct"],
        color=colour,
        linestyle=ls,
        marker=marker,
        ms=ms,
        mew=0.6,
        linewidth=1.3 if model == "chronos2_ifs" else 1.0,
        label=label,
        zorder=4 if model == "chronos2_ifs" else 3,
    )
a.set_xticks(x)
a.set_xticklabels([s.replace(" ", "\n") for s in quarters])
a.set_xlim(-0.4, len(quarters) - 0.6)
a.set_ylim(0, 22)
a.set_yticks([0, 5, 10, 15, 20])
a.set_ylabel("Day-ahead error (%)")
a.grid(axis="y")
a.tick_params(axis="x", length=0)
title(a, "Error in each quarter, 91 sites")
legend_below(fig, a, ncol=2, handlelength=2.2)

days = h["history_days"].to_numpy()
xb = np.arange(len(days))
b.plot(xb, h["nmae_pct"], color=MODEL, linewidth=1.3, marker="o", ms=3.0, zorder=3)
for xx, v in zip(xb, h["nmae_pct"], strict=True):
    steep = xx < 3
    b.annotate(
        f"{v:.1f}",
        (xx, v),
        xytext=(4, 1) if steep else (0, 4),
        textcoords="offset points",
        ha="left" if steep else "center",
        fontsize=5.6,
        color=CHARCOAL,
    )
six = int(np.flatnonzero(days == 42)[0])
b.axvline(six, color=ASH, linewidth=0.5, linestyle=(0, (2, 2)))
b.set_xticks(xb)
b.set_xticklabels([f"{d:.0f}" for d in days])
b.set_xlim(-0.5, len(days) - 0.5)
b.set_ylim(0, 36)
b.set_yticks([0, 10, 20, 30])
b.tick_params(axis="x", length=0)
b.set_xlabel("History given to the model (days)")
b.set_ylabel("Day-ahead error (%)")
b.grid(axis="y")
title(b, "How much history the model needs")
save(fig, "accuracy", data=q)
print(q.pivot(index="quarter", columns="model", values="nmae_pct").round(2).to_string())
print(h[["history_days", "nmae_pct"]].round(2).to_string(index=False))
