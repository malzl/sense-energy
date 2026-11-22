"""Slide 5, question 2, how sure: three checks of the weather effects.

A: the change in demand in cold spells and heat waves under each of 90 sets of analysis
   choices (reference weeks, event definition, site sample, holidays in or out), in order of
   size; the large point is the main analysis with its 95 % interval; the grey band is the
   range of 12 placebo runs (the events moved in time).
B: the extra reaction from the third day of an event on, when the model has the temperature
   of the day only, and when it also has the three days before (95 % intervals clustered by
   site and date). With the four days nothing is left: an event is no more than its
   temperatures.
C: the lower bound of the effect when a hidden factor can change the odds that a day is an
   event day by up to a factor Gamma (generalized marginal sensitivity model, Frauen et al.
   2023); band: 95 % interval from resampling sites. Labels: where the bound reaches zero.

analysis/event_audit.py (specification_grid.csv, placebo.csv, lag_models.csv),
analysis/causal_sensitivity.py (bounds.csv, summary.json).
"""

import json

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    CHARCOAL,
    COLD,
    HEAT,
    HEIGHT,
    MID_GREY,
    REPORTS,
    STONE,
    TEXT_ORANGE,
    WIDTH,
    figure,
    save,
    title,
)
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

EA = REPORTS / "event_audit"
grid = pd.read_csv(EA / "specification_grid.csv")
grid = grid[grid["metric"] == "reaction, no plant"]
plac = pd.read_csv(EA / "placebo.csv")
plac = plac[plac["metric"] == "reaction, no plant"]
lag = pd.read_csv(EA / "lag_models.csv")
lag = lag[lag["event_terms"] & (lag["term"] == "late")]
CS = REPORTS / "causal_sensitivity"
bounds = pd.read_csv(CS / "bounds.csv")
breakdown = json.loads((CS / "summary.json").read_text())
KINDS = (
    ("cold spell", "cold", COLD, COLD, "cold spells"),
    ("heatwave", "heat", HEAT, TEXT_ORANGE, "heat waves"),
)

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [1.45, 0.8, 1.0]})

# A: specification curve
offset, rows = 0, []
a.axhline(0, color=ASH, linewidth=0.5)
for kind, _, colour, ink, label in KINDS:
    d = grid[grid["type"] == kind].sort_values("value").reset_index(drop=True)
    x = offset + np.arange(len(d))
    q = plac[plac["type"] == kind]["value"]
    a.fill_between([x[0] - 2, x[-1] + 2], q.min(), q.max(), color=STONE, alpha=0.8, linewidth=0)
    a.scatter(x, d["value"], s=3.2, color=colour, alpha=0.75, linewidth=0, zorder=3)
    base = d[d["baseline"]].iloc[0]
    xb = offset + int(d.index[d["baseline"]][0])
    a.plot([xb, xb], [base["lo"], base["hi"]], color=CHARCOAL, linewidth=0.8, zorder=4)
    a.plot(xb, base["value"], "o", ms=4.2, color=colour, mec=CHARCOAL, mew=0.6, zorder=5)
    a.text(offset + len(d) / 2, 18.6, label, ha="center", fontsize=6.0, color=ink)
    rows.append(d.assign(x=x))
    offset += len(d) + 14
a.set_xlim(-6, offset - 8)
a.set_ylim(-4, 20)
a.set_yticks([0, 5, 10, 15, 20])
a.set_xticks([])
a.set_xlabel("Sets of choices, in order of size")
a.set_ylabel("Change in demand (%)")
a.grid(axis="y")
title(a, "90 sets of analysis choices")
a.legend(
    handles=[
        Line2D([], [], marker="o", ls="none", ms=1.8, color=ASH, label="one set of choices"),
        Line2D(
            [],
            [],
            marker="o",
            ls="-",
            lw=0.8,
            ms=3.2,
            color=CHARCOAL,
            mfc=MID_GREY,
            label="main analysis",
        ),
        Patch(color=STONE, label="placebo events"),
    ],
    loc="upper right",
    bbox_to_anchor=(1.0, 0.9),
    fontsize=5.7,
    handlelength=1.0,
    handletextpad=0.4,
    borderaxespad=0.1,
)

# B: the third day of an event, beyond its temperatures
models = ["the day's temperature", "the day and three days before"]
b.axhline(0, color=ASH, linewidth=0.5)
for j, (_, season, colour, _, label) in enumerate(KINDS):
    d = lag[lag["season"] == season].set_index("model").reindex(models)
    x = np.arange(len(models)) + (j - 0.5) * 0.26
    b.errorbar(
        x,
        d["coef"],
        yerr=[d["coef"] - d["lo"], d["hi"] - d["coef"]],
        fmt="o",
        ms=3.8,
        color=colour,
        mec="white",
        mew=0.4,
        elinewidth=1.4,
        capsize=0,
        label=label,
    )
b.text(-0.17, 3.7, "cold", ha="right", fontsize=5.8, color=COLD)
b.text(0.17, 4.45, "heat", ha="left", fontsize=5.8, color=TEXT_ORANGE)
b.set_xticks(range(len(models)))
b.set_xticklabels(["the day\nonly", "the day and\n3 days before"])
b.set_xlim(-0.55, 1.55)
b.set_ylim(-2, 5)
b.set_ylabel("Extra change from day 3 (points)")
b.tick_params(axis="x", length=0)
b.grid(axis="y")
title(b, "What is left of day 3")

# C: hidden confounding
c.axhline(0, color=ASH, linewidth=0.5)
for kind, _, colour, ink, label in KINDS:
    d = bounds[(bounds["type"] == kind) & (bounds["gamma"] <= 5)]
    c.fill_between(d["gamma"], d["ci_lo"], d["effect_lo"], color=colour, alpha=0.16, linewidth=0)
    c.plot(d["gamma"], d["effect_lo"], color=colour, linewidth=1.3, label=label)
    g0 = breakdown[kind]["gamma_point"]
    c.plot(g0, 0, "o", ms=3.4, color="white", mec=colour, mew=1.0, zorder=4)
    c.annotate(
        f"Γ = {g0:.1f}",
        (g0, 0),
        xytext=(-3, -9) if kind == "heatwave" else (3, 5),
        textcoords="offset points",
        ha="right" if kind == "heatwave" else "left",
        fontsize=5.9,
        color=ink,
    )
c.set_xscale("log")
c.set_xlim(1, 5)
c.set_xticks([1, 1.5, 2, 3, 5])
c.set_xticklabels(["1", "1.5", "2", "3", "5"])
c.minorticks_off()
c.set_ylim(-10, 12)
c.set_yticks([-10, -5, 0, 5, 10])
c.set_xlabel("Strength of a hidden factor (Γ)")
c.set_ylabel("Lowest possible change (%)")
c.grid(axis="y")
title(c, "A hidden factor")

save(fig, "checks", data=pd.concat(rows, ignore_index=True))
print(lag[["season", "model", "coef", "lo", "hi"]].round(2).to_string(index=False))
print({k: (v["gamma_point"], v["gamma_interval"]) for k, v in breakdown.items()})
