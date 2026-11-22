"""Slide 3, question 1 (can we trust the data?), three panels.

A: the metered electricity of one year against the estates return (ERIC 2023/24, April 2023
   to March 2024), at the sites with at least 80 % of the year's half-hours and a return to
   compare. The 65 verified sites sit on the line.
B: the sites with a plant in the return of 2024/25 (plausible capacity factor): grid import
   (what the meter shows) and plant output net of exports, which add up to the site's demand.
C: the 134 sites by data check (estates/coverage_flags.csv).

analysis/estates (coverage_flags.csv), analysis/chp.py (generation_2024_25.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    BLUE,
    CHARCOAL,
    HEIGHT,
    LIGHT_BLUE,
    MID_GREY,
    PLANT,
    PLANT_EDGE,
    REPORTS,
    STONE,
    WIDTH,
    figure,
    save,
    title,
)
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

f = pd.read_csv(REPORTS / "estates" / "coverage_flags.csv")
g = pd.read_csv(REPORTS / "chp" / "generation_2024_25.csv")
g = g[g["chp_elec_output_kwh"].notna() & (g["implied_mean_kw"] > 0) & g["plausible"]].copy()
g = g.sort_values("implied_mean_kw").reset_index(drop=True)

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [1.0, 1.05, 0.62]})

# A: meter against the return
kinds = {
    "ok": ("verified", BLUE, BLUE, 7),
    "chp_mismatch": ("a plant on site", PLANT, PLANT_EDGE, 9),
    "partial_supply": ("meter supplies a part or more", "white", CHARCOAL, 9),
    "aggregate_supply": ("meter supplies a part or more", "white", CHARCOAL, 9),
}
d = f[f["coverage_flag"].isin(kinds) & (f["meter_coverage_fy"] >= 0.8)].copy()
lim = (1e4, 3e7)
a.plot(lim, lim, color=ASH, linewidth=0.5, linestyle=(0, (3, 2)), zorder=1)
for key, (_label, face, edge, size) in kinds.items():
    x = d[d["coverage_flag"] == key]
    a.scatter(
        x["elec_kwh"] / 1000,
        x["meter_kwh_fy"] / 1000,
        s=size,
        color=face,
        edgecolor=edge,
        linewidth=0.5,
        zorder=3,
    )
a.set_xscale("log")
a.set_yscale("log")
a.set_xlim(lim[0] / 1000, lim[1] / 1000)
a.set_ylim(lim[0] / 1000, lim[1] / 1000)
a.set_xlabel("Estates return (MWh)")
a.set_ylabel("Meter (MWh)")
a.set_aspect("equal")
for axis in (a.xaxis, a.yaxis):
    axis.set_major_locator(FixedLocator([10, 100, 1000, 10000]))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
title(a, "Meter against the estates return")
handles = [
    Line2D([], [], marker="o", ls="none", ms=2.8, color=BLUE, label="verified"),
    Line2D(
        [], [], marker="o", ls="none", ms=3.0, mfc=PLANT, mec=PLANT_EDGE, mew=0.5, label="plant"
    ),
    Line2D(
        [],
        [],
        marker="o",
        ls="none",
        ms=3.0,
        mfc="white",
        mec=CHARCOAL,
        mew=0.5,
        label="other supply",
    ),
]
a.legend(handles=handles, loc="upper left", fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)

# B: what the meter shows at the sites with a plant
y = np.arange(len(g))
imp = g["eric_import_mean_kw"] / 1000
out = (g["chp_elec_output_kwh"] - g["exported_elec_kwh"].fillna(0)) / 8760 / 1000
b.barh(y, imp, height=0.72, color=BLUE, label="grid import: what the meter shows")
b.barh(
    y,
    out,
    left=imp,
    height=0.72,
    color=PLANT,
    edgecolor=PLANT_EDGE,
    linewidth=0.3,
    label="plant output",
)
b.set_yticks([])
b.set_ylim(-0.7, len(g) - 0.3)
b.set_xlabel("Mean power in 2024/25 (MW)")
b.spines["left"].set_visible(False)
b.grid(axis="x")
b.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderaxespad=0.1)
title(b, "Sites with a plant: import and output")
share = imp.sum() / (imp + out).sum()

# C: the 134 sites by data check
counts = f["coverage_flag"].value_counts()
parts = [
    ("verified against the return", counts["ok"], BLUE, BLUE),
    ("no return to compare", counts["no_eric"], LIGHT_BLUE, LIGHT_BLUE),
    ("plant: import only", counts["chp_mismatch"], PLANT, PLANT_EDGE),
    ("meter gaps", counts["meter_gaps"], "white", MID_GREY),
    (
        "meter supplies a part or more",
        counts["partial_supply"] + counts["aggregate_supply"],
        STONE,
        STONE,
    ),
]
bottom = 0
for label, n, face, edge in parts:
    c.bar(0, n, bottom=bottom, width=0.62, color=face, edgecolor=edge, linewidth=0.5)
    c.text(0.4, bottom + n / 2, f"{n}  {label}", va="center", fontsize=5.9, color=CHARCOAL)
    bottom += n
complete = counts["ok"] + counts["no_eric"] + counts["chp_mismatch"]
c.plot([-0.42, -0.42], [0, complete], color=CHARCOAL, linewidth=0.6)
c.text(
    -0.52,
    complete / 2,
    f"{complete} complete",
    rotation=90,
    ha="right",
    va="center",
    fontsize=5.9,
    color=CHARCOAL,
)
c.set_xlim(-0.75, 2.6)
c.set_ylim(0, bottom)
c.set_axis_off()
title(c, f"{bottom} sites by data check")

save(
    fig,
    "data_check",
    data=d[["site_code", "coverage_flag", "elec_kwh", "meter_kwh_fy", "energy_ratio"]],
)
ok = d[d["coverage_flag"] == "ok"]["energy_ratio"]
print(
    "verified:",
    len(ok),
    "median ratio",
    round(ok.median(), 3),
    ok.quantile([0.25, 0.75]).round(3).tolist(),
)
print(
    "plant sites:",
    len(g),
    "import share of demand",
    round(share, 3),
    round(imp.sum(), 1),
    round((imp + out).sum(), 1),
)
print("complete:", complete, "of", bottom)
