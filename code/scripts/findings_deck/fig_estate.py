"""Slide 11, question 5 (what changes when the estate changes?), three panels.

A: scenario. Gas-heated English NHS sites without on-site generation (estates return
   2023/24, 631 sites): the peak demand with heat pumps for all gas heat against the capacity
   of the grid connection (seasonal performance factor 2.55, boiler efficiency 0.85).
B: scenario. The 31 panel sites with gas data that are above their connection with heat
   pumps, in the weather of each year since 1960.
C: field data. Change in electricity and gas from 2023/24 to 2025/26 at the 23 sites that
   installed heat pumps in 2024/25, against the sites that did not (difference in
   differences, estates returns, 95 % interval).

analysis/estates (heat_pump_headroom_england.csv), analysis/design_winter.py (years.csv),
analysis/heat_pump_field (did.csv).
"""

import numpy as np
import pandas as pd
from _fd import (
    ASH,
    BLUE,
    CHARCOAL,
    HEAT,
    HEIGHT,
    LIGHT_BLUE,
    MID_GREY,
    PLANT,
    PLANT_EDGE,
    REPORTS,
    SHELL,
    TEXT_ORANGE,
    WIDTH,
    england_map,
    figure,
    save,
    title,
)
from matplotlib.lines import Line2D

from sense_energy.analysis.design_winter import RECENT

full = pd.read_csv(REPORTS / "estates" / "heat_pump_headroom_england.csv")
e = full.dropna(subset=["latitude"]).copy()
e["band"] = np.select([e["util_hp"] > 1.0, e["util_hp"] > 0.8], ["above", "near"], "below")
y = pd.read_csv(REPORTS / "design_winter" / "years.csv")
did = pd.read_csv(REPORTS / "heat_pump_field" / "did.csv")
did = did[(did["pair"] == "2023/24 to 2025/26, installed 2024/25") & (did["status"] == "new")].iloc[
    0
]

fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [0.9, 1.15, 0.85]})

# A: the map
england_map(a, face=SHELL)
bands = {
    "below": ("below 80 %", LIGHT_BLUE, "white"),
    "near": ("80 to 100 %", PLANT, PLANT_EDGE),
    "above": ("above 100 %", HEAT, "white"),
}
for key, (_label, face, edge) in bands.items():
    d = e[e["band"] == key]
    a.scatter(
        d["longitude"],
        d["latitude"],
        s=4.0,
        color=face,
        edgecolor=edge,
        linewidth=0.2,
        zorder=4 if key == "above" else 3,
    )
fig.legend(
    handles=[
        Line2D([], [], marker="o", ls="none", ms=3.0, mfc=f, mec=ed, mew=0.3, label=lab)
        for lab, f, ed in (
            ("heat-pump peak below 80 % of the connection", LIGHT_BLUE, "white"),
            ("80 to 100 %", PLANT, PLANT_EDGE),
            ("above the connection", HEAT, "white"),
        )
    ],
    loc="outside lower center",
    ncol=3,
    fontsize=6.0,
    handletextpad=0.2,
    columnspacing=1.6,
)
share = (full["util_hp"] > 1).mean()
title(a, f"{len(full)} sites in England", pad=2)

# B: the weather years
last = int(y["weather_year"].max())
recent = y["weather_year"] > last - RECENT
b.bar(
    y.loc[~recent, "weather_year"],
    y.loc[~recent, "hp_sites_above_capacity"],
    width=0.8,
    color=MID_GREY,
    label="1960 to 1995",
)
b.bar(
    y.loc[recent, "weather_year"],
    y.loc[recent, "hp_sites_above_capacity"],
    width=0.8,
    color=BLUE,
    label="last 30 winters",
)
cold = y.loc[y["hp_sites_above_capacity"].idxmax()]
b.annotate(
    f"{int(cold['weather_year'])}/{str(int(cold['weather_year']) + 1)[-2:]}: {int(cold['hp_sites_above_capacity'])}",
    (cold["weather_year"], cold["hp_sites_above_capacity"]),
    xytext=(6, 2),
    textcoords="offset points",
    fontsize=5.8,
    color=CHARCOAL,
)
n = int(y["hp_sites"].max())
b.set_ylim(0, n)
b.set_yticks([0, 10, 20, n])
b.set_xticks([1960, 1980, 2000, 2020])
b.set_xlabel("Weather year")
b.set_ylabel(f"Sites above the capacity (of {n})")
b.grid(axis="y")
b.legend(loc="upper right", fontsize=5.7, handlelength=1.0, borderaxespad=0.1)
title(b, "Panel sites in 66 winters")

# C: field data
rows = [
    ("electricity", did["elec_did_pct"], did["elec_did_lo"], did["elec_did_hi"], BLUE),
    ("gas", did["gas_did_pct"], did["gas_did_lo"], did["gas_did_hi"], TEXT_ORANGE),
]
c.axhline(0, color=ASH, linewidth=0.5)
for x, (_label, v, lo, hi, colour) in enumerate(rows):
    c.bar(x, v, width=0.55, color=colour)
    c.plot([x, x], [lo, hi], color=CHARCOAL, linewidth=0.6)
    c.annotate(
        f"{v:+.0f}".replace("-", "−"),
        (x, hi if v > 0 else lo),
        xytext=(0, 3 if v > 0 else -9),
        textcoords="offset points",
        ha="center",
        fontsize=5.9,
        color=CHARCOAL,
    )
c.set_xticks([0, 1])
c.set_xticklabels([r[0] for r in rows])
c.set_xlim(-0.6, 1.6)
c.set_ylim(-26, 20)
c.set_yticks([-20, -10, 0, 10, 20])
c.set_ylabel("Change against other sites (points)")
c.tick_params(axis="x", length=0)
c.grid(axis="y")
title(c, f"{int(did['sites'])} sites with new heat pumps")

save(fig, "estate", data=y[["weather_year", "hp_sites_above_capacity", "hp_sites"]])
print(e["band"].value_counts().to_dict(), "above share", round(share, 3))
recent_y = y[recent]["hp_sites_above_capacity"]
print(
    "last 30 winters:",
    recent_y.min(),
    "to",
    recent_y.max(),
    "; max year",
    int(cold["weather_year"]),
    int(cold["hp_sites_above_capacity"]),
)
print(
    did[
        [
            "sites",
            "elec_did_pct",
            "elec_did_lo",
            "elec_did_hi",
            "gas_did_pct",
            "gas_did_lo",
            "gas_did_hi",
        ]
    ].to_dict()
)
