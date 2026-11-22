"""Slide 6, question 3 (why do sites differ?), three panels.

A, B: each site's reaction to heat waves (A) and cold spells (B), tested against its own
      placebo (2,000 random blocks of its ordinary days in the same season) and adjusted for
      the number of sites (false discovery rate 5 %). A filled circle is a clear change; its
      area is the reaction after shrinkage towards the mean of all sites. Sites with complete
      meter data and no on-site generation (64 and 75); no site is named.
C: how much of the difference between sites each source explains: the NHS region (above the
   share that chance gives), the best single field of the energy certificate, the gas to
   electricity ratio of the estates return, and the same site's reaction in its other events.

analysis/event_audit.py (site_effect_tests.csv, grid_summary.csv), analysis/extreme_events.py
(alignment_fits.csv, site_reactions.csv for the coordinates).
"""

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
    SHELL,
    TEXT_ORANGE,
    TEXT_TEAL,
    WIDTH,
    england_map,
    figure,
    save,
    title,
)
from matplotlib.lines import Line2D

t = pd.read_csv(REPORTS / "event_audit" / "site_effect_tests.csv")
t = t[t["chp"] == "no on-site generation"]
geo = pd.read_csv(REPORTS / "extreme_events" / "site_reactions.csv")
geo = geo[["site_code", "latitude", "longitude"]].drop_duplicates("site_code")
t = t.merge(geo, on="site_code", how="left").dropna(subset=["latitude"])
grid = pd.read_csv(REPORTS / "event_audit" / "grid_summary.csv").set_index(["metric", "type"])
fits = pd.read_csv(REPORTS / "extreme_events" / "alignment_fits.csv")


def size(pct):
    return 3.0 + 2.0 * np.clip(pct, 0, 30)


fig, (a, b, c) = figure(WIDTH, HEIGHT, ncols=3, gridspec_kw={"width_ratios": [0.82, 0.82, 1.15]})
out = []
for ax, kind, colour in ((a, "heatwave", HEAT), (b, "cold spell", COLD)):
    england_map(ax, face=SHELL)
    d = t[t["type"] == kind]
    clear = d["q"] < 0.05
    rise, fall = d[clear & (d["effect_pct"] > 0)], d[clear & (d["effect_pct"] < 0)]
    weak = d[~clear]
    ax.scatter(
        weak["longitude"],
        weak["latitude"],
        s=3.2,
        facecolor="none",
        edgecolor=MID_GREY,
        linewidth=0.4,
        zorder=3,
    )
    rise = rise.sort_values("shrunk_pct", ascending=False)
    ax.scatter(
        rise["longitude"],
        rise["latitude"],
        s=size(rise["shrunk_pct"]),
        color=colour,
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
        zorder=4,
    )
    ax.scatter(
        fall["longitude"],
        fall["latitude"],
        s=size(-fall["shrunk_pct"]),
        color=TEXT_TEAL,
        edgecolor="white",
        linewidth=0.3,
        zorder=5,
    )
    name = "Heat waves" if kind == "heatwave" else "Cold spells"
    falls = "1 falls" if len(fall) == 1 else f"{len(fall)} fall"
    title(ax, f"{name}: {len(rise)} rise, {falls}", pad=2)
    out.append(d.assign(clear=clear))

# C: what explains the differences between sites
cert = fits[fits["descriptor"].str.startswith("certificate")].groupby("type")["adj_r2_gain"].max()
gas = fits[fits["descriptor"] == "estates return: gas over electricity"].set_index("type")
rows = []
for kind in ("heatwave", "cold spell"):
    rows += [
        (
            kind,
            "region, above chance",
            max(grid.loc[("region share minus chance", kind), "baseline"], 0.0),
        ),
        (kind, "energy certificate", 100 * max(cert[kind], 0.0)),
        (kind, "gas share", 100 * max(gas.loc[kind, "adj_r2_gain"], 0.0)),
        (kind, "own past events", grid.loc[("other events r2", kind), "baseline"]),
    ]
e = pd.DataFrame(rows, columns=["type", "source", "share_pct"])
order = ["region, above chance", "energy certificate", "gas share", "own past events"]
for j, (kind, colour, ink) in enumerate(
    (("heatwave", HEAT, TEXT_ORANGE), ("cold spell", COLD, COLD))
):
    x = e[e["type"] == kind].set_index("source").reindex(order)
    pos = np.arange(len(order))[::-1] + (0.5 - j) * 0.36
    c.barh(pos, x["share_pct"], height=0.34, color=colour)
    for p, v in zip(pos, x["share_pct"], strict=True):
        c.text(v + 1.2, p, f"{v:.0f} %", va="center", fontsize=5.8, color=ink)
c.set_yticks(np.arange(len(order))[::-1])
c.set_yticklabels(order)
c.set_xlim(0, 80)
c.set_xticks([0, 20, 40, 60, 80])
c.set_xlabel("Share of the difference between sites (%)")
c.tick_params(axis="y", length=0)
c.grid(axis="x")
title(c, "What explains the difference")

handles = [
    Line2D(
        [], [], marker="o", ls="none", ms=np.sqrt(size(5)), color=ASH, mec="white", label="+5 %"
    ),
    Line2D(
        [], [], marker="o", ls="none", ms=np.sqrt(size(20)), color=ASH, mec="white", label="+20 %"
    ),
    Line2D([], [], marker="o", ls="none", ms=2.8, color=TEXT_TEAL, label="less demand"),
    Line2D(
        [],
        [],
        marker="o",
        ls="none",
        ms=1.9,
        mfc="none",
        mec=MID_GREY,
        mew=0.4,
        label="no clear change",
    ),
    Line2D([], [], marker="s", ls="none", ms=3.2, color=HEAT, label="heat waves"),
    Line2D([], [], marker="s", ls="none", ms=3.2, color=COLD, label="cold spells"),
]
fig.legend(
    handles=handles,
    loc="outside lower center",
    ncol=6,
    fontsize=6.0,
    columnspacing=1.6,
    handletextpad=0.3,
)
save(fig, "sites", data=pd.concat(out, ignore_index=True))
print(e.pivot(index="source", columns="type", values="share_pct").round(1).to_string())
for d in out:
    k = d["type"].iloc[0]
    print(
        k,
        len(d),
        "sites; clear rise",
        int((d.clear & (d.effect_pct > 0)).sum()),
        "fall",
        int((d.clear & (d.effect_pct < 0)).sum()),
        "max shrunk",
        round(d.shrunk_pct.max(), 1),
    )
_ = CHARCOAL
