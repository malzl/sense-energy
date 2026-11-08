"""The opportunities the DEC assessors wrote down: recommendations in the advisory reports linked
to the latest certificate of each building at the site postcodes, by theme (keyword classes; a
recommendation can carry more than one) and by payback class.

data/dec.py (interim/dec_recommendations.parquet).
"""

import numpy as np
import pandas as pd
from _common import OUT

from sense_energy.config import INTERIM_DIR
from sense_energy.data.dec import PAYBACKS, RECOMMENDATION_THEMES
from sense_energy.visualization import style

r = pd.read_parquet(INTERIM_DIR / "dec_recommendations.parquet")
rows = []
for theme in RECOMMENDATION_THEMES:
    hit = r["themes"].str.split(";").apply(lambda t, th=theme: th in t)
    for payback in PAYBACKS:
        rows.append(
            {"theme": theme, "payback": payback, "n": int((hit & (r["payback"] == payback)).sum())}
        )
rows.append({"theme": "other", "payback": "short", "n": 0})
t = pd.DataFrame(rows).pivot(index="theme", columns="payback", values="n").fillna(0)
other = r[r["themes"] == ""]
for payback in PAYBACKS:
    t.loc["other", payback] = int((other["payback"] == payback).sum())
t = t.reindex(columns=list(PAYBACKS)).loc[t.sum(axis=1).sort_values().index]
y = np.arange(len(t))
colors = {"short": "#009E73", "medium": "#E69F00", "long": "#D55E00", "other": "#7A7A7A"}

fig, ax = style.figure(style.SINGLE_COLUMN)
left = np.zeros(len(t))
for payback in PAYBACKS:
    ax.barh(
        y,
        t[payback],
        left=left,
        color=colors[payback],
        height=0.7,
        label=f"{payback} payback",
        zorder=3,
    )
    left += t[payback].to_numpy()
ax.set_yticks(y)
ax.set_yticklabels(t.index)
ax.set_xlabel(f"Recommendations in {r['advisory_number'].nunique()} advisory reports")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.3)
print(t.assign(total=t.sum(axis=1)).astype(int).to_string())
style.save_figure(fig, OUT / "dec_recommendations", data=t.reset_index())
