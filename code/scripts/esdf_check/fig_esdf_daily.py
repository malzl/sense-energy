"""Error of four models on every day of the paper's test period, mean over the series of the
region: the paper's LSTM, the paper's method on the same backbone, a gradient-boosted model and
Chronos-2 with four weeks of context and no training. The shaded days are the window that the
paper scores. Argument: europe or pjm.

experiments/esdf_check.py (daily_errors.csv).
"""

import sys

import pandas as pd
from _common import ESDF, OUT

from sense_energy.visualization import style

region = sys.argv[1] if len(sys.argv) > 1 else "europe"
MODELS = ("LSTM", "ESDF (LSTM)", "LightGBM", "Chronos-2, 4 weeks")
d = pd.read_csv(ESDF / "daily_errors.csv", parse_dates=["target_day"])
d = d[(d["region"] == region) & d["model"].isin(MODELS)]
m = d.groupby(["model", "target_day"])["mae"].mean().mul(100).reset_index()
win = d[d["in_window"]]["target_day"]
fig, ax = style.figure(style.WIDE_SHORT)
ax.axvspan(win.min(), win.max(), color=style.ROLE_COLORS["zero_effect"], alpha=0.35, linewidth=0)
for model in MODELS:
    x = m[m["model"] == model].sort_values("target_day")
    ax.plot(
        x["target_day"],
        x["mae"],
        color=style.ESDF_METHOD_COLORS[model],
        linewidth=style.LINEWIDTH["secondary"],
        label=model,
    )
ax.set_ylim(0, None)
ax.set_xlabel("Target day")
ax.set_ylabel("Mean absolute error (% of range)")
fig.autofmt_xdate(rotation=0, ha="center")
style.style_axis(ax, wide_short=True)
style.add_bottom_legend(ax, lowercase=False, anchor_y=-0.32, bottom=0.36)
style.save_figure(fig, OUT / f"esdf_daily_{region}", data=m)
