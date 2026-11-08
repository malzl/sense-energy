"""The JEPA's surprise as a regime detector: the latent prediction error of each day when it is
the one held out of its window, for three sites (a CHP hospital, an acute hospital without CHP,
and the small site with the largest LED-related step), with the PELT steps of the demand series
marked.

experiments/jepa.py (surprise.parquet), analysis/estates_changes.py (demand_steps.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

sp = pd.read_parquet(PROJECT_ROOT / "code" / "reports" / "jepa" / "surprise.parquet")
steps = pd.read_csv(
    PROJECT_ROOT / "code" / "reports" / "estates" / "demand_steps.csv", parse_dates=["date"]
)
sites = pd.read_parquet(PROJECT_ROOT / "code" / "data" / "interim" / "sites.parquet").set_index(
    "site_code"
)
picks = [s for s in ("RXN02", "R0A66", "RRECO") if s in set(sp["site_code"])][:3]
names = {s: str(sites.loc[s, "site_name"]).title()[:28] for s in picks}
colors = [style.CHP_COLORS["on-site CHP"], style.CHP_GROUP_COLORS["acute, no CHP"], "#7A7A7A"]

fig, ax = style.figure(style.WIDE_SHORT)
for k, s in enumerate(picks):
    d = sp[sp["site_code"] == s].sort_values("date")
    r = d.set_index("date")["surprise"].rolling(7, center=True, min_periods=3).mean()
    ax.plot(r.index, r.to_numpy(), color=colors[k], linewidth=0.9, label=names[s], zorder=3)
    for t in steps.loc[steps["site_code"] == s, "date"]:
        ax.axvline(t, color=colors[k], linewidth=0.5, alpha=0.5, linestyle=(0, (2, 2)), zorder=2)
ax.set_ylabel("Latent surprise (7-day mean)")
ax.set_ylim(0, float(np.nanpercentile(sp["surprise"], 99.5)))
style.style_axis(ax, wide_short=True, grid="y")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.26, bottom=0.26)
style.save_figure(fig, OUT / "jepa_surprise", data=sp[sp["site_code"].isin(picks)])
