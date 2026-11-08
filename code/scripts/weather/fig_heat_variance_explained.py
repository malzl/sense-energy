"""How much of the true between-site variance of the heat slope each descriptor set explains:
the site sunshine term first (tau before and after the term), then random-effects
meta-regressions of the sunshine-controlled heat slopes on descriptor sets.

analysis/heat_drivers.py (variance_explained.csv).
"""

import numpy as np
from _common import OUT, heat_drivers

from sense_energy.visualization import style

v = (
    heat_drivers("variance_explained")
    .set_index("set")
    .reindex(list(style.VARIANCE_SET_ORDER))
    .dropna(subset=["r2_true"])
)
y = np.arange(len(v))[::-1]
share = 100 * v["r2_true"].clip(lower=0)

fig, ax = style.figure(style.SINGLE_COLUMN)
ax.barh(
    y,
    share,
    color=[
        style.TERM_COLORS["solar_kwh_m2"] if s == "site sunshine term" else "#7A7A7A"
        for s in v.index
    ],
    height=0.6,
    zorder=3,
)
for yy, (s, r) in zip(y, v.iterrows(), strict=True):
    ax.annotate(
        f"{100 * r['r2_true']:.0f} %  (n = {int(r['sites'])}, τ {r['tau_before_pct']:.1f} → {r['tau_after_pct']:.1f})",
        xy=(max(share.loc[s], 0) + 1, yy),
        va="center",
        fontsize=style.FONT_SIZE["annotation"] - 1,
    )
ax.set_yticks(y)
ax.set_yticklabels(v.index)
ax.set_xlim(0, max(60, float(share.max()) + 35))
ax.set_xlabel("True between-site variance of the heat slope explained (%)")
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "heat_variance_explained", data=v.reset_index())
