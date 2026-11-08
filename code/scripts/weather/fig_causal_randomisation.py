"""Randomisation inference: the cold and heat effects (specification B, sites without CHP) when whole
daily weather series are reassigned between sites at random, against the observed estimates. One plot.

Histograms of the permutation estimates; vertical lines are the observed estimates
(analysis/weather_causal.py).
"""

import numpy as np
from _common import OUT, causal

from sense_energy.visualization import style

r = causal("randomisation")
obs, draws = r[r["draw"] == -1].iloc[0], r[r["draw"] >= 0]

fig, ax = style.figure(style.SINGLE_COLUMN)
lo = min(100 * np.expm1(draws[list(style.DOSE_COLORS)]).min().min(), 0.0)
hi = max(
    100 * np.expm1(obs[list(style.DOSE_COLORS)]).max(),
    100 * np.expm1(draws[list(style.DOSE_COLORS)]).max().max(),
)
bins = np.linspace(lo - 0.1, hi + 0.1, 41)
for dose, c in style.DOSE_COLORS.items():
    v = 100 * np.expm1(draws[dose])
    ax.hist(v, bins=bins, color=c, alpha=0.5, linewidth=0, label=style.DOSE_NAMES[dose])
    ax.axvline(100 * np.expm1(obs[dose]), color=c, linewidth=style.LINEWIDTH["primary"])
ax.set_xlabel("Estimated demand change (% per °C)")
ax.set_ylabel("Permutations")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
for dose in style.DOSE_COLORS:
    p = (np.abs(draws[dose]) >= abs(obs[dose])).mean()
    print(
        f"{dose}: observed {100 * np.expm1(obs[dose]):.2f} %, permutation p = {p:.3f} ({len(draws)} draws)"
    )
style.save_figure(fig, OUT / "causal_randomisation", data=r)
