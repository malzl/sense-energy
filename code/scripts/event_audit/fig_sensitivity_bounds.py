"""How wrong can the event effects be? Bounds on the mean effect of a heat wave and of a cold
spell on a site's demand when a hidden factor can change the odds that a day is an event day
by up to Gamma (generalized marginal sensitivity model, Frauen et al. 2023). Lines: the lower
and upper bound; band: their 95 % interval from resampling sites. The effect is no longer
clear where the lower line crosses zero.

analysis/causal_sensitivity.py (bounds.csv).
"""

import pandas as pd
from _common import OUT

from sense_energy.config import PROJECT_ROOT
from sense_energy.visualization import style

b = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "causal_sensitivity" / "bounds.csv")
b = b[b["gamma"] <= 5]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for kind in ("cold spell", "heatwave"):
    d = b[b["type"] == kind]
    color = style.WEATHER_EVENT_COLORS[kind]
    ax.fill_between(
        d["gamma"], d["ci_lo"], d["ci_hi"], color=color, alpha=style.BAND_ALPHA, linewidth=0
    )
    ax.plot(
        d["gamma"],
        d["effect_lo"],
        color=color,
        linewidth=style.LINEWIDTH["primary"],
        label=style.EVENT_NAMES[kind],
    )
    ax.plot(
        d["gamma"],
        d["effect_hi"],
        color=color,
        linewidth=style.LINEWIDTH["secondary"],
        linestyle="--",
    )
ax.set_xscale("log")
ax.set_xticks([1, 1.5, 2, 3, 5])
ax.set_xticklabels(["1", "1.5", "2", "3", "5"])
ax.set_xlabel("Hidden factor changes the odds of an event day by up to (Γ)")
ax.set_ylabel("Effect on demand (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2)
style.save_figure(fig, OUT / "sensitivity_bounds", data=b)
