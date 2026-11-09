"""What warming does to annual electricity, site by site: the change in each site's annual
demand with every day 1, 2 and 4 degC warmer (less heating, more cooling), as a strip per shift
with the median marked.

analysis/estate_counterfactuals.py (warming_sites.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

w = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "estate_counterfactuals" / "warming_sites.csv")
w = w[w["shift_c"] > 0]
shifts = sorted(w["shift_c"].unique())
rng = np.random.default_rng(1)
colors = {1.0: "#E69F00", 2.0: "#D55E00", 4.0: "#CC79A7"}
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for k, s in enumerate(shifts):
    d = w[w["shift_c"] == s]["energy_change_pct"]
    ax.plot(
        k + rng.uniform(-0.18, 0.18, len(d)),
        d,
        linestyle="none",
        marker="o",
        markersize=2.5,
        color=colors[s],
        alpha=0.7,
        zorder=3,
    )
    ax.hlines(d.median(), k - 0.3, k + 0.3, color="#222222", linewidth=1.2, zorder=4)
ax.set_xticks(range(len(shifts)))
ax.set_xticklabels([f"+{s:g} °C" for s in shifts])
ax.set_ylabel("Change in annual electricity (%)")
ax.set_ylim(
    float(np.percentile(w["energy_change_pct"], 1)) * 1.1,
    float(np.percentile(w["energy_change_pct"], 99)) * 1.3,
)
style.style_axis(ax, grid="y")
style.save_figure(fig, OUT / "warming_energy", data=w)
