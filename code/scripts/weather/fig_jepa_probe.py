"""Which representation of a site carries its heat response: the share of the true between-site
variance of the fixed-effect heat slope recovered out of fold by a ridge probe (filled) and by
TabPFN (open), for the JEPA embedding, the Chronos-2 embedding, the observable descriptors, the
diurnal signature and their combinations.

analysis/jepa_probe.py (probe.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

p = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "jepa" / "probe.csv")
p = p[p["target"] == "heat"]
order = (
    p[p["model"] == "ridge"].set_index("representation")["share_true_variance"].sort_values().index
)
y = np.arange(len(order))
ridge = p[p["model"] == "ridge"].set_index("representation").reindex(order)
tab = p[p["model"] == "tabpfn"].set_index("representation").reindex(order)

fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0, axis="x")
ax.barh(
    y,
    100 * ridge["share_true_variance"].clip(lower=-0.3),
    height=0.6,
    color="#7A7A7A",
    label="ridge probe",
    zorder=3,
)
ax.plot(
    100 * tab["share_true_variance"].clip(lower=-0.3),
    y,
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE + 1,
    markerfacecolor="white",
    markeredgecolor=style.TERM_COLORS["cdd20"],
    markeredgewidth=1.0,
    label="TabPFN probe",
    zorder=4,
)
ax.set_yticks(y)
ax.set_yticklabels([f"{r} ({int(ridge.loc[r, 'sites'])})" for r in order])
ax.set_xlabel("True between-site variance of the heat slope recovered out of fold (%)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.22, bottom=0.26, lowercase=False)
print(ridge[["r2_oof", "share_true_variance"]].round(3).to_string())
style.save_figure(fig, OUT / "jepa_probe", data=p)
