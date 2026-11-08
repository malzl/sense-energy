"""What moderates the cold and heat slopes: each moderator's interaction coefficients from the
one-at-a-time regressions (categorical levels against their reference, continuous per SD). One plot.

95 % intervals clustered by site and date (analysis/weather_heterogeneity.py); the joint
p-value of each moderator's interactions is printed. Doses offset vertically.
"""

import numpy as np
from _common import OUT, heterogeneity

from sense_energy.visualization import style

m = heterogeneity("moderators")
m = m[m["level"] != "base"].copy()
m["label"] = np.where(
    m["kind"] == "categorical",
    m["model"] + ": " + m["level"].str.replace("_", " "),
    m["model"] + " (per SD)",
)
order = [lab for mod in style.MODERATOR_ORDER for lab in m.loc[m["model"] == mod, "label"].unique()]
y = {lab: i for i, lab in enumerate(order)}

fig, ax = style.figure((7.2, 3.8))
offsets = {"hdd14": 0.15, "cdd20": -0.15}
for dose, off in offsets.items():
    d = m[m["dose"] == dose]
    ax.errorbar(
        d["coef_pct"],
        d["label"].map(y) + off,
        xerr=[d["coef_pct"] - d["lo_pct"], d["hi_pct"] - d["coef_pct"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=style.LINEWIDTH["reference"],
        color=style.DOSE_COLORS[dose],
        label=style.DOSE_NAMES[dose],
    )
style.reference_line(ax, 0.0, axis="x")
ax.set_yticks(range(len(order)))
ax.set_yticklabels(order)
ax.set_ylim(len(order) - 0.5, -0.5)
ax.set_xlim(-7, 7)
ax.set_xlabel("Change in the slope (percentage points per °C)")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, anchor_y=-0.12, bottom=0.16)
print(m.groupby("model")["p_joint"].first().reindex(style.MODERATOR_ORDER).round(3).to_string())
style.save_figure(fig, OUT / "moderators", data=m)
