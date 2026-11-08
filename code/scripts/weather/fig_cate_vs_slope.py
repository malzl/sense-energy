"""The learned heat effect against the regression slope, one point per site: the causal forest's
out-of-fold site effect (x, with its 95 % interval) against the site's fixed-effect heat slope
net of sunshine (y). Argument: observables (default) or signature (the covariate set that adds
the site's own diurnal signature).

analysis/heat_causal_ml.py (cate_site.csv).
"""

import sys

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT, heat_features

from sense_energy.visualization import style

SET = (
    "observables + signature"
    if (len(sys.argv) > 1 and sys.argv[1] == "signature")
    else "observables"
)
c = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "heat_causal_ml" / "cate_site.csv")
c = c[c["set"] == SET].set_index("site_code")
f = heat_features()
c["sign"] = f["sign"].reindex(c.index)
for col in (
    "tau_rlearner",
    "tau_forest",
    "tau_forest_lo",
    "tau_forest_hi",
    "slope_fe",
    "slope_fe_se",
):
    c[col] = 100 * c[col]
c = c.dropna(subset=["tau_forest", "slope_fe", "slope_fe_se"])
c = c[c["slope_fe_se"] > 0]  # the same sites as the summary's agreement numbers

fig, ax = style.figure(style.SQUARE)
lim = float(np.percentile(np.abs(np.r_[c["tau_forest"], c["slope_fe"]]), 98)) * 1.1
ax.plot([-lim, lim], [-lim, lim], color="#B8B8B8", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
for sign in style.HEAT_SIGN_ORDER:
    d = c[c["sign"] == sign]
    if d.empty:
        continue
    ax.errorbar(
        d["tau_forest"],
        d["slope_fe"],
        xerr=[d["tau_forest"] - d["tau_forest_lo"], d["tau_forest_hi"] - d["tau_forest"]],
        yerr=1.96 * d["slope_fe_se"],
        fmt="none",
        ecolor="#D8D8D8",
        elinewidth=0.4,
        zorder=2,
    )
    ax.plot(
        d["tau_forest"],
        d["slope_fe"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.HEAT_SIGN_COLORS[sign],
        label=f"{sign} ({len(d)})",
        zorder=3,
    )
r = c[["tau_forest", "slope_fe"]].corr().iloc[0, 1]
ax.annotate(
    f"r = {r:.2f}, {len(c)} sites",
    xy=(0.03, 0.95),
    xycoords="axes fraction",
    fontsize=style.FONT_SIZE["annotation"],
)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Causal-forest heat effect, out of fold (% per °C)")
ax.set_ylabel("Fixed-effect heat slope net of sunshine (% per °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.32)
print(
    f"{SET}: corr forest vs slope {r:.3f}; r-learner {c[['tau_rlearner', 'slope_fe']].corr().iloc[0, 1]:.3f}"
)
style.save_figure(
    fig,
    OUT / f"cate_vs_slope_{'signature' if 'signature' in SET else 'observables'}",
    data=c.reset_index(),
)
