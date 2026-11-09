"""The JEPA counterfactual machine checked against the regression: for each site, the mean
relative change of a summer day's demand when the day is made 5 degC warmer, as predicted by the
weather-conditioned JEPA and decoded to half-hours (y), and as implied by the site's hourly
regression posteriors on the same days (x).

analysis/hourly_response.py (jepa_check.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

c = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "hourly_response" / "jepa_check.csv")
lim = float(np.nanpercentile(np.abs(np.r_[c["jepa_mean_pct"], c["reg_mean_pct"]]), 98)) * 1.1
fig, ax = style.figure(style.SQUARE)
ax.plot([-lim, lim], [-lim, lim], color="#B8B8B8", linewidth=0.6, linestyle=(0, (3, 2)), zorder=1)
style.reference_line(ax, 0.0, axis="x")
style.reference_line(ax, 0.0, axis="y")
ax.plot(
    c["reg_mean_pct"],
    c["jepa_mean_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    markeredgewidth=style.MARKER_EDGE,
    markeredgecolor="white",
    color=style.TERM_COLORS["cdd20"],
    zorder=3,
)
r = c[["reg_mean_pct", "jepa_mean_pct"]].corr().iloc[0, 1]
ax.annotate(
    f"r = {r:.2f}, {len(c)} sites\nmedian shape correlation {c['shape_corr'].median():.2f}",
    xy=(0.03, 0.95),
    xycoords="axes fraction",
    va="top",
    fontsize=style.FONT_SIZE["annotation"],
)
ax.set_xlim(-lim, lim)
ax.set_ylim(-lim, lim)
ax.set_xlabel("Regression: change for a 5 °C warmer summer day (%)")
ax.set_ylabel("JEPA counterfactual, decoded (%)")
style.style_axis(ax)
style.save_figure(fig, OUT / "jepa_cf_check", data=c)
