"""The weather-conditioned JEPA as a counterfactual machine: for each site, the size of the
change in the predicted day embedding when the target day is made 5 degC warmer (x) against the
site's fixed-effect heat slope net of sunshine (y), coloured by the sign of that slope. If the
latent response carries the heat response, risers sit to the right.

experiments/jepa.py (response_embeddings, code/reports/jepa_weather/site_embeddings.parquet).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, heat_features

from sense_energy.experiments.jepa import OUT_DIR_WEATHER
from sense_energy.visualization import style

e = pd.read_parquet(OUT_DIR_WEATHER / "site_embeddings.parquet")
f = heat_features()
d = (
    e[["h_norm", "s_norm"]]
    .join(f[["heat_pct", "heat_se_pct", "sun_pct", "sign", "mean_kw"]], how="inner")
    .dropna(subset=["heat_pct"])
)

fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0)
for sign in style.HEAT_SIGN_ORDER:
    s = d[d["sign"] == sign]
    if s.empty:
        continue
    ax.errorbar(
        s["h_norm"],
        s["heat_pct"],
        yerr=1.96 * s["heat_se_pct"],
        fmt="none",
        ecolor="#D8D8D8",
        elinewidth=0.4,
        zorder=2,
    )
    ax.plot(
        s["h_norm"],
        s["heat_pct"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.HEAT_SIGN_COLORS[sign],
        label=f"{sign} ({len(s)})",
        zorder=3,
    )
r = d[["h_norm", "heat_pct"]].corr(method="spearman").iloc[0, 1]
ax.annotate(
    f"Spearman r = {r:.2f}, {len(d)} sites",
    xy=(0.03, 0.95),
    xycoords="axes fraction",
    fontsize=style.FONT_SIZE["annotation"],
)
ax.set_xlabel("Latent response to a 5 °C warmer day (norm)")
ax.set_ylabel("Fixed-effect heat slope net of sunshine (% per °C)")
ax.set_ylim(
    -float(np.percentile(np.abs(d["heat_pct"]), 98)) * 1.1,
    float(np.percentile(np.abs(d["heat_pct"]), 98)) * 1.1,
)
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.32)
print(
    f"heat: spearman {r:.3f}; sunshine: spearman {d[['s_norm', 'sun_pct']].corr(method='spearman').iloc[0, 1]:.3f}"
)
style.save_figure(fig, OUT / "jepa_response", data=d.reset_index())
