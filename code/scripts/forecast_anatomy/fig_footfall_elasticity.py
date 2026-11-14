"""Elasticity of a site's daily demand to its own daily footfall and, as the placebo, to the
footfall of the university campus in the same city, within month and weekday with weather and
holidays held fixed (2024). 95 % intervals, Newey-West standard errors with seven lags.

analysis/forecast_anatomy.py (mobility_regression.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

r = pd.read_csv(FA / "mobility_regression.csv")
r = r[r["model"].isin(style.FOOTFALL_TERM_NAMES)]
sites = sorted(r["site_code"].unique())
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
for j, (model, name) in enumerate(style.FOOTFALL_TERM_NAMES.items()):
    d = r[r["model"] == model].set_index("site_code").reindex(sites)
    ax.errorbar(
        [k + (j - 0.5) * 0.24 for k in range(len(sites))],
        d["elasticity"],
        yerr=[d["elasticity"] - d["lo"], d["hi"] - d["elasticity"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        color=style.FOOTFALL_TERM_COLORS[model],
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=name,
    )
ax.set_xticks(range(len(sites)))
ax.set_xticklabels(sites)
ax.set_xlim(-0.5, len(sites) - 0.5)
ax.set_ylabel("Elasticity of demand to footfall")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "footfall_elasticity", data=r)
