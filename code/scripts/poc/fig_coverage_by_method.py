"""Empirical coverage of the nominal 80% interval per method, median and IQR across sites. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, display, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
models = [
    m for m in ordered_models(per["model"].unique()) if m != "seasonal_naive"
]  # a point forecast has no interval
rows = []
for m in models:
    s = per.loc[per["model"] == m, "coverage_80"] * 100
    rows.append(
        {
            "model": m,
            "method": display(m),
            "median": s.median(),
            "q25": s.quantile(0.25),
            "q75": s.quantile(0.75),
        }
    )
frame = pd.DataFrame(rows)

fig, ax = style.figure(style.SINGLE_COLUMN)
y = np.arange(len(frame))[::-1]
for yi, r in zip(y, frame.itertuples(), strict=True):
    c = style.method_color(r.method)
    ax.plot([r.q25, r.q75], [yi, yi], color=c, linewidth=style.LINEWIDTH["secondary"], zorder=2)
    ax.plot(
        [r.median],
        [yi],
        marker="o",
        markersize=style.MARKER_SIZE + 1,
        color=c,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        linestyle="none",
        zorder=3,
    )
style.reference_line(ax, 80.0, axis="x")
ax.set_yticks(y)
ax.set_yticklabels(frame["method"])
ax.set_xlabel("Coverage (%, target 80)")
ax.set_xlim(0, 100)
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "coverage_by_method", data=frame)
