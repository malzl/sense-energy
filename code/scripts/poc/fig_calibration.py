"""Quantile reliability: share of observations at or below each predicted quantile, per method
(plain variants, SARIMA and the profile baseline), pooled over sites and periods. One plot."""

import numpy as np
import pandas as pd
from _common import OUT, QCOLS, QUANTILES, display, joined, plain

from sense_energy.visualization import style

f = joined()
models = [m for m in plain(f["model"].unique()) if m != "seasonal_naive"]  # a point forecast
rows = []
for m in models:
    d = f[f["model"] == m]
    for q, c in zip(QUANTILES, QCOLS, strict=True):
        rows.append(
            {
                "model": m,
                "method": display(m),
                "nominal": q,
                "observed": float((d["y"] <= d[c]).mean()),
            }
        )
frame = pd.DataFrame(rows)

fig, ax = style.figure(style.SQUARE)
ax.plot([0, 1], [0, 1], **style.REFERENCE_LINE, zorder=1)
for m in models:
    d = frame[frame["model"] == m]
    name = display(m)
    ax.plot(
        d["nominal"],
        d["observed"],
        color=style.method_color(name),
        linestyle=style.method_linestyle(name),
        linewidth=style.LINEWIDTH["secondary"],
        marker=style.method_marker(name),
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=name,
    )
ax.set_xlabel("Nominal quantile level")
ax.set_ylabel("Observed frequency")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_xticks(np.arange(0, 1.01, 0.2))
ax.set_yticks(np.arange(0, 1.01, 0.2))
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.2, bottom=0.3)  # product names
style.save_figure(fig, OUT / "calibration", data=frame)
