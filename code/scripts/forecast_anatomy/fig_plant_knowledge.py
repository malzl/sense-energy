"""What knowledge of the plant adds to a day-ahead forecast: the change in error when a model is
told that a site has on-site generation (a trained model with the site's identity or the estate
descriptors; the zero-shot model with a flag and the plant's share, all sites seen together),
when it gets the plant's state up to the issue time, and when it gets the state of the target day
(made from the import itself: an upper bound for a schedule). Paired differences with 95 %
intervals from resampling weeks, at the sites with on-site generation and at the others.

analysis/forecast_anatomy.py (plant_differences.csv).
"""

import pandas as pd
from _common import FA, OUT

from sense_energy.visualization import style

d = pd.read_csv(FA / "plant_differences.csv")
d["pair"] = d["a"] + "|" + d["b"]
d = d[d["pair"].isin(style.PLANT_KNOWLEDGE_NAMES) & d["sites"].isin(style.PLANT_KNOWLEDGE_SETS)]
pairs = list(style.PLANT_KNOWLEDGE_NAMES)
fig, ax = style.figure(style.DOUBLE_COLUMN)
style.reference_line(ax, 0.0)
for j, (name, color) in enumerate(style.PLANT_KNOWLEDGE_SETS.items()):
    x = d[d["sites"] == name].set_index("pair").reindex(pairs)
    pos = [k + (j - 0.5) * 0.22 for k in range(len(pairs))]
    ax.errorbar(
        pos,
        x["diff_pts"],
        yerr=[x["diff_pts"] - x["lo"], x["hi"] - x["diff_pts"]],
        fmt="o",
        markersize=style.MARKER_SIZE,
        color=color,
        elinewidth=style.LINEWIDTH["secondary"],
        capsize=0,
        label=name,
    )
ax.set_xticks(range(len(pairs)))
ax.set_xticklabels([style.PLANT_KNOWLEDGE_NAMES[p] for p in pairs])
ax.set_xlim(-0.5, len(pairs) - 0.5)
ax.set_ylabel("Change in error (percentage points)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=2, bottom=0.3)
style.save_figure(fig, OUT / "plant_knowledge", data=d)
