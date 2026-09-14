"""Normalised MAE by local half-hour of the target day for the plain variant of each family,
SARIMA and the baselines. One plot."""

import numpy as np
from _common import OUT, display, joined, plain

from sense_energy.visualization import style

f = joined()
models = plain(f["model"].unique())
f = f[f["model"].isin(models)]
f["ae"] = f["err"].abs()
frame = f.groupby(["model", "tod"])["ae"].mean().reset_index()
frame["method"] = frame["model"].map(display)

fig, ax = style.figure(style.SINGLE_COLUMN)
for m in models:
    d = frame[frame["model"] == m].sort_values("tod")
    name = display(m)
    baseline = style.METHOD_ROLES[name] == "simple_baseline"
    ax.plot(
        d["tod"] / 2,
        d["ae"] * 100,
        color=style.method_color(name),
        linestyle=style.method_linestyle(name),
        linewidth=style.LINEWIDTH["reference"] if baseline else style.LINEWIDTH["primary"],
        label=name,
    )
ax.set_xlabel("Local time of the target day (h)")
ax.set_ylabel("MAE / mean demand (%)")
ax.set_xlim(0, 24)
ax.set_xticks(np.arange(0, 25, 6))
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36)  # product names
style.save_figure(fig, OUT / "nmae_by_hour", data=frame)
