"""Validation negative log-likelihood against training step for the transformer neural
processes, calendar-only (solid) and with covariates (dashed); 16:30 origins in the last
60 days of the training window. One plot."""

import pandas as pd
from _common import OUT, RESULTS

from sense_energy.visualization import style

NAMES = {"tnp": "TNP", "pttnp": "PT-TNP", "gtnp": "Gridded TNP"}
rows = []
for p in sorted((RESULTS / "tnp_models").glob("*.history.csv")):
    arch, inputs = p.name.split("_")[:2]
    h = pd.read_csv(p).assign(arch=arch, inputs="covariates" if inputs == "cov" else "calendar")
    rows.append(h)
frame = pd.concat(rows, ignore_index=True)

fig, ax = style.figure(style.SINGLE_COLUMN)
for arch, name in NAMES.items():
    for inputs, ls in (("calendar", "-"), ("covariates", "--")):
        d = frame[(frame["arch"] == arch) & (frame["inputs"] == inputs)].sort_values("step")
        if d.empty:
            continue
        ax.plot(
            d["step"] / 1000,
            d["valid_nll"],
            color=style.method_color(name),
            linestyle=ls,
            linewidth=style.LINEWIDTH["secondary"],
            label=f"{name}, {inputs}",
        )
ax.set_xlabel("Training step (thousands)")
ax.set_ylabel("Validation negative log-likelihood")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=2, anchor_y=-0.26, bottom=0.40)  # product names
style.save_figure(fig, OUT / "tnp_training", data=frame)
