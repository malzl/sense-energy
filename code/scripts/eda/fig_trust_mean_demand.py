"""Mean total demand per trust, electricity and gas where available. One plot.

Trusts are ordered by electricity demand: these are entities, not candidate
methods, so the fixed-method-order rule does not apply.
"""

import numpy as np
import pandas as pd

from sense_energy.eda import common as C
from sense_energy.visualization import style

rows = {}
for energy in ("elec", "gas"):
    _, index = C.series_for(energy, "trust")
    for _, r in index.iterrows():
        rows.setdefault(r["series_id"], {"trust": C.short_trust_name(r["organisation_name"])})[
            energy
        ] = r["mean_kwh"] * 2 / 1000  # MW
frame = (
    pd.DataFrame.from_dict(rows, orient="index").reset_index().rename(columns={"index": "trust_id"})
)
# Trusts whose total is never complete (some site always missing) have no mean;
# they are shown with a neutral marker, never as a zero bar (spec, missing data).
frame["no_complete_total"] = frame["elec"].isna()
frame = frame.sort_values(
    ["no_complete_total", "elec"], ascending=[False, True], na_position="first"
).reset_index(drop=True)

fig, ax = style.figure(style.DOUBLE_COLUMN)
y = np.arange(len(frame))
h = 0.38
ax.barh(
    y + h / 2,
    frame["elec"].fillna(0),
    height=h,
    color=style.ENERGY_COLORS["elec"],
    label="electricity",
    linewidth=0,
)
ax.barh(
    y - h / 2,
    frame["gas"].fillna(0),
    height=h,
    color=style.ENERGY_COLORS["gas"],
    label="gas (where available)",
    linewidth=0,
)
for i, r in frame.iterrows():
    if pd.isna(r["elec"]):
        ax.text(
            0.01,
            i + h / 2,
            "no complete total",
            va="center",
            ha="left",
            fontsize=style.FONT_SIZE["annotation"],
            color=style.COLORS["grey"],
        )
    if pd.isna(r["gas"]):
        ax.plot([0], [i - h / 2], marker="|", color=style.COLORS["light_grey"], markersize=4)
ax.plot(
    [],
    [],
    marker="|",
    linestyle="none",
    color=style.COLORS["light_grey"],
    markersize=6,
    label="gas not available",
)
ax.set_yticks(y)
ax.set_yticklabels(frame["trust"], fontsize=style.FONT_SIZE["tick"])
ax.set_xlabel("Mean demand (MW, trust total)")
ax.set_xlim(left=0)
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, anchor_y=-0.10, bottom=0.16)
style.save_figure(fig, C.OUT / "trust_mean_demand", data=frame)
