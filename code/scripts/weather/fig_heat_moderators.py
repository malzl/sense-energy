"""What moderates the heat slope, net of sunshine: each moderator's interaction with the heat
dose, one model per moderator (filled markers, 95 % intervals) and the joint model where the
term is in it (open markers). Argument: categorical (levels against their reference, default)
or continuous (per SD of the moderator).

analysis/heat_drivers.py (moderators_heat, moderators_heat_joint).
"""

import sys

import numpy as np
from _common import OUT, heat_drivers

from sense_energy.visualization import style

KIND = sys.argv[1] if len(sys.argv) > 1 else "categorical"
order = (
    style.HEAT_MODERATOR_CATEGORICAL if KIND == "categorical" else style.HEAT_MODERATOR_CONTINUOUS
)
one = heat_drivers("moderators_heat")
one = one[(one["dose"] == "cdd20") & (one["level"] != "base") & (one["kind"] == KIND)]
joint = heat_drivers("moderators_heat_joint")
joint = joint[(joint["dose"] == "cdd20") & (joint["level"] != "base")].set_index("level")

rows = []
for m in order:
    d = one[one["model"] == m]
    for _, r in d.iterrows():
        label = r["level"].replace("_", " ") if KIND == "categorical" else m
        rows.append((m, label, r))
y = np.arange(len(rows))[::-1]

fig, ax = style.figure(style.WIDE_SHORT)
style.reference_line(ax, 0.0, axis="x")
labels = []
for k, (yy, (_m, label, r)) in enumerate(zip(y, rows, strict=True)):
    ax.errorbar(
        r["coef_pct"],
        yy,
        xerr=[[r["coef_pct"] - r["lo_pct"]], [r["hi_pct"] - r["coef_pct"]]],
        fmt="o",
        color=style.TERM_COLORS["cdd20"],
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        elinewidth=0.7,
        capsize=1.5,
        zorder=3,
    )
    if r["level"] in joint.index:
        j = joint.loc[r["level"]]
        ax.plot(
            j["coef_pct"],
            yy - 0.28,
            linestyle="none",
            marker="o",
            markersize=style.MARKER_SIZE - 0.5,
            markerfacecolor="none",
            markeredgewidth=0.8,
            markeredgecolor="#222222",
            zorder=4,
        )
    tag = f"(p={r['p_joint']:.2f}, n={int(r['sites'])})"
    if KIND == "continuous":
        labels.append(f"{label} {tag}")
    else:  # model: level, with the joint p and the reference on the group's first row
        first = k == next(i for i, (mm, _, _) in enumerate(rows) if mm == _m)
        labels.append(f"{_m}: {label} {tag}, ref. {r['reference']}" if first else f"{_m}: {label}")
ax.set_yticks(y)
ax.set_yticklabels(labels)
ax.set_ylim(-0.8, len(rows) - 0.2)
unit = "against the reference level" if KIND == "categorical" else "per SD of the moderator"
ax.set_xlabel(f"Change in the heat slope (percentage points per °C above 20 °C), {unit}")
style.style_axis(ax, wide_short=True, grid="x")
ax.plot(
    [],
    [],
    marker="o",
    linestyle="none",
    color=style.TERM_COLORS["cdd20"],
    label="one moderator at a time",
)
ax.plot(
    [],
    [],
    marker="o",
    linestyle="none",
    markerfacecolor="none",
    markeredgecolor="#222222",
    label="joint model",
)
style.add_bottom_legend(ax, anchor_y=-0.3, bottom=0.3)
style.save_figure(fig, OUT / f"heat_moderators_{KIND}", data=one)
