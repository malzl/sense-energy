"""Peak electricity demand over available grid capacity for each panel site whose meters cover
the whole site: today (x) against with all gas heating replaced by heat pumps (y); dashed
lines = capacity, dotted diagonal = no change. One plot."""

import matplotlib.ticker as mticker
from _common import OUT, headroom_panel

from sense_energy.visualization import style

GROUPS = {
    "acute and mixed hospitals": (
        "General acute hospital",
        "Mixed service hospital",
        "Specialist hospital (acute only)",
        "Other inpatient",
    ),
    "mental health": (
        "Mental Health (including Specialist services)",
        "Mental Health and Learning Disabilities",
    ),
    "community and non-inpatient": ("Community hospital (with inpatient beds)", "Non inpatient"),
}
COLOURS = {
    "acute and mixed hospitals": "#0072B2",
    "mental health": "#CC79A7",
    "community and non-inpatient": "#009E73",
}
h = headroom_panel()
h["group"] = (
    h["site_type"]
    .map({t: g for g, ts in GROUPS.items() for t in ts})
    .fillna("community and non-inpatient")
)

fig, ax = style.figure(style.SQUARE)
lo, hi = 0.1, 4.0
ax.plot(
    [lo, hi],
    [lo, hi],
    color="#B8B8B8",
    linewidth=style.LINEWIDTH["reference"],
    linestyle=":",
    zorder=1,
)
style.reference_line(ax, 1.0)
style.reference_line(ax, 1.0, axis="x")
for g, colour in COLOURS.items():
    d = h[h["group"] == g]
    ax.plot(
        d["util_now"],
        d["util_hp"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE + 1,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=f"{g} (n = {len(d)})",
        zorder=3,
    )
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(mticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 4]))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.set_xscale("log")
ax.set_yscale("log")
for axis in (ax.xaxis, ax.yaxis):
    axis.set_major_locator(mticker.FixedLocator([0.1, 0.2, 0.5, 1, 2, 4]))
    axis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(mticker.NullFormatter())
ax.set_xlim(lo, hi)
ax.set_ylim(lo, hi)
ax.set_aspect("equal")
ax.set_xlabel("Peak / capacity today")
ax.set_ylabel("Peak / capacity with heat pumps")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.2, bottom=0.3)
style.save_figure(fig, OUT / "headroom_panel", data=h)
