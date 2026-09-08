"""Monthly demand index per trust (grey) and the median across trusts (black), elec. One plot."""

from sense_energy.eda import common as C
from sense_energy.visualization import style

ENERGY = "elec"
wide, _ = C.series_for(ENERGY, "trust")
m = C.monthly_index(C.normalise(wide))

fig, ax = style.figure(style.WIDE_SINGLE)
for col in m.columns:
    ax.plot(
        m.index, m[col], color=style.SPAGHETTI, linewidth=style.LINEWIDTH["reference"], zorder=2
    )
med = m.median(axis=1)
ax.plot(
    m.index,
    med,
    color="#000000",
    linewidth=style.LINEWIDTH["primary"],
    zorder=3,
    label=f"median of {m.shape[1]} trusts",
)
ax.plot(
    [], [], color=style.SPAGHETTI, linewidth=style.LINEWIDTH["reference"], label="individual trust"
)
ax.set_xlabel("Month")
ax.set_ylabel("Monthly demand index (trust mean = 1)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax)
out = m.copy()
out.insert(0, "median", med)
out = out.reset_index().rename(columns={"index": "month"})
style.save_figure(fig, C.OUT / f"trust_monthly_index_{ENERGY}", data=out)
