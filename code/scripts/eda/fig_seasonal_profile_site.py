"""Calendar-month demand index, median and IQR across sites, electricity and gas. One plot."""

import pandas as pd

from sense_energy.eda import common as C
from sense_energy.visualization import style

frames = []
fig, ax = style.figure(style.SINGLE_COLUMN)
for energy in ("elec", "gas"):
    wide, _ = C.series_for(energy, "site")
    s = C.seasonal_profile(C.normalise(wide))
    s["energy"] = energy
    frames.append(s)
    c = style.ENERGY_COLORS[energy]
    ax.fill_between(s["month"], s["q25"], s["q75"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(
        s["month"],
        s["median"],
        color=c,
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=style.ENERGY_LABELS[energy],
    )
ax.set_xticks(range(1, 13))
ax.set_xticklabels(list("JFMAMJJASOND"))
ax.set_xlabel("Calendar month")
ax.set_ylabel("Demand index (site mean = 1)")
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(fig, C.OUT / "seasonal_profile_site", data=pd.concat(frames, ignore_index=True))
