"""Daily demand index against daily mean temperature, binned medians and IQR across site-days. One plot.

Uses the ERA5 months extracted so far; the window is printed and belongs in the caption.
"""

import pandas as pd

from sense_energy.eda import common as C
from sense_energy.visualization import style

frames = []
fig, ax = style.figure(style.SINGLE_COLUMN)
for energy in ("elec", "gas"):
    sd = C.daily_temperature_and_demand(energy)
    b = C.bin_by_temperature(sd)
    b["energy"] = energy
    frames.append(b)
    c = style.ENERGY_COLORS[energy]
    ax.fill_between(b["t_bin"], b["q25"], b["q75"], color=c, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(
        b["t_bin"],
        b["median"],
        color=c,
        linewidth=style.LINEWIDTH["primary"],
        label=f"{style.ENERGY_LABELS[energy]} ({sd['site_code'].nunique()} sites)",
    )
    print(f"{energy}: {len(sd):,} site-days, {sd['date'].min()} -> {sd['date'].max()}")
ax.set_xlabel("Daily mean 2 m temperature (°C)")
ax.set_ylabel("Daily demand index (site mean = 1)")
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(
    fig, C.OUT / "temperature_response_site", data=pd.concat(frames, ignore_index=True)
)
