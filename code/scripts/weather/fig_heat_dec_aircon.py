"""The heat slope against what the Display Energy Certificates say about cooling plant: each
site's sunshine-controlled heat slope (y) against the floor-area share of its buildings whose
DEC records air conditioning as present (x), coloured by the main building environment on the
DEC, marker area by mean demand. Sites with no DEC at their postcode are not shown.

analysis/heat_drivers.py with data/dec.py (interim/dec_sites.parquet).
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features()
f = f[f["dec_buildings"].fillna(0) > 0].dropna(subset=["dec_ac_share", "heat"])
size = np.clip(2.5 + 1.2 * np.sqrt(f["mean_kw"] / f["mean_kw"].median()), 2.5, 7.0)
rng = np.random.default_rng(0)
jitter = rng.uniform(-0.012, 0.012, len(f))  # many sites sit exactly at 0 or 1

fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0)
for env in style.DEC_ENV_ORDER:
    d = f[f["dec_main_environment"] == env]
    if d.empty:
        continue
    for (idx, r), s in zip(d.iterrows(), size[d.index], strict=True):
        ax.plot(
            100 * r["dec_ac_share"] + 100 * jitter[f.index.get_loc(idx)],
            r["heat_pct"],
            linestyle="none",
            marker="o",
            markersize=s,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=style.DEC_ENV_COLORS[env],
            alpha=0.9,
            zorder=3,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.DEC_ENV_COLORS[env],
        label=f"{env} ({len(d)})",
    )
yl = float(np.percentile(np.abs(f["heat_pct"]), 98)) * 1.1
ax.set_ylim(-yl, yl)
ax.set_xlim(-6, 106)
ax.set_xlabel("Floor area with air conditioning on the DEC (%)")
ax.set_ylabel("Heat slope, net of sunshine (% per °C above 20 °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.36)
r = f[["dec_ac_share", "heat"]].corr().iloc[0, 1]
print(f"{len(f)} sites with a DEC; corr(AC share, heat slope) = {r:.2f}")
for cls, d in f.groupby("dec_ac_class"):
    print(f"  {cls}: {len(d)} sites, median heat slope {d['heat_pct'].median():+.2f}")
style.save_figure(fig, OUT / "heat_dec_aircon", data=f)
