"""The warm-night slope (per degC of daily minimum above 12 degC, net of cold and sunshine)
against the floor-area share of the site's buildings whose DEC names electricity as the main
heating fuel. Sites whose heating is electric should fall when the night is warm.

analysis/heat_drivers.py with data/dec.py (interim/dec_sites.parquet).
"""

import numpy as np
from _common import OUT, heat_features

from sense_energy.visualization import style

f = heat_features()
f = f[f["dec_buildings"].fillna(0) > 0].dropna(subset=["dec_electric_heating_share", "warm_night"])
size = np.clip(2.5 + 1.2 * np.sqrt(f["mean_kw"] / f["mean_kw"].median()), 2.5, 7.0)
rng = np.random.default_rng(1)
jitter = rng.uniform(-0.012, 0.012, len(f))

fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0)
for sign in style.HEAT_SIGN_ORDER:
    d = f[f["sign"] == sign]
    if d.empty:
        continue
    ax.errorbar(
        100 * d["dec_electric_heating_share"] + 100 * jitter[[f.index.get_loc(i) for i in d.index]],
        d["warm_night_pct"],
        yerr=1.96 * d["warm_night_se_pct"],
        fmt="none",
        ecolor="#D8D8D8",
        elinewidth=0.4,
        zorder=2,
    )
    for (idx, r), s in zip(d.iterrows(), size[d.index], strict=True):
        ax.plot(
            100 * r["dec_electric_heating_share"] + 100 * jitter[f.index.get_loc(idx)],
            r["warm_night_pct"],
            linestyle="none",
            marker="o",
            markersize=s,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=style.HEAT_SIGN_COLORS[sign],
            alpha=0.9,
            zorder=3,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=style.HEAT_SIGN_COLORS[sign],
        label=f"{sign} on the daily slope ({len(d)})",
    )
yl = float(np.percentile(np.abs(f["warm_night_pct"]), 98)) * 1.15
ax.set_ylim(-yl, yl)
ax.set_xlim(-6, 106)
ax.set_xlabel("Floor area heated by electricity on the DEC (%)")
ax.set_ylabel("Warm-night slope (% per °C of daily minimum above 12 °C)")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.32)
r = f[["dec_electric_heating_share", "warm_night"]].corr().iloc[0, 1]
print(f"{len(f)} sites; corr(electric-heating share, warm-night slope) = {r:.2f}")
style.save_figure(fig, OUT / "heat_dec_heating", data=f)
