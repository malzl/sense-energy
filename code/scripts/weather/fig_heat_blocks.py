"""The heat slope by block of the day (night, morning, afternoon, evening) for three groups of
sites: those whose daily heat slope net of sunshine is significantly positive, significantly
negative, or neither. Random-effects pooled slope per block and group with its 95 % interval.

analysis/heat_drivers.py: block-mean demand regressed on site x (cold, heat, sunshine) with
specification B effects. Argument: cdd20 (heat, default) or solar_kwh_m2 (sunshine).
"""

import sys

import numpy as np
from _common import OUT, heat_drivers, heat_features

from sense_energy.analysis.weather_heterogeneity import random_effects
from sense_energy.visualization import style

TERM = sys.argv[1] if len(sys.argv) > 1 else "cdd20"
UNIT = {"cdd20": "% per °C above 20 °C", "solar_kwh_m2": "% per kWh/m² of solar radiation"}[TERM]
f = heat_features()
z = f["heat"] / f["heat_se"]
f["sign"] = np.select([z > 1.96, z < -1.96], style.HEAT_SIGN_ORDER[:2], default="neither")
b = heat_drivers("block_slopes")
b = b[(b["term"] == TERM) & (b["se"] > 0)].merge(f[["sign"]], left_on="site_code", right_index=True)

fig, ax = style.figure(style.WIDE_SINGLE)
x = np.arange(len(style.BLOCK_ORDER))
offsets = {"rises with heat": -0.12, "falls with heat": 0.12, "neither": 0.0}
rows = []
for sign in style.HEAT_SIGN_ORDER:
    est, lo, hi = [], [], []
    for name in style.BLOCK_ORDER:
        d = b[(b["sign"] == sign) & (b["block_name"] == name)]
        re = random_effects(d["slope"], d["se"])
        w = 1 / (d["se"] ** 2 + re["tau"] ** 2)
        se = float(np.sqrt(1 / w.sum()))
        est.append(100 * re["pooled"])
        lo.append(100 * (re["pooled"] - 1.96 * se))
        hi.append(100 * (re["pooled"] + 1.96 * se))
        rows.append(
            {
                "sign": sign,
                "block": name,
                "pooled_pct": est[-1],
                "lo_pct": lo[-1],
                "hi_pct": hi[-1],
                "sites": len(d),
            }
        )
    n = b[b["sign"] == sign]["site_code"].nunique()
    ax.errorbar(
        x + offsets[sign],
        est,
        yerr=[np.array(est) - np.array(lo), np.array(hi) - np.array(est)],
        fmt="o-",
        color=style.HEAT_SIGN_COLORS[sign],
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        linewidth=0.9,
        elinewidth=0.6,
        capsize=1.5,
        label=f"{sign} on the daily slope ({n})",
        zorder=3,
    )
style.reference_line(ax, 0.0)
ax.set_xticks(x)
ax.set_xticklabels([s.split(" ")[0] + "\n" + s.split(" ")[1] for s in style.BLOCK_ORDER])
ax.set_xlabel("Block of the local day")
ax.set_ylabel(f"{style.TERM_NAMES[TERM][:1].upper() + style.TERM_NAMES[TERM][1:]} slope ({UNIT})")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.42, bottom=0.42)
import pandas as pd  # noqa: E402

style.save_figure(fig, OUT / f"heat_blocks_{style.TERM_NAMES[TERM]}", data=pd.DataFrame(rows))
