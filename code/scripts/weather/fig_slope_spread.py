"""How much the cold and heat slopes truly vary between sites: the distribution of the raw site
estimates against the distribution of the shrunk posteriors, for each dose. One plot.

The gap between the two is sampling noise; what remains in the posterior is real between-site
variation (tau from the random-effects decomposition, analysis/weather_heterogeneity.py).
"""

import numpy as np
from _common import OUT, heterogeneity

from sense_energy.visualization import style

s = heterogeneity("site_slopes")

fig, ax = style.figure(style.SINGLE_COLUMN)
positions = {"hdd14": (0, 1), "cdd20": (3, 4)}
for dose, (p_raw, p_post) in positions.items():
    d = s[s["dose"] == dose]
    for pos, col, alpha in ((p_raw, "slope_pct", 0.45), (p_post, "posterior_pct", 1.0)):
        v = np.sort(d[col].to_numpy())
        x = pos + np.linspace(-0.3, 0.3, len(v))
        ax.plot(
            x,
            v,
            linestyle="none",
            marker="o",
            markersize=style.MARKER_SIZE - 1.5,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=style.DOSE_COLORS[dose],
            alpha=alpha,
        )
        ax.hlines(
            np.median(v),
            pos - 0.35,
            pos + 0.35,
            color="#000000",
            linewidth=style.LINEWIDTH["primary"],
        )
style.reference_line(ax, 0.0)
ax.set_xticks([0, 1, 3, 4])
ax.set_xticklabels(["cold\nraw", "cold\nshrunk", "heat\nraw", "heat\nshrunk"])
ax.set_ylabel("Site slope (% per °C)")
lim = np.percentile(np.abs(s["slope_pct"]), 97)
ax.set_ylim(-lim, lim)
style.style_axis(ax, grid="y")
for dose, d in s.groupby("dose"):
    print(
        f"{dose}: raw SD {d['slope_pct'].std():.2f}, posterior SD {d['posterior_pct'].std():.2f}, tau {100 * d['tau'].iloc[0]:.2f}"
    )
style.save_figure(
    fig,
    OUT / "slope_spread",
    data=s[["site_code", "dose", "slope_pct", "se_pct", "posterior_pct", "tau", "i2"]],
)
