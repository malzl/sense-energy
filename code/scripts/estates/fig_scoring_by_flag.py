"""Day-ahead nMAE of the forecast-weather variants and seasonal naive on all 91 PoC sites, on the
sites without flagged supplies, and on the sites whose meters ERIC verifies. One plot."""

import pandas as pd
from _common import OUT, RESULTS

from sense_energy.visualization import style

MODELS = [
    "seasonal_naive",
    "lightgbm_ifs",
    "chronos2_ifs",
    "timesfm3_ifs",
    "tabpfn_ts_ifs",
    "tirex2_ifs",
    "t0beta_ifs",
]
SUBSETS = {
    "all sites": ("o", "#000000"),
    "without flagged supplies": ("s", "#7A7A7A"),
    "verified only": ("^", "#0072B2"),
}
s = pd.read_csv(RESULTS / "scoring_by_coverage_flag.csv")
s = s[s["model"].isin(MODELS)]
names = [style.METHOD_NAMES[m] for m in MODELS]
order = style.ordered(names)
y = {n: i for i, n in enumerate(order[::-1])}

fig, ax = style.figure(style.SINGLE_COLUMN)
for k, (sub, (marker, colour)) in enumerate(SUBSETS.items()):
    d = s[s["subset"] == sub]
    n = int(d["sites"].max())
    ax.plot(
        d["nmae"] * 100,
        [y[style.METHOD_NAMES[m]] + (k - 1) * 0.18 for m in d["model"]],
        linestyle="none",
        marker=marker,
        markersize=style.MARKER_SIZE,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=f"{sub} (n = {n})",
    )
ax.set_yticks(list(y.values()))
ax.set_yticklabels(list(y.keys()))
ax.set_xlabel("MAE / mean demand (%)")
ax.set_xlim(left=0)
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.22, bottom=0.32)
style.save_figure(fig, OUT / "scoring_by_flag", data=s)
