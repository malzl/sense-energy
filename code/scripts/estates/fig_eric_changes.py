"""What changed between the two estates returns, and whether the meter saw it: for every panel
site without CHP that reports a change from ERIC 2023/24 to 2024/25, the change in LED coverage
(percentage points, x) against the largest step in its daily demand within or near the 2024/25
reporting year (per cent, y); new solar and new heat pumps marked. Sites without a detected
step sit on the dashed zero line.

analysis/estates_changes.py (eric_changes.csv, demand_steps.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT, RESULTS

from sense_energy.visualization import style

m = pd.read_csv(RESULTS / "eric_changes.csv").set_index("site_code")
m = m[m["any_change"] & ~m["chp_site"]].copy()
m["led_delta"] = m["led_pct_2425"] - m["led_pct_2324"]
m["step_pct"] = 100 * m["step_change"].fillna(0)
rng = np.random.default_rng(0)
m["led_delta_j"] = m["led_delta"] + rng.uniform(-1.5, 1.5, len(m))

fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
plain = m[~m["solar_new"] & ~m["heat_pumps_new"]]
ax.plot(
    plain["led_delta_j"],
    plain["step_pct"],
    linestyle="none",
    marker="o",
    markersize=style.MARKER_SIZE,
    markeredgewidth=style.MARKER_EDGE,
    markeredgecolor="white",
    color="#7A7A7A",
    label=f"LED change only ({len(plain)})",
    zorder=3,
)
sol = m[m["solar_new"]]
ax.plot(
    sol["led_delta_j"],
    sol["step_pct"],
    linestyle="none",
    marker="s",
    markersize=style.MARKER_SIZE + 1,
    markeredgewidth=style.MARKER_EDGE,
    markeredgecolor="white",
    color="#E69F00",
    label=f"solar generation new ({len(sol)})",
    zorder=4,
)
hp = m[m["heat_pumps_new"]]
ax.plot(
    hp["led_delta_j"],
    hp["step_pct"],
    linestyle="none",
    marker="^",
    markersize=style.MARKER_SIZE + 1,
    markeredgewidth=style.MARKER_EDGE,
    markeredgecolor="white",
    color="#D55E00",
    label=f"heat pumps new ({len(hp)})",
    zorder=4,
)
ax.set_xlabel("Change in LED coverage, 2023/24 to 2024/25 (points)")
ax.set_ylabel("Largest demand step in or near 2024/25 (%)")
ax.set_ylim(-60, max(60, float(m["step_pct"].max()) * 1.1))
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.24, bottom=0.34)
led = m[(m["led_delta"] >= 10) & ~m["solar_new"] & ~m["heat_pumps_new"]]
print(
    f"{len(m)} non-CHP sites with a change; LED-only sites with a step: median step {led['step_pct'].median():+.1f} %, {int((led['step_pct'] < 0).sum())} down of {len(led)}"
)
style.save_figure(fig, OUT / "eric_changes", data=m.reset_index())
