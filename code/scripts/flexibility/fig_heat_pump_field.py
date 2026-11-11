"""Change in gas against change in electricity between the 2023/24 and 2024/25 estates returns,
every English site without CHP on an unchanged floor area: sites that report their first heat
pumps against sites that report none. Changes beyond the axes are drawn at the edge.

analysis/heat_pump_field.py (site_changes.csv).
"""

import numpy as np
import pandas as pd
from _common import HEAT_PUMPS, OUT

from sense_energy.visualization import style

m = pd.read_csv(HEAT_PUMPS / "site_changes.csv")
m = m[m["usable"] & m["status"].isin(["never", "new"])].copy()
m["gas_pct"] = (100 * (m["gas_kwh_b"] / m["gas_kwh_a"] - 1)).clip(-100, 100)
m["elec_pct"] = (100 * (np.exp(m["dlog_elec"]) - 1)).clip(-60, 100)
m["role"] = m["status"].map({"never": "no heat pump", "new": "new heat pumps"})
fig, ax = style.figure(style.SQUARE)
style.reference_line(ax, 0.0)
style.reference_line(ax, 0.0, axis="x")
for role, size, alpha in (
    ("no heat pump", 2, 0.35),
    ("new heat pumps", style.MARKER_SIZE + 1, 1.0),
):
    d = m[m["role"] == role]
    ax.plot(
        d["gas_pct"],
        d["elec_pct"],
        linestyle="none",
        marker="o",
        markersize=size,
        markeredgewidth=0,
        color=style.HEAT_PUMP_STATUS_COLORS[role],
        alpha=alpha,
        label=role,
        zorder=3 if role == "new heat pumps" else 2,
    )
ax.set_xlabel("Change in gas (%)")
ax.set_ylabel("Change in electricity (%)")
style.style_axis(ax)
style.add_bottom_legend(ax, bottom=0.26)
style.save_figure(
    fig, OUT / "heat_pump_field", data=m[["site_code", "role", "gas_pct", "elec_pct"]]
)
