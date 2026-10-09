"""Acute trusts: electricity 2023/24 (ERIC, sum over sites) against occupied bed-days (KH03,
all sectors, overnight), trusts in our panel highlighted. One plot."""

import pandas as pd
from _common import OUT, RESULTS, plain_log_ticks

from sense_energy.config import INTERIM_DIR
from sense_energy.visualization import style

t = pd.read_csv(RESULTS / "trust_intensity.csv")
t = t[
    t["trust_type"].astype(str).str.upper().str.contains("ACUTE")
    & (t["elec_kwh"] > 0)
    & (t["occupied_beds_all"] > 10)
].copy()
t["bed_days"] = t["occupied_beds_all"] * 366
ours = set(pd.read_parquet(INTERIM_DIR / "nhs_activity_trusts.parquet")["ods_code"].dropna())
t["in_panel"] = t["trust_code"].isin(ours)

fig, ax = style.figure(style.SINGLE_COLUMN)
for flag, name, colour, z in (
    (False, "other acute trusts", "#B8B8B8", 2),
    (True, "trusts in our panel", "#0072B2", 3),
):
    d = t[t["in_panel"] == flag]
    ax.plot(
        d["bed_days"] / 1e3,
        d["elec_kwh"] / 1e6,
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=colour,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=f"{name} (n = {len(d)})",
        zorder=z,
    )
ax.set_xscale("log")
ax.set_yscale("log")
plain_log_ticks(ax.xaxis, [10, 30, 100, 300, 1000])
plain_log_ticks(ax.yaxis, [2, 5, 10, 20, 50])
ax.set_xlabel("Occupied bed-days 2023/24 (thousands)")
ax.set_ylabel("Electricity 2023/24 (GWh)")
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "bed_day_intensity", data=t)
