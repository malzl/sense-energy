"""The largest panel site in the headroom scenario, over the coldest week of 2023/24: metered
demand today and with all gas heating from heat pumps; dashed line = available capacity. One plot."""

import matplotlib.dates as mdates
import pandas as pd
from _common import OUT, RESULTS, headroom_panel

from sense_energy.visualization import style

h = headroom_panel()
site = h.sort_values("elec_kwh_fy").iloc[-1]
p = pd.read_parquet(RESULTS / "heat_pump_profiles.parquet")
p = p[p["site_code"] == site["site_code"]].copy()
p["datetime"] = pd.to_datetime(p["datetime"], utc=True)
p = p.set_index("datetime").sort_index()
daily = p["temp_c"].resample("D").mean()
start = daily.rolling(7).mean().idxmin() - pd.Timedelta(days=6)
w = p.loc[start : start + pd.Timedelta(days=7)]
tl = w.index.tz_convert("Europe/London")

fig, ax = style.figure(style.WIDE_SINGLE)
ax.plot(
    tl,
    w["now_kw"],
    color=style.SCENARIO_COLORS["today"],
    linewidth=style.LINEWIDTH["secondary"],
    label="today",
)
ax.plot(
    tl,
    w["hp_kw"],
    color=style.SCENARIO_COLORS["with heat pumps"],
    linewidth=style.LINEWIDTH["primary"],
    label="with heat pumps",
)
style.reference_line(ax, float(site["capacity_kw"])).set_label("available capacity")
ax.set_ylabel("Demand (kW)")
ax.set_xlabel(f"Local time, week of {tl[0]:%d %b %Y} ({site['site_code']})")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%a", tz="Europe/London"))
style.style_axis(ax)
style.add_bottom_legend(ax)
style.save_figure(fig, OUT / "heat_pump_week", data=w.reset_index())
