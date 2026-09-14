"""One site, one target day: observed demand, Chronos-2 and LightGBM + IFS ENS medians with 80% bands. One plot."""

import matplotlib.dates as mdates
import pandas as pd
from _common import OUT, display, forecasts

from sense_energy.config import PROCESSED_DIR
from sense_energy.visualization import style

MODELS = ["chronos2", "lightgbm_ifs"]

fc = pd.concat([forecasts(m) for m in MODELS])
SITE = fc.groupby("site_code")["point"].mean().idxmax()  # the largest site in the experiment
fc = fc[fc["site_code"] == SITE]
day = sorted(fc["origin"].unique())[len(fc["origin"].unique()) // 2]  # a mid-window origin
fc = fc[fc["origin"] == day]
y = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet", columns=[SITE])
y.index = pd.to_datetime(y.index, utc=True)
obs = y.loc[fc["target"].min() : fc["target"].max(), SITE] * 2  # kWh/half-hour -> kW

fig, ax = style.figure(style.WIDE_SINGLE)
t_local = obs.index.tz_convert("Europe/London")
ax.plot(
    t_local,
    obs.to_numpy(),
    color="#000000",
    linewidth=style.LINEWIDTH["primary"],
    label="observed",
    zorder=4,
)
for m in MODELS:
    d = fc[fc["model"] == m].sort_values("target")
    tl = d["target"].dt.tz_convert("Europe/London")
    c = style.method_color(display(m))
    ax.fill_between(tl, d["q10"] * 2, d["q90"] * 2, color=c, alpha=style.BAND_ALPHA, linewidth=0)
    ax.plot(tl, d["q50"] * 2, color=c, linewidth=style.LINEWIDTH["secondary"], label=display(m))
ax.set_xlabel(
    f"Local time, {pd.Timestamp(day).tz_convert('Europe/London') + pd.Timedelta(days=1):%d %b %Y}"
)
ax.set_ylabel("Demand (kW)")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz="Europe/London"))
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False)  # method names are product names
style.save_figure(fig, OUT / "example_day", data=fc.assign(observed_kw=fc["target"].map(obs)))
