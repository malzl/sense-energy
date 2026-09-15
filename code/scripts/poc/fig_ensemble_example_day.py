"""The largest site, on its target day with the widest member spread: the 51 member medians of Chronos-2 driven by each IFS ENS member
(grey), the mixture's median and 80% band, and the observed demand. One plot."""

import matplotlib.dates as mdates
import pandas as pd
from _common import OUT, RESULTS, display, forecasts

from sense_energy.config import PROCESSED_DIR
from sense_energy.visualization import style

MODEL = "chronos2_ifs_members"
TZ = "Europe/London"

members = pd.read_parquet(RESULTS / "ensemble_members_chronos2.parquet")
members["target"] = pd.to_datetime(members["target"], utc=True)
members["origin"] = pd.to_datetime(members["origin"], utc=True)
mix = forecasts(MODEL)
mix["origin"] = pd.to_datetime(mix["origin"], utc=True)
# the largest site in the experiment, on the origin where its member spread is widest
site_mean = members.groupby("site_code")["median"].mean()
SITE = site_mean.idxmax()
spread = members[members["site_code"] == SITE].groupby("origin")["median"].std()
DAY = spread.idxmax()
mm = members[(members["site_code"] == SITE) & (members["origin"] == DAY)]
mx = mix[(mix["site_code"] == SITE) & (mix["origin"] == DAY)].sort_values("target")
y = pd.read_parquet(PROCESSED_DIR / "elec" / "site" / "wide_raw.parquet", columns=[SITE])
y.index = pd.to_datetime(y.index, utc=True)
obs = y.loc[mx["target"].min() : mx["target"].max(), SITE] * 2  # kWh per half hour -> kW

fig, ax = style.figure(style.WIDE_SINGLE)
for k, (_, d) in enumerate(mm.groupby("member")):
    d = d.sort_values("target")
    ax.plot(
        d["target"].dt.tz_convert(TZ),
        d["median"] * 2,
        color=style.SPAGHETTI,
        linewidth=0.5,
        alpha=0.7,
        zorder=2,
        label="member medians" if k == 0 else None,
    )
c = style.method_color(display(MODEL))
tl = mx["target"].dt.tz_convert(TZ)
ax.fill_between(
    tl, mx["q10"] * 2, mx["q90"] * 2, color=c, alpha=style.BAND_ALPHA, linewidth=0, zorder=3
)
ax.plot(
    tl,
    mx["q50"] * 2,
    color=c,
    linewidth=style.LINEWIDTH["secondary"],
    label=display(MODEL),
    zorder=4,
)
ax.plot(
    obs.index.tz_convert(TZ),
    obs.to_numpy(),
    color="#000000",
    linewidth=style.LINEWIDTH["primary"],
    label="observed",
    zorder=5,
)
ax.set_xlabel(f"Local time, {(pd.Timestamp(DAY).tz_convert(TZ) + pd.Timedelta(days=1)):%d %b %Y}")
ax.set_ylabel("Demand (kW)")
ax.set_ylim(bottom=0)
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=TZ))
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False)  # product names
style.save_figure(fig, OUT / "ensemble_example_day", data=mm.assign(site_code=SITE))
