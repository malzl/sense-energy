"""Finding 4: heat-wave and cold-wave days against the other days of the same site and month.
Left: day-ahead error. Right: share of the possible peak cut that a battery gets when it plans
on the forecast. 91 sites, every day of the test window, forecast with no weather input. A wave
day has a site temperature beyond the 5th or 95th percentile of the days before the test."""

import pandas as pd
from _deck import BLUE, MID_GREY, ORANGE, REPORTS, figure, save
from matplotlib.patches import Patch

FA = REPORTS / "forecast_anatomy"
e = pd.read_csv(FA / "daily_error_wave_matched.csv")
v = pd.read_csv(FA / "daily_value_wave_matched.csv")
e, v = e[e["group"] == "all sites"], v[v["group"] == "all sites"]
rows = [
    ("heat wave", "heat wave", "Heat-wave days", ORANGE),
    ("heat wave", "other days", "Other days", MID_GREY),
    ("cold wave", "cold wave", "Cold-wave days", BLUE),
    ("cold wave", "other days", "Other days ", MID_GREY),
]
d = pd.DataFrame(rows, columns=["matched_to", "wave", "label", "colour"])
key = ["matched_to", "wave"]
d = d.merge(e[[*key, "site_days", "nmae_pct"]], on=key).merge(
    v[[*key, "captured_point_pct"]], on=key
)
d = d.iloc[::-1].reset_index(drop=True)
pos = [0, 1, 2.6, 3.6]
fig, (a, c) = figure(90, 39, ncols=2, sharey=True)
a.barh(pos, d["nmae_pct"], color=d["colour"], height=0.72)
a.set_xlim(0, 20)
a.set_xticks([0, 5, 10, 15, 20])
a.set_xlabel("Day-ahead error (%)")
c.barh(pos, d["captured_point_pct"], color=d["colour"], height=0.72)
c.set_xlim(0, 100)
c.set_xticks([0, 25, 50, 75, 100])
c.set_xlabel("Share of possible peak cut (%)")
a.set_yticks(pos)
a.set_yticklabels(d["label"])
for ax in (a, c):
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
handles = [
    Patch(color=ORANGE, label="heat wave"),
    Patch(color=BLUE, label="cold wave"),
    Patch(color=MID_GREY, label="other days, same site and month"),
]
fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.0)
cols = ["matched_to", "wave", "site_days", "nmae_pct", "captured_point_pct"]
save(fig, "waves", data=d[cols])
print(d[cols].round(1).to_string(index=False))
