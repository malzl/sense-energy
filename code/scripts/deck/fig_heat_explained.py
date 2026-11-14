"""Finding 2, chart: share of the true spread of the heat response between sites that each set
of site descriptors explains (random-effects meta-regression). Sets that explain nothing are
drawn at zero; the data file keeps the raw values."""

import pandas as pd
from _deck import MID_GREY, ORANGE, REPORTS, figure, save

v = pd.read_csv(REPORTS / "heat_drivers" / "variance_explained.csv").set_index("set")
rows = {
    "building use": "Building use",
    "urban form and density": "Location type",
    "estates descriptors": "Estates return",
    "cooling plant (DEC)": "Energy certificate",
    "load shape": "Load statistics",
    "diurnal signature": "Time-of-day pattern\nof the meter",
}
d = pd.DataFrame({"label": list(rows.values()), "r2_true": [v.loc[k, "r2_true"] for k in rows]})
d["share_pct"] = 100 * d["r2_true"].clip(lower=0)
d = d.iloc[::-1].reset_index(drop=True)
fig, ax = figure(42, 39)
colours = [ORANGE if s > 20 else MID_GREY for s in d["share_pct"]]
ax.barh(d["label"], d["share_pct"].where(d["share_pct"] > 0, 0.6), color=colours, height=0.62)
ax.set_xlim(0, 50)
ax.set_xticks([0, 25, 50])
ax.set_xlabel("Share explained (%)")
ax.tick_params(axis="y", length=0)
ax.grid(axis="x")
save(fig, "heat_explained", data=d)
