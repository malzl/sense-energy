"""Finding 3: day-ahead error of each model (left) and the share of the possible peak cut that a
battery gets when it plans on the model's forecast (right). 91 sites, 69 test days."""

import pandas as pd
from _deck import ASH, BLUE, ORANGE, REPORTS, figure, save
from matplotlib.patches import Patch

models = {
    "seasonal_naive": ("Seasonal naive", ASH),
    "lightgbm_ifs": ("LightGBM", ORANGE),
    "tabpfn_ts_ifs": ("TabPFN-TS", BLUE),
    "timesfm3_ifs": ("TimesFM 3.0", BLUE),
    "chronos2_ifs": ("Chronos-2", BLUE),
}
s = pd.read_csv(REPORTS / "poc" / "scores_per_site.csv")
b = pd.read_csv(REPORTS / "poc" / "scores_per_site_baselines.csv")
nmae = pd.concat([s, b]).groupby("model")["nmae"].mean() * 100
fv = pd.read_csv(REPORTS / "flexibility" / "forecast_value.csv")
fv = fv[fv["plan"] == "point"].set_index("model")["captured_share"] * 100
d = pd.DataFrame(
    {
        "model": list(models),
        "label": [v[0] for v in models.values()],
        "colour": [v[1] for v in models.values()],
    }
)
d["nmae_pct"] = d["model"].map(nmae)
d["captured_pct"] = d["model"].map(fv)
d = d.iloc[::-1].reset_index(drop=True)
fig, (a, c) = figure(90, 39, ncols=2, sharey=True)
a.barh(d["label"], d["nmae_pct"], color=d["colour"], height=0.62)
a.set_xlim(0, 20)
a.set_xticks([0, 5, 10, 15, 20])
a.set_xlabel("Day-ahead error (%)")
c.barh(d["label"], d["captured_pct"], color=d["colour"], height=0.62)
c.set_xlim(0, 100)
c.set_xticks([0, 25, 50, 75, 100])
c.set_xlabel("Share of possible peak cut (%)")
for ax in (a, c):
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
handles = [
    Patch(color=col, label=lab)
    for col, lab in (
        (ASH, "baseline"),
        (ORANGE, "trained on the sites"),
        (BLUE, "zero-shot, no site training"),
    )
]
fig.legend(handles=handles, loc="outside upper center", ncol=3, handlelength=1.0)
save(fig, "forecast", data=d[["model", "label", "nmae_pct", "captured_pct"]])
print(d[["label", "nmae_pct", "captured_pct"]].round(1).to_string())
