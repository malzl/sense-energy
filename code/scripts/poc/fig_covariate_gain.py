"""Change in CRPS skill from adding the known-ahead covariates (+ ERA5: perfect weather;
+ IFS ENS: forecast weather) relative to the family's plain variant, paired per site:
median and IQR across sites. One plot."""

import pandas as pd
from _common import OUT, display, dot_iqr, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
keys = ordered_models(per["model"].unique())
rows = []
for k in keys:
    name = display(k)
    if style.method_variant(name) == "none":
        continue
    family = style.method_family(name)
    base = next((b for b in keys if display(b) == family), None)
    if base is None:
        continue
    a = per[per["model"] == k].set_index("site_code")["crps_skill_vs_naive"]
    b = per[per["model"] == base].set_index("site_code")["crps_skill_vs_naive"]
    d = (a - b.reindex(a.index)).dropna() * 100
    rows.append(
        {
            "model": k,
            "method": name,
            "median": d.median(),
            "q25": d.quantile(0.25),
            "q75": d.quantile(0.75),
            "n_sites": len(d),
            "share_improved": float((d > 0).mean()),
        }
    )
frame = pd.DataFrame(rows)

fig, ax = style.figure(style.SINGLE_COLUMN)
dot_iqr(ax, frame)
style.reference_line(ax, 0.0, axis="x")
ax.set_xlabel("Change in CRPS skill vs the plain variant (percentage points)")
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "covariate_gain", data=frame)
