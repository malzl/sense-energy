"""What the causal forest splits on: feature importance of the site covariates and calendar
modifiers for the heat effect, both covariate sets side by side (top 15 of the set with the
signature).

analysis/heat_causal_ml.py (importance_observables.csv, importance_signature.csv).
"""

import numpy as np
import pandas as pd
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

D = PROJECT_ROOT / "code" / "reports" / "heat_causal_ml"
a = pd.read_csv(D / "importance_observables.csv").set_index("covariate")["importance"]
b = pd.read_csv(D / "importance_signature.csv").set_index("covariate")["importance"]
top = b.sort_values(ascending=False).head(15).index
t = pd.DataFrame({"observables": a.reindex(top), "observables + signature": b.reindex(top)}).fillna(
    0
)
t = t.iloc[::-1]
y = np.arange(len(t))

fig, ax = style.figure(style.SINGLE_COLUMN)
ax.barh(y + 0.18, t["observables"], height=0.36, color="#7A7A7A", label="observables", zorder=3)
ax.barh(
    y - 0.18,
    t["observables + signature"],
    height=0.36,
    color=style.TERM_COLORS["cdd20"],
    label="observables + signature",
    zorder=3,
)
ax.set_yticks(y)
ax.set_yticklabels([s.replace("_", " ") for s in t.index], fontsize=style.FONT_SIZE["tick"] - 1)
ax.set_xlabel("Causal-forest feature importance")
style.style_axis(ax, grid="x")
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.2, bottom=0.22)
style.save_figure(fig, OUT / "cate_importance", data=t.reset_index())
