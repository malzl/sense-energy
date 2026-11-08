"""The JEPA site embedding in two dimensions (first two principal components), one point per
site, coloured by the sign of the site's daily heat slope and marked by building use; marker
area by mean demand. Sites that rise with heat should sit apart from those that fall if the
embedding carries the response.

experiments/jepa.py (site_embeddings.parquet), analysis/heat_drivers.py (site_features.csv).
"""

import numpy as np
import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT, heat_features
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from sense_energy.visualization import style

e = pd.read_parquet(PROJECT_ROOT / "code" / "reports" / "jepa" / "site_embeddings.parquet")
cols = [c for c in e.columns if c.startswith("e")]
z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(e[cols]))
d = pd.DataFrame(z, index=e.index, columns=["pc1", "pc2"])
f = heat_features()
d["sign"] = f["sign"].reindex(d.index).fillna("not in the heat sample")
d["mean_kw"] = f["mean_kw"].reindex(d.index)
size = np.clip(
    2.5 + 1.2 * np.sqrt(d["mean_kw"].fillna(d["mean_kw"].median()) / d["mean_kw"].median()),
    2.5,
    7.0,
)
colors = {**style.HEAT_SIGN_COLORS, "not in the heat sample": "#D0D0D0"}

fig, ax = style.figure(style.SQUARE)
for sign in [*style.HEAT_SIGN_ORDER, "not in the heat sample"]:
    s = d[d["sign"] == sign]
    if s.empty:
        continue
    for (_idx, r), ms in zip(s.iterrows(), size[s.index], strict=True):
        ax.plot(
            r["pc1"],
            r["pc2"],
            linestyle="none",
            marker="o",
            markersize=ms,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            color=colors[sign],
            alpha=0.9,
            zorder=3 if sign != "not in the heat sample" else 2,
        )
    ax.plot(
        [],
        [],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        color=colors[sign],
        label=f"{sign} ({len(s)})",
    )
ax.set_xlabel("JEPA embedding, first principal component")
ax.set_ylabel("Second principal component")
style.style_axis(ax)
style.add_bottom_legend(ax, ncol=2, anchor_y=-0.22, bottom=0.3)
style.save_figure(fig, OUT / "jepa_map", data=d.reset_index())
