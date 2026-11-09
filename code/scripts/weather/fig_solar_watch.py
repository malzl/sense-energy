"""Watching solar go on roofs: each site's sunshine slope (daytime demand change per kWh/m2 of
solar radiation) in the summers of 2023, 2024 and 2025, for the sites whose slope breaks
downwards by more than five points and the sites whose estates returns report new or grown solar;
the latter in orange.

analysis/solar_watch.py (breaks.csv).
"""

import pandas as pd  # noqa: E402
from _common import OUT, PROJECT_ROOT

from sense_energy.visualization import style

b = pd.read_csv(PROJECT_ROOT / "code" / "reports" / "solar_watch" / "breaks.csv").set_index(
    "site_code"
)
eric = b["eric_solar_new_or_grown"].astype("boolean").fillna(False).astype(bool)
show = b[(b["break_down"] & (b["change_pct"] < -5)) | eric]
years = [2023, 2024, 2025]
fig, ax = style.figure(style.SINGLE_COLUMN)
style.reference_line(ax, 0.0)
labels = []
for s, r in show.iterrows():
    v = [r[f"sun_{y}_pct"] for y in years]
    is_eric = bool(eric.get(s, False))
    ax.plot(
        years,
        v,
        color="#E69F00" if is_eric else "#7A7A7A",
        linewidth=1.2 if is_eric else 0.8,
        marker="o",
        markersize=2.5,
        zorder=3 if is_eric else 2,
    )
    last = next((x for x in reversed(v) if pd.notna(x)), None)
    if last is not None:
        labels.append([last, last, str(r["site_name"])[:24]])
# push labels apart to a minimum vertical gap, keeping their order
labels.sort(key=lambda t: t[0])
gap = 1.25
for i in range(1, len(labels)):
    labels[i][1] = max(labels[i][1], labels[i - 1][1] + gap)
for y0, y1, text in labels:
    ax.annotate(
        text,
        xy=(2025, y0),
        xytext=(2025.12, y1),
        va="center",
        fontsize=style.FONT_SIZE["annotation"] - 2,
        color="#444444",
        arrowprops={"arrowstyle": "-", "color": "#BBBBBB", "linewidth": 0.4},
    )
ax.plot([], [], color="#E69F00", label="solar new or grown in the estates returns")
ax.plot([], [], color="#7A7A7A", label="no solar reported")
ax.set_xticks(years)
ax.set_xlim(2022.8, 2026.2)
ax.set_xlabel("Summer (April–September)")
ax.set_ylabel("Sunshine slope (% per kWh/m²)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.22, bottom=0.3)
style.save_figure(fig, OUT / "solar_watch", data=show.reset_index())
