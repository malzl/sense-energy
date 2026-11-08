"""How alike sites are by distance: the correlation between pairs of sites of their day-to-day
temperature anomalies, and of their specification-B demand residuals, by distance band. One plot.

Mean pairwise correlation across site pairs in each band (analysis/weather_causal.py); the
number of pairs per band is in the data file.
"""

from _common import OUT, causal

from sense_energy.visualization import style

s = causal("spatial_correlation")
bands = list(dict.fromkeys(s["band_km"]))

fig, ax = style.figure(style.SINGLE_COLUMN)
for name, color in style.SPATIAL_SERIES_COLORS.items():
    d = s[s["series"] == name].set_index("band_km").reindex(bands)
    ax.plot(
        range(len(bands)),
        d["mean_correlation"],
        color=color,
        linewidth=style.LINEWIDTH["primary"],
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        label=name,
    )
style.reference_line(ax, 0.0)
ax.set_xticks(range(len(bands)))
ax.set_xticklabels(bands)
ax.set_ylim(-0.1, 1.0)
ax.set_xlabel("Distance between the two sites (km)")
ax.set_ylabel("Mean correlation between site pairs")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
print(s.round(3).to_string(index=False))
style.save_figure(fig, OUT / "causal_spatial", data=s)
