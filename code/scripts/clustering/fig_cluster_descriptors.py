"""Electricity sites by night/day ratio and weekend/weekday ratio, coloured by cluster, marker
area proportional to mean demand. One plot."""

from _common import OUT, clusters

from sense_energy.visualization import style

table = clusters("elec", "site")
area = 6 + 120 * table["mean_kw"] / table["mean_kw"].max()

fig, ax = style.figure(style.SINGLE_COLUMN)
for i in sorted(table["cluster"].unique()):
    d = table[table["cluster"] == i]
    ax.scatter(
        d["night_day_ratio"],
        d["weekend_ratio"],
        s=area[d.index],
        color=style.cluster_color(i),
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        alpha=0.85,
        zorder=3,
        label=f"cluster {i} (n = {len(d)})",
    )
ax.set_xlabel("Night / day demand (02:00–05:00 over 10:00–16:00)")
ax.set_ylabel("Weekend / weekday demand")
ax.set_xlim(left=0)
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="both")
style.add_bottom_legend(ax, ncol=3, anchor_y=-0.26, bottom=0.34)
style.save_figure(
    fig,
    OUT / "cluster_descriptors_elec_site",
    data=table.reset_index()[
        ["series_id", "cluster", "night_day_ratio", "weekend_ratio", "mean_kw"]
    ],
)
