"""Organisation type by cluster for electricity sites: marker area proportional to the count. One plot."""

from _common import OUT, clusters

from sense_energy.visualization import style

table = clusters("elec", "site")
counts = table.groupby(["organisation_type", "cluster"]).size().unstack(fill_value=0)
counts = counts.loc[counts.sum(axis=1).sort_values(ascending=True).index]
frame = counts.stack().rename("n").reset_index()

fig, ax = style.figure(style.SINGLE_COLUMN)
y = {t: i for i, t in enumerate(counts.index)}
for _, r in frame.iterrows():
    if r["n"] == 0:
        continue
    ax.scatter(
        r["cluster"],
        y[r["organisation_type"]],
        s=8 + 12 * r["n"],
        color=style.cluster_color(r["cluster"]),
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        zorder=3,
    )
ax.set_yticks(list(y.values()))
ax.set_yticklabels([t.title().replace("And", "and") for t in y])
ax.set_xticks(sorted(frame["cluster"].unique()))
ax.set_xlabel("Cluster")
ax.set_xlim(-0.6, frame["cluster"].max() + 0.6)
for n in (1, 10, 40):
    ax.scatter(
        [],
        [],
        s=8 + 12 * n,
        color="#7A7A7A",
        edgecolor="white",
        linewidth=style.MARKER_EDGE,
        label=f"{n} sites",
    )
style.style_axis(ax, grid="both")
style.add_bottom_legend(ax, anchor_y=-0.22, bottom=0.30)
style.save_figure(fig, OUT / "cluster_composition_elec_site", data=frame)
