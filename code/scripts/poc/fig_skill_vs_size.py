"""Per-site CRPS skill against site size (mean demand) for the plain variant of each learned
family; one marker per site. One plot."""

from _common import NAIVE, OUT, display, per_site, plain

from sense_energy.visualization import style

per = per_site()
models = [
    m
    for m in plain(per["model"].unique())
    if m != NAIVE and style.METHOD_ROLES[display(m)] != "simple_baseline"
]
frame = per[per["model"].isin(models)][["model", "site_code", "y_mean", "crps_skill_vs_naive"]]
frame = frame.assign(method=frame["model"].map(display))

fig, ax = style.figure(style.SINGLE_COLUMN)
for m in models:
    d = frame[frame["model"] == m]
    name = display(m)
    ax.plot(
        d["y_mean"] * 2,  # kWh per half hour -> kW
        d["crps_skill_vs_naive"],
        linestyle="none",
        marker=style.method_marker(name),
        markersize=style.MARKER_SIZE - 1,
        color=style.method_color(name),
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE * 0.6,
        alpha=0.85,
        label=name,
    )
style.reference_line(ax, 0.0)
ax.set_xscale("log")
ax.set_xlabel("Mean demand (kW)")
ax.set_ylabel("CRPS skill vs seasonal naive")
style.style_axis(ax)
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36)  # product names
style.save_figure(fig, OUT / "skill_vs_size", data=frame)
