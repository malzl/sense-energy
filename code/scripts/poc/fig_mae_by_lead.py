"""Normalised MAE by lead time for the plain variant of each family, SARIMA and the baselines. One plot."""

from _common import OUT, by_lead, display, per_site, plain

from sense_energy.visualization import style

lead = by_lead()
scale = per_site().groupby("model")["y_mean"].mean()  # site-mean demand, to express MAE as a share
lead["nmae"] = lead["mae"] / lead["model"].map(scale)
models = plain(lead["model"].unique())
lead = lead[lead["model"].isin(models)]

fig, ax = style.figure(style.SINGLE_COLUMN)
for m in models:
    d = lead[lead["model"] == m].sort_values("lead_bucket")
    name = display(m)
    baseline = style.METHOD_ROLES[name] == "simple_baseline"
    ax.plot(
        d["lead_bucket"] + 3,
        d["nmae"],
        color=style.method_color(name),
        linestyle=style.method_linestyle(name),
        linewidth=style.LINEWIDTH["reference"] if baseline else style.LINEWIDTH["primary"],
        marker=style.method_marker(name),
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=name,
    )
ax.set_xlabel("Lead time from 16:30 issue (hours, 6 h bins)")
ax.set_ylabel("MAE / mean demand")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36)  # product names
style.save_figure(fig, OUT / "mae_by_lead", data=lead.assign(method=lead["model"].map(display)))
