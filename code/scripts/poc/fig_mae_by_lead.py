"""Normalised MAE by lead time for each candidate method. One plot."""

from _common import OUT, by_lead, display, ordered_models, per_site

from sense_energy.visualization import style

lead = by_lead()
scale = per_site().groupby("model")["y_mean"].mean()  # site-mean demand, to express MAE as a share
lead["nmae"] = lead["mae"] / lead["model"].map(scale)
models = ordered_models(lead["model"].unique())

fig, ax = style.figure(style.SINGLE_COLUMN)
for m in models:
    d = lead[lead["model"] == m].sort_values("lead_bucket")
    name = display(m)
    ax.plot(
        d["lead_bucket"] + 3,
        d["nmae"],
        color=style.method_color(name),
        linewidth=style.LINEWIDTH["primary"]
        if "naive" not in name.lower() and "Profile" not in name
        else style.LINEWIDTH["reference"],
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgecolor="white",
        markeredgewidth=style.MARKER_EDGE,
        label=name,
    )
ax.set_xlabel("Lead time from 16:30 issue (hours, 6 h bins)")
ax.set_ylabel("MAE / mean demand")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y")
style.add_bottom_legend(
    ax, lowercase=False, ncol=3, anchor_y=-0.26, bottom=0.36
)  # method names are product names
style.save_figure(fig, OUT / "mae_by_lead", data=lead.assign(method=lead["model"].map(display)))
