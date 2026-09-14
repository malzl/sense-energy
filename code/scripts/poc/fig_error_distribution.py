"""Distribution of the signed point-forecast error as a share of mean demand, per method
(boxes: IQR, whiskers 5-95%), pooled over sites, origins and half-hours. One plot."""

from _common import OUT, boxes, display, joined, ordered_models

from sense_energy.visualization import style

f = joined()
models = ordered_models(f["model"].unique())
methods = [display(m) for m in models]
groups = [f.loc[f["model"] == m, "err"].to_numpy() * 100 for m in models]

fig, ax = style.figure(style.WIDE_SHORT)
boxes(ax, groups, methods)
style.reference_line(ax, 0.0)
ax.set_ylabel("Forecast − observed (% of mean demand)")
style.style_axis(ax, grid="y", wide_short=True)
fig.subplots_adjust(bottom=0.38)
summary = f.groupby("model")["err"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]) * 100
style.save_figure(
    fig,
    OUT / "error_distribution",
    data=summary.reset_index().assign(method=lambda d: d["model"].map(display)),
)
