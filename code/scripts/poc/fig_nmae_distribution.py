"""Distribution across sites of the normalised MAE per method (boxes: IQR, whiskers 5-95%). One plot."""

from _common import OUT, boxes, display, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
models = ordered_models(per["model"].unique())
methods = [display(m) for m in models]
groups = [per.loc[per["model"] == m, "nmae"].dropna().to_numpy() * 100 for m in models]

fig, ax = style.figure(style.WIDE_SHORT)
boxes(ax, groups, methods)
ax.set_ylabel("MAE / mean demand (%)")
ax.set_ylim(bottom=0)
style.style_axis(ax, grid="y", wide_short=True)
fig.subplots_adjust(bottom=0.38)
data = per[["model", "site_code", "nmae"]].assign(method=per["model"].map(display))
style.save_figure(fig, OUT / "nmae_distribution", data=data)
