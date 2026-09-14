"""Sharpness: mean width of the 80% interval as a share of mean demand, median and IQR across sites. One plot."""

from _common import NAIVE, OUT, dot_iqr, iqr_rows, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
per["width_share"] = per["width_80"] / per["y_mean"]
models = [m for m in ordered_models(per["model"].unique()) if m != NAIVE]
frame = iqr_rows(per, "width_share", models)

fig, ax = style.figure(style.SINGLE_COLUMN)
dot_iqr(ax, frame)
ax.set_xlabel("80% interval width / mean demand")
ax.set_xlim(left=0)
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "width_by_method", data=frame)
