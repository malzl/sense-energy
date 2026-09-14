"""CRPS skill against seasonal naive per candidate method: median and IQR across sites. One plot."""

from _common import NAIVE, OUT, dot_iqr, iqr_rows, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
models = [m for m in ordered_models(per["model"].unique()) if m != NAIVE]
frame = iqr_rows(per, "crps_skill_vs_naive", models)

fig, ax = style.figure(style.SINGLE_COLUMN)
dot_iqr(ax, frame)
style.reference_line(ax, 0.0, axis="x")
ax.set_xlabel("CRPS skill vs seasonal naive (1 − CRPS/CRPS$_{naive}$)")
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "crps_skill_by_method", data=frame)
