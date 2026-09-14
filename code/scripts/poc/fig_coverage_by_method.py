"""Empirical coverage of the nominal 80% interval per method, median and IQR across sites. One plot."""

from _common import NAIVE, OUT, dot_iqr, iqr_rows, ordered_models, per_site

from sense_energy.visualization import style

per = per_site()
models = [m for m in ordered_models(per["model"].unique()) if m != NAIVE]  # no interval
frame = iqr_rows(per, "coverage_80", models, scale=100)

fig, ax = style.figure(style.SINGLE_COLUMN)
dot_iqr(ax, frame)
style.reference_line(ax, 80.0, axis="x")
ax.set_xlabel("Coverage (%, target 80)")
ax.set_xlim(0, 100)
style.style_axis(ax, grid="x")
style.save_figure(fig, OUT / "coverage_by_method", data=frame)
