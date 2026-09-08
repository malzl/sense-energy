"""Share of half-hours observed per site and month (elec, raw). One plot."""

from sense_energy.eda import common as C
from sense_energy.visualization import style

ENERGY = "elec"

wide, index = C.series_for(ENERGY, "site", flavour="raw")
monthly = wide.resample("MS").count() / wide.resample("MS").size().to_numpy()[:, None]
order = index.sort_values("first")["series_id"]
monthly = monthly[[s for s in order if s in monthly.columns]]

fig, ax = style.figure(style.WIDE_SINGLE)
im = ax.imshow(
    monthly.T.to_numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
)
months = monthly.index
tick_pos = [i for i, m in enumerate(months) if m.month == 1]
ax.set_xticks(tick_pos)
ax.set_xticklabels([months[i].strftime("%Y") for i in tick_pos])
ax.set_yticks([])
ax.set_xlabel("Month")
ax.set_ylabel(f"Sites, ordered by first reading (n = {monthly.shape[1]})")
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cb.set_label("Share of half-hours observed")
cb.ax.tick_params(
    labelsize=style.FONT_SIZE["tick"], width=style.TICK_WIDTH, length=style.TICK_LENGTH
)
cb.outline.set_linewidth(style.AXIS_LINEWIDTH)
style.style_axis(ax, all_spines=True)

out = monthly.T.reset_index().rename(columns={"index": "series_id"})
out.columns = ["series_id"] + [m.strftime("%Y-%m") for m in months]
style.save_figure(fig, C.OUT / f"availability_{ENERGY}_site", data=out)
