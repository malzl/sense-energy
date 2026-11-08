"""How far each gridded thermometer sits from the nearest Met Office station: per-site RMSE of the
day-to-day temperature anomaly (within site x month) against the station, by station distance.
One plot.

Points are sites; the anomaly is what identifies the weather effects, so this is the error that
matters (analysis/thermometers.py).
"""

from _common import OUT, thermometers

from sense_energy.visualization import style

a = thermometers("agreement")

fig, ax = style.figure(style.SINGLE_COLUMN)
for name in style.THERMOMETER_ORDER:
    d = a[a["thermometer"] == name]
    if d.empty:
        continue
    ax.plot(
        d["station_km"],
        d["anomaly_rmse_c"],
        linestyle="none",
        marker="o",
        markersize=style.MARKER_SIZE,
        markeredgewidth=style.MARKER_EDGE,
        markeredgecolor="white",
        color=style.THERMOMETER_COLORS[name],
        label=f"{style.THERMOMETER_NAMES[name]}, median {d['anomaly_rmse_c'].median():.2f} °C",
    )
ax.set_ylim(bottom=0)
ax.set_xlabel("Distance from the site to its station (km)")
ax.set_ylabel("RMSE of the daily anomaly vs the station (°C)")
style.style_axis(ax, grid="y")
style.add_bottom_legend(ax, ncol=1, anchor_y=-0.25, bottom=0.34)
print(
    a.groupby("thermometer")[
        ["bias_mean_c", "rmse_mean_c", "rmse_max_c", "anomaly_corr", "anomaly_rmse_c"]
    ]
    .median()
    .round(3)
    .to_string()
)
style.save_figure(fig, OUT / "thermometer_agreement", data=a)
