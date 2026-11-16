"""What event days cost the forecast in value: captured share of the achievable peak cut on
heat-wave and cold-spell days minus the same on the other days of the same site and month, by
summer and winter, for two definitions of an event day. 95 % intervals from resampling sites;
seasons with fewer than 20 sites are left out.

analysis/forecast_record.py (event_days.csv).
"""

from _common import OUT, best_model, table

from sense_energy.visualization import style

e = table("event_days")
model = best_model(table("by_period"))
e = e[(e["model"] == model) & (e["sites_group"] == "all sites") & (e["sites"] >= 20)]
e = e[e["season"] != "all"].sort_values(["type", "season"])
seasons = list(dict.fromkeys(e["season"]))
fig, ax = style.figure(style.WIDE_SINGLE)
style.reference_line(ax, 0.0)
for j, (name, marker) in enumerate(style.EVENT_DEFINITION_MARKERS.items()):
    d = e[e["definition"] == name].set_index("season").reindex(seasons)
    colors = [style.WEATHER_EVENT_COLORS[t] if isinstance(t, str) else "none" for t in d["type"]]
    for k, season in enumerate(seasons):
        r = d.loc[season]
        if r.isna().all():
            continue
        ax.errorbar(
            k + (j - 0.5) * 0.22,
            r["captured_diff_pts"],
            yerr=[
                [r["captured_diff_pts"] - r["captured_diff_lo"]],
                [r["captured_diff_hi"] - r["captured_diff_pts"]],
            ],
            fmt=marker,
            markersize=style.MARKER_SIZE + 1,
            color=colors[k],
            elinewidth=style.LINEWIDTH["secondary"],
            capsize=0,
            label=style.EVENT_DEFINITION_NAMES[name] if k == 0 else None,
        )
ax.set_xticks(range(len(seasons)))
ax.set_xticklabels([s.replace(" ", "\n") for s in seasons])
ax.set_ylabel("Captured peak cut, event days minus other days (points)")
style.style_axis(ax, grid="y")
leg = style.add_bottom_legend(ax, ncol=2, anchor_y=-0.24, bottom=0.3)
for h in leg.legend_handles:
    h.set_color("#7A7A7A")
style.save_figure(fig, OUT / "record_event_value", data=e)
