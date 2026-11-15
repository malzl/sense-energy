"""Each site's reaction on the map (argument: heatwave or cold): demand on its event days
against its reference days, diverging colours centred on zero and cut at +/- 25 %. Sites with
on-site generation have a black edge.

analysis/extreme_events.py (site_reactions.csv).
"""

from _common import OUT, PLAIN, england_axes, kind_arg, table

from sense_energy.visualization import style

kind, tag = kind_arg()
s = table("site_reactions")
s = s[(s["type"] == kind) & s["clean"]].dropna(subset=["latitude", "longitude"])
fig, ax = england_axes()
for plant, edge in ((False, "white"), (True, "black")):
    d = s[(s["chp"] != PLAIN) == plant]
    sc = ax.scatter(
        d["longitude"],
        d["latitude"],
        c=d["reaction_pct"],
        cmap="RdBu_r",
        vmin=-25,
        vmax=25,
        s=20,
        edgecolor=edge,
        linewidth=style.MARKER_EDGE,
        zorder=4 if plant else 3,
    )
cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02, extend="both")
cb.set_label(f"Reaction of demand in a {style.EVENT_NAMES[kind]} (%)", fontsize=8)
cols = ["site_code", "latitude", "longitude", "reaction_pct", "degrees", "chp"]
style.save_figure(fig, OUT / f"map_reaction_{tag}", data=s[cols])
