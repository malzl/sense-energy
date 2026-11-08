"""Shared paths for the weather figures (one plot per script)."""

from __future__ import annotations

import pandas as pd

from sense_energy.config import INTERIM_DIR, PROJECT_ROOT

EVENTS = PROJECT_ROOT / "code" / "reports" / "weather_events"
COOLING = PROJECT_ROOT / "code" / "reports" / "cooling"
OUT = PROJECT_ROOT / "code" / "reports" / "figures" / "weather"


def weather_daily() -> pd.DataFrame:
    return pd.read_parquet(INTERIM_DIR / "weather_daily.parquet")


def national_events() -> pd.DataFrame:
    return pd.read_csv(EVENTS / "national_events.csv", parse_dates=["start", "end"])


RESPONSE = PROJECT_ROOT / "code" / "reports" / "weather_response"


def response(name: str) -> pd.DataFrame:
    return pd.read_csv(RESPONSE / f"{name}.csv")


CAUSAL = PROJECT_ROOT / "code" / "reports" / "weather_causal"


def causal(name: str) -> pd.DataFrame:
    return pd.read_csv(CAUSAL / f"{name}.csv")


def plot_response_curve(variable: str, name: str, xlabel: str, ylabel: str) -> None:
    """Binned treatment-response curve of sites without CHP under specifications A, B and D
    (analysis/weather_causal.py): one plot, 95 % intervals two-way clustered by site and date."""
    from sense_energy.visualization import style

    c = causal("response_curve")
    c = c[c["variable"] == variable]
    fig, ax = style.figure(style.SINGLE_COLUMN)
    for spec in ("A", "B", "D"):
        d = c[c["spec"] == spec].sort_values("bin_mid")
        color = style.SPEC_COLORS[spec]
        ax.fill_between(
            d["bin_mid"], d["lo_pct"], d["hi_pct"], color=color, alpha=style.BAND_ALPHA, linewidth=0
        )
        ax.plot(
            d["bin_mid"],
            d["coef_pct"],
            color=color,
            linewidth=style.LINEWIDTH["primary"] if spec != "A" else style.LINEWIDTH["secondary"],
            marker="o",
            markersize=style.MARKER_SIZE - 1,
            markeredgewidth=style.MARKER_EDGE,
            markeredgecolor="white",
            label=style.SPEC_NAMES[spec],
        )
    style.reference_line(ax, 0.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    style.style_axis(ax)
    style.add_bottom_legend(ax)
    print(c.pivot_table(index="bin", columns="spec", values="coef_pct").round(1).to_string())
    style.save_figure(fig, OUT / name, data=c)


THERMO = PROJECT_ROOT / "code" / "reports" / "thermometers"


def thermometers(name: str) -> pd.DataFrame:
    return pd.read_csv(THERMO / f"{name}.csv")


def england_axes(fig_size=None):
    """A square map axis with the England outline, aspect-corrected, axis off."""
    import numpy as np

    from sense_energy.eda import common as C
    from sense_energy.visualization import style

    fig, ax = style.figure(fig_size or style.SQUARE)
    eng = C.england()
    eng.plot(ax=ax, facecolor="#F1F0EA", edgecolor="#C3C2B7", linewidth=0.5, zorder=1)
    minx, miny, maxx, maxy = eng.total_bounds
    ax.set_xlim(minx - 0.2, maxx + 0.2)
    ax.set_ylim(miny - 0.1, maxy + 0.1)
    ax.set_aspect(1 / np.cos(np.deg2rad(53)))
    ax.set_axis_off()
    return fig, ax


def map_colorbar(fig, ax, mappable, label: str, **kw):
    from sense_energy.visualization import style

    cb = fig.colorbar(mappable, ax=ax, fraction=0.035, pad=0.02, **kw)
    cb.set_label(label, fontsize=style.FONT_SIZE["label"])
    cb.ax.tick_params(
        labelsize=style.FONT_SIZE["tick"], width=style.TICK_WIDTH, length=style.TICK_LENGTH
    )
    cb.outline.set_linewidth(style.AXIS_LINEWIDTH)
    return cb


def site_points() -> pd.DataFrame:
    g = pd.read_parquet(PROJECT_ROOT / "code" / "data" / "geo" / "sites_geo.parquet")
    return g[g["latitude"].notna()][["site_code", "site_name", "latitude", "longitude"]]


HETERO = PROJECT_ROOT / "code" / "reports" / "weather_heterogeneity"


def heterogeneity(name: str) -> pd.DataFrame:
    return pd.read_csv(HETERO / f"{name}.csv")


HEAT = PROJECT_ROOT / "code" / "reports" / "heat_drivers"


def heat_drivers(name: str) -> pd.DataFrame:
    return pd.read_csv(HEAT / f"{name}.csv")


def heat_features() -> pd.DataFrame:
    """Site features with slopes in percentage points (log points x 100)."""
    f = pd.read_csv(HEAT / "site_features.csv").set_index("site_code")
    num = f.select_dtypes("number").columns
    pct = [c for c in num if c.startswith(("heat", "sun", "cold", "hot_afternoon", "warm_night"))]
    f[[f"{c}_pct" for c in pct]] = 100 * f[pct]
    return f
