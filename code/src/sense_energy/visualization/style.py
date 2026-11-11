"""Shared figure style - the master figure specification, implemented once.

Every plotting script imports this and never defines its own labels, colours
or sizes. Deviations are allowed only for scientific correctness and must be
recorded in the figure script that makes them.

Summary of the specification this encodes (standing instruction, 2026-08-04,
amended 2026-08-25):

* one independent plot per script; no titles, panel labels, annotations,
  text boxes or decoration - captions and panel letters are added in LaTeX
* exports: ``<base>.pdf`` (vector, transparent, fonts embedded) and
  ``<base>.png`` (white, 600 dpi), plus the plotted data as ``<base>_data.csv``
* sans-serif, never bold; labels 8 pt, ticks 7 pt, legend 7 pt
* axes 0.7 pt, ticks 0.6 pt / 3 pt out, top and right spines removed
* legend outside, centred below the axes, borderless, fixed method order
* colours by conceptual role from an Okabe-Ito palette
* fixed orders for horizons and methods - never by result
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Dimensions (inches). No arbitrary sizes.
# --------------------------------------------------------------------------- #

SINGLE_COLUMN = (3.5, 2.8)  # default
WIDE_SINGLE = (5.2, 3.2)  # many horizons / dates / model names
DOUBLE_COLUMN = (7.2, 3.8)  # only when smaller widths are illegible
SQUARE = (3.5, 3.5)
WIDE_SHORT = (7.2, 2.6)  # distribution / strip / trajectory-construction panels (amendment)

# --------------------------------------------------------------------------- #
# Typography and marks
# --------------------------------------------------------------------------- #

FONT_FAMILY = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]
FONT_SIZE = {"label": 8, "tick": 7, "legend": 7, "annotation": 7}
FONT_SIZE_WIDE_SHORT_TICK = 8  # amendment: tick font for the wide/short format

LINEWIDTH = {"primary": 1.5, "secondary": 1.0, "reference": 0.8}
MARKER_SIZE = 4
MARKER_EDGE = 0.5
BAND_ALPHA = 0.18
MAX_LINESTYLES = 4

AXIS_LINEWIDTH = 0.7
TICK_WIDTH = 0.6
TICK_LENGTH = 3

GRID_LINEWIDTH = 0.5
GRID_ALPHA = 0.15

REFERENCE_LINE = {"color": "#7A7A7A", "linewidth": 0.8, "linestyle": "--"}

# --------------------------------------------------------------------------- #
# Colours - by conceptual role, never by appearance order.
# --------------------------------------------------------------------------- #

COLORS = {  # Okabe-Ito
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
    "grey": "#7A7A7A",
    "light_grey": "#B8B8B8",
}

ROLE_COLORS = {
    "reference": "#000000",
    "simple_baseline": "#7A7A7A",
    "classical_model": "#56B4E9",
    "foundation_model": "#0072B2",
    "specialist_model": "#009E73",
    "exact_bayes": "#E69F00",
    "pfn": "#D55E00",
    "zero_effect": "#B8B8B8",
}

SEQUENTIAL_CMAPS = ("viridis", "cividis", "magma")

#: Entity colours for exploratory figures, by conceptual role. Energy vectors
#: take two categorical slots; day types take the reference/baseline greys so
#: the two pairings never compete inside one figure.
ENERGY_COLORS = {"elec": "#0072B2", "gas": "#E69F00"}
ENERGY_LABELS = {"elec": "electricity", "gas": "gas"}
DAYTYPE_COLORS = {"weekday": "#000000", "weekend": "#7A7A7A"}
SPAGHETTI = "#B8B8B8"  # many background series; the highlighted statistic is black
DIVERGING_CMAPS = ("RdBu_r", "coolwarm")

# --------------------------------------------------------------------------- #
# Nomenclature (binding). Update here, nowhere else.
# --------------------------------------------------------------------------- #

DATASET_NAMES = {
    "nhs": "NHS half-hourly consumption",  # Energy Systems Catapult extract
    "era5": "ERA5",
    "aifs_ens": "AIFS-ENS",
    "agile": "Octopus Agile",
}

#: Display names for this project's candidate methods, in the fixed order used
#: in every figure. Keys are the experiment/model registry names. A method's
#: "+ ERA5" / "+ IFS ENS" variants add the known-ahead covariates (calendar,
#: Agile price, NESO day-ahead forecasts) with perfect (reanalysis) or forecast
#: (00z IFS ENS run of D-1) weather.
METHOD_NAMES = {
    "seasonal_naive": "Seasonal naive",
    "profile_mean": "Profile mean",
    "profile_quantiles": "Profile quantiles",
    "ridge": "Ridge",
    "sarima": "SARIMA",
    "lightgbm": "LightGBM",
    "lightgbm_era5": "LightGBM + ERA5",
    "lightgbm_ifs": "LightGBM + IFS ENS",
    "lightgbm_ifs_noid": "LightGBM + IFS ENS, no site ID",
    "lightgbm_ifs_eric": "LightGBM + IFS ENS + ERIC",
    "chronos2": "Chronos-2",
    "chronos2_era5": "Chronos-2 + ERA5",
    "chronos2_ifs": "Chronos-2 + IFS ENS",
    "chronos2_ifs_ctrl": "Chronos-2 + IFS control",
    "chronos2_ifs_members": "Chronos-2 + IFS members",
    "timesfm3": "TimesFM 3.0",
    "timesfm3_era5": "TimesFM 3.0 + ERA5",
    "timesfm3_ifs": "TimesFM 3.0 + IFS ENS",
    "tabpfn_ts": "TabPFN-TS",
    "tabpfn_ts_era5": "TabPFN-TS + ERA5",
    "tabpfn_ts_ifs": "TabPFN-TS + IFS ENS",
    "tirex2": "TiRex-2",
    "tirex2_era5": "TiRex-2 + ERA5",
    "tirex2_ifs": "TiRex-2 + IFS ENS",
    "t0beta": "T0-beta",
    "t0beta_era5": "T0-beta + ERA5",
    "t0beta_ifs": "T0-beta + IFS ENS",
    "tnp": "TNP",
    "tnp_era5": "TNP + ERA5",
    "tnp_ifs": "TNP + IFS ENS",
    "pttnp": "PT-TNP",
    "pttnp_era5": "PT-TNP + ERA5",
    "pttnp_ifs": "PT-TNP + IFS ENS",
    "gtnp": "Gridded TNP",
    "gtnp_era5": "Gridded TNP + ERA5",
    "gtnp_ifs": "Gridded TNP + IFS ENS",
}
METHOD_ORDER = list(METHOD_NAMES.values())


def method_family(method: str) -> str:
    """'LightGBM + IFS ENS' -> 'LightGBM'."""
    return method.split(" + ")[0]


def method_variant(method: str) -> str:
    """'LightGBM + IFS ENS' -> 'IFS ENS'; a plain method -> 'none'."""
    return " + ".join(method.split(" + ")[1:]) if " + " in method else "none"


FAMILY_ROLES = {
    "Seasonal naive": "simple_baseline",
    "Profile mean": "simple_baseline",
    "Profile quantiles": "simple_baseline",
    "Ridge": "classical_model",
    "SARIMA": "classical_model",
    "LightGBM": "specialist_model",
    "Chronos-2": "foundation_model",
    "TimesFM 3.0": "foundation_model",
    "TabPFN-TS": "pfn",
    "TiRex-2": "foundation_model",
    "T0-beta": "foundation_model",
    "TNP": "pfn",  # meta-learned in-context regressors, like prior-fitted networks
    "PT-TNP": "pfn",
    "Gridded TNP": "pfn",
}
METHOD_ROLES = {m: FAMILY_ROLES[method_family(m)] for m in METHOD_ORDER}

#: Colour = model family, one fixed Okabe-Ito slot each (the two foundation
#: models cannot share the role colour, so TimesFM takes the reddish-purple
#: slot). Covariate variants share their family's colour and differ by marker
#: (points) or line style (lines): plain / + ERA5 / + IFS ENS. Baselines grey.
FAMILY_COLORS = {
    "Seasonal naive": "#7A7A7A",
    "Profile mean": "#B8B8B8",
    "Profile quantiles": "#B8B8B8",
    "Ridge": "#56B4E9",
    "SARIMA": "#56B4E9",
    "LightGBM": "#009E73",
    "Chronos-2": "#0072B2",
    "TimesFM 3.0": "#CC79A7",
    "TabPFN-TS": "#D55E00",
    "TiRex-2": "#E69F00",  # the orange slot is free in this project
    "T0-beta": "#000000",  # last free slot; never plot it next to an observed series
    # Okabe-Ito is exhausted: the neural-process family takes Paul Tol's colour-blind-safe
    # "muted" wine, purple and teal (exception recorded here, not in the scripts)
    "TNP": "#882255",
    "PT-TNP": "#AA4499",
    "Gridded TNP": "#44AA99",
}
METHOD_COLORS = {m: FAMILY_COLORS[method_family(m)] for m in METHOD_ORDER}
VARIANT_MARKERS = {
    "none": "o",
    "ERA5": "s",
    "IFS ENS": "^",
    "IFS control": "v",
    "IFS members": "D",
    "IFS ENS, no site ID": "X",
    "IFS ENS + ERIC": "P",
}
VARIANT_LINESTYLES = {
    "none": "-",
    "ERA5": "--",
    "IFS ENS": ":",
    "IFS control": "-.",
    "IFS members": ":",
    "IFS ENS, no site ID": "-.",
    "IFS ENS + ERIC": "-.",
}
METHOD_MARKERS = {m: VARIANT_MARKERS[method_variant(m)] for m in METHOD_ORDER}
METHOD_LINESTYLES = {m: VARIANT_LINESTYLES[method_variant(m)] for m in METHOD_ORDER}

#: Reconciliation approaches (hierarchical forecasting), fixed order, names and
#: Okabe-Ito colours; "base" is the unreconciled direct forecast (reference).
RECONCILIATION_NAMES = {
    "base": "Base (direct)",
    "bottom_up": "Bottom-up",
    "top_down": "Top-down",
    "middle_out": "Middle-out (trust)",
    "ols": "OLS",
    "wls_struct": "WLS (structural)",
    "wls_var": "WLS (variance)",
    "mint_shrink": "MinT (shrinkage)",
}
RECONCILIATION_ORDER = list(RECONCILIATION_NAMES.values())
RECONCILIATION_COLORS = {
    "Base (direct)": "#000000",
    "Bottom-up": "#7A7A7A",
    "Top-down": "#B8B8B8",
    "Middle-out (trust)": "#56B4E9",
    "OLS": "#E69F00",
    "WLS (structural)": "#009E73",
    "WLS (variance)": "#0072B2",
    "MinT (shrinkage)": "#D55E00",
}
RECONCILIATION_LINESTYLES = {
    "Base (direct)": "--",
    "Bottom-up": "-",
    "Top-down": "-",
    "Middle-out (trust)": "-",
    "OLS": "-",
    "WLS (structural)": "-",
    "WLS (variance)": "-",
    "MinT (shrinkage)": "-",
}

#: Categorical colours for data-driven groups (clusters), Okabe-Ito in a fixed
#: sequence; cluster 0 is always the first colour.
CLUSTER_COLORS = [
    "#0072B2",
    "#E69F00",
    "#009E73",
    "#D55E00",
    "#CC79A7",
    "#56B4E9",
    "#F0E442",
    "#7A7A7A",
]


def cluster_color(i: int) -> str:
    return CLUSTER_COLORS[int(i) % len(CLUSTER_COLORS)]


#: Meter-coverage check against ERIC 2023/24 (doc/estates_analysis.md); fixed names and colours.
COVERAGE_FLAG_NAMES = {
    "ok": "whole site metered",
    "partial_supply": "partial supply",
    "aggregate_supply": "aggregate supply",
    "chp_mismatch": "on-site CHP",
    "meter_gaps": "meter gaps",
}
COVERAGE_FLAG_COLORS = {
    "whole site metered": "#009E73",
    "partial supply": "#D55E00",
    "aggregate supply": "#E69F00",
    "on-site CHP": "#CC79A7",
    "meter gaps": "#B8B8B8",
}
#: Electrification scenario: today's demand and demand with all gas heat from heat pumps.
SCENARIO_COLORS = {"today": "#7A7A7A", "with heat pumps": "#D55E00"}
#: Price cases of the flexibility analysis (analysis/flexibility.py); fixed order and colours.
PRICE_CASE_ORDER = (
    "wholesale",
    "wholesale + red 2p",
    "wholesale + red 5p",
    "wholesale + red 10p",
    "Agile",
)
PRICE_CASE_NAMES = {
    "wholesale": "wholesale index",
    "wholesale + red 2p": "+ 2p red band",
    "wholesale + red 5p": "+ 5p red band",
    "wholesale + red 10p": "+ 10p red band",
    "Agile": "Agile tariff",
}
PRICE_CASE_COLORS = {
    "wholesale": "#7A7A7A",
    "wholesale + red 2p": "#0072B2",
    "wholesale + red 5p": "#0072B2",
    "wholesale + red 10p": "#0072B2",
    "Agile": "#E69F00",
}
#: Storage that holds a site under its connection (analysis/design_winter.py).
STORE_CLASS_ORDER = (
    "battery scale (under 2 h)",
    "thermal store scale (2 to 24 h)",
    "beyond storage (over 24 h)",
)
STORE_CLASS_COLORS = {
    "within capacity": "#B8B8B8",
    "battery scale (under 2 h)": "#009E73",
    "thermal store scale (2 to 24 h)": "#E69F00",
    "beyond storage (over 24 h)": "#D55E00",
}
#: Weather years of the design-winter analysis.
WEATHER_YEAR_COLORS = {
    "last 30 winters": "#0072B2",
    "earlier winters": "#B8B8B8",
    "reference year": "#000000",
}
#: Winters shown for each site: the reference year, the one-in-twenty winter, the worst on record.
DESIGN_CASE_COLORS = {
    "reference year": "#7A7A7A",
    "one-in-twenty winter": "#0072B2",
    "worst winter on record": "#D55E00",
}
DESIGN_CASE_MARKERS = {
    "reference year": "o",
    "one-in-twenty winter": "^",
    "worst winter on record": "s",
}
#: Heat-pump status between two estates returns (analysis/heat_pump_field.py).
HEAT_PUMP_STATUS_COLORS = {"no heat pump": "#B8B8B8", "new heat pumps": "#D55E00"}
#: Solar programme sites (analysis/solar_programme.py).
SOLAR_PROGRAMME_COLORS = {
    "funded, break": "#D55E00",
    "funded, no break": "#7A7A7A",
    "not funded, break": "#0072B2",
}
#: Threshold plans of the peak-shaving test (analysis/flexibility.py).
PLAN_NAMES = {"point": "planned on the median", "q70": "planned on the 70 % quantile"}
#: Extreme-weather event types (analysis/weather_events.py); fixed order and colours.
WEATHER_EVENT_ORDER = ("heatwave", "cold spell", "snow", "heavy rain", "high wind")
WEATHER_EVENT_COLORS = {
    "heatwave": "#D55E00",
    "cold spell": "#0072B2",
    "snow": "#56B4E9",
    "heavy rain": "#009E73",
    "high wind": "#7A7A7A",
}
#: Sites with and without fossil-fuel CHP (grid import vs demand); CHP matches the coverage flag.
CHP_NAMES = {False: "no on-site CHP", True: "on-site CHP"}
CHP_COLORS = {"no on-site CHP": "#0072B2", "on-site CHP": "#CC79A7"}
#: Site groups of the CHP comparison (analysis/chp.py).
CHP_GROUP_ORDER = ("on-site CHP", "acute, no CHP", "other sites")
CHP_GROUP_COLORS = {"on-site CHP": "#CC79A7", "acute, no CHP": "#0072B2", "other sites": "#7A7A7A"}
#: Site groups of the weather-response analysis (analysis/weather_response.py).
SITE_GROUP_ORDER = ("inpatient", "outpatient", "on-site CHP")
SITE_GROUP_COLORS = {"inpatient": "#0072B2", "outpatient": "#56B4E9", "on-site CHP": "#CC79A7"}
#: Quasi-experimental weather designs (analysis/weather_causal.py): the two temperature doses
#: take the cold-spell and heatwave colours; specifications in the order they nest.
DOSE_NAMES = {"hdd14": "cold: per °C below 14 °C", "cdd20": "hot: per °C above 20 °C"}
DOSE_COLORS = {"hdd14": "#0072B2", "cdd20": "#D55E00"}
SPEC_ORDER = ("A", "B", "C", "D")
SPEC_COLORS = {"A": "#7A7A7A", "B": "#0072B2", "C": "#56B4E9", "D": "#D55E00"}
SPEC_NAMES = {
    "A": "site × month",
    "B": "+ date",
    "C": "+ region × date",
    "D": "B + weather",
}
EXTREME_DAY_NAMES = {"hot28": "28–30 °C day", "hot30": "30 °C+ day"}
EXTREME_DAY_COLORS = {"hot28": "#E69F00", "hot30": "#D55E00"}
SPATIAL_SERIES_COLORS = {"temperature anomaly": "#0072B2", "demand residual, spec B": "#7A7A7A"}
#: Site thermometers (analysis/thermometers.py): the 25 km grid is the baseline grey.
THERMOMETER_ORDER = ("ERA5", "ERA5-Land", "stations")
THERMOMETER_NAMES = {
    "ERA5": "ERA5 (25 km grid)",
    "ERA5-Land": "ERA5-Land (9 km grid)",
    "stations": "Met Office stations",
}
THERMOMETER_COLORS = {"ERA5": "#7A7A7A", "ERA5-Land": "#0072B2", "stations": "#D55E00"}
#: Building-use groups of the heterogeneity analysis (analysis/weather_heterogeneity.py).
USE_ORDER = (
    "inpatient, acute",
    "inpatient, community",
    "mental health",
    "non-inpatient",
    "support",
)
USE_COLORS = {
    "inpatient, acute": "#0072B2",
    "inpatient, community": "#56B4E9",
    "mental health": "#009E73",
    "non-inpatient": "#E69F00",
    "support": "#7A7A7A",
}
MODERATOR_ORDER = (
    "building use",
    "demand-shape cluster",
    "trust type",
    "site size (log mean kW)",
    "mean build year",
    "summer mean daily maximum (°C)",
    "gas / electricity ratio",
    "LED share (%)",
)
# --- what moves heat (analysis/heat_drivers.py) ---------------------------
TERM_NAMES = {"hdd14": "cold", "cdd20": "heat", "solar_kwh_m2": "sunshine"}
TERM_COLORS = {"hdd14": "#0072B2", "cdd20": "#D55E00", "solar_kwh_m2": "#E69F00"}
HEAT_TYPE_ORDER = (
    "rises with heat",
    "rises with heat, falls with sunshine",
    "falls with sunshine only",
    "falls with heat",
    "neither",
)
HEAT_TYPE_COLORS = {
    "rises with heat": "#D55E00",
    "rises with heat, falls with sunshine": "#CC79A7",
    "falls with sunshine only": "#E69F00",
    "falls with heat": "#0072B2",
    "neither": "#7A7A7A",
}
HEAT_SIGN_ORDER = ("rises with heat", "falls with heat", "neither")
HEAT_SIGN_COLORS = {
    "rises with heat": "#D55E00",
    "falls with heat": "#0072B2",
    "neither": "#7A7A7A",
}
BLOCK_ORDER = ("night (00–06)", "morning (06–12)", "afternoon (12–18)", "evening (18–24)")
BLOCK_COLORS = {
    "night (00–06)": "#0072B2",
    "morning (06–12)": "#56B4E9",
    "afternoon (12–18)": "#D55E00",
    "evening (18–24)": "#CC79A7",
}
URBAN_ORDER = ("major conurbation", "minor conurbation", "city and town", "rural")
URBAN_COLORS = {
    "major conurbation": "#D55E00",
    "minor conurbation": "#E69F00",
    "city and town": "#0072B2",
    "rural": "#009E73",
}
HEAT_MODERATOR_CATEGORICAL = (
    "building use",
    "air conditioning (DEC)",
    "building environment (DEC)",
    "urban form",
    "tenure",
    "heat pumps (ERIC)",
    "overheating reported (ERIC)",
    "solar generation (ERIC)",
)
HEAT_MODERATOR_CONTINUOUS = (
    "air-conditioned floor share (DEC)",
    "cooled floor share, AC or mixed mode (DEC)",
    "electric-heating floor share (DEC)",
    "building electricity (log kWh/m², DEC)",
    "operational rating (DEC)",
    "population density (log)",
    "electricity intensity (log kWh/m²)",
    "peak demand (W/m²)",
    "night / day ratio",
    "weekend / weekday ratio",
    "winter / summer ratio",
    "site size (log mean kW)",
    "mean build year",
    "gas / electricity ratio",
    "summer mean daily maximum (°C)",
)
VARIANCE_SET_ORDER = (
    "site sunshine term",
    "building use",
    "urban form and density",
    "estates descriptors",
    "load shape",
    "all descriptors",
    "cooling plant (DEC)",
    "heating and intensity (DEC)",
    "all descriptors and DEC",
    "diurnal signature",
)
DEC_ENV_ORDER = (
    "air conditioning",
    "mixed mode",
    "heating, mechanical ventilation",
    "heating, natural ventilation",
    "unconditioned",
)
DEC_ENV_COLORS = {
    "air conditioning": "#D55E00",
    "mixed mode": "#E69F00",
    "heating, mechanical ventilation": "#56B4E9",
    "heating, natural ventilation": "#0072B2",
    "unconditioned": "#7A7A7A",
}

PERSISTENCE_NAMES = {
    "hot_day2": "hot day 2",
    "hot_day3plus": "hot day 3+",
    "cold_day2": "cold day 2",
    "cold_day3plus": "cold day 3+",
}
#: Grid headroom classes: peak over available capacity.
HEADROOM_CLASSES = {"below 80 %": "#009E73", "80 to 100 %": "#E69F00", "above capacity": "#D55E00"}

#: Generic lead times for half-hourly demand; fixed, never reordered by result.
HORIZON_ORDER_HOURS = [0.5, 1, 3, 6, 12, 24, 48, 168]

#: Axis-label vocabulary. One term per quantity, units included.
QUANTITY_LABELS = {
    "horizon": "Forecast horizon (hours)",
    "demand": "Demand (kWh per half hour)",
    "temperature": "Temperature (\u00b0C)",
    "mae": "MAE (kWh)",
    "rmse": "RMSE (kWh)",
    "mase": "MASE",
    "crps": "CRPS (kWh)",
    "crps_skill": "CRPS skill",
    "coverage": "Coverage (%, target 80)",
    "bias": "Bias (kWh)",
}


def method_color(method: str) -> str:
    """Fixed colour for a method name (see METHOD_COLORS)."""
    if method not in METHOD_COLORS:
        raise KeyError(f"'{method}' has no colour in style.py - add it there, not in the script")
    return METHOD_COLORS[method]


def method_marker(method: str) -> str:
    """Fixed marker for a method: plain 'o', + ERA5 's', + IFS ENS '^'."""
    return METHOD_MARKERS[method]


def method_linestyle(method: str) -> str:
    """Fixed line style for a method: plain solid, + ERA5 dashed, + IFS ENS dotted."""
    return METHOD_LINESTYLES[method]


def ordered(methods: Sequence[str]) -> list[str]:
    """Return the given methods in the fixed dissertation order, never by performance."""
    order = {m: i for i, m in enumerate(METHOD_ORDER)}
    unknown = [m for m in methods if m not in order]
    if unknown:
        raise KeyError(f"Methods without a fixed order: {unknown}")
    return sorted(methods, key=order.__getitem__)


# --------------------------------------------------------------------------- #
# Matplotlib configuration
# --------------------------------------------------------------------------- #


def configure_matplotlib() -> None:
    """Apply the specification to rcParams. Call once at the top of every script."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_FAMILY,
            "font.weight": "normal",
            "axes.labelweight": "normal",
            "axes.titleweight": "normal",
            "figure.titleweight": "normal",
            "mathtext.default": "regular",
            "text.usetex": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": FONT_SIZE["label"],
            "axes.labelsize": FONT_SIZE["label"],
            "xtick.labelsize": FONT_SIZE["tick"],
            "ytick.labelsize": FONT_SIZE["tick"],
            "legend.fontsize": FONT_SIZE["legend"],
            "axes.linewidth": AXIS_LINEWIDTH,
            "xtick.major.width": TICK_WIDTH,
            "ytick.major.width": TICK_WIDTH,
            "xtick.minor.width": TICK_WIDTH,
            "ytick.minor.width": TICK_WIDTH,
            "xtick.major.size": TICK_LENGTH,
            "ytick.major.size": TICK_LENGTH,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "axes.axisbelow": True,
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.alpha": GRID_ALPHA,
            "lines.linewidth": LINEWIDTH["primary"],
            "lines.markersize": MARKER_SIZE,
            "lines.markeredgewidth": MARKER_EDGE,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.2,
            "legend.handletextpad": 0.5,
            "figure.dpi": 100,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "image.cmap": "viridis",
        }
    )


def figure(size: tuple[float, float] = SINGLE_COLUMN):
    """A configured (fig, ax) at one of the allowed sizes."""
    if size not in (SINGLE_COLUMN, WIDE_SINGLE, DOUBLE_COLUMN, SQUARE, WIDE_SHORT) and not (
        size[0] == 7.2 and 2.4 <= size[1] <= 2.8
    ):
        raise ValueError(f"{size} is not an allowed figure size")
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=size)
    style_axis(ax, wide_short=(size[0] == 7.2 and size[1] <= 2.8))
    return fig, ax


def style_axis(
    ax, grid: str | None = None, all_spines: bool = False, wide_short: bool = False
) -> None:
    """Spines, ticks and optional grid per the specification.

    ``grid`` is ``None`` (default), ``"y"`` (bar plots, score comparisons) or
    ``"both"``; only use it when it helps numerical comparison.
    """
    for side in ("top", "right"):
        ax.spines[side].set_visible(all_spines)
    for side in ("left", "bottom", "top", "right"):
        ax.spines[side].set_linewidth(AXIS_LINEWIDTH)
    tick_size = FONT_SIZE_WIDE_SHORT_TICK if wide_short else FONT_SIZE["tick"]
    ax.tick_params(width=TICK_WIDTH, length=TICK_LENGTH, direction="out", labelsize=tick_size)
    ax.set_axisbelow(True)
    if grid:
        ax.grid(
            True,
            axis=grid if grid != "both" else "both",
            which="major",
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
    else:
        ax.grid(False)
    ax.set_title("")


def add_bottom_legend(
    ax,
    lowercase: bool = True,
    ncol: int | None = None,
    anchor_y: float = -0.22,
    bottom: float = 0.30,
    fontsize: int | None = None,
    handles=None,
    labels=None,
):
    """Legend outside, centred below the axes, borderless, one row when possible.

    ``anchor_y`` -0.22 is the default; the wide/short amendment uses -0.30 or
    lower with a matching ``bottom`` margin. Entries are lowercased per the
    amendment.
    """
    if handles is None or labels is None:
        handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    if lowercase:  # descriptive entries are lowercase; pass lowercase=False for product names
        labels = [
            lab[:1].lower() + lab[1:] if lab and not lab[:2].isupper() else lab for lab in labels
        ]
    ncol = ncol or len(handles)
    legend = ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, anchor_y),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.2,
        handletextpad=0.5,
        fontsize=fontsize or FONT_SIZE["legend"],
        title=None,
    )
    ax.figure.subplots_adjust(bottom=bottom)
    return legend


def reference_line(ax, value: float, axis: str = "y"):
    """A statistical reference line (0 for differences, 1 for ratios, nominal for coverage)."""
    if axis == "y":
        return ax.axhline(value, **REFERENCE_LINE, zorder=1)
    return ax.axvline(value, **REFERENCE_LINE, zorder=1)


def save_figure(fig, base: str | Path, png_dpi: int = 600, data: Any = None) -> dict[str, Path]:
    """Write ``<base>.pdf`` (transparent vector), ``<base>.png`` (white, dpi) and the data.

    ``base`` is a path without extension, relative to the repository root.
    ``data`` may be a DataFrame (-> ``_data.csv``) or any JSON-serialisable
    object (-> ``_data.json``). Prints the written paths and closes the figure.
    """
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    pdf = base.with_suffix(".pdf")
    fig.savefig(pdf, format="pdf", transparent=True, bbox_inches="tight", pad_inches=0.02)
    written["pdf"] = pdf

    png = base.with_suffix(".png")
    fig.savefig(
        png,
        format="png",
        dpi=png_dpi,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    written["png"] = png

    if data is not None:
        if hasattr(data, "to_csv"):
            csv = base.parent / f"{base.name}_data.csv"
            data.to_csv(csv, index=False)
            written["data"] = csv
        else:
            js = base.parent / f"{base.name}_data.json"
            js.write_text(json.dumps(data, indent=1, default=str))
            written["data"] = js

    plt.close(fig)
    for kind, path in written.items():
        print(f"{kind:<5} {path}")
    return written
