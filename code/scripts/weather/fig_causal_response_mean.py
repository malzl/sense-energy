"""Treatment-response curve on the daily mean temperature, sites without CHP, under specifications
A (site x month), B (+ date effects) and D (B + sunshine, wind, rain). One plot.

Reference 14-16 degC; open-ended bands drawn at -3 and 25 degC (analysis/weather_causal.py).
"""

from _common import plot_response_curve

plot_response_curve(
    "daily mean",
    "causal_response_mean",
    "Daily mean temperature (°C)",
    "Demand vs a 15 °C day (%)",
)
