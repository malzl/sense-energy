"""Treatment-response curve on the daily maximum temperature, sites without CHP, under specifications
A (site x month), B (+ date effects) and D (B + sunshine, wind, rain). One plot.

Reference 18-20 degC; open-ended bands drawn at 0 and 33 degC (analysis/weather_causal.py).
"""

from _common import plot_response_curve

plot_response_curve(
    "daily maximum",
    "causal_response_tmax",
    "Daily maximum temperature (°C)",
    "Demand vs an 18–20 °C day (%)",
)
