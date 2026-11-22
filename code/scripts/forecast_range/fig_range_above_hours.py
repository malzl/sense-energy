"""How often the demand is above the 90 % quantile of the day-ahead forecast, by hour of the
day and season. The dashed line is the nominal 10 %. Whole record, zero-shot model with the
weather forecast, 91 sites; mean over sites.

analysis/forecast_range.py (season_hour.csv).
"""

from _common import misses_by_hour

misses_by_hour("above_pct", "Demand above the range (% of half-hours)", "range_above_hours")
