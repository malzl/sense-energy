"""How often the demand is below the 10 % quantile of the day-ahead forecast, by hour of the
day and season. The dashed line is the nominal 10 %. Whole record, zero-shot model with the
weather forecast, 91 sites; mean over sites.

analysis/forecast_range.py (season_hour.csv).
"""

from _common import misses_by_hour

misses_by_hour("below_pct", "Demand below the range (% of half-hours)", "range_below_hours")
