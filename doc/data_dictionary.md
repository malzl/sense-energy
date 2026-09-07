# Data dictionary

Every column of every source, with units and provenance. Update this in the same
PR that adds or changes a data source.

## Conventions

- All timestamps are stored **tz-aware UTC**. Half-hourly timestamps denote the
  **start** of the settlement period. Local time (`Europe/London`) is a display
  concern only — this matters because BST transitions produce 46- and 50-period days.
- Energy is in **kWh** per period unless stated. Power is in **kW**.
- `site_id` is the primary site key across all tables.

## `raw/hh_meter_data.csv`

| Column | Type | Unit | Description | Notes |
|---|---|---|---|---|
| `site_id` | str | — | Site / MPAN identifier | TBD |
| `timestamp` | datetime | UTC | Settlement period start | TBD |
| `value` | float | kWh | Consumption in the period | TBD |
| `unit` | str | — | Unit label | TBD |

**Source:** TBD · **Provider:** TBD · **Coverage:** TBD · **Refresh:** TBD

## `raw/site_metadata.csv`

| Column | Type | Unit | Description | Notes |
|---|---|---|---|---|
| `site_id` | str | — | Site identifier | Joins to meter data |
| `trust_name` | str | — | NHS trust | TBD |
| `site_type` | str | — | Acute / community / mental health | TBD |
| `gia_m2` | float | m² | Gross internal area | TBD |
| `beds` | int | — | Bed count | TBD |
| `latitude` | float | ° | WGS84 | For weather matching |
| `longitude` | float | ° | WGS84 | For weather matching |

## `raw/weather.csv` (or `external/`)

| Column | Type | Unit | Description | Notes |
|---|---|---|---|---|
| `site_id` | str | — | Site identifier | Nearest station / grid point |
| `timestamp` | datetime | UTC | Observation time | TBD |
| `temperature_c` | float | °C | Dry bulb air temperature | Primary demand driver |
| `humidity_pct` | float | % | Relative humidity | TBD |
| `wind_speed_ms` | float | m/s | Wind speed | Affects infiltration losses |
| `solar_irradiance_wm2` | float | W/m² | Global horizontal irradiance | Solar gain, on-site PV |

**Source:** TBD (Met Office / Open-Meteo / ERA5) · **Licence:** TBD

## Derived columns

Built by `code/src/sense_energy/features/`. See the module docstrings for exact definitions.

| Column | Source module | Description |
|---|---|---|
| `hour`, `day_of_week`, `month`, `is_weekend` | `features/calendar.py` | Local-time calendar parts |
| `tod_sin`/`tod_cos`, `dow_*`, `doy_*` | `features/calendar.py` | Cyclical encodings |
| `is_holiday` | `features/calendar.py` | England & Wales bank holiday |
| `heating_degrees`, `cooling_degrees` | `features/weather.py` | Degree hours, base 15.5 °C |
| `value_lag_<n>` | `features/lags.py` | Target lagged *n* periods |
| `value_roll_mean_<w>` | `features/lags.py` | Rolling mean, shifted by 1 |
