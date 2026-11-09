# Data dictionary

## Conventions

- All timestamps are tz-aware **UTC**; a half-hourly timestamp denotes the
  **start** of the settlement period. Local time (`Europe/London`) is a display
  concern — this matters because BST transitions produce 46- and 50-period days.
- Energy is **kWh per half hour** unless stated.
- The bottom grain is the **meter** (`mpxn`), not the site.

## Source: `energy_systems_catapult.nhs_energy_consumption.nhs_trust_consumption_data.parquet`

Energy Systems Catapult, NHS trust consumption. 454 MB, **21,315,603 rows**,
31 columns, 2021-03-01 → 2026-05-08, 284 sites. Written by DuckDB v1.4.3.

Site attributes are denormalised onto every reading row.

### Reading columns

| Column | Type | Description |
|---|---|---|
| `datetime` | timestamp[us, UTC] | Settlement period start |
| `consumption` | double | **Unit depends on `reading_type`** — see below |
| `reading_type` | string | Code 1–14; determines quantity and unit |
| `reading_type_name` | string | Human-readable form of `reading_type` |
| `energy_type` | string | `elec` (19.7M rows) or `gas` (1.6M) |
| `mpxn` | string | Meter identifier (MPAN/MPRN) — **the true grain** |
| `monitor_external_reference` | string | Alternative meter reference |

### ⚠️ `reading_type` — the most important column in the file

`consumption` mixes units and granularities. Filtering is mandatory.

| Code | Name | Quantity | Rows | Keep? |
|---|---|---|---|---|
| `1` | 30 minute aggregated kWh | Active energy | 10,338,634 | ✅ |
| `2` | Estimate 30 minute aggregated kWh | Active energy (estimated) | 516,574 | ✅ flagged |
| `3` | Part actual 30 minute aggregate kWh | Active energy (partial) | 36,500 | ✅ flagged |
| `6` | Export 30 minute aggregated kWh | Export (generation) | 79,952 | ❌ not demand |
| `8` | Reactive Export 30 min aggregated **kVArh** | Reactive power | 4,606,851 | ❌ not energy |
| `9` | Reactive Export estimate **kVArh** | Reactive power | 327,852 | ❌ not energy |
| `10` | Reactive Import 30 min aggregated **kVArh** | Reactive power | 5,029,235 | ❌ not energy |
| `11` | Reactive Import estimate **kVArh** | Reactive power | 366,941 | ❌ not energy |
| `4` | Cumulative kWh | Meter register | 8,457 | ❌ wrong granularity |
| `5` | Monthly aggregated kWh | Monthly | 2,403 | ❌ wrong granularity |
| `14` | Time of use cumulative kWh | Register | 2,100 | ❌ wrong granularity |
| `NULL` | — | Unknown | 104 | ❌ |

**9.6M of 21.3M rows (45%) are reactive power in kVArh, not energy.** Averaging
`consumption` without filtering silently mixes kVArh into kWh.

### Site columns (constant per `site_code`; extracted to `sites.parquet`)

| Column | Type | Description |
|---|---|---|
| `site_code` | string | Site identifier |
| `site_name` | string | Site name |
| `postcode` | string | Postcode — the route to weather matching |
| `organisation_name` | string | NHS trust |
| `organisation_type` | string | ACUTE - TEACHING / LARGE / MEDIUM / SMALL / SPECIALIST, COMMUNITY, MENTAL HEALTH AND LEARNING DISABILITY, AMBULANCE |
| `comissioning_region` | string | **[sic]** — misspelled in source; renamed `commissioning_region` downstream |
| `integrated_care_board` | string | ICB |
| `local_authority` | string | Local authority |
| `occupied_floor_area` | double | m² |
| `site_gross_internal_area` | double | m² — the denominator for kWh/m² |
| `site_heated_volume` | double | m³ |
| `site_construction_year_band` | string | Construction era band |
| `site_use_type` | string | Site use classification |
| `fossil_fuel_led_chp_units_operated_on_site` | int32 | On-site CHP count |

### Unused source columns

`main_heating_fuel` is **empty for all 21,315,603 rows**. The pre-aggregated
`total_*_energy_consumption_kwh`, `chp_*`, `*_reported_kwh_m2` and `kWh_m2`
columns are trust/period roll-ups, not half-hourly, and are not carried into
interim.

---

## Interim: `code/data/interim/consumption_halfhourly.parquet`

Built by `sense-energy build-interim`. 11,013,277 rows, 38.6 MB (zstd).
**235 meters, 146 sites**, 2022-12-07 → 2026-05-08.

Grain: one row per (`mpxn`, `energy_type`, `datetime`) on a complete 30-minute
grid. Meter grain is preserved deliberately so the series can be aggregated up a
hierarchy — `mpxn → site_code → organisation_name → integrated_care_board →
commissioning_region` — via `cleaning.aggregate_to_level`.

| Column | Type | Description |
|---|---|---|
| `mpxn` | string | Meter identifier |
| `site_code` | string | Joins to `sites.parquet` |
| `energy_type` | string | `elec` or `gas` |
| `datetime` | datetime64[us, UTC] | Period start, on a gap-free grid |
| `consumption_kwh` | float64 | Active energy, kWh. NaN = missing (11.82%) |
| `is_estimated` | bool | Came from `reading_type` 2 or 3 (3.10%) |
| `is_interpolated` | bool | Linearly filled across a gap ≤ 4 periods (1.38%) |
| `is_outlier` | bool | Implausible high reading: robust z > 10 **and** > 5× meter median **and** > 3× meter p99.5 (148 readings, 0.001%) |

### Cleaning decisions

| Decision | Rationale |
|---|---|
| Keep `reading_type` 1/2/3 only | Everything else is a different quantity or granularity |
| Meters **summed, never de-duplicated** | 33 of 146 sites have multiple meters (Homerton: 20). 3.1M rows share `site_code`+`datetime` and are distinct meters, not duplicates |
| 101 true duplicates dropped | Identical on (`mpxn`, `energy_type`, `reading_type`, `datetime`); last kept |
| 211,478 actual/estimate collisions resolved | Same meter and timestamp under two reading types; the actual (`1`) wins |
| Exact zeros → NaN (1,120,688; 10.49%) | Meter dropout written as zero. Concentrated — 8 sites are >50% zero, 94 of 145 under 1% — which is what distinguishes it from genuine low demand |
| Gaps ≤ 4 periods interpolated | Longer outages stay NaN rather than inventing a load profile |
| Outliers flagged, never dropped | A genuine spike is signal. Three conditions must all hold (z > 10, > 5× median, > 3× p99.5): a plain MAD z-score flagged up to 45% of readings on low-consumption or bimodal meters, whose idle/running regimes are not faults. Now 148 readings in total |

### Known data quality issues

- **Gas maximum of 32,611 kWh/half-hour** (65 MW thermal) against a gas mean of
  204 — implausible, flagged as an outlier and treated as missing in the processed tables. Electricity peaks at 2,505
  kWh/half-hour (5 MW), which is plausible for a large acute site.
- **Coverage varies widely**: median 1,192 days per site, minimum 388.
- **Only 146 of 284 sites** have half-hourly active energy at all.
- **No weather data** in this source. Temperature is the strongest single
  predictor of hospital demand and must be joined from elsewhere, matched on
  `postcode`.

## Interim: `code/data/interim/sites.parquet`

284 rows, 14 columns — the site columns above, with `comissioning_region`
renamed to `commissioning_region`.

---

## Geography: `code/data/geo/`

The NHS extract carries **no coordinates**. `postcode` is the only direct
locator — present for 275 of 284 sites, and 141 of the 146 with half-hourly
data. Everything below is fetched by `sense-energy fetch-geo`.

### Boundary layers

ONS Open Geography Portal, **Open Government Licence v3** — "Contains OS data
© Crown copyright and database right". Generalised-clipped (`BGC`) versions:
full-resolution coastlines are far larger and indistinguishable at national scale.

| File | Layer | Features | Joins to |
|---|---|---|---|
| `countries_uk.geojson` | Countries December 2024 UK BGC | 4 | map background |
| `nhs_regions_en.geojson` | NHS England Regions January 2024 BGC | 7 | `sites.commissioning_region` |
| `icb_en.geojson` | Integrated Care Boards April 2023 BGC | 42 | `sites.integrated_care_board` |

Each is also written as an ESRI Shapefile (`.shp` + `.dbf`/`.shx`/`.prj`/`.cpg`)
via `sense-energy fetch-geo --shapefile`. GeoJSON is the native download and the
better archival form — one file, explicit CRS, no 10-character field-name
truncation — so the shapefiles are a convenience for tools that need them.

⚠️ ONS layer names carry the boundary type as a suffix. `_NC` means **no
coordinates** — a lookup table with null geometry, not a boundary layer.
`BFC`/`BFE` are full resolution, `BGC` generalised, `BSC`/`BUC` (super/ultra)
more so.

### `sites_geo.parquet`

284 rows — the site dimension with centroids appended.

| Column | Description |
|---|---|
| *(all `sites.parquet` columns)* | |
| `latitude`, `longitude` | Postcode centroid, EPSG:4326 |
| `eastings`, `northings` | OSGB grid (EPSG:27700) |
| `country`, `region_ons`, `admin_district` | ONS geographies — cross-check on the site table |
| `postcode_is_terminated` | Centroid came from a retired postcode |

**Source:** [postcodes.io](https://postcodes.io) — a free service over ONS open
data. Only postcodes are sent; no consumption data leaves the machine. Hospital
postcodes are public information, but see [data_governance.md](data_governance.md)
before extending this to anything site-level.

### Geocoding outcome

- **275 of 284 sites** located; all 258 distinct postcodes resolved.
- **5 postcodes are terminated** (retired 2010–2025) and were resolved through
  the terminated-postcodes endpoint. NHS estates records outlive the postcode
  register, so this fallback is required, not incidental.
- **9 sites have no postcode at all.** Five are `Site(s) Unknown - <trust>`
  placeholders — trust-level aggregates carried in the data as if they were
  sites. They are unmappable, and should be excluded from site-level modelling
  rather than treated as a location.
- **141 of the 146 sites with demand data are mappable**; 134 appear on the map,
  the rest having no non-outlier electricity readings.
- Every located site is in **England** — consistent with an NHS England source.

---

## Weather: `code/data/external/weather/era5/` → `interim/weather_*.parquet`

ERA5 from the Copernicus Climate Data Store (`reanalysis-era5-single-levels`).
Licence: Copernicus, free with attribution. Pulled by `sense-energy fetch-weather`,
one NetCDF per product-month; extracted by `sense-energy build-weather`.
Why reanalysis rather than forecasts: [ADR 0003](adr/0003-weather-and-price-data-sources.md).

| Product | Grid | Cadence | Members | File |
|---|---|---|---|---|
| `reanalysis` | 0.25° | hourly | 1 (member 0) | `weather_reanalysis.parquet` |
| `ensemble_members` | 0.5° | 3-hourly | 10 (EDA) | `weather_ensemble_members.parquet` |

Area `[55, -5, 50, 2]` (N, W, S, E) covers every located site. Site values are
the nearest grid point to the postcode centroid.

| Column | Units | Notes |
|---|---|---|
| `site_code`, `datetime` (UTC), `member` | | member 0 = deterministic |
| `t2m` | K | 2 m temperature. Subtract 273.15 |
| `d2m` | K | 2 m dewpoint → relative humidity |
| `tcc`, `lcc` | 0–1 | total / low cloud cover |
| `tp` | m | total precipitation, **accumulated over the previous hour** (3 h for EDA) |
| `sf` | m water equivalent | snowfall, accumulated |
| `u10`, `v10` | m s⁻¹ | wind components; speed = √(u²+v²) |
| `i10fg` | m s⁻¹ | instantaneous 10 m gust |
| `ssrd` | J m⁻² | surface solar radiation down, accumulated; ÷3600 → W m⁻² mean |
| `strd` | J m⁻² | surface thermal radiation down, accumulated |
| `sp` | Pa | surface pressure |

⚠️ The new CDS delivers a zip holding one NetCDF per step type when a request
mixes instantaneous and accumulated variables, even with
`download_format: unarchived`. `weather.open_download` merges them.

### Daily: `interim/weather_daily.parquet`

Built by `sense-energy build-weather-daily` (`data/weather_daily.py`) from
`weather_reanalysis.parquet`, nothing fetched. 351,450 rows: 275 sites ×
1 Dec 2022 → 31 May 2026. One row per (`site_code`, `date`), `date` a
**Europe/London** calendar day (23 or 25 hours on BST transition days).
Accumulated fields are dated by the hour they accumulate over.

| Column | Units | Definition |
|---|---|---|
| `n_hours` | h | hourly values in the local day |
| `t_mean`, `t_min`, `t_max` | °C | 2 m temperature |
| `t_eve` | °C | mean 2 m temperature over local 18:00–23:59, the part of the day a daily mean blurs |
| `hdd_15_5` | °C day | heating degree days, integration method: mean over hours of max(15.5 − T, 0) |
| `cdd_15_5`, `cdd_18`, `cdd_22` | °C day | cooling degree days, integration method (CIBSE TM41): mean over hours of max(T − base, 0) |
| `cdd_dm_15_5`, `cdd_dm_18`, `cdd_dm_22` | °C day | daily-mean method: max(T_mean − base, 0) |
| `cdd_eurostat` | °C day | Eurostat: T_mean − 21 on days with T_mean > 24 °C (almost always 0 in England) |
| `hours_ge_25`, `hours_ge_28` | h | hours with T ≥ 25 / 28 °C |
| `td_mean`, `td_max` | °C | dew point |
| `rh_mean` | % | relative humidity (Magnus) |
| `tw_max` | °C | wet-bulb temperature, daily max (Stull 2011) |
| `h_mean`, `h_max` | kJ kg⁻¹ | moist-air enthalpy (latent + sensible cooling need) |
| `wind_mean`, `wind_max` | m s⁻¹ | 10 m wind speed |
| `gust_max` | m s⁻¹ | 10 m gust, daily max |
| `solar_kwh_m2` | kWh m⁻² | surface solar irradiation |
| `precip_mm` | mm | total precipitation |
| `snow_mm_we` | mm water equivalent | snowfall |

ERA5's 0.25° grid smooths urban heat islands, afternoon peaks, convective rain
and gusts: the hottest site-day here is 34.2 °C against 35.1 °C observed at Kew
on 26 May 2026. Read the hot, wet and windy tails as lower bounds. Uses:
[weather_and_cooling.md](weather_and_cooling.md).

## Site thermometers: ERA5-Land and NOAA ISD stations

Two independent measurements of the temperature at each site, for the measurement-error
question in [weather_and_cooling.md](weather_and_cooling.md#5-quasi-experimental-estimates).

### ERA5-Land: `code/data/external/weather/era5_land/land/` → `interim/weather_land.parquet` → `interim/weather_daily_land.parquet`

CDS `reanalysis-era5-land`, hourly, 0.1° (about 9 km), land points only; `t2m`, `d2m`, `sp`
(fluxes stay with ERA5: ERA5-Land accumulates them from 00 UTC, not hourly). Pulled by
`sense-energy fetch-weather --config code/configs/weather_era5_land.yaml --product land`,
extracted by `build-weather --config ... --product land` to the nearest *land* grid point
(coastal sites would otherwise land in the sea), and summarised by
`build-weather-daily --product land --suffix _land` with the same columns as `weather_daily`
minus solar, rain, snow and wind.
351,450 site-days for 275 sites. Against the stations its daily mean has an RMSE of 0.72 °C
(ERA5: 0.62) and its daily maximum 1.26 °C (ERA5: 0.94): the land-only tiles lose the sea's
moderation at coastal sites, so the finer grid is the *worse* site thermometer. Kept as a check,
not used as the primary temperature.

### NOAA ISD: `code/data/external/noaa_isd/` → `interim/isd_*.parquet`, `interim/weather_daily_isd.parquet`

NOAA's Integrated Surface Database, the Met Office synoptic and airfield observations as
exchanged on the WMO network, open data without a key, one CSV per station and year from
`https://noaa-global-hourly-pds.s3.amazonaws.com/{year}/{USAF}{WBAN}.csv` (`sense-energy
fetch-isd`, 76 stations, 1.6 GB) plus the station history `isd-history.csv`. The mirror holds
files to **24 August 2025** (nothing for 2026), so the station thermometer covers 34 of the
record's 41 months, every summer included. `sense-energy build-isd` writes:

| Table | Grain | Columns |
|---|---|---|
| `isd_hourly.parquet` | station × clock hour | `t_c`, `td_c` (°C; `TMP`/`DEW` tenths with quality codes 0, 1, 4, 5, 9; daily/monthly summaries skipped; observations averaged within the hour) |
| `isd_site_stations.parquet` | site × candidate | the 3 nearest stations within 60 km, ranked |
| `isd_site_hourly.parquet` | site × hour | the nearest candidate that reported that hour, with `station_id`, `distance_km` |
| `weather_daily_isd.parquet` | site × local day | `t_mean`, `t_min`, `t_max`, `t_eve`, `n_hours` (≥ 18 required), `distance_km` |

361,260 site-days for 275 sites; station distance median 13.5 km (10th–90th percentile 5.7–29.2 km).
Against the stations ERA5's daily mean has a bias of +0.06 °C and an RMSE of 0.79 °C, its daily
maximum an RMSE of 1.18 °C, and on the stations' 30+ °C days it reads 1.3 °C too cool.

## Weather-effect heterogeneity: `code/reports/weather_heterogeneity/`

`sense-energy weather-heterogeneity` (`analysis/weather_heterogeneity.py`), on the sites without
CHP with the doses and specification B of `weather_causal`.

| File | Grain | Contents |
|---|---|---|
| `site_slopes.csv` | site × dose | `slope`, `se` (log points), `slope_pct`, `se_pct`, `posterior` / `posterior_pct` (empirical-Bayes), `shrinkage`, `pooled`, `tau`, `i2`, plus the site descriptors |
| `site_descriptors.csv` | site | `use` (ERIC site type collapsed to five groups), `cluster` (k = 4 demand shape), `trust_type`, `log_mean_kw`, `mean_build_year`, `summer_tmax`, `gas_to_elec`, `led_pct`, `region` |
| `moderators.csv` | moderator × dose × level | one-at-a-time dose × moderator interactions, `coef_pct` in percentage points per °C (categorical: against the reference; continuous: per SD), `p_joint` for the moderator |
| `moderators_joint.csv` | dose × term | the same with building use and all continuous moderators together (87 sites) |
| `hourly_by_use.csv` | hour × use × band | demand on days below 2 °C and at or above 22 °C against a 14–16 °C day, by building use |

Site-dose slopes whose clustered variance is undefined (three in the current run) are dropped
from the shrinkage and listed in the log.

## What moves heat: `code/reports/heat_drivers/`

`sense-energy heat-drivers` (`analysis/heat_drivers.py`), on the sites without CHP and without a
`meter_gaps` coverage flag (see `site_coverage_flags.parquet`), specification B of `weather_causal`.
Slopes are in log points (`*_pct` columns: × 100).

| File | Grain | Contents |
|---|---|---|
| `plain_terms.csv` | site × term | cold (`hdd14`) and heat (`cdd20`) slopes against temperature alone, on this module's sample |
| `site_terms.csv` | site × term | the same with a site-specific sunshine term (`solar_kwh_m2`, ERA5 surface solar radiation, kWh/m²/day) and a heat × weekend × site term: `cdd20` is the weekday heat slope, `cdd20_weekend` the weekend slope, `cdd20_weekend_diff` their difference |
| `night_day_terms.csv` | site × term | the heat dose split: `hot_afternoon` (per °C of daily maximum above 24 °C) and `warm_night` (per °C of daily minimum above 12 °C), with cold and sunshine |
| `block_slopes.csv` | site × block × term | cold, heat and sunshine slopes on the mean demand of each six-hour local block (`block` 0–3: night 00–06, morning, afternoon, evening) |
| `site_features.csv` | site | everything above side by side (`heat_plain`, `heat`, `sun`, `heat_weekend*`, `heat_b0..3`, `sun_b0..3`, `hot_afternoon`, `warm_night`, each with `_se`), derived `heat_afternoon_excess` (afternoon minus night block), `sun_day` (precision-weighted morning + afternoon), `sun_night`; `type` (rises with heat / rises with heat, falls with sunshine / falls with sunshine only / falls with heat / neither, at 5 %), `sign` (the daily heat slope's sign group), flags `afternoon_peaked`, `weekday_only`, `night_sunshine_negative`, `cooling_afternoon`, `heating_tail`; `use`, `mean_kw`, `region`, `urban`, `overheats`, `solar_eric`, `coverage_flag` |
| `site_descriptors.csv` | site | the heterogeneity descriptors plus `elec_kwh_per_m2`, `log_elec_per_m2`, `max_demand_kw`, `peak_w_m2`, `heat_pumps` / `heat_pumps_any`, `overheating_occurrences` / `overheats`, `solar_generated_kwh`, `solar_consumed_kwh` / `solar_eric`, `tenure` / `tenure_group`, `urban`, `density_km2` / `log_density`, `repd_solar_mw`, `repd_solar_status`, `night_day_ratio`, `weekend_ratio`, `winter_summer_ratio`, `daily_cv` |
| `group_decomposition.csv` | model × sign group × term | the three sign groups' pooled slopes, model `daily mean` (cold, heat, sunshine) and `night and day` (cold, hot afternoon, warm night, sunshine) |
| `moderators_heat.csv` | moderator × term × level | one-at-a-time interactions of cold, heat and sunshine with each moderator (`dose` column), `coef_pct` in percentage points per unit of the term; `p_joint` tests the heat interactions of the moderator |
| `moderators_heat_joint.csv` | term | building use, urban form, density, electricity intensity, peak W/m², night/day and weekend ratios together |
| `variance_explained.csv` | descriptor set | random-effects meta-regression of the sunshine-controlled heat slopes: `tau_before_pct`, `tau_after_pct`, `r2_true` (share of the true between-site variance explained; the first row is the sunshine term itself, τ before and after it) |
| `excluded_sites.csv` | site | the flagged sites left out, with their heat slopes from the heterogeneity run |
| `summary.json` | — | the headline numbers |

## Forecasts: `code/data/external/weather/aifs_ens/` → `interim/forecast_aifs_ens.parquet`

ECMWF AIFS-ENS from ECMWF open data, **CC BY 4.0 - "Contains ECMWF open data"**.
Harvested by `sense-energy harvest-forecasts` from the open-data archive, which holds ~4 days;
there is no historical AIFS before 2025. One NetCDF per run, `YYYYMMDDTHHz.nc`, dims
`(step, number, latitude, longitude)`.

**Retired 9 November 2026.** The daily cron harvest was removed: no experiment used AIFS (they use the IFS ensemble), and the harvested runs (5 Sep – 9 Nov 2026) start after the electricity record ends (8 May 2026). Do not reschedule it; AIFS for past dates can be hindcast locally on the GPU instead. The 130 runs on disk are kept.

| | |
|---|---|
| Runs kept | 00z, 12z (06z/18z exist) |
| Lead times | 0–72 h, 6-hourly (configurable to 360 h) |
| Members | 51: `number` 0 = control (`cf`), 1–50 = perturbed (`pf`) |
| Grid | 0.25°, same area as ERA5 |

| Column | Units | Notes |
|---|---|---|
| `site_code`, `run_time`, `valid_time`, `step_hours`, `member` | UTC | `valid_time = run_time + step` |
| `t2m`, `d2m`, `skt` | K | |
| `tcc`, `lcc` | % | **percent here, fraction in ERA5** |
| `tp`, `sf` | kg m⁻² (= mm) | **accumulated from run start**, not per step; difference consecutive steps |
| `ssrd`, `strd` | J m⁻² | accumulated from run start |
| `u10`, `v10` | m s⁻¹ | |
| `sp` | Pa | |

On members: AIFS-ENS samples its members from a diffusion model with
EDA-perturbed initial conditions rather than perturbing model physics, but the
data are 50 perturbed fields plus a control per parameter and step.

## Prices: `interim/prices_*.parquet`

Built by `sense-energy fetch-prices`. Rationale in
[ADR 0003](adr/0003-weather-and-price-data-sources.md).

### `prices_agile.parquet` - the day-ahead-known signal

Octopus Energy Agile import tariff, half-hourly, 14 GSP regions. A published,
deterministic transform of the N2EX day-ahead hourly auction (multiplier +
peak adder, capped), released the afternoon of D-1. Retail p/kWh, **not**
wholesale £/MWh. One row per (`region`, `datetime`) from the product on sale
that day; `prices_agile_all_products.parquet` holds every product.

| Column | Notes |
|---|---|
| `datetime`, `valid_to` | UTC, 30-min |
| `product` | which Agile product supplied the rate - products differ in caps, so a change is a step |
| `region` | GSP group A–P (no I, O). Sites carry `gsp_group` in `sites_geo.parquet` |
| `rate_exc_vat`, `rate_inc_vat` | p/kWh |

Product chain over the window: `AGILE-FLEX-22-11-25` (to 2023-12-11) →
`AGILE-23-12-06` (to 2024-04-02) → `AGILE-24-04-03` (to 2024-09-30) →
`AGILE-24-10-01`. `AGILE-18-02-21` keeps publishing throughout and is the
fallback.

### `prices_elexon_mid.parquet` - ex-post wholesale reference

Elexon Market Index Data: APX (`APXMIDP`) and N2EX (`N2EXMIDP`) volume-weighted
price and volume per settlement period, £/MWh. Published after delivery -
**evaluation only, never a feature**.

### Not available

ENTSO-E day-ahead prices for GB (`10YGB----------A`): none published since GB
left EU market coupling in January 2021, and the REST host answers 404 for
every zone at the time of writing. Nord Pool's data portal requires a
subscription.


---

## Forecasts (historical): `code/data/external/weather/tigge_ens/` → `interim/forecast_tigge_ens.parquet`

ECMWF IFS ENS from the **TIGGE** archive via the **ECMWF Data Store (ECDS)**,
which replaced the Public Datasets Web API for TIGGE on 2026-05-27. Research
use under the TIGGE licence (CC BY-NC 4.0 for the ECMWF origin), 48 h delay.
`sense-energy fetch-tigge` (resumable, one data-store request per type-month)
then `sense-energy build-tigge`. Needs an ECDS token in `.env`.

This is *what the forecast said* for every day of the training window — the
counterpart to ERA5 (*what happened*). A day-ahead model backtested on this,
not on ERA5, sees the accuracy it would really have had.

| | |
|---|---|
| Runs | 00z, 12z |
| Lead times | 0–72 h, 6-hourly (TIGGE goes to 384 h) |
| Members | 51: `cf` control = member 0, `pf` = 1–50 |
| Grid | ECDS serves the model's **native reduced grid** (739 points in the box, ~9 km), not a regular 0.5° box; sites take the nearest point |
| Storage | Each month is extracted to the located sites on arrival (`control_forecast_sites/`, `perturbed_forecast_sites/`, float32 parquet) and the GRIB deleted (perturbed months are ~1 GB) |

| ECDS variable | Short name | Units (as delivered) | Notes |
|---|---|---|---|
| `2_m_temperature` / `2_m_dewpoint_temperature` | `t2m` / `d2m` | K | |
| `maximum_/minimum_2_m_temperature_in_the_last_6_hours` | `mx2t6` / `mn2t6` | K | over the preceding 6 h; NaN at step 0 |
| `10_m_u/v_component_of_wind` | `u10` / `v10` | m s⁻¹ | |
| `surface_pressure` | `sp` | Pa | |
| `skin_temperature` | `skt` | K | |
| `total_precipitation` / `snow_fall_water_equivalent` | `tp` / `sf` | **kg m⁻² (= mm)** | **accumulated from run start** |
| `total_cloud_cover` | `tcc` | **%** | fraction in ERA5 |
| `surface_net_solar_radiation` / `surface_net_thermal_radiation` | `ssr` / `str` | J m⁻² | **net** in TIGGE (ERA5 gives *downward*) — not directly comparable; accumulated |
| `sunshine_duration` | `sund` | s | accumulated |

No wind-gust field exists in the TIGGE single-level set. Output columns mirror
`forecast_aifs_ens.parquet`: `site_code`, `run_time`, `valid_time`,
`step_hours`, `member`, then the variables.

---

## NESO: `code/data/external/neso/`, `code/data/geo/neso/` → `interim/*.parquet`

NESO data portal (CKAN, `api.neso.energy`), NESO Open Data Licence —
"Contains NESO open data". `sense-energy fetch-neso` (manifest-driven, by
resource name) then `sense-energy build-neso`.

### Future Energy Scenarios — the "data workbook", sheet by sheet

The FES workbook is published on the portal as one dataset per sheet. Kept for
FES 2023, 2024 and 2025. Pathways: FES 2024 *Holistic Transition, Electric
Engagement, Hydrogen Evolution, Counterfactual* (+ *Five Year Forecast*);
FES 2025 replaces Counterfactual with *Falling Behind* (+ *Ten Year Forecast*).

| Interim table | Sheet | Grain | Use |
|---|---|---|---|
| `fes_ed1_demand_summary` | ED1 | data item × pathway × year | **GB annual (GWh) and peak (GW) demand**; `statistic ∈ {Annual [Fiscal], Peak, Min (06:00), Min (14:00)}`; `neso.peak_demand()` filters the peaks |
| `fes_es1_supply` | ES1 | connection × pathway × technology × year | capacity/output; `connection == "Distributed"` is the **embedded generation** fleet |
| `fes_flx1_flexibility` | FLX1 | flexibility type × pathway × year | interconnectors, storage, DSR |
| `fes_building_blocks` | BB | pathway × building block × GSP × year | per-GSP technology counts/capacities, joined to definitions |
| `fes_regional_demand` | regional | scenario × GSP × year | **peak / AM / PM demand per GSP (MW)**, 2023–2050; 2024 is the latest published |
| `fes_regional_distributed_generation` | regional | scenario × GSP × technology × year × size band | capacity and contribution at winter peak / summer AM / PM |
| `fes_gsp_info` | regional | GSP | GSP id, group, name, lat/lon |

`fes_year` on every row says which publication the value came from; scenario
codes are expanded via `neso.SCENARIO_NAMES` (HE/EE/HT/CF/FB…). Years in the
regional files arrive as two digits and are expanded.

### National demand — known-beforehand covariates

| Interim table | Source | Grain | Notes |
|---|---|---|---|
| `national_demand_halfhourly` | Historic Demand Data | 30 min, UTC | `nd` national demand, `tsd` transmission system demand, `england_wales_demand`, **embedded wind/solar generation and capacity**, interconnector flows. 2019–2026 |
| `embedded_forecast_archive` | Embedded Wind & Solar Forecasts archive | 30 min target × issue time | every forecast NESO issued (`issued_at`, `lead_hours`). Use `neso.latest_forecast_before(min_lead_hours)` so a feature never uses a forecast issued after the model's own issue time. 2022–2026 |

Settlement periods are local-clock half hours; conversion to UTC is exact on
the 46- and 50-period clock-change days.

### GSP boundaries → sites

`code/data/geo/neso/`: GSP regions 2026-02-09 (362 regions, EPSG:4326 and
27700, GeoPackage + GeoJSON), 2022-03-14 (333), the tRESP GSP areas 2025 (235),
and the GSP/GNode/region lookup with GSP group per node. Sites gain two columns
in `sites_geo.parquet`:

| Column | Source | Notes |
|---|---|---|
| `gsp_region` | NESO 2026-02 polygons, point-in-polygon | e.g. `LEGA_1`; coastal misses snapped to the nearest region within 5 km |
| `gsp_group_neso` | same polygons | A–P; cross-checked against the Octopus-derived `gsp_group`, disagreements logged |

`gsp_region` is the join key to `fes_regional_demand.gsp_id` /
`fes_regional_distributed_generation.gsp_id`, which is how a hospital gets its
GSP's scenario demand pathway.


---

## Processed: `code/data/processed/` — see [data_manifest.md](data_manifest.md)

Analysis-ready datasets built by `sense-energy build-processed` (`make processed`):
electricity and gas × meter / site / trust, each as a raw and a profile-KNN-imputed
wide matrix and long table with a series index, plus `site_attributes`. The
manifest is generated by the build and records every step, parameter, exclusion,
count and file hash.


---

## NESO covariates: `interim/neso_covariates_halfhourly.parquet`, `interim/neso_site_covariates.parquet`

Built by `sense-energy build-neso-covariates` (config `neso_covariates.yaml`).
Half-hourly UTC grid 2022-12-01 → 2026-05-31 (61,297 rows), MW unless stated.
**Suffixes say when a value is knowable**: `_actual` is outturn (only legal as
a lag); `_da` is known by the **day-ahead cutoff, 16:00 UTC on D−1** (the
hour Agile prices are published), each with its `*_issued_at` / `*_published_at`.

| Column group | Source | Notes |
|---|---|---|
| `nd_actual_mw`, `tsd_actual_mw`, `england_wales_demand_actual_mw` | Historic Demand Data | national / transmission-system / England & Wales demand |
| `embedded_{wind,solar}_actual_mw`, `embedded_{wind,solar}_capacity_mw`, `embedded_{wind,solar}_cf_actual` | Historic Demand Data | embedded generation and capacity factor |
| `pump_storage_pumping_actual_mw`, `interconnector_net_import_actual_mw` | Historic Demand Data | net import = sum of all interconnector flows |
| `genmix_*_actual_mw`, `genmix_*_actual_pct`, `carbon_intensity_actual_gco2_kwh` | Historic GB Generation Mix | `DATETIME` taken as UTC |
| `nd_forecast_da_mw`, `nd_forecast_da_published_at` | Day-ahead HH forecast performance | NESO's own half-hourly day-ahead ND forecast, published ~09:45 UTC on D−1; aligned by settlement date/period. MAPE vs outturn 2.5% |
| `triad_avoidance_estimate_mw` | same | NESO's estimate of demand avoided at triad-risk periods |
| `embedded_{wind,solar}_forecast_da_mw`, `embedded_forecast_da_issued_at` | Embedded forecast archive | latest forecast issued before the cutoff |
| `nd_forecast_{1,2,7,14}d_{peak,trough}_da_mw`, `*_issued_at` | Cardinal-point forecasts | daily forecast peak (max of `P` points) and trough (min of `T` points) for the target day, latest issue before the cutoff, broadcast to the day's periods. 2-day file starts 2023-04 |
| `nd_forecast_da_error_actual_mw` | derived | outturn − day-ahead forecast; ex-post |
| `settlement_date_local`, `day_ahead_cutoff_hour_utc` | derived | |

`neso_site_covariates.parquet` (one row per site, 275 with a GSP): the FES 2024
regional breakdown for the site's GSP region (composite regions summed):
`fes_gsp_peak_2024_mw`, `fes_gsp_peak_2030_{electric_engagement,holistic_transition,hydrogen_evolution}_mw`
(peak demand summed over demand types), `fes_gsp_dg_{solar,wind,battery}_2024_mw`
(distributed generation capacity, both size bands), `fes_gsp_n_parts`.

Not included: NESO's carbon-intensity *forecast* (only the live file is on the
portal; the historic forecast archive is on the separate Carbon Intensity API).

---

## Mobility (ESC / The Oxford Partnership): `interim/mobility_*.parquet`

*Aggregate mobile data visitation for UK campus* — one trust only. Daily
mobile-phone visit counts for calendar 2024 at three Royal Devon University
Healthcare sites and a university campus comparator, plus static visitor
profiles. Built by `sense-energy build-mobility` from
`code/data/external/esc_mobility/` (git-ignored).

| id | Site | 2024 mean visits/day | Demand coverage 2024 |
|---|---|---|---|
| `RH801` | Royal Devon & Exeter Hospital (Wonford), general acute, GIA 125,557 m² | 5,183 | electricity 100% (3 active meters), gas partial |
| `RH802` | Heavitree Hospital, day hospital | 1,575 | electricity 100% |
| `RH8K6` | NHS Nightingale Hospital Exeter, day hospital | 461 | electricity 100%, gas 93% |
| `university` | campus comparator, not an NHS site | 6,262 | — |

| Table | Columns | Notes |
|---|---|---|
| `mobility_visits_daily` | `id, site_code, date, visits` | RH802 carries one extra day (2025-01-01) as delivered |
| `mobility_visitor_profiles` | `id, site_code, profile, category, pct, id_corrected` | profiles: `age_band`, `household_income_quintile`, `consumer_style` (MBI Consumer Styles A–J), `travel_distance_m`. In the styles and travel-distance files the fourth row was labelled `RH801` a second time where the other files say `university`; it is relabelled and flagged `id_corrected` |

**Licence** (EULA ESC3888-3.0, Schedule 1): research purposes only; no
attempt to identify any person, premise or site; access-controlled storage;
delete all copies on expiry; Schedule 3's secure-environment option is not
ticked, so local copies are permitted. **Required attribution in any output**:
*"This [report/product] contains data provided by The Oxford Partnership. Other
than providing certain data sets, The Oxford Partnership has not been involved
in the development of this [report/product] and accepts no responsibility
whatsoever for the accuracy, completeness or reliability of this
[report/product], nor for any advice contained within it or for the use or
reliance any reader may give to it."*

**How it can be used**: as an observed occupancy proxy for a three-site case
study in 2024 — not as a forecasting feature (footfall is not known ahead of
time unless forecast itself), and not estate-wide (one trust).


---

## Forecasts (historical, 51 members): `code/data/external/weather/ifs_ens/sites/` → `interim/forecast_ifs_ens/YYYY-MM.parquet`

ECMWF IFS ENS from the **AWS mirror of ECMWF open data** (CC BY 4.0,
"Contains ECMWF open data"), harvested by `sense-energy harvest-ifs-ens`:
each ensemble step's `.index` gives every message's byte offset, so only the
wanted parameters and members are fetched by HTTP range (32 connections),
decoded with ecCodes, cropped to the site box and reduced to the nearest grid
point per site. One parquet per run; `build-ifs-ens` concatenates by month.

| | |
|---|---|
| Period | 18 Jan 2023 → (the mirror's first day); 00z and 12z |
| Members | **51**: `member` 0 = control, 1–50 perturbed |
| Lead times | 0–48 h, 6-hourly (3-hourly exists to 144 h) |
| Grid | **0.4°** to 28 Feb 2024 (`grid = 0p4`), **0.25°** from 29 Feb 2024 (`grid = 0p25`) |
| Parameters | `t2m d2m u10 v10 tp ssrd strd sp` (K, m s⁻¹, m accumulated, J m⁻² accumulated, Pa) |

⚠️ **Dewpoint and radiation (`d2m ssrd strd`) exist only from 2024-03-06.** Before
that the ensemble stream carried `2t 10u 10v tp sp` plus fields we do not keep
(`msl skt` and upper-air) — so those three columns are absent for
2023-01-18 → 2024-03-06, independent of the grid change. No cloud cover exists in
the open-data ensemble at all. Columns: `site_code, run_time, valid_time,
step_hours, member, grid` + parameters.

**Mirror gaps**: 13 runs are absent from the AWS mirror altogether — 26 Apr
12z → 30 Apr 2023 (9 runs) and 1 → 2 May 2023 (4 runs); no source can fill
them except TIGGE. Other runs occasionally answer `503 Slow Down` (S3
throttling); the harvester backs off and re-passes until they land.

Relationship to TIGGE: same model family; TIGGE control (42 months, 14
variables) and TIGGE perturbed (10 members, still being served by ECDS)
remain; this source supplies the full 51-member spread from Jan 2023.

## Public NHS activity statistics (`interim/nhs_activity_*.parquet`)

Pulled by `sense-energy fetch-nhs-activity` from the NHS England statistics site (see `doc/external_activity_data.md`).

| File | Key | Content |
|---|---|---|
| `nhs_activity_ae_monthly.parquet` | `ods_code`, `period` (month start) | A&E attendances by department type, booked appointments, 4-hour breaches, 4–12 h and 12 h+ waits from decision to admit, emergency admissions via A&E and other; provider level, Apr 2023 – Mar 2026 |
| `nhs_activity_ambsys_monthly.parquet` | `ods_code`, `period` | Ambulance Systems Indicators (A0–A139: calls, incidents by category, response-time mean/90th, hear-and-treat, see-and-treat), per ambulance service, Aug 2017 – Aug 2026 |
| `nhs_activity_kh03_quarterly.parquet` | `ods_code`, `sector`, `snapshot` (quarter end) | overnight beds available and occupied, occupancy rate, 2001 – Jun 2024 |
| `nhs_activity_trusts.parquet` | `organisation_name` | the demand panel's trusts with their ODS code (site-code prefix or name match), AmbSYS code for ambulance services, and flags for presence in each source |

## ERIC 2023/24 and KH03 occupied beds (`interim/eric_*`, `interim/kh03_*`)

Source archives (ESC extracts) unpacked in `external/eric/` and `external/nhs_beds/`; built by `sense-energy estates-analysis` (loaders `data/eric.py`, `data/nhs_beds.py`). Analysis and figures: `doc/estates_analysis.md`.

| File | Key | Content |
|---|---|---|
| `eric_2023_24_site.parquet` | `site_code` | ERIC 2023/24 (1 Apr 2023 – 31 Mar 2024), 2,869 English NHS sites: total electricity as the sum of the five consumption components (`elec_kwh`; the headline field is filled for ~5 % of sites and kept as `elec_headline_kwh`), gas, oil, steam, hot water, renewable heat, solar generated, maximum demand (kW), available capacity (kVA), floor area, heated volume, floor-area-weighted mean construction year, LED coverage, heat pumps, CHP units, old boilers, EV charge points, single rooms, high/significant-risk backlog, overheating occurrences that triggered a risk assessment (clinical areas above 26 °C, field S16_07: `overheating_occurrences`), trust-owned and third-party solar consumed (`solar_consumed_kwh`, `third_party_solar_kwh`), site type and tenure, postcode centroid (`latitude`, `longitude`, postcodes.io), intensities (`elec_kwh_per_m2`, `gas_kwh_per_m2`, `gas_to_elec`) |
| `eric_2023_24_trust.parquet` | `trust_code` | the trust-level ERIC table as published (210 trusts) |
| `eric_2024_25_site.parquet`, `eric_2024_25_trust.parquet` | `site_code` / `trust_code` | ERIC 2024/25 (1 Apr 2024 – 31 Mar 2025, published 16 Oct 2025; NHS England files in `external/eric_2024_25/`, latin-1, column names normalised to the 2023/24 spelling), 2,901 sites and 207 trusts, same columns as 2023/24 plus `occupied_floor_area_m2`, `chp_elec_output_kwh`, `chp_thermal_output_kwh`, `chp_gas_input_kwh`, `chp_other_input_kwh`, `chp_elec_purchased_kwh`, `chp_thermal_purchased_kwh`, `chp_thermal_wasted_kwh`, `exported_elec_kwh`, `exported_thermal_kwh`, `chp_elec_kw`, `chp_thermal_kw`, and an age band 2025–2034; the headline `elec_headline_kwh` field was dropped from the collection (missing). `collection` names the year. Built by `sense-energy build-eric --year 2024_25` |
| `eric_2023_24_postcodes.parquet` | `postcode` | postcode centroid cache for the ERIC sites |
| `kh03_occupied_by_sector.parquet` | `ods_code`, `quarter_end`, `sector` | average occupied beds per quarter, `overnight` and `day_only`, sectors general & acute / maternity / mental illness / learning disabilities; 2001 (overnight) or 2010 (day only) to June 2024 |
| `kh03_occupied_by_specialty.parquet` | `ods_code`, `quarter_end`, `specialty` | the same by 78 treatment-specialty codes, 2010 – June 2024 |
| `site_coverage_flags.parquet` | `site_code` | metered vs ERIC electricity 2023/24 per panel site: `coverage_flag` (`ok`, `partial_supply`, `aggregate_supply`, `chp_mismatch`, `meter_gaps`, `no_eric`), `energy_ratio`, `peak_ratio` |

## Site context: `interim/site_context.parquet`

`sense-energy build-site-context` (`data/site_context.py`), one row per site of `sites.parquet`. Three
open sources:

| Column(s) | Source | Content |
|---|---|---|
| `oa21`, `lsoa21`, `lsoa_name`, `bua`, `ruc11`, `ruc21` | postcodes.io (ONS open data) | 2021 census geographies of the site postcode (output area, LSOA, built-up area) and the rural-urban classification 2011 (output-area codes A1–F2) and 2021; empty for the six terminated postcodes |
| `urban` | derived | `ruc11` collapsed: major conurbation (A1), minor conurbation (B1), city and town (C1, C2), rural (D–F) |
| `density_km2` | ONS Census 2021 table TS006 via the Nomis API, `external/ons/census2021_ts006_density_lsoa21.csv` (35,672 LSOAs, England and Wales) | usual residents per km² of the site's LSOA |
| `repd_solar_schemes`, `repd_solar_mw`, `repd_solar_status`, `repd_solar_operational_mw`, `repd_solar_first_operational`, `repd_battery_mw` | DESNZ Renewable Energy Planning Database, quarterly extract `external/repd/REPD_Publication_Q2_2026.csv` (schemes of 150 kW and above, England) | solar PV and battery schemes at the site's full postcode: count, summed capacity, best planning status, capacity operational by the end of the electricity extract (8 May 2026) and its first operational date. In the Q2 2026 extract 24 sites have a scheme at their postcode and none was operational within the sample: these are the 2025 NHS rooftop-solar applications, awaiting construction |

## Display Energy Certificates: `code/data/external/dec/` → `interim/dec_certificates.parquet`, `interim/dec_sites.parquet`

`sense-energy build-dec` (`data/dec.py`). The MHCLG energy performance of buildings data API
(`api.get-energy-performance-data.communities.gov.uk`, bearer token `EPB_API_KEY` in `.env`, Open
Government Licence v3.0): `GET /api/display/search?postcode=` for every site postcode, then
`GET /api/certificate?certificate_number=` for each certificate found; both cached as JSON under
`external/dec/search/` and `external/dec/certificates/`, so a rebuild with `--no-fetch` reads the cache.
Every occupied public building over 250 m² displays a DEC, renewed yearly above 1,000 m²; a hospital
postcode carries one certificate per building per year, back to 2008.

| File | Grain | Contents |
|---|---|---|
| `dec_certificates.parquet` | certificate | flattened record: `certificate_number`, `uprn`, address, `postcode_norm`, `property_type`, `schema_type`, `registration_date`, `nominated_date`, assessment period, `floor_area_m2`, `main_heating_fuel`, `building_environment` and its collapsed `environment` (air conditioning / mixed mode / heating with mechanical or natural ventilation / unconditioned), `hvac_system`, `ac_present` and `ac_yes`, `ac_rated_kw` (missing when the certificate flags the rating unknown), `operational_rating`, metered `electricity_kwh` and `thermal_kwh` (all non-electric fuels, `thermal_fuels`) of the assessment year, `elec_kwh_m2` / `thermal_kwh_m2` and their typical benchmarks, renewables, `electric_heating` (main fuel is grid electricity) |
| `dec_recommendations.parquet` | recommendation | the advisory report linked from each latest certificate (`related_certificate_number`, newer schemas only): `advisory_number`, `dec_number`, `payback` (short / medium / long / other), `code`, `recommendation` text, `co2_impact`, `themes` (cooling, heating, hot water, lighting, controls, fabric, ventilation, renewables, by keyword), the report's floor area and building environment |
| `dec_sites.parquet` | site | the latest certificate of each building at the site postcode (buildings keyed by UPRN, else by first address line), aggregated by floor area: `dec_buildings`, `dec_floor_area_m2`, `dec_ac_share` (floor share with air conditioning present), `dec_ac_buildings`, `dec_ac_kw` / `dec_ac_kw_m2` (where rated output is known, `dec_ac_kw_known_share`), `dec_cooled_share` (air conditioning or mixed mode), `dec_mech_vent_share`, `dec_env_*_share` per environment, `dec_main_environment`, `dec_electric_heating_share`, `dec_elec_kwh_m2`, `dec_thermal_kwh_m2`, `dec_operational_rating`, `dec_renewables_electrical`, `dec_electricity_kwh`, `dec_latest_nominated` / `dec_oldest_nominated`, `dec_property_types`, `dec_buildings_ever` (buildings with any certificate, against `dec_buildings` whose latest is from 2019 on), and the recommendation counts `rec_buildings`, `rec_total`, `rec_short_payback`, `rec_<theme>` with `rec_<theme>_building_share` |

| `epc_certificates.parquet`, `epc_sites.parquet` | certificate / site | the same for non-domestic EPCs (`sense-energy build-epc`, API kind `non-domestic`, cache `external/dec/epc_search/`, `epc_certificates/`): asset-rated buildings DECs do not cover (leased premises, new builds), with `asset_rating` instead of an operational rating; `epc_sites` carries the `epc_*` twins of the `dec_*` columns, latest certificate per building since 2016 (ten-year validity). `heat_drivers` fills a site's `dec_*` descriptors from its EPCs when it has no DEC (`cert_source` = DEC or EPC) |

Matching is by the site's full postcode, so buildings of a hospital that carry a different postcode
are missed and a shared postcode can bring in a neighbour's buildings; `heat_drivers` reports
`dec_coverage`, the DEC floor area over the ERIC gross internal area, as the check.

## Next-step analyses: `code/reports/estates/eric_changes*`, `forecast_heat_days/`, `heat_causal_ml/`

| File | Grain | Contents |
|---|---|---|
| `estates/eric_changes.csv` | panel site in both ERIC returns | 2023/24 and 2024/25 values and deltas of solar generated and consumed, heat pumps, LED share, CHP units, floor area, electricity, gas, EV points, overheating, max demand, capacity; flags `solar_new`, `solar_grew`, `heat_pumps_new`, `heat_pumps_more`, `led_up_10pts`, `chp_changed`, `any_change`, `chp_site`; the nearest demand step in or near 2024/25 (`steps_in_window`, `step_date`, `step_change`) |
| `estates/demand_steps.csv` | step | PELT changepoints in each site's log daily demand net of day-of-week (`analysis/estates_changes.py`): `date`, `change` (relative level change), `level_before_kw`, segment lengths |
| `forecast_heat_days/scores_by_day_class.csv` | model × day class | day-ahead skill by the target day's class (ERA5 daily maximum ≥ 28 °C, 25–28, other June–September, daily mean ≤ 2 °C, rest): site-mean `nmae`, `nmae_median`, `ncrps`, `coverage_80`, `skill_vs_naive`, `nmae_vs_lightgbm_ifs`; `per_site.csv` the per-site values |
| `heat_causal_ml/cate_site.csv` | site × covariate set | learned heat effects per site (`tau_rlearner`, `tau_forest` with `tau_forest_lo/hi`), the fixed-effect slope (`slope_fe`, `slope_fe_se`), `set` = observables or observables + signature |
| `heat_causal_ml/importance_*.csv` | covariate | causal-forest feature importance per covariate set |
| `heat_causal_ml/summary.json` | — | agreement of the learned effects with the regression slopes (correlation, calibration slope, share of true variance recovered), placebo with shuffled covariates, calendar modifier effects |

## JEPA site embeddings: `code/reports/jepa/`

`sense-energy train-jepa` (`experiments/jepa.py`) and `sense-energy jepa-probe` (`analysis/jepa_probe.py`).
A joint-embedding predictive architecture trained on the half-hourly panel: 28-day windows of one
site's demand tokenised by day (48 half-hours scaled by the site's trailing 28-day mean, weekday
one-hot, season), seven target days per window (a block of four and three singles) whose
embeddings under an EMA target encoder the predictor must produce from the remaining 21 days;
smooth-L1 loss in embedding space plus a VICReg variance term.

| File | Grain | Contents |
|---|---|---|
| `site_embeddings.parquet` | site | `e0..e{d-1}`: mean target-encoder embedding of the site's valid days; `s_e0..`: the same over June–September; `days` embedded |
| `day_embeddings.parquet` | site × local day | the embedding of every valid day (from non-overlapping full windows; the tail window averaged where it overlaps) |
| `surprise.parquet` | site × local day | latent prediction error of the day when it alone is held out of its window: a learned anomaly / regime score |
| `chronos_embeddings.parquet` | site | mean-pooled Chronos-2 encoder states of the site's hourly demand over twelve 14-day windows spread through the history, level-normalised (the foundation-model comparison) |
| `probe.csv` | target × representation × probe model | out-of-fold `r2_oof`, `corr` and `share_true_variance` (DerSimonian-Laird) of the fixed-effect heat, cold and sunshine slopes predicted from each representation (JEPA, JEPA June–September, Chronos-2, observables, diurnal signature, combinations) by a ridge probe and by TabPFN, five site folds, repeated |
| `probe_classification.csv` | target × representation | out-of-fold accuracy of building use and DEC environment from each representation, against the majority class |
| `clusters.csv` | representation | k-means (k = 4) on each representation: adjusted Rand index against the demand-shape clusters and the share of heat-slope variance cluster membership explains |
| `jepa.pt`, `train_log.csv`, `summary.json`, `probe_summary.json` | — | weights and configuration, training curve (loss, embedding SD), headline numbers |

`code/reports/jepa_weather/` holds the same files for the weather-conditioned model
(`sense-energy train-jepa --weather`: the day tokens and the predictor's target tokens carry the
day's `t_mean`, `t_max`, `t_min`, `solar_kwh_m2`, `wind_mean`, `precip_mm`, standardised over the
panel), plus in `site_embeddings.parquet` the counterfactual **response embeddings**: `h0..h{d-1}`
and `h_norm`, the mean change in the predicted embedding of the last seven days of every full
window when their temperatures are raised by 5 °C; `s0..` and `s_norm` the same for +2 kWh/m² of
solar radiation; `windows` averaged.

## Latent PFN: `code/reports/latent_pfn/`

`sense-energy train-latent-pfn` (`experiments/latent_pfn.py`), the LaT-PFN route on the
weather-conditioned JEPA: a decoder (embedding → scaled 48-slot profile, MLP) and an in-context
transformer that predicts query days' embeddings from a context set of the same site's days
(embedding, weather, calendar) and the queries' weather and calendar, trained across sites on the
period before June 2025.

| File | Grain | Contents |
|---|---|---|
| `forecasts.parquet` | site × local day × variant | day-ahead profiles (`slot0..slot47`, kWh per half-hour) on the proof-of-concept site-days for `latent PFN + ERA5` (realised weather for the target day), `latent PFN + IFS ENS` (the 00z run of the day before, aggregated to daily variables) and `oracle: decoded true embedding` |
| `scores_by_day_class.csv` | variant × day class | site-mean nMAE by the kind of target day for the latent variants and, on the same site-days, seasonal naive, LightGBM (IFS, ERA5), Chronos-2 and TimesFM |
| `decoder.pt`, `pfn.pt`, `train_log.csv`, `summary.json` | — | weights, training curves, headline numbers |

## The weather-normalised estate: hourly response, savings, warming, heatwaves, solar, CHP dispatch

Write-up: `doc/weather_normalised_estate.md` (HTML page `doc/html/weather_normalised_estate.html`).

| File | Grain | Contents |
|---|---|---|
| `hourly_response/hourly_terms.csv` | site × hour × term | per local hour, the site's response of the hour's demand to the cold dose, heat dose and solar radiation (`slope`, `se`, log points) and its empirical-Bayes `posterior` (`sense-energy hourly-response`) |
| `hourly_response/pooled.csv` | hour × term | random-effects pooled response, its SE, between-site SD (`tau`), `i2` |
| `hourly_response/site_profiles.csv` | site | `heat_peak_hour`, `heat_redband_share` (share of the positive heat response in 16:00–19:00), daily mean, red-band, night and evening-excess heat response (%), cold response, `peak_hour_hot` / `peak_hour_other` (hour of the highest weekday demand on summer days ≥ 25 °C maximum and on other summer weekdays), `peak_ratio_hot` |
| `hourly_response/jepa_check.csv`, `jepa_check_days.csv` | site; site-day | the weather-conditioned JEPA's decoded change for a 5 °C warmer summer day against the regression's on the same days: `jepa_mean_pct`, `reg_mean_pct`, `shape_corr` |
| `installation_savings/nac.csv` | site × financial year | normalised annual consumption (`nac_kw`, mean kW under the common reference days), raw mean, model fit (`r2`, `cv_rmse`), cold and heat slopes (`sense-energy installation-savings`) |
| `installation_savings/site_changes.csv` | site | NAC and raw change 2023/24 → 2025/26, ERIC change flags, LED change in points, `group` (no change / LED only / solar new or grown / heat pumps installed) |
| `installation_savings/group_effects.csv`, `led_dose.csv` | group; term | group medians and the difference in medians from unchanged sites with bootstrap intervals; cross-site regression of the NAC change on the LED change (per 10 points), solar and heat pumps (HC3) |
| `estate_counterfactuals/warming_sites.csv`, `warming_summary.csv` | site × shift; shift | annual electricity change, summer peak, peak / available capacity, half-hours above 90 % of capacity, winter utilisation today and with heat pumps, for +1, +2, +4 °C (`sense-energy estate-counterfactuals`) |
| `estate_counterfactuals/heatwave_site_days.csv`, `heatwave_events.csv` | site-day; event × group | 2025 heatwave site-days: observed and counterfactual temperature, kWh, attributed kWh, share, wholesale cost (market index price), CO₂ (grid intensity); CHP sites' grid import via the group's binned response |
| `estate_counterfactuals/heatwave_surprise.csv` | group | the weather-blind JEPA's latent surprise on heatwave days against other summer days, Mann-Whitney p |
| `solar_watch/slopes.csv`, `breaks.csv` | site × summer; site | sunshine slope of daytime demand per summer (2023–2025), the 2023 → 2025 change, z, downward and upward break flags, ERIC solar and LED changes, REPD status (`sense-energy solar-watch`) |
| `chp_dispatch/daily.csv`, `hourly.csv`, `classification.csv` | coefficient; site | per CHP site and pooled: daily import response to the cold and heat doses and the daily price; hourly response to the power price and the working-day red band; each against the sites without CHP; flags `heat_led`, `price_following`, `red_band_avoiding` (`sense-energy chp-dispatch`) |
