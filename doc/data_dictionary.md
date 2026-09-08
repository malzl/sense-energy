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
| `is_outlier` | bool | Per-meter robust z-score > 10 (1.31%) |

### Cleaning decisions

| Decision | Rationale |
|---|---|
| Keep `reading_type` 1/2/3 only | Everything else is a different quantity or granularity |
| Meters **summed, never de-duplicated** | 33 of 146 sites have multiple meters (Homerton: 20). 3.1M rows share `site_code`+`datetime` and are distinct meters, not duplicates |
| 101 true duplicates dropped | Identical on (`mpxn`, `energy_type`, `reading_type`, `datetime`); last kept |
| 211,478 actual/estimate collisions resolved | Same meter and timestamp under two reading types; the actual (`1`) wins |
| Exact zeros → NaN (1,120,688; 10.49%) | Meter dropout written as zero. Concentrated — 8 sites are >50% zero, 94 of 145 under 1% — which is what distinguishes it from genuine low demand |
| Gaps ≤ 4 periods interpolated | Longer outages stay NaN rather than inventing a load profile |
| Outliers flagged, never dropped | A genuine spike is signal. Threshold is loose (z > 10), targeting meter faults |

### Known data quality issues

- **Gas maximum of 32,611 kWh/half-hour** (65 MW thermal) against a gas mean of
  204 — implausible, flagged as an outlier. Electricity peaks at 2,505
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

## Forecasts: `code/data/external/weather/aifs_ens/` → `interim/forecast_aifs_ens.parquet`

ECMWF AIFS-ENS from ECMWF open data, **CC BY 4.0 - "Contains ECMWF open data"**.
Harvested by `sense-energy harvest-forecasts`, which must run **daily**: the
archive holds ~4 days and there is no historical AIFS before 2025. One NetCDF
per run, `YYYYMMDDTHHz.nc`, dims `(step, number, latitude, longitude)`.

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
`sense-energy fetch-tigge` (resumable, one data-store request per type-month;
GRIB only) then `sense-energy build-tigge`. Needs an ECDS token in `.env`.

This is *what the forecast said* for every day of the training window — the
counterpart to ERA5 (*what happened*). A day-ahead model backtested on this,
not on ERA5, sees the accuracy it would really have had.

| | |
|---|---|
| Runs | 00z, 12z |
| Lead times | 0–72 h, 6-hourly (TIGGE goes to 360 h) |
| Members | 51: `cf` control = member 0, `pf` = 1–50 |
| Grid | 0.5°, same area as ERA5 |

| ECDS variable | Short name | Units | Notes |
|---|---|---|---|
| `2_m_temperature` / `2_m_dewpoint_temperature` | `t2m` / `d2m` | K | |
| `maximum_/minimum_2_m_temperature_in_the_last_6_hours` | `mx2t6` / `mn2t6` | K | over the preceding 6 h |
| `10_m_u/v_component_of_wind` | `u10` / `v10` | m s⁻¹ | |
| `surface_pressure` | `sp` | Pa | |
| `skin_temperature` | `skt` | K | |
| `total_precipitation` / `snow_fall_water_equivalent` | `tp` / `sf` | m | **accumulated from run start** |
| `total_cloud_cover` | `tcc` | 0–1 | |
| `surface_net_solar_radiation` / `surface_net_thermal_radiation` | `ssr` / `str` | J m⁻² | **net** in TIGGE (ERA5 gives *downward*) — not directly comparable; accumulated |
| `sunshine_duration` | `sund` | s | accumulated |

No wind-gust field exists in the TIGGE single-level set.

Output columns mirror `forecast_aifs_ens.parquet`: `site_code`, `run_time`,
`valid_time`, `step_hours`, `member`, then the variables.

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
