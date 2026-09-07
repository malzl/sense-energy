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
