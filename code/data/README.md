# Data

**Nothing in this tree is committed to git.** `.gitignore` excludes every subdirectory;
only this file and the `.gitkeep` placeholders are tracked.

| Directory   | Contents                                                        | Written by |
|-------------|-----------------------------------------------------------------|------------|
| `raw/`      | Source files exactly as received. **Treat as read-only.**       | Humans     |
| `external/` | Third-party data: weather, bank holidays, tariffs, degree days   | Humans / API scripts |
| `interim/`  | Cleaned, reshaped intermediates                                  | Pipeline   |
| `processed/`| Model-ready tables (`demand.parquet`, `features.parquet`)        | Pipeline   |

## Adding a new source

1. Drop the file in `raw/` with a name that includes the source and extract date,
   e.g. `hh_meter_data_2026-09-01.csv`.
2. Add a reader in [`code/src/sense_energy/data/loaders.py`](../src/sense_energy/data/loaders.py).
3. Document every field, its units and its provenance in
   [`doc/data_dictionary.md`](../../doc/data_dictionary.md).
4. Never edit a file in `raw/` — corrections belong in the cleaning code, so they
   are versioned and reproducible.

## Expected raw files

| File                  | Description                                             |
|-----------------------|---------------------------------------------------------|
| `hh_meter_data.csv`   | Half-hourly electricity consumption per site            |
| `site_metadata.csv`   | Site register: trust, type, floor area, beds, lat/lon   |
| `weather.csv`         | Temperature, humidity, wind, irradiance per site        |

## `geo/`

Boundary layers and geocoded site centroids, fetched by `sense-energy fetch-geo`.
Git-ignored like the rest of the tree. Contents and licensing are documented in
[doc/data_dictionary.md](../../doc/data_dictionary.md#geography-codedatageo).

## `external/weather/`

| Path | Source | Fetched by |
|---|---|---|
| `era5/reanalysis/`, `era5/ensemble_members/` | ERA5 via CDS, one NetCDF per month | `sense-energy fetch-weather` (resumable) |
| `aifs_ens/` | ECMWF AIFS-ENS open data, one NetCDF per run | `sense-energy harvest-forecasts` (**daily**) |

Both are git-ignored. The AIFS archive only exists from the day the harvester
starts running - it cannot be backfilled.

## `external/neso/` and `geo/neso/`

NESO FES tables, national demand history, embedded forecast archive (~2.3 GB) and
GSP boundaries. `sense-energy fetch-neso` downloads by resource name from the
CKAN API; `sense-energy build-neso` tidies them into `interim/`.

## `external/weather/tigge_ens/`

IFS ENS historical forecasts from TIGGE via the ECMWF Data Store, one GRIB per
type-month. `sense-energy fetch-tigge` (needs an ECDS token and the TIGGE licence).
