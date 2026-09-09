# Data manifest - processed demand datasets

Generated 2026-09-09T09:13:47+00:00 by `sense-energy build-processed` at commit `618da4f`.
Regenerate with `make processed`; this file and `code/data/processed/manifest.json` are written by the build and should not be edited by hand.

## Lineage

```
Energy Systems Catapult extract (raw)
  -> sense-energy build-interim   : reactive/cumulative rows dropped, meter grain, zeros->NaN,
                                     30-min grid, outliers flagged, gaps <= 2 h interpolated
  -> sense-energy build-processed : this manifest
```

## Layout

`code/data/processed/<energy>/<level>/` for energy in {elec, gas} and level in {meter, site, trust}:

| File | Content |
|---|---|
| `wide_raw.parquet` / `.csv` | time x series matrix, kWh per half hour, missing = NaN |
| `wide_imputed.parquet` / `.csv` | same, gaps filled by profile-KNN; heavily-imputed series omitted |
| `long_raw.parquet` | `series_id, datetime, kwh` - present values only |
| `long_imputed.parquet` | `series_id, datetime, kwh, imputed` (bool at meter level; imputed share of members above) |
| `series_index.parquet` / `.csv` | one row per series: window, coverage, imputation share, flags, site/trust attributes |

Plus `site_attributes.parquet` / `.csv` (site register with coordinates, GSP group/region) for clustering.

Conventions: timestamps are UTC period *starts* on a common 30-minute grid; units are kWh per half hour; `series_id` is the MPxN at meter level, `site_code` at site level and a normalised trust name at trust level.

## Steps

1. **Load interim**
   - rows: `11013277`
   - meters: `235`
   - sites: `146`
   - window: `['2022-12-07 00:00:00+00:00', '2026-05-08 23:30:00+00:00']`
   - source: `interim/consumption_halfhourly.parquet`

2. **Exclude placeholder sites**
   - regex: `Unknown`
   - sites_dropped: `['RFSUnknown', 'RH8Unknown', 'RL1Unknown', 'RQXUnknown', 'RXNUnknown']`
   - rows_dropped: `290592`

3. **Common time grid**
   - start: `2022-12-07 00:00:00+00:00`
   - end: `2026-05-08 23:30:00+00:00`
   - periods: `59952`
   - timezone: `UTC, period start`

4. **Site attributes**
   - rows: `275`
   - columns: `27 items: ['site_code', 'site_name', 'postcode', 'organisation_name', 'organisation_type', 'commissioning_region'] ...`

5. **elec: meter frame**
   - meters: `170`
   - outliers_set_missing: `142`

6. **elec: profile-KNN imputation at meter level**
   - method: `profile_knn`
   - k: `5`
   - window_weeks: `8`
   - same_day_type: `True`
   - min_observed_fraction_of_day: `0.25`
   - max_gap_periods: `48`
   - level_scaling: `True`
   - max_imputed_fraction_per_series: `0.1`
   - cells_missing_before: `1743070`
   - cells_imputed: `40572`
   - share_of_missing_filled: `0.0233`
   - days_skipped: `{'too_sparse': 34590, 'gap_over_24h': 110, 'no_neighbours': 2055}`
   - series_flagged_heavily_imputed: `['1100050353005']`

7. **elec: inactive meters excluded from totals**
   - min_member_coverage: `0.1`
   - meters: `['1030071208238', '1200010097523', '1419581430000', '2200030368412', '2200032196710', '2200042744540', '2200043162436']`

8. **elec: site totals**
   - sites: `134`
   - rule: `total = sum of meters only where every active meter has a value (observed or imputed); else NaN`
   - raw_complete_share: `0.8566`
   - imputed_complete_share: `0.8603`

9. **elec: trust totals**
   - trusts: `24`
   - rule: `total = sum of sites only where every active site has a value; else NaN`
   - raw_complete_share: `0.734`
   - imputed_complete_share: `0.7457`

10. **gas: meter frame**
   - meters: `56`
   - outliers_set_missing: `6`

11. **gas: profile-KNN imputation at meter level**
   - method: `profile_knn`
   - k: `5`
   - window_weeks: `8`
   - same_day_type: `True`
   - min_observed_fraction_of_day: `0.25`
   - max_gap_periods: `48`
   - level_scaling: `True`
   - max_imputed_fraction_per_series: `0.1`
   - cells_missing_before: `2307184`
   - cells_imputed: `26533`
   - share_of_missing_filled: `0.0115`
   - days_skipped: `{'too_sparse': 46534, 'gap_over_24h': 738, 'no_neighbours': 1776}`
   - series_flagged_heavily_imputed: `[]`

12. **gas: inactive meters excluded from totals**
   - min_member_coverage: `0.1`
   - meters: `['13542103', '1643806708', '7678762807', '81755703', '9330756405', '9382951702']`

13. **gas: site totals**
   - sites: `21`
   - rule: `total = sum of meters only where every active meter has a value (observed or imputed); else NaN`
   - raw_complete_share: `0.186`
   - imputed_complete_share: `0.1917`

14. **gas: trust totals**
   - trusts: `10`
   - rule: `total = sum of sites only where every active site has a value; else NaN`
   - raw_complete_share: `0.1495`
   - imputed_complete_share: `0.1603`

## Imputation method

Profile-KNN within each series (Peppanen et al. 2016). Each UTC day is a 48-slot vector. For a day with gaps that is at least 25% observed and contains no gap longer than 48 periods, the k=5 nearest *complete* days of the same series and day type (weekday/weekend, local time) within ±8 weeks are found by RMSE over the observed slots; the missing slots take the neighbours' mean profile, scaled to the day's observed level (scale clipped to [0.25, 4]). Whole missing days are never filled. Series needing more than 10% imputation are flagged `imputed_heavily` and excluded from `wide_imputed`. Site and trust totals are formed only where every active member has a value.

## Tables

| energy/level | series (raw) | series (imputed) | periods | raw cells present | imputed cells present | median coverage in span | mean imputed share |
|---|---|---|---|---|---|---|---|
| elec/meter | 170 | 169 | 59,952 | 82.9% | 83.2% | 100.0% | 0.40% |
| elec/site | 134 | 133 | 59,952 | 85.7% | 86.0% | 100.0% | 0.26% |
| elec/trust | 24 | 23 | 59,952 | 73.4% | 74.6% | 98.4% | 0.35% |
| gas/meter | 56 | 56 | 59,952 | 31.3% | 32.1% | 79.6% | 0.79% |
| gas/site | 21 | 21 | 59,952 | 18.6% | 19.2% | 51.1% | 0.61% |
| gas/trust | 10 | 10 | 59,952 | 14.9% | 16.0% | 39.4% | 0.49% |

## Files

| path | rows | cols | MB | sha256[:12] |
|---|---|---|---|---|
| `site_attributes.parquet` | 275 | 27 | 0.04 | `2d6b3dbd2253` |
| `site_attributes.csv` | 275 | 27 | 0.11 | `bed9aebe7894` |
| `elec/meter/wide_raw.parquet` | 59,952 | 170 | 10.3 | `87c4599c3d23` |
| `elec/meter/wide_raw.csv` | 59,952 | 170 | 67.78 | `a2b6f5209d9c` |
| `elec/meter/wide_imputed.parquet` | 59,952 | 169 | 10.57 | `d487dba53e4f` |
| `elec/meter/wide_imputed.csv` | 59,952 | 169 | 67.6 | `cffac8e55ff4` |
| `elec/meter/long_raw.parquet` | 8,448,770 | 3 | 14.9 | `2c8d95c87e88` |
| `elec/meter/long_imputed.parquet` | 8,433,140 | 4 | 15.21 | `774a3755970e` |
| `elec/meter/series_index.parquet` | 170 | 23 | 0.03 | `b9b8178b71f2` |
| `elec/meter/series_index.csv` | 170 | 23 | 0.05 | `3094899a14c7` |
| `elec/site/wide_raw.parquet` | 59,952 | 134 | 9.18 | `0e5b3c105906` |
| `elec/site/wide_raw.csv` | 59,952 | 134 | 55.13 | `664ed3fc0810` |
| `elec/site/wide_imputed.parquet` | 59,952 | 133 | 9.39 | `569b4f4abcfe` |
| `elec/site/wide_imputed.csv` | 59,952 | 133 | 54.91 | `7c7b8e46b9d3` |
| `elec/site/long_raw.parquet` | 6,881,456 | 3 | 13.34 | `6781a1647989` |
| `elec/site/long_imputed.parquet` | 6,859,825 | 4 | 13.54 | `f5dfb3040884` |
| `elec/site/series_index.parquet` | 134 | 19 | 0.02 | `905ed8ca3c28` |
| `elec/site/series_index.csv` | 134 | 19 | 0.04 | `2d25bac5afe9` |
| `elec/trust/wide_raw.parquet` | 59,952 | 24 | 4.69 | `a93ac21f7c31` |
| `elec/trust/wide_raw.csv` | 59,952 | 24 | 11.17 | `653c8b8ed19e` |
| `elec/trust/wide_imputed.parquet` | 59,952 | 23 | 4.83 | `35ea144abda8` |
| `elec/trust/wide_imputed.csv` | 59,952 | 23 | 10.92 | `d20a8a9b2a7f` |
| `elec/trust/long_raw.parquet` | 1,056,066 | 3 | 5.08 | `c93c2c8f4e4f` |
| `elec/trust/long_imputed.parquet` | 1,028,290 | 4 | 5.12 | `f7c3c622e1b9` |
| `elec/trust/series_index.parquet` | 24 | 16 | 0.01 | `73df946619a5` |
| `elec/trust/series_index.csv` | 24 | 16 | 0.01 | `2c319bca604c` |
| `gas/meter/wide_raw.parquet` | 59,952 | 56 | 2.01 | `036dc5bf069e` |
| `gas/meter/wide_raw.csv` | 59,952 | 56 | 12.24 | `9879651ae154` |
| `gas/meter/wide_imputed.parquet` | 59,952 | 56 | 2.24 | `5ec889087f46` |
| `gas/meter/wide_imputed.csv` | 59,952 | 56 | 12.41 | `75d04de51df3` |
| `gas/meter/long_raw.parquet` | 1,050,128 | 3 | 2.42 | `277e2faa8752` |
| `gas/meter/long_imputed.parquet` | 1,076,661 | 4 | 2.63 | `e9deba87a018` |
| `gas/meter/series_index.parquet` | 56 | 23 | 0.02 | `bd46e5999286` |
| `gas/meter/series_index.csv` | 56 | 23 | 0.02 | `03ee3f18416a` |
| `gas/site/wide_raw.parquet` | 59,952 | 21 | 1.02 | `53449b759217` |
| `gas/site/wide_raw.csv` | 59,952 | 21 | 4.49 | `c854ca4f073e` |
| `gas/site/wide_imputed.parquet` | 59,952 | 21 | 1.09 | `b51b8bf333de` |
| `gas/site/wide_imputed.csv` | 59,952 | 21 | 4.54 | `00a19e657b23` |
| `gas/site/long_raw.parquet` | 234,154 | 3 | 1.04 | `2d30a301944f` |
| `gas/site/long_imputed.parquet` | 241,294 | 4 | 1.1 | `7ce70c71764e` |
| `gas/site/series_index.parquet` | 21 | 19 | 0.02 | `f48f65678e2d` |
| `gas/site/series_index.csv` | 21 | 19 | 0.01 | `32a6707a8819` |
| `gas/trust/wide_raw.parquet` | 59,952 | 10 | 0.71 | `94cbdd010bf4` |
| `gas/trust/wide_raw.csv` | 59,952 | 10 | 2.85 | `ea4c13ee99f7` |
| `gas/trust/wide_imputed.parquet` | 59,952 | 10 | 0.79 | `819ce7096d1f` |
| `gas/trust/wide_imputed.csv` | 59,952 | 10 | 2.89 | `21bc00355ab5` |
| `gas/trust/long_raw.parquet` | 89,648 | 3 | 0.61 | `c28137a859b7` |
| `gas/trust/long_imputed.parquet` | 96,125 | 4 | 0.71 | `ebbe39ebb9ed` |
| `gas/trust/series_index.parquet` | 10 | 16 | 0.01 | `b8d9eb09c71b` |
| `gas/trust/series_index.csv` | 10 | 16 | 0.0 | `8159e04cef41` |

## Known limitations

- Gaps longer than a day, and days observed less than the threshold, remain missing in the imputed tables by design.
- Site/trust `imputed` in the long tables is the share of members whose value was imputed at that time, not a boolean.
- Gas has far fewer meters and a much weaker daily cycle; profile-KNN is less well suited to it.
- Coverage differs across series: check `series_index` before forming rectangular blocks for clustering or foundation models.
