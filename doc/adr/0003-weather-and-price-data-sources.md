# ADR 0003: Weather and price sources - reanalysis for history, AIFS-ENS from now on

- **Status:** Accepted
- **Date:** 2026-09-08

## Context

Temperature is the strongest single predictor of hospital demand and the NHS
extract carries no weather at all. The modelling window is 2022-12 to 2026-05.
The wish was ECMWF *forecast* data with *every ensemble member*, plus a
day-ahead electricity price that is genuinely known before delivery.

What each candidate source can actually provide:

| Source | Ensemble | Covers 2022-2026? | Access |
|---|---|---|---|
| ECMWF IFS ENS (51 members) | yes | archived in MARS/TIGGE | separate ECMWF licence / API key; not on CDS |
| ECMWF AIFS-ENS (51 members) | yes | **no - operational since July 2025** | free, open data, rolling ~4-day archive |
| ERA5 reanalysis | 10-member EDA at 0.5 deg 3-hourly, plus deterministic 0.25 deg hourly | yes | CDS, free |
| ENTSO-E day-ahead prices for GB | - | **no - GB left EU market coupling Jan 2021**; REST host also 404 | - |
| Octopus Agile unit rates | - | yes, from 2018 | free, no auth, 14 GSP regions |
| Elexon Market Index Data | - | yes | free, no auth |

## Decision

1. **History (training and backtesting):** ERA5 from CDS, both products -
   `reanalysis` as the truth series and `ensemble_members` as the only
   ensemble product available for the period. It is *analysis* uncertainty,
   not forecast uncertainty; models trained on it will look better than they
   will in production (see ADR 0002).
2. **Going forward (operational and evaluation):** ECMWF AIFS-ENS from open
   data, all 51 members, harvested daily because the archive is rolling. It
   starts accumulating from the day the harvester is first scheduled and can
   never be backfilled.
3. **Price, known beforehand:** Octopus Agile half-hourly rates, canonical
   product per day, regional by GSP group. It is a published transform of the
   N2EX day-ahead auction - a *retail* signal with caps, not the wholesale
   price. Elexon MID is stored as the ex-post wholesale reference and must not
   be used as a feature.
4. ENTSO-E is not used. The token is kept in `.env` in case the platform is
   needed for interconnector or generation data later.

## Consequences

- Two weather tables with different semantics: `weather_reanalysis` /
  `weather_ensemble_members` (what happened) and `forecast_aifs_ens` (what was
  predicted, by lead time). Features must be built from the one that matches
  the question.
- The AIFS archive only exists from the day the cron job starts. Schedule it
  before anything else if forecast-based evaluation matters.
- **Amendment 2026-09-10:** ECDS serves TIGGE perturbed members at 9–18 h per
  month from tape. The 51-member IFS ENS is also on the AWS mirror of ECMWF open
  data from 18 Jan 2023 as plain S3 with per-message indexes, so it is now the
  primary perturbed source (`harvest-ifs-ens`, days not weeks). TIGGE keeps
  running for the control run and for Dec 2022 – 17 Jan 2023 coverage. Dewpoint and
  radiation only exist in that stream from 2024-03-06; cloud cover is absent throughout.
- **Amendment 2026-09-08:** true historical forecast ensembles are now pulled
  from TIGGE (IFS ENS, 51 members, 0.5 deg, 6-hourly, from 2006) via the ECMWF
  Data Store (ECDS) - `sense-energy fetch-tigge`. This becomes the
  *forecast-at-issue-time* series for the training window; ERA5 remains the
  truth series and AIFS-ENS the operational feed. The Public Datasets Web API
  that used to serve TIGGE was decommissioned on 2026-05-27; ECDS needs its own
  token (ECDS profile page) and the TIGGE licence accepted on the dataset page.
- Agile rates are known from the afternoon of D-1. A forecast issued earlier
  that day cannot use them; check the issue time before adding the feature.
