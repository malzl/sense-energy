# Public NHS activity data as covariates for the trusts and sites

Assessment of 15 September 2026 (what exists, at what resolution, for which of our 24 trusts, and what is
already pulled). Loader: `sense-energy fetch-nhs-activity` (`code/src/sense_energy/data/nhs_activity.py`).

## Pulled (NHS England statistics site, direct downloads)

| Source | Resolution | Our trusts | Content | Interim file |
|---|---|---|---|---|
| A&E attendances and emergency admissions (MSitAE) | monthly, provider | 18 of 24 (acute and community trusts with any A&E/UTC activity) | attendances by department type, 4-hour breaches, 12-hour waits, emergency admissions | `nhs_activity_ae_monthly.parquet`, Apr 2023 – Mar 2026, 6,727 provider-months |
| Ambulance Systems Indicators (AmbSYS) | monthly, ambulance service | 2 of 2 ambulance trusts | calls, incidents by category, response times, hear-and-treat / see-and-treat shares | `nhs_activity_ambsys_monthly.parquet`, Aug 2017 – Aug 2026 |
| Bed availability and occupancy (KH03, overnight) | quarterly, trust, by sector | 23 of 24 | available and occupied beds, occupancy rate | `nhs_activity_kh03_quarterly.parquet`, 2001 – Jun 2024 (later quarters only as spreadsheets on the KH03 page) |

Trust ↔ ODS code mapping: `nhs_activity_trusts.parquet` (site-code prefix, else normalised name; ambulance
services matched to their AmbSYS code separately).

## Available but not pulled

| Source | Resolution | Note |
|---|---|---|
| Urgent and Emergency Care daily situation reports | **daily**, acute trust: G&A and critical-care beds occupied/available, ambulance handovers, discharges | winter only (early December to early April each year; 2025-26 file covers 4 Dec 2025 – 2 Apr 2026); spreadsheets on the UEC SitRep pages. The only public daily occupancy signal; overlaps our test winter |
| Mental Health Services Monthly Statistics (MHSDS) | monthly, provider | referrals, contacts, bed days for the mental-health trusts (Hertfordshire Partnership, Midlands Partnership, Berkshire); CSV data files on NHS Digital |
| Community Services Statistics (CSDS) | monthly, provider | contacts and referrals for the community trusts (Cambridgeshire, Hertfordshire, Norfolk, Central London); NHS Digital |
| Provisional monthly HES (admitted patient care, outpatients) | monthly, provider | replaced the discontinued Monthly Hospital Activity provider series (last 2017/18); open CSVs on NHS Digital |
| Estates Returns Information Collection (ERIC) | annual, **site** | floor areas, building age, energy and water consumption, occupancy proxies, backlog; 2019/20 – 2024/25 on NHS Digital (data.gov.uk mirror stops at 2016/17). Our site table already carries the ERIC-style statics from the ESC extract; the annual energy totals would validate the meter data |

NHS Digital (`digital.nhs.uk`) returns 403 to scripted requests; those files need a browser download or the
NHS Digital publication API, then drop into `code/data/external/`.

## What they are good for

- **Cross-sectional structure**: kWh per attendance, per occupied bed day, per incident; normalising sites of
  different function; explaining the shape clusters (`doc/clustering.md`).
- **Slow covariates** for the models: monthly activity level, bed occupancy (quarterly), which capture
  level shifts (ward closures, new units) that a six-week context cannot see. Not day-ahead dynamics.
- **Daily occupancy** (UEC sitreps) is the one public series that could improve day-ahead forecasts at acute
  sites, for the winter months only, and only at trust level.
- **No public patient data exists at site or half-hourly resolution**; anything finer needs the trusts
  themselves (PAS/EPR feeds) or the mobility dataset as an occupancy proxy.
