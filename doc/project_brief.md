# Project brief

> Fill this in at kick-off. It is the document you will re-read every time scope drifts.

## Problem

Energy demand forecasting for NHS hospitals.

## Why it matters

<!-- e.g. procurement exposure to volatile wholesale prices; DSR/flexibility revenue;
     net-zero (Delivering a Net Zero NHS) reporting; capacity planning for
     electrification of heat and EV fleets. -->

## Scope

- **Sites in scope:** TBD
- **Energy vectors:** electricity / gas / heat — TBD
- **Temporal resolution:** half-hourly (settlement periods)
- **Forecast horizon(s):** TBD (day-ahead? week-ahead? annual budget?)
- **Update cadence:** TBD

## Out of scope

<!-- Stating this explicitly is as valuable as stating the scope. -->

## Stakeholders

| Role | Name | Interest |
|---|---|---|
| Data owner | TBD | Approves data access |
| Estates / energy manager | TBD | Consumes the forecast |
| Information governance | TBD | Approves data handling |

## Success criteria

| Criterion | Target | How measured |
|---|---|---|
| Beats seasonal naive | MASE < 1.0 | Rolling-origin backtest |
| Day-ahead accuracy | MAPE < X% | Held-out period |
| Reproducibility | `make data && make train && make evaluate` from clean checkout | CI |

## Key risks

| Risk | Impact | Mitigation |
|---|---|---|
| Data access delays | Blocks everything | Start with synthetic/public data |
| Meter gaps and quality issues | Biased models | Validation report before modelling |
| Regime change (new plant, ward closure, CHP) | Historical data misleads | Changepoint checks; site metadata |
| COVID-era data unrepresentative | Distorted seasonality | Consider excluding 2020-21 |

## Timeline

| Milestone | Date |
|---|---|
| Data received | TBD |
| Data quality report | TBD |
| Baselines established | TBD |
| Candidate model selected | TBD |
