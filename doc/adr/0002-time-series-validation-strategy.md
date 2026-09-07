# ADR 0002: Use rolling-origin validation, never random k-fold

- **Status:** Accepted
- **Date:** 2026-09-07

## Context

Demand data is strongly autocorrelated and enriched with lag features. Random
k-fold cross-validation places future observations in the training set and lets
lag features carry information across the split, producing accuracy estimates
that do not survive contact with production.

## Decision

All validation uses forward-chaining rolling-origin splits
(`sense_energy.evaluation.backtest.rolling_origin_splits`), with a `gap` between
train and test equal to the operational forecast lead time.

## Consequences

- Fewer effective folds, so metrics are noisier — report per-fold results, not
  just the mean.
- Backtest numbers are lower than random-CV numbers would be. This is correct.
- Backtesting with *observed* weather still overstates production accuracy;
  where possible, backtest against archived weather **forecasts**.
