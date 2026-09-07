# Methodology

## Framing

Site-level half-hourly demand forecasting. Each site is a time series; models may
be fit per site (`local`) or across all sites with `site_id` as a feature
(`global`). Global models generally win when per-site history is short, and let a
new site borrow strength from the rest of the estate.

## Validation

**Rolling-origin (forward-chaining) cross-validation, always.** Random k-fold
leaks future information through both the split and the lag features, and will
flatter every model you try.

```
fold 1: [====train====][gap][test]
fold 2: [======train======][gap][test]
fold 3: [========train========][gap][test]
```

The `gap` between train and test equals the operational lead time — if the
forecast is issued at 11:00 for the following day, the model cannot use readings
from the intervening hours. `sense_energy.evaluation.backtest.rolling_origin_splits`
implements this.

## Leakage checklist

- [ ] Every lag ≥ the forecast horizon plus lead time
- [ ] Rolling windows shifted by at least one period
- [ ] No target-derived aggregate (site mean, normalisation constant) fitted on the full series
- [ ] Weather features are **forecast** weather at prediction time, not observations
- [ ] Test period strictly after train period in every fold

That fourth point is the one that bites: backtesting with observed temperature
overstates accuracy, because in production you only have a weather forecast.

## Metrics

| Metric | Use |
|---|---|
| **MAE** | Primary; interpretable in kWh |
| **RMSE** | Penalises the large errors that drive peak-charge exposure |
| **MASE** | Model vs. seasonal naive; < 1 means it earns its keep |
| **MAPE / sMAPE** | Reporting to stakeholders; unstable near zero demand |
| **Bias** | Systematic over/under-forecasting — matters for procurement |

Report metrics per site *and* pooled. A pooled average hides a model that works
on large acute sites and fails on small community ones.

## Model progression

1. **Seasonal naive** (same half-hour last week) — the bar to clear.
2. **Profile mean** by (day-of-week × time-of-day).
3. **Regularised linear** on calendar + degree-day features — interpretable.
4. **Gradient boosting (LightGBM)** — expected workhorse.
5. Optionally: SARIMAX with exogenous weather, or a global neural forecaster
   (N-BEATS / TFT) if the estate is large and the horizon long.

Do not skip steps 1–2. A benchmark you can explain to an estates manager is worth
more than a marginal accuracy gain you cannot.

## Known modelling issues in this domain

- **BST transitions** produce 46- and 50-period days. Store UTC; derive local.
- **Regime changes** — CHP commissioning, ward closures, PV installation — break
  stationarity. Check for changepoints and record them in site metadata.
- **COVID period** (2020–21) has atypical occupancy; consider excluding.
- **Hospitals have a high baseload** relative to offices, and weaker weekend
  reduction — profiles from commercial buildings do not transfer.
