# Load forecasts under heat and cold waves: a check of Lu et al. (2026)

A reading of "Electricity load forecasting under extreme heat and cold waves" (Lu, Qin, Zhou,
Bai, Wen, Kang and Wang, *Nature Communications* 17, 10041, published 21 August 2026) and of its
code, with an independent check on the paper's own public data, 14 November 2026, on CPU.
Command: `esdf-check`. Module: `code/src/sense_energy/experiments/esdf_check.py`. Tables in
`code/reports/esdf_check/`, figures from `code/scripts/esdf_check/`. The paper's repository
(github.com/hkuedl/Heat-Cold_Wave_Load_Forecasting, MIT licence, commit 0036608 of 14 October
2026) is kept in `code/data/external/esdf/`: its public data (Europe, PJM), its saved forecasts
and its Python files. The paper's code was read, not run.

What the page answers: what the paper does and what its code scores; whether its numbers hold
on the days that it does not score; how references that it leaves out do on the same samples;
and what follows for the NHS panel.

## The answers in short

- **The method works, by about a tenth.** Over all days of the paper's test periods, the
  proposed model has 9 % less error than the same network without it (Europe 2.61 against 2.86,
  PJM 3.33 against 3.65, in per cent of the load range). That holds on hot days, cold days and
  ordinary days alike.
- **The headline gain comes from one window per region.** The paper reports the error on 8
  consecutive days for all 18 European countries (25 February to 4 March 2018) and on 15
  consecutive days for all 19 PJM zones (29 June to 13 July 2023). In Europe the method is 20 %
  ahead of its best baseline inside that window and 4 % ahead on the 297 test days.
- **In PJM the scored window is not a heat wave.** By the 95th percentile of the training
  period, 1.5 of its 15 days are hot. The processed PJM file has no data from 25 July to
  1 September 2023, the hottest weeks of that summer.
- **A model with no training is level with it.** Chronos-2, given four weeks of history and the
  same temperature, has 11 % less error than the method in Europe (13 of 18 countries) and 6.5 %
  more in PJM (better in 6 of 19 zones). With the paper's seven days of history it is clearly
  worse. A gradient-boosted model and a linear regression with the paper's inputs do not reach
  the method.
- **The large errors are public holidays.** They are 3.8 % of the European test days and
  12.5 % of the method's error; every model is 2.5 to 4.4 times worse on them. No model in
  the paper gets the calendar.
- **For the NHS panel the lever is small in error and large in value.** Heat-wave and
  cold-wave days hold 19 % of our forecast error on 12 % of the site-days; against the same
  site and month their excess is 5 % of all error, where days on which a plant starts or stops
  hold 42 %. But on heat-wave days the forecast keeps 33 % of the achievable peak cut against
  50 % on the other days of the same months (section 6).

## 1. What the paper proposes

Extreme Synthesis-Disentanglement Forecasting (ESDF), in three stages:

1. **Synthesis.** A score-based diffusion model (U-Net, variance-exploding SDE, 1,000 steps,
   predictor-corrector sampling) learns the joint distribution of eight days of hourly load and
   temperature. A classifier on noisy samples (cold wave, heat wave, other) steers sampling
   towards extreme regimes. The model makes as many synthetic samples as there are real ones
   (1,200 per European country, 800 per PJM zone).
2. **Disentanglement.** A domain separation network: one shared encoder and one private encoder
   each for real and synthetic samples, a forecast head and a reconstruction head. The loss is
   the L1 forecast error on both domains, 0.1 times the reconstruction error, a maximum mean
   discrepancy term that aligns the shared codes (weight 0.5 or 1) and an orthogonality penalty
   between shared and private codes (weight 0.5 or 1).
3. **Inference.** The shared and the real-domain encoder with the real-domain head.

The paper reports 80 series in six regions (Europe 18, PJM 19, Texas, Guangdong 20, Hunan 14,
India 2), an accuracy of 97.1 % (one minus the normalised error) and 9.8 to 33.5 % less
normalised absolute error than the strongest baseline of each region. Baselines: MLP, LSTM,
CNN, N-BEATS, ImpactNet, Informer, Autoformer. Public data: Europe (Open Power System Data)
and PJM; the Chinese series are not released.

## 2. What the code scores

Read from `forecasting_using_generated_samples/` (Europe) and
`forecasting_using_generated_samples_PJM/`; line numbers at commit 0036608.

| Point | What the code does | Where |
|---|---|---|
| Task | The 24 hourly loads of day D from the hourly load and temperature of D−7 to D−1 and the **observed** temperature of D. One sample is an 8 × 24 slice. | `Forecasting_model_training.py`, `train_dataloader`; `model(x, label[:, 24:48])` |
| Unit | Load is scaled to 0–1 by the minimum and maximum of the training period. A reported error of 0.03 is 3 % of that range, about 2.3 % of mean load in Europe and 3.4 % in PJM. | `data_scaler` |
| Test set | Samples of the last year, but the reported error is taken on `[24·day : 24·(day+period)]`: `day = 19, period = 8` (Europe), `day = 22, period = 15` (PJM). One window, the same for every series of the region. | lines 29–30; PJM lines 33–34 |
| Wave label | Daily temperature is the mean of the day's highest and lowest hour. A day is a heat (cold) wave day when a positive excess over the 95th (below the 5th) percentile exists; the acceleration term only scales the index, so no duration rule is in effect. | `Dataset_Loader_2D.py` lines 53–62 |
| Guidance | The weight of the classifier gradient is `numda = 0` in the published Europe code and 10 in the PJM code. With 0 the Europe "cold wave" samples are unconditional draws. | `diff_Model_2D.py` line 12 |
| Validation | A random 20 % of the training samples (the paper says 10 %). Neighbouring samples share seven of eight days, so the validation set is not independent of the training set. | line 132 |
| Seeds | One seed (2) for every run. | throughout |
| Test curve | The test error is computed after every epoch for the convergence figures. The saved model is the one with the best validation loss. | `training_process`, `training_process_dsn` |
| PJM data | The processed file has 113 of the 153 days from May to September 2023: 25 July to 1 September and 16 September are missing. Test samples are 7 June to 20 July. | `Data/.../PJM_reformed_data/` |

The samples were rebuilt from the paper's loaders and compared with the `true` column of its
saved forecasts: **all 37 public series match exactly** (297 samples per European country, 44
per PJM zone).

{{table: code/reports/esdf_check/samples_check.csv | cols=region,series,samples,paper_samples,max_abs_diff_true,hot_days,cold_days,window_first_day,window_last_day,window_hot_or_cold_days | round=6}}

## 3. The reported numbers reproduce from the saved forecasts

The window error recomputed from the repository's saved forecasts against the paper's results
workbook, mean over the series of the region:

{{table: code/reports/esdf_check/reproduction.csv | round=4}}

All eight PJM numbers and the four Europe baselines that have one saved run reproduce exactly.
The Europe numbers of the method and of its three backbones differ by 1 to 8 % between the
workbook and the saved files (another run).

## 4. Beyond the scored window

Error in per cent of the training range, mean over the series; `window` is what the paper
reports, `all_days` every test day, `hot_cold_days` the days whose temperature is beyond the 5th
or 95th percentile of the training period (Europe: 31 hot and 20 cold days per country of 297;
PJM: 3 hot days of 44). The last columns count the series in which a model is below ESDF (LSTM).

**Europe, 18 countries, test days 6 February to 29 November 2018**

{{figure: esdf_subsets_europe | caption=Europe: error on the paper's scored window (filled) and on all 297 test days (open). Grey: the paper's baselines; vermilion: the paper's method; colours: references that the paper does not have.}}

{{chart: code/reports/esdf_check/table_europe.csv | kind=bar | x=all_days_mae_pct | y=model | xlabel=Mean absolute error on all test days (% of the training load range) | ylabel=model | title=Europe, 18 countries, all 297 test days | height=420}}

{{table: code/reports/esdf_check/table_europe.csv | round=2}}

**PJM, 19 zones, test days 7 June to 20 July 2023**

{{figure: esdf_subsets_pjm | caption=PJM: error on the paper's scored window (filled) and on all 44 test days (open).}}

{{chart: code/reports/esdf_check/table_pjm.csv | kind=bar | x=all_days_mae_pct | y=model | xlabel=Mean absolute error on all test days (% of the training load range) | ylabel=model | title=PJM, 19 zones, all 44 test days | height=420}}

{{table: code/reports/esdf_check/table_pjm.csv | round=2}}

- **Against its own backbone the method gains about 9 % everywhere.** ESDF (LSTM) against
  the LSTM: Europe 2.61 against 2.86 on all days, PJM 3.33 against 3.65. The gain is not
  specific to extremes: in PJM it is 8.8 % inside the window and 8.8 % on all days.
- **The Europe headline is a window effect.** Inside the eight days the method is 20 % below
  the best baseline (2.62 against 3.29); on all test days it is 4 % below N-BEATS (2.61 against
  2.71) and 3 % below the LSTM with the synthetic samples simply added (2.68).
- **Eighty series are not eighty tests.** The 18 European countries are scored on the same
  eight days of one cold spell, the 19 PJM zones on the same fifteen days. With one seed and no
  interval, the number of independent events behind the public results is two.

{{figure: esdf_gain | caption=Error of ESDF (LSTM) relative to the LSTM without synthetic samples, one point per series, by set of days. Bars: medians. Below zero the method is better.}}

{{figure: esdf_daily_europe | caption=Europe: error on every test day, mean over the 18 countries. Shaded: the eight days that the paper scores. The spikes are public holidays (2 April, 1 May, 10 and 21 May, 15 August, 1 November).}}

{{figure: esdf_daily_pjm | caption=PJM: error on every test day, mean over the 19 zones. Shaded: the fifteen days that the paper scores.}}

## 5. References that the paper does not have

All get the paper's inputs: the seven days of hourly load and temperature and the observed
temperature of the target day (`Week before` uses the load only).

- **Chronos-2 with no training and four weeks of history: 11 % below the method in Europe,
  6.5 % above it in PJM.** Europe: 2.33 against 2.61 on all days (better in 13 of 18 countries), 1.99
  against 2.51 on hot and cold days (15 of 18), 2.22 against 2.62 in the scored window. PJM:
  3.54 against 3.33 (better in 6 of 19 zones).
- **With the paper's seven days it is not enough**: 3.51 (Europe) and 4.20 (PJM). One week of
  context holds one example of each weekday.
- **Trained references with the paper's inputs stay behind**: LightGBM 2.97 and 3.64, a linear
  regression per hour 3.38 and 4.36. The paper's neural baselines are not weak for this task.
- **Why PJM differs.** PJM summer load follows temperature closely, and the method has learnt
  that response from two years of the zone; a model that reads four weeks in context has not.
  A second reading cannot be excluded: the European national loads of 2015 to 2018 are public
  and can be in the training data of a pre-trained model. On the NHS panel, which no public
  model can have seen, the same model also leads (`doc/poc_results.md`).

**Public holidays.** No model of the paper sees the calendar. On the public holidays of the 18
countries (3.8 % of the test days) every model has 2.5 to 4.4 times its error of the other
days, and the method is worse than its backbone there (10.2 against 8.6):

{{table: code/reports/esdf_check/table_holidays_europe.csv | round=2}}

## 6. What follows for the NHS panel

The paper's premise is that models fail on rare extreme days. Tested on the panel with a
forecast for every day of the test window (91 sites, 342 days, Chronos-2 without covariates;
`doc/forecast_error_anatomy.md`), with the paper's labels (site temperature beyond the 5th or
95th percentile of the days before June 2025):

{{table: code/reports/forecast_anatomy/daily_error_wave_matched.csv | cols=matched_to,group,wave,sites,site_days,nmae_pct,nmae_level_oracle_pct,median_bias_pct | round=1}}

- **The premise holds, mildly.** Against the other days of the same site and month, heat-wave
  days have 14.7 % against 13.2 % error and cold-wave days 13.6 % against 11.4 %. The forecast
  is too low on both (median bias −1.6 % and −3.3 %).
- **The lever is small in error terms.** Heat-wave days were 9.7 % of the site-days of this
  test window (the summer of 2025) and hold 14.9 % of the error; cold-wave days 2.6 % and 4.0 %.
  Bringing them to the error of their neighbours removes 4.9 % of the panel's error (heat
  3.6 %, cold 1.3 %). Days with a plant start or stop hold 42 %.
- **The lever is larger in decision terms.** On heat-wave days the battery that plans on the
  forecast keeps 32.9 % of the achievable peak cut against 49.9 % on the other days of the
  same site and month; on cold-wave days 43.6 % against 53.0 %. With forecast weather among
  the inputs the best model keeps 40.0 % on heat-wave days (section 7 of the forecast page).
- **What transfers.** For the bulk of the error, nothing: it comes from plant state, which no
  sample generator can supply. For heat-wave days the method is a candidate, because there the
  forecast is low by a few per cent and the cost is in the decision; its gain on regional loads
  is a tenth, and the test would be the peak cut on heat-wave days, not the mean error. Worth
  keeping in any case: scoring on every day and by kind of day, a reference with no training in
  every comparison, and the calendar among the inputs.

## 7. Limits of this check

- Only the public series were checked: Europe and PJM, 37 of the paper's 80. The Chinese series
  that carry the cold-wave and heat-wave results for Guangdong and Hunan are not released.
- The method was scored from the saved forecasts of the repository, not retrained. One run per
  model, as in the paper; the differences of a few per cent between models are inside what a
  second seed could move.
- The references use fixed settings and no tuning. A tuned gradient-boosted model or a
  fine-tuned pre-trained model can do better than shown.
- The observed temperature of the target day is an input of every model, as in the paper. With
  a weather forecast the errors of all models rise.

## Reproduce

```
git clone https://github.com/hkuedl/Heat-Cold_Wave_Load_Forecasting   # into code/data/external/esdf
.venv/bin/python -I -m sense_energy.cli esdf-check --threads 8        # 12 minutes on 8 CPU threads
.venv/bin/python code/scripts/esdf_check/run_all.py
```
