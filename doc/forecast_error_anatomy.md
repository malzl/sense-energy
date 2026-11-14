# Where the day-ahead error sits, and what can remove it

The test of the two hypotheses behind the forecast finding of the kickoff deck, 14 November
2026, all on CPU. Commands: `forecast-probes` (new forecasts), `forecast-anatomy` (tables).
Modules: `code/src/sense_energy/analysis/forecast_anatomy.py`,
`experiments/forecast_probe.py`, `data/strikes.py`. Tables in
`code/reports/forecast_anatomy/`, figures from `code/scripts/forecast_anatomy/`.

The proof of concept (`doc/poc_results.md`) ended with a research avenue, "train a compact
prior-data fitted network with occupancy and mobility signals; measure it in decisions, not
only in error", and two hypotheses:

- **H1.** Occupancy and mobility signals decrease the error on atypical days.
- **H2.** On-site generation causes most of the rest of the error.

What the page answers: where the error of the best models is (which sites, which days, level
or shape, error or decision value); whether activity moves electricity at all (doctor strikes,
footfall); what more history or a later issue time gives; and what heat and cold waves do.

New forecasts, all Chronos-2 with no training: every day of the test window (91 sites, 342
days, 31,122 forecasts), four shorter histories, two issue times on the target day, and seven
variants with footfall at the three sites that have it. New data: the dates of the national
doctor strikes (British Medical Association and government notices).

## The answers in short

- **H2 holds, in part.** The 12 sites with on-site CHP have 50 % of the metered demand and
  79 % of the forecast error in kWh; three of them have 57 %. Half of that is their size: over
  their total demand (import plus reported generation) the error is 6.4 %, where sites of that
  size without a plant have about 3.4 %. What the plant adds is 37 % of all the panel's error.
- **The plant error is steps.** On 15 % of their days the import of a CHP site steps by more
  than its own mean within two hours, with no pattern from the last four weeks. Those days are
  2 % of all site-days and hold 42 % of all error. A forecast issued at 06:00 on the day itself
  does not see them either: the error at these sites falls from 19.1 % to 17.1 %.
- **H1 has no support.** On 34 doctor-strike weekdays away from holidays, demand between 08:00
  and 18:00 changes by −0.4 % (95 % interval −1.6 to +0.8); no hour of the day moves by 2 %.
  The elasticity of demand to daily footfall is 0.00 at the three sites with counts. A model
  that is given the true footfall of the target day forecasts no better: +0.006 points of error
  (−0.012 to +0.024), where calendar, price and weather inputs take 0.86 points off.
- **A daily signal has little to give.** A perfect forecast of each day's mean level takes the
  error of the best model from 11.5 % to 10.1 % and its decision value from 53 % to 63 % of the
  achievable peak cut. The rest is the shape of the day.
- **History: two weeks are enough.** 12.7 % error with two weeks of context, 12.2 % with six;
  one week gives 16.4 %.
- **Heat waves cost value, not accuracy.** Against the other days of the same site and month,
  heat-wave days have 14.7 % against 13.2 % error, but a battery that plans on the forecast
  keeps 33 % against 50 % of the achievable peak cut. With forecast weather among the inputs
  the best model keeps 40 %.
- **The research avenue changes.** The compact models trained on the panel are behind the
  pre-trained ones (13.7 to 14.6 % against 11.5 %), and the signals the avenue named carry no
  information for electricity. The signal that is missing is the state of the plant.

## 1. Level and shape (`level_oracle_models`)

For each site-day the forecast is rescaled to the day's true mean. The error that is left is
shape; the part that went is what any daily signal (occupancy, footfall, plant state as a
daily value) could give at most. A signal that changes the class of the day's shape, such as
a holiday flag, can give more.

{{figure: level_oracle_models | caption=Day-ahead error of each model (filled) and of the same forecast with the day's true level (open). 91 sites, 69 test days, site mean of the normalised absolute error.}}

{{chart: code/reports/forecast_anatomy/table_models.csv | kind=bar | x=nmae_pct | y=model | xlabel=Normalised absolute error (%) | ylabel=model | title=Day-ahead error by model, 91 sites and 69 test days | height=300}}

{{table: code/reports/forecast_anatomy/table_models.csv | round=1}}

- **A fifth of the error of the pre-trained models is level**: 11.5 % becomes 10.1 % for
  Chronos-2 with forecast weather. The models trained on the panel have more to gain (28 to
  32 % of their error is level), the seasonal naive forecast 40 %.
- **With the true level the models are within two points of each other** (10.1 to 12.3 %).
  What separates them today is mostly how well they place the level of the day.

## 2. Decisions (`value_plans`)

The decision is the one of `doc/flexibility_and_design_winters.md`: a battery (20 % of mean
demand, two hours) sets a threshold for the day from the forecast and defends it in real time.
The score is the share of the peak cut that perfect foresight achieves.

{{figure: value_plans | caption=Share of the achievable peak cut by plan: on the median forecast, on its 70 % quantile, and on the median with the day's true level.}}

{{table: code/reports/forecast_anatomy/table_groups.csv | round=1}}

- **53 % with the best forecast, 55 % with its 70 % quantile, 63 % with the true daily level.**
  A perfect daily signal is worth ten points; 37 points stay out of reach of any signal that
  only moves the level.
- **By group**: inpatient sites 57.5 % (63.1 % on the quantile), CHP sites 52.0 % (51.4 %),
  outpatient sites 49.0 % (52.3 %). Planning on a high quantile helps where the error is noise
  and not where it is a plant.

## 3. On-site generation (`error_share_groups`, `error_by_step`, `chp_size_adjusted`)

{{figure: error_share_groups | caption=Share of the panel's metered demand and of its forecast error by site group, best model.}}

**Steps.** For each site-day: the largest difference between the mean of the next two hours and
the mean of the last two hours, after the site's usual profile (median of the same half-hour
and weekday in the last four weeks) is taken out, as a share of the site's mean demand.

{{figure: error_by_step | caption=Share of the panel's error by the size of the day's largest unexplained step, by site group. Best model, 69 test days.}}

{{chart: code/reports/forecast_anatomy/error_by_group_step.csv | kind=bar | x=step_bin | y=error_share_pct | color=group | xlabel=Largest unexplained step of the day (share of mean demand) | ylabel=Share of the panel's error (%) | title=Where the error is: by size of the day's largest unexplained step | height=320}}

{{table: code/reports/forecast_anatomy/error_by_group_step.csv | cols=group,step_bin,site_days,day_share_of_group_pct,nmae_kwh_weighted_pct,error_share_pct,demand_share_pct | round=1}}

- **Steps larger than the site's mean demand: 14 % of CHP site-days, 43 % of all error** (69
  test days; 15 % and 42 % on all 342 days). On these days the error is 40 % of the import. At
  sites without CHP such days are 4 to 7 % of the days and 1 % of the error.
- **The steps are a plant that starts or stops.** In the example the import is 0.4 MW at
  night, 2.7 MW from 09:00 to 18:00 and 0.4 MW again; the forecast peaks at 0.8 MW.

{{figure: switch_day_example | caption=One CHP site on 27 January 2026: observed grid import, the day-ahead median of the best model and its 80 % interval.}}

**Size.** The meter of a CHP site records import. With the plant's reported share of the
site's electricity (estates returns, `chp_dispatch`), total demand is import / (1 − share). The
error of a site without a plant falls with its size (slope −0.21 in logs); the line gives the
error to expect at the total demand of each CHP site.

{{figure: chp_size_adjusted | caption=Error against site size. Blue: sites without CHP and their fitted line. Filled: CHP sites, error over import. Open: the same error over total demand.}}

{{table: code/reports/forecast_anatomy/chp_size_adjusted.csv | cols=site_code,mean_kw,chp_share,total_kw,nmae_pct,error_of_total_pct,expected_nmae_pct,error_mwh,expected_error_mwh,plant_added_mwh,error_share_pct,big_step_days,days | round=2}}

- **CHP sites: 50 % of the import, 75 % of the total demand, 79 % of the error.** Their error
  is 18.9 % of import and 6.4 % of total demand; sites without a plant would have 3.4 % at that
  size. The plant adds 47 % of these sites' error, which is 37 % of the panel's.
- **The line is extrapolated.** The largest site without CHP has 1.5 MW and an error of 1.9 %;
  the three largest CHP sites have 2.8 to 3.5 MW of total demand. With that site's rate in
  place of the line for those three, the plant's part is 47 %.
- **One CHP site has no excess**: a plant that runs flat (capacity factor 0.84, no large step
  in 67 test days) leaves an import that is forecast better than its size predicts (2.1 %
  against 4.4 %). The error comes with the switching, not with the plant.
- **Known-ahead inputs help a little more where the plant follows no signal**: calendar, price
  and weather take 0.9 points off at the nine sites whose plant follows price or the red band
  and 1.3 points at the three that do not (`chp_dispatch_gain.csv`).

## 4. Doctor strikes as a shock to activity (`strike_effect`, `strike_profile`)

Nineteen national rounds of industrial action by junior (resident) doctors and consultants fall
inside the record: 70 strike days, of which 34 are weekdays away from holidays, 18 are weekdays
next to Christmas or Easter and 18 are weekend days. On these days hospitals cancelled elective
operations and outpatient appointments in large numbers. Daily demand against the same site,
month and weekday, with weather by site group and calendar controls; pseudo-strikes one week
before and after as placebos; standard errors clustered by site and date.

{{figure: strike_effect | caption=Change in demand between 08:00 and 18:00 on strike days and on the placebo days, all sites, with 95 % intervals.}}

{{table: code/reports/forecast_anatomy/strike_effect.csv | round=2}}

- **No effect of the working day.** 08:00 to 18:00: −0.40 % (−1.60 to +0.81). Whole day:
  −0.68 % (−1.62 to +0.27). Working hours against the night of the same day: +0.56 % (−0.44
  to +1.57), the wrong sign for an activity effect.
- **Hour by hour nothing moves by 2 %.** The largest value is −1.9 % at 08:00 (interval −4.1
  to +0.4); the week after, with no strike, shows −1.5 % in the same hour.
- **Next to Christmas and Easter the working hours are 2.2 % lower** (−4.0 to −0.4). Those
  weeks drift with the holidays whether there is a strike or not, which is why they are kept
  apart.
- **By site type** the acute sites without CHP show −0.42 % (−2.05 to +1.23) in working
  hours; no type shows an effect in working hours.
- **The forecast is not too high on strike days.** In the every-day run its median bias is
  −0.9 to −1.6 % on strike days against −0.2 to −0.5 % on typical days.

{{figure: strike_profile | caption=The strike effect by hour of the day with its 95 % interval; the pseudo-strikes one week before and after as lines.}}

## 5. Footfall (`footfall_partial`, `footfall_elasticity`, `mobility_forecast_difference`)

Daily footfall counts exist for three sites of one trust in 2024 (The Oxford Partnership,
research licence).

**In a regression.** Log daily demand on log daily footfall within month and weekday, weather
and holidays held fixed, Newey-West standard errors; the footfall of the university campus in
the same city as the placebo.

{{figure: footfall_partial | caption=Daily demand against daily footfall, both with month, weekday, weather and holidays taken out.}}

{{table: code/reports/forecast_anatomy/mobility_regression.csv | round=3}}

- **Elasticity 0.00 at all three sites**: −0.002 (−0.18 to +0.18) at the CHP site, −0.012
  (−0.047 to +0.023) and +0.001 (−0.028 to +0.029) at the other two. The raw correlations of
  0.43 and 0.50 are the weekly rhythm that both series share.
- **The placebo is "significant" at the CHP site** (0.28, p = 0.003): a warning about what a
  noisy import series can show.

**In the forecast.** Chronos-2 for every day from 13 February to 31 December 2024 with the
proof of concept's inputs (calendar, price, weather) and the footfall added in context: the
count of the last complete day; the count of the target day itself (a perfect footfall
forecast); the campus count of the target day (placebo); and the last two again with one week
of history. Paired differences with 95 % intervals from resampling weeks.

{{figure: mobility_forecast_difference | caption=Change in error when the model gets the footfall, two sites without CHP. Below zero the footfall helps.}}

{{table: code/reports/forecast_anatomy/table_mobility.csv | cols=sites,kind,model,site_days,nmae_pct,crps_pct,coverage_80 | round=3 | head=14}}

{{table: code/reports/forecast_anatomy/mobility_differences.csv | round=3}}

- **A perfect footfall forecast adds nothing**: 5.589 % against 5.583 % at the two sites
  without CHP, +0.006 points (−0.012 to +0.024). On the days that are not typical (holidays,
  strike days, unusual counts): −0.001 (−0.041 to +0.034).
- **The count of the last full day adds nothing** (+0.010, −0.013 to +0.036), and the campus
  placebo does the same as the site's own count (+0.002).
- **With one week of history it does not help either**: 8.44 % against 8.41 %.
- **The model does use inputs that carry information**: calendar, price and weather take 0.86
  points off (−1.12 to −0.62).

## 6. Kinds of day, every day of the test window (`daily_error_by_kind`)

One forecast per site for each of the 342 test days, Chronos-2 without covariates. A day has
one kind, in this order: holiday or Christmas; doctor strike; hot or cold day (maximum of 28 °C
or more, mean of 2 °C or less); level shift (the day's mean differs by more than 15 % from the
same weekday of the last four weeks); typical.

{{figure: daily_error_by_kind | caption=Error by kind of day, forecast (filled) and forecast with the day's true level (open). 30,875 site-days.}}

{{table: code/reports/forecast_anatomy/daily_error_by_kind.csv | cols=kind,site_days,day_share_pct,nmae_pct,nmae_level_oracle_pct,level_share_pct,error_share_pct,demand_share_pct | round=1}}

{{table: code/reports/forecast_anatomy/daily_value_by_kind.csv | round=1}}

- **Typical days: 64 % of the days, 29 % of the error, 10.1 %.**
- **Level-shift days: 22 % of the days, 53 % of the error**, and 49 points of the 53 are at
  the CHP sites. With the true level their error is still 13.2 %.
- **Holidays: 4 % of the days, 7 % of the error, 20.1 %**, and here the level is the problem
  (14.6 % with the true level). This run has no calendar input; with it the best model has
  12 to 23 % on holidays, so the calendar flag helps but does not solve them.
- **Strike days: 12.7 %** against 12.3 % on all days, with no sign of lower demand
  (section 4).

## 7. Heat and cold waves (`wave_error`, `wave_value`)

With the label of Lu et al. (2026): a wave day has a site temperature (mean of the day's
highest and lowest hour) beyond the 5th or 95th percentile of the site's days before the test
window. The summer of 2025 made 9.7 % of the test site-days heat-wave days; 2.6 % are cold-wave
days. The comparison is with the other days of the same site and calendar month, so the season
is not counted as weather. The paper itself is checked in `doc/heat_cold_wave_paper_check.md`.

{{figure: wave_error | caption=Error on heat-wave and cold-wave days (filled) against the other days of the same site and month (open), by site group.}}

{{table: code/reports/forecast_anatomy/daily_error_wave_matched.csv | cols=matched_to,group,wave,sites,site_days,nmae_pct,nmae_level_oracle_pct,median_bias_pct | round=1}}

{{figure: wave_value | caption=Share of the achievable peak cut on wave days and on the other days of the same site and month, by plan. All sites.}}

{{table: code/reports/forecast_anatomy/daily_value_wave_matched.csv | round=1}}

{{table: code/reports/forecast_anatomy/value_wave_matched_stored.csv | round=1}}

- **Error: a little more.** Heat-wave days 14.7 % against 13.2 %, cold-wave days 13.6 %
  against 11.4 %. The forecast is too low on both (median bias −1.6 % and −3.3 %). Bringing
  the wave days to the error of their neighbours would remove 4.9 % of the panel's error (heat
  3.6 %, cold 1.3 %).
- **Value: a third less.** On heat-wave days the plan on the forecast keeps 32.9 % of the
  achievable peak cut against 49.9 % on the other days of the same months; on cold-wave days
  43.6 % against 53.0 %. A forecast that is a few per cent low sets the threshold too low and
  the battery is empty before the peak.
- **Weather inputs recover part of it.** On the 69 stored test days the best model, which has
  the weather forecast, keeps 40.0 % on heat-wave days against 50.0 %; the same model without
  weather 31.9 % against 50.5 %; LightGBM with weather 41.9 % against 50.5 %.
- **The level is half of the loss.** With the true daily level the plan keeps 50.7 % on
  heat-wave days against 60.1 %.

## 8. History and issue time (`context_length`, `issue_time_error`, `issue_value`)

{{figure: context_length | caption=Error of Chronos-2 without covariates by the history it gets, by site group.}}

{{table: code/reports/forecast_anatomy/table_context.csv | round=1}}

- **Two weeks are enough**: 12.7 % with two weeks, 12.2 % with six. One week (one example of
  each weekday) gives 16.4 %, one day 31.1 %. A site that starts to deliver data has a usable
  forecast after a fortnight.

**Issue time.** The same model issued at 00:00 and at 06:00 on the target day for the rest of
that day, scored on the periods that all three issues forecast (from 06:30). For the decision,
the battery defends the day-ahead threshold until the new issue and then a new threshold, the
lowest that it can hold on the new forecast from its state of charge.

{{figure: issue_time_error | caption=Error by issue time on the common periods, by site group.}}

{{figure: issue_value | caption=Share of the achievable peak cut by the time of the last plan, by site group.}}

{{table: code/reports/forecast_anatomy/table_issue.csv | round=1}}

- **A later issue helps a little**: 13.5 % the day before, 12.9 % at midnight, 12.8 % at 06:00.
  At CHP sites 19.1 %, 16.9 % and 17.1 %: the plant switches later in the day, after the issue.
- **In value two points**: 51.1 %, 51.8 % and 53.1 % of the achievable peak cut.

## 9. What this says about the research avenue

- **Occupancy and mobility: drop them as forecast inputs for electricity.** Two independent
  shocks to activity (strikes at 132 sites, footfall at three) leave demand where it was, and
  a perfect daily level signal is bounded at 1.5 points of error. Daily bed occupancy remains
  untested (see limits), but it would have to act differently from elective activity and
  footfall to matter.
- **A compact model: the panel does not ask for one yet.** The three neural-process models
  trained on the panel (1.1 to 1.5 million parameters) have 13.7 to 14.6 % against 11.5 % for
  the pre-trained model, and two weeks of context already give the pre-trained model its
  accuracy. Training a prior-fitted network needs the GPUs, which are off limits for now; what
  it would have to add is plant state, and no such input exists yet.
- **Plant state is the avenue.** Thirty-seven per cent of the error is added by plants, on 2 %
  of the site-days. The sites know when a plant is down for service, and half-hourly
  generation meters exist. A schedule or a generation series for two or three CHP sites would
  show how much of this can be forecast.
- **Decisions, not error, show the weather problem.** Heat waves take a tenth more error and a
  third of the value. Any method for rare days should be scored on the decision.

## 10. Limits

- **One panel, one test period.** 91 sites, June 2025 to May 2026. The 12 CHP sites are few,
  and three of them carry most of the result.
- **The plant share comes from the estates returns**, reported per year by the trusts, for 9
  of the 12 CHP sites; the size comparison is extrapolated beyond the largest site without CHP.
- **Strikes are not occupancy.** They cut elective and outpatient work; inpatients stayed.
  Daily bed occupancy was not tested: the published daily figures (urgent and emergency care
  situation reports, winter months, per trust) sit behind a bot check on the NHS England site
  and need a manual download; the other public series are monthly or quarterly.
- **Footfall is three sites of one trust in one year.**
- **The decision is one decision**: a peak cut with a small battery and a threshold for the
  day. Other decisions weigh errors differently.
- **The every-day run has no covariates**; sections 6 and 7 use it for its sample, and the 69
  stored days for the models with inputs.

## Reproduce

```
CUDA_VISIBLE_DEVICES="" sense-energy forecast-probes --which context  --threads 10   #  8 minutes
CUDA_VISIBLE_DEVICES="" sense-energy forecast-probes --which daily    --threads 10   # 53 minutes
CUDA_VISIBLE_DEVICES="" sense-energy forecast-probes --which intraday --threads 8    # 20 minutes
CUDA_VISIBLE_DEVICES="" sense-energy forecast-probes --which mobility --threads 10   # 90 minutes
CUDA_VISIBLE_DEVICES="" sense-energy forecast-anatomy                                #  5 minutes
python code/scripts/forecast_anatomy/run_all.py
```
