# What we found so far: the findings map

16 November 2026. This page puts every result of the project under one of five questions and
gives each a grade. It is the entry point to the other pages, which were written thread by
thread in the order the questions came up. The kickoff deck (`doc/kickoff_deck/`) follows it
slide for slide.

The grades:

| Grade | Meaning |
|---|---|
| **Solid** | An interval that excludes zero, and a second, independent confirmation: another method, another period or sample, or a placebo or randomisation test. No known contradiction. |
| **Indicative** | The direction is supported, but by one method, one period or few sites, or the size moves with the analysis choices. |
| **Open** | Tested without a clear answer, or an interpretation that has not been tested. |
| **Scenario** | A calculation that depends on stated assumptions. Not a measurement. |

The checks behind the grades were run for this page (commands `event-audit` and
`forecast-record`; sections 6 and 7): the event findings under 90 combinations of analysis
choices and against 12 placebo runs, and the forecast findings on every day of the record
(1,201 days) in place of the 69 test days.

## The five questions in short

| Question | Answer | Grade |
|---|---|---|
| 1. Can we trust the data? | 134 sites with 3.4 years of half-hourly electricity; 92 have complete meter data; the meters reproduce the annual estates returns at 65 verified sites. At sites with on-site generation the meter shows grid import only. | Solid |
| 2. What changes the demand of a site? | Temperature: +1.1 % per °C below 14 °C and +1.5 % per °C above 20 °C, carried over three days, heat in the evening. Patient activity: nothing measurable. | Solid (size of the heat effect: indicative) |
| 3. Why do sites differ? | The reaction is a stable property of the site. Region and trust explain no more than chance; the registers explain the cold reaction in part (heating fuel) and the heat reaction hardly at all. | Solid; sites with a plant indicative |
| 4. How good is a day-ahead forecast, and what is it worth? | A model with no training has 11.2 % error in every year, better than trained models. A battery that plans on it gets half of the achievable peak cut, and 40 % less on heat-wave days. Three quarters of the error sits at 12 sites with a plant. | Solid |
| 5. What changes when the estate changes? | With heat pumps for all gas heat, 13 of 31 sites exceed their grid connection. Field data a year after installation: +8 points of electricity. | Scenario; field data indicative |

## 1. Can we trust the data?

| | Finding | Number | Evidence | Grade |
|---|---|---|---|---|
| 1.1 | The panel | 134 sites with half-hourly electricity, 7 December 2022 to 8 May 2026; 92 with complete meter data, 42 flagged (35 with meter gaps) | coverage rule, `estates_analysis.md` §1 | Solid |
| 1.2 | The meters reproduce the estates returns | annual ratio median 1.00 (quartiles 0.98 to 1.00), peak ratio 1.00 (0.95 to 1.03), at the 65 sites that verify, of 114 matched | the 65 are those within 0.65 to 1.5 of the return; the agreement inside that window is far tighter than the window | Solid |
| 1.3 | At sites with a plant the meter shows grid import only | 9.2 MW of import against 23.3 MW of implied demand at 20 sites: 40 % | arithmetic on the 2024/25 return; three unit errors found in the returns | Indicative |
| 1.4 | The temperatures are good enough | ERA5 against 76 stations: daily mean within 0.62 °C; 0.9 °C too cool on hot days; every weather effect is the same with three temperature sources | `weather_and_cooling.md` §6 | Solid |
| 1.5 | Public patient-activity data stop at the trust and the month | A&E monthly for 18 of 24 trusts, beds quarterly | source survey; the daily winter reports need a manual download | Solid |

## 2. What changes the demand of a site?

| | Finding | Number | Evidence | Grade |
|---|---|---|---|---|
| 2.1 | Cold raises demand | +1.1 % per °C below 14 °C (0.7 to 1.5). In a cold spell, 6.7 °C colder than the reference days: +11 % (9 to 13) | fixed effects with date effects; weather reassigned between sites at random: 0 of 200 as large; three temperature sources; neighbours' weather adds nothing. Event measure: 9 to 17 % over 90 sets of choices, every interval clear of zero; 12 placebo runs give −2 to +6 % | Solid |
| 2.2 | Heat raises demand | +1.5 % per °C above 20 °C (0.2 to 2.8); +1.8 (0.5 to 3.2) with sunshine held fixed. In a heat wave: +5.7 % (2.6 to 8.7) | randomisation: 6 of 200 as large; three temperature sources give 1.4 to 2.3. Event measure: 3.5 to 6.8 % over 90 sets of choices, every interval clear of zero; placebo runs −2.5 to +2.3 % | Direction solid; size indicative |
| 2.3 | Heat acts in the afternoon and evening, cold all day | heat-wave days: +9 % from 16:00 to 22:00, +1 % from 08:00 to 12:00; cold-spell days: +9 to +13 % at every hour | three analyses agree: by block of the day (+2.4 % per °C in the afternoon, +0.3 % at night), by hour (3.3 % per °C at 16:00), and the event days. Evening minus morning: 5.9 to 9.0 points over 90 sets of choices; placebo −2.3 to +2.3 | Solid |
| 2.4 | The temperature acts over several days | per °C colder: 0.76 % today, 0.51 % from the day before, 0.11 and 0.18 from the two days before that, 1.6 % together. Per °C of daily maximum above 24 °C: 0.54, 0.72, 0.21, 0.15, 1.6 % together | fixed effects earlier: the same day and one day of carry-over. Event measure: the reaction from the third day of a heat wave is 4.4 points above the first day (2.1 to 6.8; clear in all 90 sets of choices; placebo −3.6 to +3.4) | Solid |
| 2.5 | An event is no more than its temperatures | with the temperatures of the day and of the three days before, nothing is left for "the third day of an event": heat +0.4 points (−1.3 to +2.0), cold +0.3 (−0.9 to +1.5). With the day's temperature alone: +2.7 (1.1 to 4.4) and +2.2 (0.9 to 3.5) | fixed effects earlier: no extra reaction on days of 30 °C and more (−1.5 %, −4.3 to +1.3) | Solid |
| 2.6 | Patient activity does not change electricity demand | doctor strikes: −0.4 % from 08:00 to 18:00 (−1.6 to +0.8), 34 weekdays, 132 sites. Visitors: elasticity 0.00 at three sites; a forecast that is given the true visitor count is no better (+0.006 points, −0.012 to +0.024). Occupied beds by quarter: correlation −0.04, 19 trusts | four tests with different data; placebo week next to the strikes | Solid for electricity |
| 2.7 | Sunshine lowers the import | −0.7 % per kWh/m² of daily sunshine; 44 of 83 sites | one analysis; no interval on the pooled value | Indicative |
| 2.8 | Tomorrow's heat already raises today's demand | +0.45 % per °C (−0.15 to +1.04) | anticipation, or error in the temperature of a site; not separated | Open |
| 2.9 | The effects survive hidden confounding of moderate strength | a hidden factor must change the odds that a day is an event day by 1.8 times (heat waves) and 2.6 times (cold spells) before the effect can be zero; 1.3 and 2.2 times with the sampling interval | sharp bounds of the generalized marginal sensitivity model (Frauen et al. 2023), worst case. Observed calendar factors change those odds by up to 3.2 and 8.5 times, and holding them fixed moves the effect by 0.2 and 0.4 points | Solid for cold; indicative for heat |

Tested and rejected on the way: that the nights of the previous days make the heat build up
(the day maxima of the previous days do; with night temperatures alone the third-day term stays
at +2.2 points, 0.4 to 3.9); that strike days or visitor numbers mark days that are hard to
forecast.

## 3. Why do sites differ?

| | Finding | Number | Evidence | Grade |
|---|---|---|---|---|
| 3.1 | Sites differ far beyond noise, for heat even in sign | heat slope from −3.0 to +5.4 % per °C across the middle 80 % of sites; 94 to 97 % of the spread is real; 33 sites rise with heat and 21 fall, of 83 with complete meters | standard errors corrected on 9 November (they were 1.7 to 3.2 times too small) | Solid |
| 3.2 | The reaction belongs to the site, not to the place | the region accounts for 9 % (heat) and 5 % (cold) of the differences between sites, chance gives 8 % and 7 %; trusts are at chance too; neighbours do not react alike while their weather does | region above chance in 0 % (heat) and 6 % (cold) of the 90 sets of choices, as the placebo runs | Solid |
| 3.3 | A site's past events predict its next one | error of the predicted reaction: 9.6 points from the other sites in the event, 9.9 from the other sites of the region, 6.7 from the site's own other events (heat); 10.8, 11.2 and 8.5 (cold). The site's other events explain 50 % (heat) and 67 % (cold) | 36 to 59 % and 49 to 67 % over the middle half of the 90 sets of choices; placebo runs at most 22 % | Solid |
| 3.4 | The cold reaction follows the heating fuel | a site with a standard deviation more gas per unit of electricity reacts 0.5 points per °C less (0.2 to 0.8); in cold spells 3.9 points less (1.1 to 6.6) | two measures, two samples (87 and 53 sites); a mechanism | Solid |
| 3.5 | The registers do not explain the heat reaction | estates descriptors: none of the real spread; certificate fields: 0 to 9 %; gas ratio 11 % | two measures. The site's own daily profile explains 45 % in sample and 30 to 50 % out of sample: indicative | Solid (that the registers do not) |
| 3.6 | Sites that fall on hot days are small buildings with a heating tail | 21 of 83 sites with 8 % of the demand; they fall as much with warm nights as with hot afternoons | the mechanism is a reading | Indicative |
| 3.7 | At sites with a plant, the meter multiplies every change in demand | import in heat waves +25 % (9 to 42) against +6 % elsewhere, 12 sites. The plant supplies about 70 % of the demand, so a demand reaction like that of the other sites, with the plant unchanged, gives +17 to +23 % at the meter | 14 to 31 % over 90 sets of choices, but placebo runs reach +18 %. Earlier pages: +32 % on days of 24 °C and more (23 sites), +12 % in the 2025 heat waves (17 sites) | Direction solid; size and mechanism indicative |
| 3.8 | In cold spells the plants give more output; in heat waves a change is not established | cold: import +4 % where +15 to +32 % is expected with the plant unchanged, so output +5 to +18 % at the six plants that follow heat. Heat: output −3 to −7 % at those plants (−15 to +6) | output is the yearly mean of the estates return, not a measurement; two assumptions for the demand reaction | Indicative |
| 3.9 | Three shapes of the day; the use of a site sets its shape | 64 flat sites, 44 moderate, 21 office-like, 2 night-heavy | descriptive; the number of groups is a judgement | Indicative |

## 4. How good is a day-ahead forecast, and what is it worth?

| | Finding | Number | Evidence | Grade |
|---|---|---|---|---|
| 4.1 | A model with no training beats the trained models | 11.5 % error against 13.0 % for the gradient-boosted model (−1.5 points, −1.9 to −1.1), 13.7 % for a neural process trained on the panel and 18.4 % for the same day last week, on the 69 test days | on every day of the record: 11.2 % (10.0 to 12.3), and 11.0 to 11.5 % in the four years, against 17.7 % for the same day last week | Solid |
| 4.2 | Calendar, price and the weather forecast add 0.7 points; the weather forecast is as good as the observed weather | −0.65 points (−1.07 to −0.38); forecast against observed weather +0.05 (0.01 to 0.08) | whole record: 11.2 % against 11.8 % without inputs | Solid |
| 4.3 | The error sits at the sites with a plant | 78 % of the error at 12 sites with 48 % of the metered demand, in every year (73 to 79 %); three sites hold 55 % | half of it is their size; days with a step larger than the site's mean import are 1.5 % of site-days and hold 37 % of the error | Solid |
| 4.4 | A battery that plans on the forecast gets half of the achievable peak cut | 52 % (48 to 58) on the whole record, 52 to 57 % in the four years; same day last week: 39 % | depends on the storage: 43 % with one hour, 65 % with four; not on the power | Solid for a two-hour battery |
| 4.5 | Heat-wave days cost value, not accuracy | captured peak cut 27 % on heat-wave days against 45 % on the other days of the same site and month (−18.4 points, −27.7 to −10.0); error 12.1 % against 11.8 % (0.4 points, −0.5 to 1.3) | summer 2023: −21.2 points, −28.5 to −11.8; summer 2025: −18.4 points, −28.5 to −10.3; with the site's own percentiles: −13.0 points, −18.8 to −8.1; with each of five batteries the heat-wave days keep 57 to 70 % of what the other days get | Solid |
| 4.6 | Cold spells do not clearly cost value | captured peak cut 55 % on cold-spell days against 57 % on the other days of the same site and month (−1.1 points, −4.3 to 1.8); clear in 0 of 4 winters | the earlier figure (44 % against 53 %) came from the one winter of the test year | Withdrawn |
| 4.7 | Knowing that a site has a plant does not help; knowing when it runs would | a flag: +0.02 points (−0.18 to +0.21). The state of the plant on the target day: 16.5 to 12.5 % at the plant sites | the state signal is made from the import itself: an upper bound | Indicative |
| 4.8 | Visitor counts add nothing to the forecast | +0.006 points (−0.012 to +0.024) | two sites; agrees with 2.6 | Indicative |
| 4.9 | Coherent forecasts across meters, sites and trusts improve the national total | 8.0 to 6.8 % | no interval; 61 test days | Indicative |
| 4.10 | A published method for heat and cold waves (Lu et al. 2026) is level with the zero-shot model | 11 % behind in Europe, 6.5 % ahead in PJM | the paper's own public data; one seed | Indicative |
| 4.11 | A rule for special days works on the bank holidays and only there | bank holidays: 23.5 to 18.5 % error (−4.9 points, −7.8 to −2.3). Special days (bank holidays and the Christmas period, 4.1 % of days): 16.7 to 14.9 % with the rule on bank holidays (−1.8, −2.8 to −0.9). All days: 11.33 to 11.27 % (−0.06, −0.09 to −0.03). Normal days (11.1 %) do not change | rule of Arora and Taylor (2013, 2018) on the whole-record forecasts from 2024 on, 76,009 site-days. The rule on the Christmas period adds 3.6 points (2.9 to 4.4) and on the proximity days 3.2 points (2.7 to 3.7); the same pattern without inputs (bank holidays −13.8 points) | Solid for bank holidays; the rule fails on the Christmas period and the proximity days |

## 5. What changes when the estate changes?

| | Finding | Number | Evidence | Grade |
|---|---|---|---|---|
| 5.1 | Heat pumps for all gas heat put sites above their grid connection | 13 of 31 panel sites; 48 % of 631 sites in England | assumes all gas heat replaced, a seasonal performance factor of 2.55, boiler efficiency 0.85 | Scenario |
| 5.2 | The choice of winter hardly matters | 10 to 15 of 31 sites in the last 30 winters, 14 in a one-in-twenty winter, 19 in 1962/63 | 66 weather years on the same scenario | Scenario |
| 5.3 | A year after heat pumps, sites use more electricity | +8.3 points (3.5 to 14.6), 23 sites; gas −9.4 points (−21 to +7, not clear); performance factor 2.8 at 13 clean switches (1.3 to 7.4) | estates returns, coarse; the field suggests that about a tenth of the gas is replaced, not all of it | Indicative |
| 5.4 | Hospitals use rooftop solar themselves | all of it up to 1 kWp per kW of mean demand | assumed performance ratio | Scenario |
| 5.5 | A battery earns £23 per kWh a year on wholesale prices | £88 to £96 on an Agile-type tariff | perfect price foresight, no degradation | Scenario |
| 5.6 | Warming lowers annual electricity | −1.2 % at +4 °C | uniform shift, fixed response | Scenario |
| 5.7 | LED lighting saves electricity | −0.6 % per ten points of LED share (−1.2 to 0.0) | p = 0.05, 59 sites | Open |
| 5.8 | Solar and heat-pump installations show at the meter | 3 of the 4 sites with reported solar | 8 downward breaks among 107 sites, and upward ones as large | Indicative |

## 6. The audit of the event findings

`event-audit` (`analysis/event_audit.py`, tables in `code/reports/event_audit/`).

**Ninety sets of choices.** Five sets of reference weeks, three definitions of an event (the
catalogue, heat thresholds one degree lower, the site's own 5th and 95th percentiles), three
site samples (complete meters, all 134 sites, the 91 forecast sites), holidays in or out.
**Twelve placebo runs.** The catalogue's events moved by five to eight weeks and by one and two
years; a reaction on those days is a fault of the measure or chance.

{{figure: specification_curve_reaction | caption=The reaction of the sites without on-site generation under each of the 90 sets of choices (small points, in order of size), the baseline with its interval (large point) and the range of the placebo runs (grey band).}}

{{figure: specification_curve_build_up | caption=The reaction from the third day of an event minus that of the first day.}}

{{figure: specification_curve_evening | caption=Evening (16:00 to 22:00) minus morning (08:00 to 12:00).}}

{{figure: specification_curve_plant | caption=Sites with a plant minus the others. For heat waves every set of choices gives a clear difference, but the placebo band reaches most of it.}}

{{figure: specification_curve_region | caption=Share of the differences between sites that the region explains, above chance. The baseline, the 90 sets of choices and the placebo runs all sit around zero.}}

{{figure: specification_curve_own_events | caption=Share of a site's reaction that its other events explain.}}

{{table: code/reports/event_audit/grid_summary.csv | round=2}}

**Does a site's past predict its next event?** Each site's reaction in each national event,
predicted without that event (and without that site, for the means of other sites).

{{figure: prediction | caption=Error of four predictions of a site's reaction in an event that the prediction has not seen. Sites without on-site generation.}}

{{table: code/reports/event_audit/prediction_scores.csv | round=2}}

The region is a worse guide than the national mean of the event. The site's own other events
cut the error by a fifth (cold) to a third (heat). "Two events are sufficient", a hypothesis of
the earlier deck, is not supported: in date order the error keeps falling up to the fifth
earlier event (12.1 to 10.0 points for cold spells on the same later events).

{{table: code/reports/event_audit/prediction_curve.csv | round=2}}

**Is an event more than its temperatures?** The daily reaction of the sites without on-site
generation on the temperature of the day and of the three days before, each as a difference
from the reference days; cold season on degrees of daily mean, warm season on degrees of daily
maximum above 24 °C (heat acts above a threshold: the plain difference in the daily maximum
explains nothing in the warm season).

{{figure: lag_weights | caption=The effect on today's demand of a degree today and of a degree one, two and three days before.}}

{{figure: event_term | caption=The extra reaction from the third day of an event, when the model has the day's temperature and when it also has the three days before.}}

{{table: code/reports/event_audit/lag_models.csv | round=2}}

**What the plant does.** The observed reaction of the import against the reaction that the
import would show if the site's demand reacted like that of the sites without a plant and the
plant did not change. Plant output is the rated power times the capacity factor of the estates
return.

{{figure: plant_decomposition | caption=Sites with a plant: reaction of the summed import (bar; line: the range when one site at a time is left out) and the reaction expected with the plant unchanged (grey bar: demand reacts like all sites without a plant; tick: like the acute sites without a plant).}}

{{table: code/reports/event_audit/plant_decomposition.csv | round=1}}

**Which site effects are real?** (added 17 November 2026; `site_effect_tests.csv`,
`site_effect_spread.csv`.) Each site's reaction is tested against its own placebo: 2,000 random
sets of blocks of its ordinary days in the same season, with the lengths of its events. The
spread of those placebo reactions is the site's standard error; the p-values are adjusted for
the number of sites (false discovery rate 5 %).

{{figure: site_effects_heatwave | caption=Heat waves: each site's reaction with its placebo interval, in order of size. Filled: different from zero after the adjustment. Black points: after shrinkage towards the mean of the group.}}

{{figure: site_effects_cold_spell | caption=Cold spells: the same.}}

{{table: code/reports/event_audit/site_effect_spread.csv | round=2}}

- **Heat waves, 64 sites without on-site generation:** mean +5.4 % (standard error 1.0). 21
  sites rise and 1 falls after the adjustment (26 and 3 at p < 0.05, where chance gives 3).
  The true spread between sites is 6.4 points; 79 % of the observed spread is real.
- **Cold spells, 75 sites:** mean +9.0 % (0.7). 50 rise, none falls. True spread 4.8 points.
- **Sites with on-site generation:** for these the measure is biased upwards. Random blocks of
  ordinary summer days give +1 to +21 % at single sites, because the import is volatile.
  With that removed, their mean reaction to heat waves is +11.9 % (standard error 4.7), not
  +25 %, and the difference from the other sites (+6.5 points) is not clear. Finding 3.7 has
  to be read with this.
- **A ranking of raw site reactions overstates the ends.** Of the ten highest raw reactions,
  seven are different from zero after the adjustment; of the ten lowest, two. After shrinkage
  the largest reaction is +26 %, not +92 %.

**How wrong can the event effects be?** (added 22 November 2026; command
`causal-sensitivity`, `analysis/causal_sensitivity.py`, tables in
`code/reports/causal_sensitivity/`.) The event effects compare event days with ordinary days of
the same site and season. A hidden factor that makes event days more likely and also moves
demand would bias them. The generalized marginal sensitivity model of Frauen, Melnychuk and
Feuerriegel (NeurIPS 2023) bounds the effect for a given strength Γ of such a factor: Γ is the
largest factor by which it can change the odds that a day is an event day. The bounds are
sharp, and they are a worst case: they let the hidden factor line up with demand as badly as
possible.

{{figure: sensitivity_bounds | caption=Bounds on the mean effect of a heat wave and of a cold spell for a hidden factor of strength Γ. Solid: lower bound; dashed: upper bound; band: 95 % interval from resampling sites.}}

{{table: code/reports/causal_sensitivity/bounds.csv | round=2}}

- **Heat waves** (64 sites): +5.3 % at Γ = 1. The lower bound reaches zero at Γ = 1.84, its
  interval at Γ = 1.34.
- **Cold spells** (75 sites): +9.4 %. The lower bound reaches zero at Γ = 2.64, its interval at
  Γ = 2.18.
- **What a Γ of that size is.** The same odds ratio for what we do observe: weekends 1.06 and
  1.12 (the weather does not know the weekday), school holidays 3.2 for heat waves and 1.5 for
  cold spells, the calendar month 2.4 and 8.5. These factors are stronger than the breakdown
  values, yet holding them fixed moves the effect from 4.9 to 4.8 % (heat) and from 9.3 to 8.9 %
  (cold): they are tied to the timing of events and hardly to the reaction, because the
  reference days already remove the season.
- **Reading.** A hidden factor would have to be tied to the timing of events about as strongly
  as school holidays are, and to demand far more strongly than any calendar factor we can
  measure. The cold effect is safe against that; the heat effect has less room.

{{table: code/reports/causal_sensitivity/benchmarks.csv | round=2}}

{{table: code/reports/causal_sensitivity/observed_adjustment.csv | round=2}}

**Road traffic near the site** (`traffic_moderator.csv`). The Department for Transport counts
in the SENSE data pack are yearly averages for count points, not daily series, so they cannot
serve as a daily covariate. As a descriptor of the place (the flow at the busiest count point
within one kilometre, median 23,000 vehicles a day) they do not go with the reaction of a site:
−1.6 points per standard deviation for heat waves (−4.5 to +1.3) and −1.8 for cold spells (−5.1
to +1.4).

{{table: code/reports/event_audit/traffic_moderator.csv | round=2}}

## 7. The forecast on the whole record

`forecast-probes --which record` and `forecast-record` (`analysis/forecast_record.py`, tables
in `code/reports/forecast_record/`). Chronos-2, zero-shot, for every day from 18 January 2023 to
8 May 2026 at the 91 forecast sites, on a GPU: without inputs, and with calendar, price,
national forecasts and the weather forecast of the day before. A site-day counts if all 48
half-hours are observed and at least 80 % of the six weeks before it.

{{figure: record_error_years | caption=Day-ahead error in each year, with the interval from resampling sites.}}

{{figure: record_value_years | caption=Share of the achievable peak cut in each year.}}

{{table: code/reports/forecast_record/by_period.csv | cols=window,model,sites,site_days,days,nmae_pct,nmae_lo,nmae_hi,naive_nmae_pct,captured_point_pct,captured_lo,captured_hi,captured_q70_pct,plant_error_share_pct,top3_error_share_pct,plant_large_step_day_share_pct,plant_large_step_error_share_pct | round=1}}

{{figure: record_event_value | caption=Captured peak cut on event days minus the other days of the same site and month, by season, for two definitions of an event day.}}

{{table: code/reports/forecast_record/event_days.csv | cols=definition,type,season,sites_group,model,sites,event_site_days,nmae_event_pct,nmae_other_pct,nmae_diff_pts,nmae_diff_lo,nmae_diff_hi,bias_event_pct,captured_event_pct,captured_other_pct,captured_diff_pts,captured_diff_lo,captured_diff_hi | round=1}}

{{figure: record_decision | caption=Captured peak cut for five batteries.}}

{{table: code/reports/forecast_record/decision.csv | round=1}}

{{table: code/reports/forecast_record/model_differences.csv | round=2}}

**How much history the model needs** (added 22 November 2026; `forecast_probe.run_context_long`,
`code/reports/forecast_anatomy/context_long.csv`). Spoek et al. (2026) find twelve weeks of
history best for zero-shot district heat load forecasts. On our 69 test days, without inputs:
one week 16.4 % error, two weeks 12.7 %, six weeks 12.2 %, twelve weeks 12.1 %, twenty-four
weeks 12.2 %. Twelve weeks are better than six by 0.09 points (0.03 to 0.16); twenty-four are
not (−0.05, −0.13 to +0.04); two weeks are worse than six by 0.54 points (0.35 to 0.72). The
paper's result holds on our data, with a small gain; the earlier statement that two weeks are
sufficient was too generous.

{{table: code/reports/forecast_anatomy/context_long.csv | round=2}}

**A rule for special days** (added 22 November 2026; command `special-days`,
`analysis/special_days.py`, tables in `code/reports/special_days/`). Arora and Taylor (2013,
2018) forecast anomalous load with a rule: each special day refers to the most recent day of
the same kind at the same time of year, weekday with weekday. Here the rule corrects the
zero-shot forecast: the forecast for the special day times the ratio of demand to forecast on
the corresponding past special day, by half-hour.

{{figure: special_days | caption=Day-ahead error by kind of special day: the forecast, the forecast with the rule, and the load of the most recent Sunday. Forecast with calendar and weather inputs, 2024 to May 2026.}}

{{table: code/reports/special_days/scores.csv | cols=model,kind,sites_group,sites,site_days,days,nmae_forecast_pct,nmae_rule_half_hour_pct,nmae_rule_day_pct,nmae_sunday_pct,rule_minus_forecast_pts,lo,hi,median_bias_forecast_pct | round=2}}

- **Bank holidays** are where the forecast fails: 23.5 % error with the calendar as an input
  (32.3 % without), against about 11 % on ordinary days, and the forecast is too high (+5 %
  and +12 %). The rule brings it to 18.3 % (−5.2 points, −8.2 to −2.4). The load of the most
  recent Sunday does as well (17.2 %).
- **The days next to a holiday and the Christmas period are not anomalous at these sites**:
  11.6 to 15.1 % error. There the rule adds the noise of one past day and makes the forecast
  worse by 3 to 4 points. The rule is for bank holidays only.
- **By holiday** (`scores_by_name.csv`): the gain is on Christmas Day, Good Friday, New Year's
  Day and the late summer bank holiday (29 to 36 % error down to 17 to 21 %). On Easter Monday,
  May Day, the spring bank holiday and Boxing Day the forecast is already at 12 to 20 % and the
  rule does not help.
- **Over all days** bank holidays are 2.3 % of days and 2.8 % of the error; the rule lowers
  the error of all days from 11.33 to 11.27 % (−0.06 points, −0.09 to −0.03).

{{table: code/reports/special_days/overall.csv | round=2}}

**The rule on all days, normal days, special days and proximity days** (added 22 November 2026;
`day_classes.csv`). Arora and Taylor score a rule for special days on four sets of days. Here:
special days are the bank holidays and the Christmas period (24 and 27 to 31 December, 2
January); proximity days are the working day before and the working day after a bank holiday;
all other days are normal days. Three policies say where the rule is applied; a day outside
the policy keeps the forecast, so the normal days are the same with and without the rule.

{{figure: special_day_classes | caption=Day-ahead error on all days, normal days, special days (and their two parts) and proximity days: the forecast, and the forecast with the rule under three policies. Forecast with calendar and weather inputs, 2024 to May 2026, 91 sites.}}

{{table: code/reports/special_days/day_classes.csv | cols=model,policy,day_class,site_days,days,share_of_days_pct,share_of_error_pct,rule_applied_pct,nmae_forecast_pct,nmae_with_rule_pct,rule_minus_forecast_pts,lo,hi | round=2}}

For the forecast with calendar and weather inputs (76,009 site-days on 855 days):

| Days | Share of days | Forecast | Rule on bank holidays | Rule on special days | Rule on special and proximity days |
|---|---|---|---|---|---|
| All days | 100 % | 11.33 % | 11.27 % (−0.06, −0.09 to −0.03) | 11.35 % (+0.02, −0.02 to +0.05) | 11.42 % (+0.09, 0.05 to 0.13) |
| Normal days | 93.6 % | 11.09 % | 11.09 % | 11.09 % | 11.09 % |
| Special days | 4.1 % | 16.68 % | 14.88 % (−1.80, −2.77 to −0.88) | 16.84 % (+0.16, −0.82 to +1.06) | 16.84 % |
| of these: bank holidays | 2.3 % | 23.46 % | 18.53 % (−4.93, −7.84 to −2.25) | 18.53 % | 18.53 % |
| of these: Christmas period | 1.8 % | 11.90 % | 11.90 % | 15.49 % (+3.59, 2.93 to 4.39) | 15.49 % |
| Proximity days | 2.3 % | 13.45 % | 13.45 % | 13.45 % | 16.65 % (+3.21, 2.70 to 3.73) |

- **The rule works on the bank holidays and only there.** Applied to the bank holidays it
  lowers the error of the special days by 1.8 points and the error of all days by 0.06 points;
  both intervals exclude zero.
- **Applied to all special days it gains nothing** (+0.16 points on the special days): the
  loss on the Christmas period cancels the gain on the bank holidays. The Christmas period is
  no anomaly at these sites (11.9 % error against 11.1 % on normal days), so one past day adds
  noise.
- **Applied to the proximity days it does harm** (+3.2 points). The proximity days are harder
  than normal days (13.5 % against 11.1 %), but the same day a year before does not say how.
- **Without inputs the same pattern is larger**: bank holidays 32.5 to 18.8 % (−13.8 points,
  −19.3 to −8.8), special days −5.6 points with the rule on bank holidays, all days −0.19
  points (−0.25 to −0.13); Christmas period +3.3 and proximity days +1.2 points.
- A hospital is not the national grid: the national load of the two papers falls on the days
  around a holiday as well, and a hospital's does not. The policy for these sites is the rule
  on bank holidays.

## 8. What this page changes

**Withdrawn or corrected**

- *Cold spells cost the forecast value* (44 % against 53 % of the peak cut): one winter. Over
  four winters the difference is not clear (finding 4.6).
- *Grid import at sites with a plant rises by 25 % in heat waves because the plant is turned
  down*: the rise is real in direction, but most of it is the meter showing a third of the
  load, and the placebo runs reach +18 %. A fall in plant output is not established (3.7, 3.8).
- *Two events are sufficient to predict a site's reaction* and *the nights of the previous days
  explain the build-up*: both tested and not supported (3.3, 2.5).
- *39 sites fall on hot days against 48 that rise* (key results of the PI report): from before
  the correction of the standard errors; it is 26 against 43, and 21 against 33 on the sites
  with complete meters.
- *Every learned model beats the naive forecast by 26 to 35 %* and *the weather forecast is
  within 0.1 points of the observed weather for every model* (proof-of-concept page): true for
  the leading models, not for all in the table.
- *The two night-heavy sites probably have storage heating* (clustering page): both run a
  plant.

**Confirmed on the whole record**

- The error of the best model (11.5 % on the 69 test days) is 11.0 to 11.5 % in each of the
  four years; the share of the achievable peak cut (53 %) is 52 to 57 %.
- Days on which the import of a site with a plant steps by more than its mean are 1.5 % of all
  site-days and hold 37 % of the error (1.8 % and 42 % in the test year).
- Heat-wave days cost the forecast value in both large summers and under both definitions of
  a heat-wave day.

**How numbers for the same thing relate**

- *The heat effect.* +1.5 % per °C of daily mean above 20 °C (finding 2.2) is the number to
  quote. 1.6 % per °C of daily maximum above 24 °C is the same effect added over four days.
  The 0.3 to 0.8 % per °C of the event page divides the reaction by the whole difference in
  the daily maximum from the reference days, most of which lies below the threshold.
- *Sites.* 134 with electricity; 114 matched to the estates return; 109 or 107 without a plant;
  92 with complete meters (83 without a plant in the heat analysis); 91 in the forecast panel.
- *Event days.* The Met Office and UKHSA criteria per site (the catalogue) are the reference
  from now on. The site's own 5th and 95th percentiles and days of 28 °C and more were used on
  earlier pages; section 7 gives the forecast results for the first two side by side.

**Not answered**

- Plant output: the meters show import only. Everything about plants in section 3 is a
  calculation with yearly means from the estates return.
- The size of cooling plant: recorded on 19 % of recent certificates.
- Gas, and daily bed occupancy (the daily winter reports need a manual download).
- The project brief was never filled in: these five questions are a reading of the work done,
  not a plan that was set at the start.
