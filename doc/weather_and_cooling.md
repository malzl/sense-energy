# Cooling need, extreme-weather events and the demand response

`sense-energy build-weather-daily` (table `interim/weather_daily.parquet`, see the
[data dictionary](data_dictionary.md#daily-interimweather_dailyparquet)), `sense-energy cooling-analysis`
(`analysis/cooling.py`, tables in `code/reports/cooling/`) and `sense-energy weather-events`
(`analysis/weather_events.py`, tables in `code/reports/weather_events/`); figures from
`code/scripts/weather/` in `code/reports/figures/weather/`. 26 October 2026.

Everything is derived from the hourly ERA5 reanalysis already extracted at the 275 located sites
(1 December 2022 – 31 May 2026); nothing new was downloaded. The half-hourly electricity record
runs 7 December 2022 – 8 May 2026 (gas to 9 March 2025), so it covers three summers but not the
record heat of late May 2026.

## 1. Cooling need

The integration-method cooling degree days (base 15.5 °C) of the median site were 197, 183 and
291 °C day over June–August 2023, 2024 and 2025; summer 2025 had 119 hours at or above 25 °C
against 46–50 in the two summers before (`cooling_need_monthly`). The hottest site-days in the
table are 25–26 May 2026 (34.2 °C at the hottest site; Kew observed a UK May record of 35.1 °C),
after the electricity record ends.

## 2. Electricity response to cooling need

Daily mean demand on non-holiday weekdays, regressed on HDD (15.5 °C) and CDD with a fixed effect
for every calendar month, so the slope comes from day-to-day weather within a month
(`cooling_response_site`, 95 % HAC intervals).

| Group | Sites | Median slope (% per °C day) | 10th – 90th percentile | Significantly > 0 | Demand-weighted |
|---|---|---|---|---|---|
| no on-site CHP | 107 | 0.8 | −1.7 – 3.9 | 51 | 1.4 |
| on-site CHP | 21 | 3.9 | 1.7 – 10.4 | 14 | 6.0 |

- **Most sites barely respond to heat.** Without CHP, a 30 °C day (about 8 °C day above 15.5 °C)
  adds about 6 % at the median site. General healthcare buildings respond most (median 3 % per
  °C day); mental-health and community sites hardly at all.
- **CHP sites import much more on hot days, but not all of it is cooling.** A heat-led CHP unit
  turns down when there is no heat demand, so grid import rises with temperature even if the
  building's demand does not. The two cannot be separated without CHP generation data; acute
  hospitals without CHP respond 1.4 – 2.3 % per °C day, acute hospitals with CHP 2 – 12 %.
- **Base 15.5 °C fits best** for 80 of 128 sites (AIC across 15.5, 18 and 22 °C): hospitals' high
  internal gains put their cooling balance point low. Daily maximum temperature and integrated
  CDD(15.5) track summer demand slightly better than humidity measures (enthalpy, wet bulb), but
  within-month weather explains under 10 % of summer weekday variance at the median site.

## 3. Extreme-weather events

| Type | Definition (site-day, then runs) | National events | Site events |
|---|---|---|---|
| heatwave | Met Office: ≥ 3 days with T_max ≥ county threshold (25 – 28 °C) | 8 (7 in the electricity record) | 907 |
| cold spell | UKHSA cold-alert criterion: daily mean ≤ 2 °C for ≥ 2 days | 14 | 3,263 |
| snow | ≥ 5 mm water equivalent in a day | 8 | 498 |
| heavy rain | ≥ 25 mm in a day | 19 | 1,069 |
| high wind | daily max gust ≥ 22 m/s (about 50 mph) | 37 | 5,378 |

A national event is a run of dates on which at least 5 % of sites are in a site event of that type
(one-day gaps bridged). The 28 °C counties (London, Surrey, Berkshire, Buckinghamshire,
Bedfordshire, Hertfordshire, Cambridgeshire) and East Riding (26 °C) follow the Met Office 2022
list; other counties take a regional approximation (`site_thresholds.csv`). Lowering every
threshold by 1 °C for ERA5's warm-tail bias raises the heatwave site events from 907 to 1,445 but
leaves the national count at 8 (`weather_event_timeline`).

The catalogue recovers the named storms (Otto, Noa, Ciarán, Debi, Pia, Gerrit, Henk, Isha,
Jocelyn, Kathleen, Bert, Darragh, Éowyn, Amy), Storm Babet's rain on 20 October 2023, the four
England heatwaves of summer 2025 and the cold spells of December 2022, January 2024 and
January 2025.

**Demand on event days** (median over affected site-days, against the same weekday one and two
weeks either side; `weather_event_demand`): cold spells +4 to +12 % (median +7 %), snow +5 to
+8 %, heatwaves +3 to +6 % (median +4 %), heavy rain +0 to +5 %, high wind −6 to +6 % (the low
points are the Christmas-week storms, where the holiday dip dominates).

## 4. Nonlinear temperature response

`sense-energy weather-response` (`analysis/weather_response.py`, tables in
`code/reports/weather_response/`; pyfixest). Log daily demand on 2 °C bins of daily mean temperature
(reference 14 – 16 °C) with site × year-month and site × weekday fixed effects, bank holiday,
Christmas-week and school-summer controls; 95 % intervals clustered by ISO week. Each bin answers "how
much more does this site use on a day this warm or cold than on a 15 °C day in the same month".
140,326 site-days, 132 sites, December 2022 – May 2026.

| Daily mean | inpatient (43) | outpatient (66) | on-site CHP (23) |
|---|---|---|---|
| < −2 °C | +24.8 % [21.3, 28.4] | +31.5 % [27.8, 35.3] | −0.2 % [−8.5, 8.9] |
| 0 – 2 °C | +15.9 % | +21.0 % | +0.5 % |
| 8 – 10 °C | +6.1 % | +7.6 % | −3.3 % |
| 14 – 16 °C | 0 (reference) | 0 | 0 |
| 20 – 22 °C | +2.6 % | +4.6 % | +29.5 % |
| 22 – 24 °C | +6.3 % [4.7, 7.9] | +7.2 % [5.6, 8.8] | +26.6 % |
| ≥ 24 °C | +10.0 % [6.5, 13.6] | +14.8 % [9.7, 20.2] | +32.3 % [11.7, 56.8] |

(`temperature_response_curve`)

- **A hockey stick, not a line.** Demand is flat between 14 and 20 °C. Below that it rises almost
  linearly, about 1.5 – 2 % per degree. Above 20 °C it rises *convexly*: +2.6 % at 20 – 22 °C,
  +6.3 % at 22 – 24 °C, +10 % above 24 °C. A linear temperature term or a single cooling-degree-day
  slope understates the hot tail. Cold days still move demand more than hot days do: −2 °C costs
  2.5 × as much as +24 °C.
- **The daily peak behaves like the mean** on the cold side; on the hot side outpatient peaks rise
  faster (+17 % at ≥ 24 °C).
- **Hot days shift demand into the evening** (`temperature_response_hourly`, inpatient). On ≥ 24 °C
  days, inpatient demand at 19:00 is +23 %, at 11:00 – 12:00 roughly −2 %, and on 22 – 24 °C days the
  midday hours are −8 to −9 %. Days below −2 °C lift every hour by a similar 22 – 29 %. The midday dip remains at the
  39 inpatient sites with no solar in ERIC 2023/24, so it is daylight (lighting) or unreported
  generation, not reported PV. The evening shift lands on the 17:00 – 20:00 system peak.
- **Persistence** (`weather_persistence`): beyond its own temperature, the 2nd hot day adds +2.1 %
  [0.3, 4.1] and the 3rd and later +2.6 % [0.6, 4.7] at inpatient sites (buildings heat up), nothing at
  outpatient sites. Long cold spells add +2.5 – 3.0 % from the 3rd day.
- **Other weather**, on top of the bins: wind on cold days +0.4 % per m/s (infiltration and heat
  loss), solar irradiation −1.0 to −1.6 % per kWh/m² (daylight), dew point on warm days ≈ 0: humidity
  adds nothing beyond temperature.
- **Extreme heat, on the daily maximum** (`temperature_response_tmax`; bands on the daily maximum
  to 30 – 32 °C and ≥ 32 °C, reference 18 – 20 °C). Demand is flat up to a 24 °C maximum, then
  rises by a growing step per band: inpatient sites +1.1 % (24 – 26 °C), +2.8 % (26 – 28), +5.3 %
  (28 – 30), +7.6 % (30 – 32) and +14.2 % [9.9, 18.6] at 32 °C and above; outpatient sites +10.1 %
  and +17.8 %. The daily peak rises more at the hottest sites, +27 % at outpatient sites on ≥ 32 °C
  days. The two hottest bands rest on 132 and 16 inpatient site-days (11 and 4 dates, all in
  2023 – 2025), so the ≥ 32 °C step is an order of magnitude, not a precise number; the record
  heat of June 2026 lies beyond the electricity record.
- **CHP sites are the mirror image.** Flat on the cold side (the CHP covers the extra heating
  electricity while it runs for heat), +27 – 32 % on warm days (it turns down), and +10 – 13 % more on
  the 2nd and 3rd hot day. This is import, not demand; see section 2.

## 5. Quasi-experimental estimates

`sense-energy weather-causal` (`analysis/weather_causal.py`, tables in `code/reports/weather_causal/`;
pyfixest). Weather is assigned by nature: hospitals do not choose their weather, and within a site
and a calendar month, which days are hot or cold is as good as random. What is not random is how
much heat or cold each site is exposed to overall (southern and urban sites run warmer and are
more often acute hospitals); every design below absorbs that with site-level fixed effects and
they differ in which *time-varying* confounders they rule out. Two doses, the two arms of the
hockey stick: degrees of daily mean below 14 °C (cold) and above 20 °C (heat). Sites without CHP
unless stated; 95 % intervals two-way clustered by site and date.

### 5.1 Nested fixed effects (`causal_specs`)

| Specification | Cold, % per °C | within-SD (°C) | Heat, % per °C | within-SD (°C) |
|---|---|---|---|---|
| A: site × month, site × weekday, calendar | +1.38 [1.11, 1.66] | 2.22 | +2.42 [1.36, 3.50] | 0.32 |
| B: A + date effects | +1.09 [0.71, 1.47] | 0.55 | +1.50 [0.16, 2.85] | 0.19 |
| C: A + region × date effects | +1.08 [0.28, 1.89] | 0.31 | +1.21 [−0.27, 2.72] | 0.13 |
| D: B + sunshine, wind, rain | +1.14 [0.76, 1.51] | 0.55 | +1.80 [0.47, 3.16] | 0.19 |

Date effects remove everything common to all sites on a day (bank holidays, strikes, school
terms, the grid, the news): what remains is how much warmer or colder one site was than the
others on the same day, relative to its own month. That is a difference-in-differences with a
continuous dose. The within-SD column is the dose variation left to identify the effect: date
effects remove 94 % of it, because the panel is one weather system (section 5.5). The cold effect
is unchanged by them; the heat effect halves and loses precision, then recovers once the day's
sunshine is held fixed (hot days are sunny, and sunshine lowers lighting demand).

The binned curves of section 4 re-estimated under B and D (`causal_response_mean`,
`causal_response_tmax`) keep their shape and shrink by about a third at both ends: on the daily
mean, below −2 °C +28 % (A) → +18 % (B) → +19 % (D), at ≥ 24 °C +12.6 → +7.9 → +10.6 %; on the daily
maximum, ≥ 32 °C +16.5 → +9.3 → +12.7 %. A confounder would bend one end of the curve; a noisier
thermometer scales the whole of it, which is what the date effects do.

### 5.2 Stacked event-study difference-in-differences (`causal_event_study`)

For each national heatwave or cold spell, sites inside a site-level event against clean
controls with no such event within ±10 days; stack-specific site and date effects; reference
k ≤ −5; events with another onset in their pre-period, or fewer than five treated or control
sites, are dropped (`stack_log.csv`: the cold spells of January 2023 and January 2026 hit nearly
every site and have no clean controls).

| Event | Stacks | Treated / control sites | Temperature ATT | Demand ATT | % per °C (SE) | Pre-trend p |
|---|---|---|---|---|---|---|
| heatwave | 6 | 85 / 102 | +1.12 °C | +0.1 % [−2.6, +2.8] | +0.1 (1.2) | 0.16 |
| cold spell | 7 | 102 / 81 | −1.19 °C | +1.6 % [−2.2, +5.4] | +1.3 (1.6) per °C colder | 0.71 |

Pre-event coefficients are flat (no anticipation), the post-event ones return to zero (no
rebound: heatwave −2.1 % [−5.6, +1.3], cold −0.6 %). The design's weakness is its dose: during
a national event the clean controls are still hot or cold, only 1.1 – 1.2 °C less so, so the
expected demand gap (1.5 – 2 %) sits inside intervals 3 – 5 times wider than the fixed-effect
estimates'. The point estimates agree with section 5.1; the event study adds the parallel-trends
evidence, not precision. For CHP sites the heatwave ATT is −7.8 % [−24, +13]: no power.

### 5.3 Falsification (`causal_lags`, `causal_randomisation`)

- **Randomisation inference.** Reassigning whole daily weather series between sites at random,
  200 times, specification B gives a cold effect as large as the observed one 0 times in 200
  (permutation SD 0.22 % per °C against 1.09 observed) and a heat effect as large 6 times in 200
  (p = 0.03).
- **Leads.** Tomorrow's and the next day's temperature, given today's and the past week's, are
  jointly significant (p = 0.022), driven by tomorrow's heat (+0.45 % per °C [−0.15, +1.04]);
  the cold leads are nil and the two-day lead is zero. Controlling for the same evening's
  temperature does not remove it (p = 0.052; the lead stays +0.53), so it is not the daily mean
  blurring the evening. Two readings remain: anticipation (heat-health alerts go out one to two
  days ahead and estates teams run cooling early) or measurement error in a site's ERA5
  temperature, which lets adjacent days proxy for the true value. The lead is under half the
  same-day effect either way.
- **Lags.** The effect of a degree is largest the same day (cold +1.03, heat +1.11 % per °C),
  carries into the next day (+0.31, +0.60) and is nil after that; cumulative over a week, cold
  +1.41 [0.57, 2.26] and heat +1.82 [−0.33, 4.02] % per °C-day, with no systematic rebound.

### 5.4 Extreme days (`temperature_response_tmax`, `causal_extreme`)

Indicators for a daily maximum of 28 – 30 and 30+ °C on top of the linear heat dose (323 site-days
at 30+ °C). With date effects, a 30+ °C day carries −1.5 % [−4.3, +1.3] beyond what its mean
temperature implies (28 – 30 °C: −1.0 %); without them −3.8 % [−6.7, −0.9]. There is no extra kick
on the hottest days: demand keeps rising with temperature through 30 °C (section 4) at about
1.2 % per °C of daily maximum, and if anything falls short of a linear extrapolation in the daily
mean, which is what cooling plant running at its limit would look like. The daily peak rises
more than the mean on such days (+12 % inpatient, +27 % outpatient at ≥ 32 °C).

### 5.5 Space (`causal_spatial`)

- **One weather system.** Day-to-day temperature anomalies correlate 0.998 between sites within
  25 km, 0.98 at 50 – 100 km, 0.94 at 100 – 200 km and 0.89 at 200 – 400 km. Specification B's
  residuals correlate 0.01 or less at every distance: after date effects there are no regional
  demand shocks left to confound, and nothing for region × date effects to remove.
- **Neighbour placebo.** The mean dose of the sites 30 – 150 km away (median 99 neighbours), given
  the site's own: for cold it does nothing (own +1.03 [0.43, 1.63], neighbours +0.36 [−0.19,
  +0.92]); for heat the two are too collinear to split (own +0.73 [−0.70, 2.17], neighbours +2.13
  [0.81, 3.48]; with date effects +2.29 [−0.07, 4.70]) and only their sum, about 2.8 % per °C of a
  regionally hot day, is well determined. On the neighbourhood dose alone, heat is +2.85 [1.75,
  3.97] (A) and +3.03 [0.83, 5.29] (B) % per °C, cold +1.39 in both. Section 6 tests whether this
  is the thermometer: it is not. A regional channel is the remaining reading – heat-health alerts
  are issued by region, and a site in an alerted region may run its cooling harder whatever its
  own reading says – and the UKHSA alert record would test it.
- **Inference.** Specification B standard errors: site and date two-way 0.19 (cold) and 0.67 (heat)
  % per °C; region × week 0.08 and 0.32; ICB (20 clusters) 0.15 and 0.77. Clustering by region ×
  week ignores the same-day correlation across regions that the date dimension captures, so the
  two-way clustering is the conservative choice and ICB clusters agree with it.
- **Effect modification** (`heterogeneity.csv`): older buildings, larger sites and warmer local
  climates do not change the slopes detectably (interactions −0.3 to +1.3 % per °C, p ≥ 0.16).

**Reading.** A degree of cold below 14 °C adds 1.1 – 1.4 % to a site's daily electricity, a degree of
heat above 20 °C 1.5 – 2 % (2.8 – 3 % per degree of a regionally hot day). Both survive the removal
of every national and regional day shock, randomisation of weather across sites and pre-trend
tests, and section 6 shows the heat figure is not an artefact of the 25 km grid.

## 6. Which thermometer? ERA5, ERA5-Land and the Met Office stations

`sense-energy fetch-isd`, `build-isd` (`data/isd.py`: NOAA's Integrated Surface Database, the Met
Office synoptic and airfield observations, open data, 76 stations, hourly, to 24 August 2025),
ERA5-Land (`fetch-weather --config code/configs/weather_era5_land.yaml`: 0.1°, land points only)
and `sense-energy thermometers` (`analysis/thermometers.py`, tables in `code/reports/thermometers/`).
Each site takes the nearest reporting station among the three within 60 km (median 13.5 km).
Everything is re-estimated on the 106,425 site-days all three thermometers cover, so the
estimates differ only by the thermometer.

| | ERA5, 25 km grid point | ERA5-Land, 9 km land point | Met Office station, median 13.5 km |
|---|---|---|---|
| Agreement with the station: daily mean / maximum RMSE, bias | 0.62 / 0.94 °C, +0.07 | 0.72 / 1.26 °C, −0.12 | – |
| Anomaly correlation, anomaly RMSE | 0.989, 0.44 °C | 0.980, 0.56 °C | – |
| Cold, % per °C below 14 (B / D) | +1.07 / +1.11 | +1.02 / +1.06 | +1.08 / +1.09 |
| Heat, % per °C above 20 (B / D) | +1.68 / +1.97 [0.63, 3.31] | +1.97 / +2.31 [1.08, 3.56] | +1.41 / +1.69 [0.51, 2.88] |
| Heat on the neighbourhood dose alone (B) | +3.47 | +3.36 | +2.81 |
| 30+ °C day beyond the dose (B) | −1.6 [−4.7, +1.6] | +0.9 [−3.2, +5.2] | +0.3 [−3.0, +3.6] |
| Leads, joint p | 0.031 | < 0.001 | 0.114 |

(`thermometer_agreement`, `thermometer_effects`, `thermometer_response`; maps
`map_thermometer_network`, `map_thermometer_error_era5` / `_era5_land`, `map_hot_day_bias`,
`map_heatwave_exposure`.)

- **The error is coastal.** On the map, ERA5's worst sites against the stations (anomaly RMSE
  0.8 – 1.3 °C against a median of 0.44) line the Norfolk, Suffolk and south coasts and Torbay –
  where a 25 km cell straddles sea – and its hot-day cold bias follows the same coast (median
  −0.9 °C, to −4 °C at the coastal sites; 221 sites with 10 or more station days at 28 °C+).
  Inland, the Midlands and London, the grid point is as good as the station.
- **The heat is unevenly spread.** By the station thermometer the hottest day of summer 2025
  ranged from 27.9 °C on the Pennine and coastal sites to 33.6 °C in London, with a median of
  three days at 30 °C+ and up to nine: the cross-site spread that the date-effect designs
  identify from is a 5 – 6 °C band across England on the same days.


- **The grid was not hiding the heat effect.** A station 13 km from the site, independent of any
  reanalysis, agrees with ERA5 on the day-to-day anomalies that identify the effects to 0.44 °C
  and gives the same slopes: cold identical on all three thermometers, heat 1.7 – 2.3 % per °C
  with every estimate inside the others' intervals. The attenuation reading of section 5 (that
  the 25 km grid understated heat) is not supported; the date effects shrink the curve because
  they remove 94 % of the variation, not because the thermometer is bad.
- **Finer is not better.** ERA5-Land's 9 km land-only tiles agree with the stations *less* well
  than ERA5's 25 km cells (anomaly RMSE 0.56 against 0.44 °C, daily-maximum RMSE 1.26 against
  0.94 °C): at coastal and estuary sites the land tile loses the sea's moderation that the
  coarser cell and the station both see. Its leads test fails outright (p < 0.001), the mark of
  a thermometer whose error is proxied by adjacent days. ERA5 stays the primary thermometer;
  the stations are the independent check.
- **The regional average still predicts better than any local reading**, with all three
  thermometers, so that is not a measurement effect: it is a regional channel (heat-health
  alerts are issued by region) or collinearity, and the sum of the local and regional doses,
  about 2.8 % per °C, is what the data pin down.
- **The lead halves** with the station thermometer (joint p 0.11), so part of it was grid error
  proxied by adjacent days; what remains is small and compatible with anticipation.
- **Where ERA5 is weakest is the hottest days**: on the stations' 30+ °C days it reads 1.3 °C too
  cool. The 30+ °C indicator is within noise of zero on all three thermometers – still no extra
  demand beyond the temperature.

## 7. Who responds: heterogeneity of the slopes

`sense-energy weather-heterogeneity` (`analysis/weather_heterogeneity.py`, tables in
`code/reports/weather_heterogeneity/`). Sites without CHP, specification B (site × month,
site × weekday and date effects), so every national day shock is removed before any
between-site comparison.

### 7.1 Every site's slope (`site_slopes_hdd14`, `site_slopes_cdd20`, `slope_spread`, maps)

Dose × site interactions in one regression give 107 cold and 106 heat slopes with standard
errors; a random-effects decomposition (DerSimonian–Laird) separates the true between-site
spread (τ) from sampling noise, and empirical-Bayes shrinkage gives each site a posterior slope.

| | Cold, % per °C below 14 | Heat, % per °C above 20 |
|---|---|---|
| Pooled slope | +1.14 | +0.96 |
| True between-site SD (τ) | 1.30 | 3.91 |
| Share of observed variance that is real (I²) | 0.99 | 0.99 |
| Posterior, 10th / 50th / 90th percentile | −0.2 / +0.7 / +3.1 | −3.9 / +0.5 / +7.0 |
| Sites significantly above / below zero | 82 / 8 | 48 / 39 |

- **The average hides the sites.** Both slopes vary far more between sites than their standard
  errors can explain (I² = 0.99). The cold slope runs from about zero to +3 % per °C; the heat
  slope from −4 to +7, with 39 sites that significantly *fall* on hot days against 48 that rise.
  "The heat effect" is an average over sites that differ in sign. The cold and heat slopes of a
  site are only weakly related (correlation 0.21).
- Three site-dose slopes (RCBL8 heat, RWRRK heat, RY318 both) have undefined clustered
  variances and are left out of the shrinkage.

### 7.2 Moderators (`moderators`)

Each moderator enters as dose × moderator interactions on top of the main doses, one at a time
(categorical levels against a reference, continuous per SD) and jointly on the 87 sites with
every descriptor.

| Moderator | Cold slope | Heat slope | Joint p |
|---|---|---|---|
| Building use (vs acute inpatient) | community inpatient +0.8, mental health +0.9, non-inpatient +1.2, support +0.5 points | all within ±1.2, none significant | < 0.001 |
| Gas / electricity ratio (per SD) | −0.51 [−0.81, −0.21] | −1.0 [−2.2, +0.2] | 0.001 |
| Trust type | community +0.6 (p 0.09) | – | 0.041 |
| Demand-shape cluster | – | – | 0.61 |
| Site size (log kW) | – | +0.55 (p 0.14) | 0.26 |
| Mean build year, summer climate, LED share | – | – | 0.27 – 0.44 |

In the joint model (p < 0.001) building use and the gas ratio survive together: acute inpatient
hospitals respond to cold with +0.05 % per °C, every other use with +1.2 to +1.4 points more, and
a one-SD higher gas-to-electricity ratio removes 0.45 points.

- **Cold has structure, and it is the heating fuel.** The more a site heats with gas, the less
  its electricity responds to cold; the acute hospitals, which have the largest gas plant, barely
  respond at all. Cold raises electricity where heat is electric – non-inpatient buildings,
  mental-health units, community hospitals. That is the physically sensible answer, and the
  practical one for electrification: a site that moves from gas to heat pumps will acquire a
  cold slope of the order of +1 to +1.5 % per °C.
- **Heat has no explained structure.** Not one of the eight moderators predicts the heat slope,
  jointly or alone. The large real dispersion (τ = 3.9 points) is driven by something ERIC does
  not record – most plausibly whether a site has cooling plant at all, and how it is controlled.
  That is the strongest argument yet for asking trusts for cooling and CHP plant data.
- Trust type matters (p = 0.04) only as a proxy for building use.

### 7.3 Hour of day by building use (`hourly_by_use_cold`, `hourly_by_use_hot`)

Hourly curves for days below 2 °C and at or above 22 °C daily mean against a 14–16 °C day, by
building use, with site × month and site × weekday effects (date effects would leave the hot
band resting on a handful of site-days in the small groups; groups with fewer than 150 site-days
in a band are left out).

- **Cold days** lift non-inpatient (+22 to +29 %), mental-health (+19 to +25 %) and community
  inpatient (+15 to +29 %) sites through the whole day, strongest in the morning and afternoon;
  the five acute hospitals rise only 6 to 8 %, flat across the day – gas carries their heating.
- **Hot days** at mental-health and non-inpatient sites (the two groups with enough hot
  site-days) are flat until midday and then climb to +12 to +15 % at 17:00–19:00 – the evening
  shift of section 4, now visible group by group, and landing on the system peak.

## 8. What moves heat

`sense-energy heat-drivers` (`analysis/heat_drivers.py`, tables in `code/reports/heat_drivers/`,
figures `heat_*`, `sunshine_slopes`, `map_heat_types`). Section 7 left the heat slope unexplained:
a real spread of ±4 points around +1 % per °C, 39 sites falling on hot days, and none of the
estates descriptors predicting any of it. This section takes the slope apart on the same panel
(sites without CHP, specification B, standard errors clustered by site and date), with three
additions: a site-specific **sunshine** term, the demand of each **six-hour block** of the day
and the **weekend**, and the heat dose split into **hot afternoons and warm nights**. Two new
open datasets feed the moderators: the ONS rural-urban classification and Census 2021 population
density of every site postcode (`data/site_context.py`, via postcodes.io and Nomis), and the DESNZ
Renewable Energy Planning Database for solar schemes at the site postcode. The estates return
contributes two fields not used before: overheating occurrences in clinical areas (S16_07) and
solar consumed on site.

**Sample.** 24 of the 107 sites carry the `meter_gaps` coverage flag (a meter set that comes and
goes within the year); they are left out here, because meters that drop in and out *within a
month* put swings into exactly the variation the site slopes are estimated from, and they
turned out to own the tails of section 7: of the 24, Bridlington's heat slope was −15 % per °C
and two sites' +12 %, with monthly demand swinging threefold. 83 sites remain
(`excluded_sites.csv` lists the rest with their section-7 slopes).

### 8.1 Sunshine (`heat_sunshine_control`, `sunshine_slopes`, `heat_blocks_sunshine`)

Hot days are sunny days, and 55 of the 83 sites import less electricity on sunny days at a given
temperature: the pooled sunshine slope is −0.9 % per kWh/m² of daily surface solar radiation,
which is −5 % between an overcast and a clear summer day. The slope has the shape of daylight:
−1.1 % in the morning block, −1.4 % in the afternoon, −0.5 % in the evening and +0.2 % at night
(`heat_blocks_sunshine`). Sites that report solar generation to ERIC have a sunshine slope 1.9
points more negative than the rest (p = 0.06), community inpatient sites 3.9 points more
negative (p = 0.005). It is daylight-linked lighting across the estate, plus panels where there
are panels.

It is not, however, where the heat dispersion comes from. Adding the sunshine term moves the
pooled heat slope from +0.6 to +1.1 % per °C (the daylight saving was masking some cooling) and
leaves the between-site SD at 4.0–4.1 points; the two heat slopes correlate 0.90, and 22 sites
still fall significantly with heat net of sunshine (28 before). The heat and sunshine slopes of a
site are unrelated (r = 0.05). REPD confirms there is little rooftop PV behind these meters yet:
24 sites have a scheme at their postcode, all 2025 applications awaiting construction, none
operational within the sample.

### 8.2 Time of day and the week (`heat_blocks_heat`, `heat_signature`, `heat_weekend`)

The heat response is an afternoon-and-evening phenomenon: pooled across sites it is +0.2 % per °C
at night, +0.8 % in the morning, +3.6 % in the afternoon and +2.1 % in the evening. Among the 41
sites that rise with heat, 32 have an afternoon block slope significantly above their night
block: cooling plant that works with the sun. The 22 fallers fall through the whole day, night
included (their block slopes run −1.3 to −3.7 %), which no sunshine mechanism can produce.

Weekday and weekend heat slopes correlate 0.66; 13 risers have a significantly smaller weekend
response (plant that runs with the working day). The median site's weekend response is 0.9
points *larger* than its weekday one, which is a percentage effect: the same cooling kilowatts
against a lower weekend baseline.

### 8.3 Hot afternoons and warm nights (`heat_night_day`, `heat_group_decomposition`)

Splitting the dose into degrees of daily maximum above 24 °C and degrees of daily minimum above
12 °C separates the two populations (group × dose interactions, one regression):

| Sites by the daily heat slope | Hot afternoon (% per °C of T_max > 24) | Warm night (% per °C of T_min > 12) | Sunshine (% per kWh/m²) |
|---|---|---|---|
| rise with heat (41, 74 % of demand) | +1.45 (0.38) | +0.73 (0.26) | −1.27 (0.41) |
| fall with heat (22, 8 % of demand) | −1.24 (0.35) | −1.18 (0.26) | −1.05 (0.25) |
| neither (20) | +0.51 (0.70) | +0.36 (0.25) | −0.77 (0.48) |

{{table: code/reports/heat_drivers/group_decomposition.csv | cols=model,sign,term_name,coef_pct,lo_pct,hi_pct,p,sites | round=2}}

Risers respond twice as much to hot afternoons as to warm nights: cooling. Fallers respond to
both, and as much to the night minimum as to the afternoon maximum: 17 of the 22 have a
significantly negative warm-night slope (`heating_tail` flag). That is the signature of
heating that is still running in the warm season and switches off when the night is warm:
electric heating and the auxiliaries of gas heating in small buildings (the fallers' median
demand is 9 kW against 21 kW for risers), plus water heating that follows mains-water
temperature. The extreme cases say the same: Kelling Hospital runs 50 kW in winter and 19 kW in
summer, Malton 43 kW in April and 30 kW in July, and South Devon's renal unit (dialysis water
heated to 37 °C) falls 16 % per °C. "Falls with heat" is a heating-season tail, not a cooling
behaviour, and it carries 8 % of the demand in the sample.

### 8.4 Moderators and how much they explain (`heat_moderators_categorical`, `heat_moderators_continuous`, `heat_variance_explained`)

With cold, heat and sunshine each interacted with the moderator (sunshine so that a daylight
effect cannot masquerade as a heat moderator), one at a time:

- **Nothing in location.** Urban form (major conurbation / city and town / rural; joint p =
  0.92) and the LSOA population density (p = 0.38) do not move the heat slope: the urban heat
  island does not show up in the response, consistent with the thermometer result of section 6.
- **Nothing in use, age or heating fuel.** Building use (p = 0.35), build year, gas-to-electricity
  ratio, LED share, heat pumps, winter/summer ratio, size: none.
- **What moves it is how electrically intensive the estate is.** Peak demand per m² (+0.7 points
  per SD, p = 0.05 alone; the joint model puts it on electricity intensity instead: +1.9 points
  per SD, p = 0.001), tenure (joint p = 0.02: leased sites +0.9, PFI/LIFT/other −0.9) and solar
  generation in ERIC (+2.0 points, p = 0.01), which reads as a marker of trusts that invest in
  their plant. Overheating reports in clinical areas go with a larger heat slope (+0.8, p = 0.13):
  sites that overheat respond, presumably with portable and retrofitted cooling.
{{table: code/reports/heat_drivers/variance_explained.csv | cols=set,sites,parameters,tau_before_pct,tau_after_pct,r2_true | round=2}}

- **How much is explained.** A random-effects meta-regression of the sunshine-controlled site
  slopes on each descriptor set gives the share of the *true* between-site variance it accounts
  for: building use, urban form and density, the estates descriptors, the load shape, and all of
  them together each explain nothing (−8 % to +6 %). The site's own diurnal signature (afternoon
  excess and sunshine slope) explains 46 %. The heat slope is about plant on the site, which shows
  in the shape of the day and not in anything the estates return records.

### 8.5 Types (`heat_signature`, `map_heat_types`)

By significance at 5 %: 16 sites rise with heat and nothing else (45 % of demand), 25 rise with
heat and fall with sunshine (29 %), 12 fall with sunshine only (13 %), 22 fall with heat (8 %),
8 neither (5 %). Three quarters of the metered demand sits on sites whose electricity rises on hot
afternoons. The map shows the types mixed within regions and ICBs: the type is a property of the
building, not of the place.

### 8.6 What this changes

- The cooling response of the estate is cleaner and larger than the raw heat slope says: +1.1 %
  per °C of daily mean above 20 °C net of daylight, +1.45 % per °C of daily maximum above 24 °C
  at the sites that have cooling, and concentrated in the afternoon and evening.
- "Sites that fall on hot days" are small, heating-dominated buildings in their shoulder season,
  with 8 % of the demand; they should not be modelled as cooling anomalies.
- Sunshine deserves a place in the day-ahead feature set in its own right: −0.9 % per kWh/m² is
  a daily swing of several percent, separate from temperature.
- The direct data on cooling plant is the Display Energy Certificate register; section 8.7 reads
  it. It confirms the direction (the sites whose buildings are classed as air-conditioned have the
  largest heat slopes) but records presence rather than size, so the diurnal signature stays the
  better measure of plant.

### 8.7 What the certificates say (`heat_dec_aircon`, `heat_dec_heating`, `dec_recommendations`)

With the MHCLG energy-performance API (`data/dec.py`, `sense-energy build-dec`; bearer token in
`.env`) every Display Energy Certificate at the site postcodes was pulled: 3,266 certificates since
2008, 285 buildings with a certificate from 2019 on, at 187 of the 275 postcodes; 67 of the 83 sites
of this section have one, and the certified floor area matches the ERIC gross internal area at the
median (ratio 1.08; a shared postcode can bring in neighbours, `dec_coverage` flags it). Each
certificate names the building environment, whether air conditioning is present, the main heating
fuel, the metered electricity and fuel of the assessment year and the operational rating; the
newer ones link an advisory report with the assessor's recommendations.

- **"Air conditioning present" is almost universal and separates nothing.** The question covers
  any system, server rooms included: 51 of the 67 sites are "mostly air-conditioned" by floor area
  and 16 have none. The heat slope of the two groups differs by 1.0 point (p = 0.46; medians 1.4
  against 0.4 % per °C), and the air-conditioned floor share explains 1.5 % of the true between-site
  variance.
- **The building environment does separate, at the top.** The dominant class is heating with
  natural ventilation at 38 sites, mechanical ventilation at 13, mixed mode at 13 and air
  conditioning at 3. Those three sites (34 kW median) have a heat slope 3.1 points above the
  naturally ventilated reference (p < 0.001; median 4.7 % per °C against 1.0); mixed mode sits at
  2.1 and mechanical ventilation at 0.3. In the joint model the cooled floor share (air conditioning
  or mixed mode) adds 0.9 points per SD (p = 0.02, 38 sites). The register classes most hospital
  buildings as heated and naturally ventilated while their demand rises on hot afternoons: the
  plant that does it is below the threshold of "building environment", and its size (the AC kW
  rating) is recorded on 19 % of recent certificates only.
- **The fallers are gas-heated on paper.** Electricity is the main heating fuel of some floor area
  at 4 of 67 sites; the warm-night slope correlates −0.28 with that share, in the expected
  direction, but the fallers' certificates name gas. Their warm-night fall is the auxiliaries and
  hot water of gas heating and electric spot heating, not electric space heating as the main
  system.
- **The non-domestic EPCs do not pool with the DECs.** The same API gives 1,581 EPCs at the site
  postcodes (`sense-energy build-epc`, `epc_sites.parquet`), which would cover 8 of the 16 sites
  without a DEC. Their building-environment field is a design classification from the SBEM
  assessment: 34 % of EPCs at these postcodes are "Air Conditioning" against 1 % of DECs, and
  filling the eight sites from EPCs flips the environment result (joint p from < 0.001 to 0.86).
  The EPC fields are kept as a separate table, not as moderators.
- **The recommendations are the "opportunities".** 115 advisory reports linked to the current
  certificates carry 1,552 recommendations: fabric 335, controls 322, heating 284, renewables 214,
  cooling 170 (163 of them short payback: review the air-conditioning strategy, stop heating and
  cooling running together, weather-compensated controls), ventilation 159, lighting 104, hot water
  78 (`dec_recommendations`). They are stored per site as counts and building shares by theme
  (`rec_*` in `dec_sites.parquet`) and are available as features or clustering inputs; 27 of the 83
  sites here have a report, too few for a moderator yet.

### 8.8 Causal machine learning on the heat response (`cate_vs_slope_observables`, `cate_vs_slope_signature`, `cate_importance`)

`sense-energy heat-causal-ml` (`analysis/heat_causal_ml.py`, tables in `code/reports/heat_causal_ml/`).
The same panel (83 sites, 99,181 site-days) is put to two orthogonal learners. Log demand,
the heat dose and the controls (cold dose, sunshine) are demeaned on specification B's fixed
effects by alternating projections and the controls partialled out, so the learners see only the
within-site, within-month, within-day variation the regressions used. An R-learner (LightGBM on
the residual ratio, weighted by the squared residual dose) and an econml causal forest (ridge
nuisance models, honest trees) are cross-fitted over five site folds, so every site's effect comes
from a model that never saw it. Covariates: the estates, context and DEC descriptors of section 8.4
plus three calendar modifiers (weekend, bank holiday, school summer); a second run adds the site's
diurnal signature. A placebo shuffles the site covariates across sites.

- **The forest recovers a quarter of the heterogeneity from observables; the linear moderators
  recovered none.** Out of fold, the forest's site effects correlate 0.21 with the
  fixed-effect slopes and recover 24 % of their true
  between-site variance (calibration slope 0.19: the learned effects are
  spread 5.7 points against a true 4.2, so they are
  over-dispersed and should be shrunk before use). The R-learner with observables alone finds
  nothing (-0.11); with the signature it reaches 0.21 and
  18 %. The placebo sits at -0.06 and
  -0.01.
- **What the forest splits on** is the load shape and the climate, not the estates fields: the
  night/day ratio (0.26 of importance), the winter/summer
  ratio, size, the gas ratio and the summer maximum; with the signature in, the daytime sunshine
  slope takes 0.44 and the afternoon excess 0.15.
  The DEC fields do not enter the top splits.
- **The calendar does not modify the heat response.** With site x weekday effects already in, the
  forest's effect is 1.70 % per °C on weekdays and 1.82 at weekends,
  and identical on bank holidays and in the school summer holiday. The weekend difference of
  section 8.2 is a percentage effect on a lower baseline, not a different response.
- **Who the forest gets right.** 35 sites have a learned effect significantly above zero
  and 15 below; the large risers are recovered, the fallers are not (the forest
  places most of them near zero): their heating tail has no observable to split on.

The mean learned effect is 1.73 % per °C, an unweighted site-day average against
the regression's pooled +1.1; the difference is the weighting. Next in this thread: shrink the
forest effects (empirical Bayes against the fixed-effect slopes), and use the dated ERIC changes
(section 7 of `estates_analysis.md`) as staggered treatments for installation effects.

## 9. Does any of this reach the day-ahead forecasts? (`forecast_heat_days`)

`analysis/forecast_heat_days.py` scores every model's test-window forecasts (June 2025 to May 2026)
by the kind of target day: ERA5 daily maximum at or above 28 °C (214 site-days), 25–28 °C
(235), other June–September days (1816), daily mean at or below 2 °C (73) and
the rest (3917); nMAE per site as in the main experiment, then the site mean.

| Model | 28 °C and above | 25–28 °C | other summer | at or below 2 °C | rest of year |
|---|---|---|---|---|---|
| LightGBM + IFS | 12.8 | 12.3 | 12.2 | 16.0 | 13.4 |
| LightGBM + ERA5 (perfect weather) | 12.8 | 12.4 | 12.0 | 15.5 | 13.1 |
| LightGBM, no weather | 14.3 | 13.1 | 12.4 | 16.2 | 13.7 |
| Chronos-2 + IFS | 11.5 | 13.8 | 11.0 | 14.2 | 11.6 |
| TimesFM-3 + IFS | 11.6 | 13.4 | 11.2 | 15.9 | 11.8 |
| TabPFN-TS + IFS | 12.2 | 13.6 | 11.2 | 19.9 | 12.3 |

{{chart: code/reports/forecast_heat_days/scores_by_day_class.csv | kind=bar | x=nmae | y=model | color=day_class | title=Day-ahead nMAE by kind of target day (site mean) | height=420}}

- **Hot days are harder, by about half a point.** Every model errs more on 28 °C+ days than on
  other summer days; the cold days (at or below 2 °C) are harder still, by three to four points.
- **It is not the weather forecast.** LightGBM with realised ERA5 weather does no better on the
  hottest days than with the IFS forecast (12.8 against 12.8), and without
  weather at all it loses 1.4 points. The gap is in how the models use heat, which
  is why the heat work prescribes the features: daily maximum and evening temperature rather than
  the hourly series, the station correction of the hottest days, the day's solar radiation, and
  the calendar (school summer holiday, Christmas, distance to a bank holiday) which the baseline set
  carried only as weekday and bank-holiday flags.

**The heat features do not move the hot days (`lightgbm_ifs_heat`).** A LightGBM variant with
the prescribed features (the IFS day's forecast maximum, minimum and evening temperature, the
station-corrected maximum and hot-day flags at 25 and 28 °C, degree terms, the day's solar
radiation, and the extended calendar: school summer holiday, Christmas period, next-day holiday,
days to and since a bank holiday) scores 12.83 % nMAE on 28 °C+ days against
12.85 for the baseline IFS variant, 12.61 against 12.32 on 25–28 °C days,
15.49 against 16.02 on the coldest days and 13.06 against 13.06
over the whole test window. The cold days gain half a point (the calendar), the hot days nothing.
Site identity, the recent level and the lag features carry the model; the hot-day penalty is in
what no day-ahead feature carries, the day's plant switching, and the foundation models' lead on
hot days (Chronos-2 11.5) comes from their shape modelling, not from
weather. The honest reading: the response work explains demand on hot days after the fact better
than it helps predict it the day before.

### 8.9 A learned site embedding (`jepa_probe`, `jepa_map`, `jepa_surprise`)

`sense-energy train-jepa` and `sense-energy jepa-probe` (`experiments/jepa.py`,
`analysis/jepa_probe.py`, outputs in `code/reports/jepa/`). A joint-embedding predictive
architecture was trained on the half-hourly panel: 28-day windows of one site's demand, one
token per day (48 half-hours scaled by the site's trailing month, weekday, season), seven target
days per window whose embeddings under a slowly moving target encoder the predictor must produce
from the other 21; 0.48 million parameters, 4,000 steps on 131 sites, 131,656 days
embedded. The test was pre-registered: does the site's embedding predict its fixed-effect heat
slope out of fold better than the diurnal signature (46 %)? Alongside it the same probe ran on
the Chronos-2 encoder's embedding of each site's demand (the foundation-model teacher), on the
observable descriptors and on the signature, with a ridge probe and with TabPFN.

{{table: code/reports/jepa/probe.csv | cols=target_name,representation,model,sites,dims,r2_oof,share_true_variance | round=2}}

- **No, on the heat slope.** The JEPA embedding recovers 1 % of the true
  between-site heat variance (9 % from June–September days only), the
  Chronos-2 embedding -0 %, the observables 7 %, the
  diurnal signature 36 % (ridge) and 45 % (TabPFN).
  Adding the embedding to the signature adds nothing. By the criterion set in advance, the thread
  stops here.
- **Yes, on the responses that show in the day's shape.** The same embeddings carry the sunshine
  slope well (Chronos-2 63 %, JEPA 52 %) and
  Chronos-2 carries the cold slope (42 %) where the observables manage
  18 %. Daylight-linked lighting and electric heating leave a mark on the profile
  that a weather-blind representation picks up; an afternoon cooling response does not, because
  it is conditional on a driver the model never sees. (The signature's 90 % on sunshine is
  circular: the daytime sunshine slope is one of its two components.)
- **The embeddings encode operation, not category.** Building use is classified at
  66 % by the JEPA embedding and 53 % by Chronos-2 against a
  62 % majority class; the DEC environment at 51 % against
  59 %. k-means on the JEPA embedding agrees moderately with the demand-shape
  clusters (adjusted Rand 0.28) and, like them, explains none of the heat slope.
- **The surprise score is the usable by-product.** The latent prediction error of a day held out
  of its window spikes at the CHP hospitals exactly where the changepoint search put steps and
  stays flat at small sites (`jepa_surprise`): a learned regime detector that needs no penalty
  tuning, for the meter-gap and plant-switching flags.

What a second version should do is condition on the driver: give the day tokens and the
predictor the day's weather (temperature, sunshine), so that the embedding must encode how the
site responds rather than what its profile looks like; or predict demand-day embeddings from
weather-day embeddings in a two-tower setting. The LaT-PFN route (a prior-fitted network trained
to forecast in a learned latent space, pre-trained on synthetic series and this panel) is the
larger version of the same idea and the natural next experiment after that.
