# The weather-normalised estate: savings, warming, heatwaves, solar and CHP dispatch

Six questions answered with one engine, 9 November 2026. The engine is the site-level response
to weather by hour of the day (`analysis/hourly_response.py`): for every local hour, the hour's
demand regressed on site × (cold dose, heat dose, sunshine) interactions with site × year-month,
site × weekday and date effects, on the 83 sites without CHP or meter gaps, shrunk across sites
hour by hour. A site's demand in hour h under another temperature is the observed demand times
exp(β_heat[s, h] × change in the heat dose + β_cold[s, h] × change in the cold dose). Commands:
`hourly-response`, `installation-savings`, `estate-counterfactuals`, `solar-watch`, `chp-dispatch`.

**Correction (9 November 2026).** The first version of this page used standard errors clustered
by site and date for the site-specific coefficients. For a coefficient that belongs to one site
those understate the uncertainty two- to threefold; the numbers here use Driscoll–Kraay standard
errors (`weather_causal.site_vcov`). Point estimates barely moved; the shrinkage weights and every
count of "significant" sites did. What changed: the solar watch now finds 8 downward and 5 upward
breaks instead of 19 and 23, and the LED reading of the upward breaks does not survive; 10 CHP
plants test heat-led instead of 13; one site, not two, passes its capacity at +4 °C. The installation
savings use a different model and are unaffected.

**Correction (11 November 2026).** The market index price table carries two providers, and one of
them (N2EX) reports zero in 99.8 % of half-hours. The heatwave costing joined both, which counted
every half-hour of the sites without CHP twice (energy and carbon doubled, cost unaffected) and
halved the daily mean price used for the CHP sites (their cost halved). The CHP price test used
the same halved price, so its coefficients per £100/MWh were twice too large. All price work now
reads the traded provider only (`data.prices.load_mid`). Percentages and the heat-led and
red-band classifications did not move; one plant (Royal Devon and Exeter) no longer tests as
price-following.

## 1. Where in the day the weather response lives (`hourly_pooled`, `hourly_heat_profiles`, `hot_day_peak`)

The heat response is an afternoon and evening phenomenon at almost every site. Pooled, a degree
of daily mean above 20 °C adds 0.4 % at night (00:00–06:00) and 2.5 % on average between 12:00
and 19:00, peaking at 3.3 % in the 16:00 hour; the cold response is flat through the day at about
0.8 % per degree (0.5–1.2 %); sunshine lowers demand by 1.2 % per kWh/m² between 09:00 and 16:00
and not at night.

{{chart: code/reports/hourly_response/pooled.csv | kind=line | x=hour | y=pooled | color=term | title=Pooled response by local hour (log points per °C or per kWh/m²) | xlabel=local hour | ylabel=pooled response | height=300}}

- **The red band carries a fifth of each site's heat response.** At the median site, 22 % of the
  positive heat response falls in 16:00–19:00 (interquartile 16–28 %), 1.8 times the band's share
  of the day: warm afternoons raise demand when distribution charges are highest.
- **Hot days move the peak later.** On summer weekdays at or above 25 °C the site's demand peak
  falls later than on other summer weekdays at 37 of 81 sites and earlier at 10; the hot-day peak
  is 3 % higher at the median site.
- **The site profiles differ.** 61 % of sites peak between 12:00 and 19:00, the rest are spread
  through the night and morning, where small responses dominate the noise.

{{table: code/reports/hourly_response/site_profiles.csv | cols=site_code,heat_peak_hour,heat_redband_share,heat_daily_mean_pct,heat_evening_excess_pct,cold_daily_mean_pct,peak_hour_hot,peak_hour_other,peak_ratio_hot | sort=-heat_daily_mean_pct | round=2 | head=30}}

**The JEPA counterfactual machine, checked (`jepa_cf_check`).** The weather-conditioned JEPA
predicts each summer day's embedding with the day's temperature raised by 5 °C; a decoder turns
both predictions into half-hours. Against the regression on the same 6,601 site-days it gets the
size roughly right (median +5.4 % against +4.3 %), ranks sites weakly (r = 0.35 across sites) and
has no hourly shape (median correlation of the two hour profiles 0.00). As a counterfactual machine it
knows that a warm day costs about five per cent; it does not know when, or where.

## 2. Weather-normalised savings of the reported installations (`installation_savings`)

ERIC 2024/25 reports changes against 2023/24 at 45 of the 109 panel sites in both returns. Each
site's electricity is put on a common weather footing, the measurement-and-verification way
(normalised annual consumption): a daily model per site and financial year on the cold and heat
doses, solar radiation, weekday, bank holidays, Christmas and the school summer holiday
(median R² 0.75–0.79, CV(RMSE) 11 %), predicting the same reference days for every year. Before
is 2023/24, after 2025/26, 2024/25 the installation year; sites with CHP left out.

{{table: code/reports/installation_savings/group_effects.csv | cols=group,sites,median_nac_change_pct,median_raw_change_pct,did_median_pct,did_lo,did_hi | round=1}}

- **Sites that report heat pumps use a third more electricity.** The three sites that installed
  heat pumps use 32 % more electricity than the sites with no change (bootstrap 95 % interval +18
  to +51 %). It is added electric load, not gas being replaced: in the returns one of the three
  has no gas, one kept its gas and one changed its reporting scope (corrected 11 November 2026;
  see section 7 of `flexibility_and_design_winters.md`).
- **LED saves 0.6 % per ten points of coverage** (cross-site regression, 59 sites, interval −1.2
  to 0.0, p = 0.05): a site going from no LED to full LED saves about 6 %. The 16 LED-only sites
  fall 3.9 points more than unchanged sites at the median (interval −11 to +4).
- **Solar: −8 points, too few sites to say more** (three sites, interval −35 to +10).
- **Weather normalisation moves these by under half a point,** because 2023/24 and 2025/26 had
  similar weather. The framework matters more in a year like 2022.

{{table: code/reports/installation_savings/led_dose.csv | round=3}}

## 3. A warmer climate against the connections (`warming_capacity`, `warming_energy`)

Every day of the record shifted by +1, +2 and +4 °C (uniform shifts as sensitivity levels; the
hottest days warm more than the mean in the UKCP18 projections, so the peaks here are a lower
bound), through each site's hourly heat and cold response.

{{table: code/reports/estate_counterfactuals/warming_summary.csv | round=3}}

- **Annual electricity falls slightly.** At the median site −0.4 %, −0.8 % and −1.2 % at +1, +2
  and +4 °C: less heating outweighs more cooling; a quarter of sites rise, the top tenth by 0.8 %
  or more at +4 °C (at most 2.9 %).
- **Summer peaks barely move, with one exception.** Median utilisation of available capacity
  stays at 0.33. One site (RWR63) passes 90 % of capacity at +2 °C and exceeds it at +4 °C, for 21
  half-hours a summer; a second (RREA9) reaches 94 % at +4 °C for 7 half-hours.
- **Electrification binds first, everywhere.** No site's summer peak at +4 °C reaches its winter
  peak with heat pumps replacing gas (headroom analysis). The connection question for the NHS
  estate is winter heat, not summer cooling.

## 4. What the 2025 heatwaves added (`heatwave_attribution`)

For every site-day inside a 2025 heatwave (Met Office definition), the counterfactual is the
site's mean temperature over the surrounding 29 days without heatwave days; the difference
between observed and counterfactual demand, half-hour by half-hour, is attributed to the heat,
priced at the market index price and costed at the grid's carbon intensity.

{{table: code/reports/estate_counterfactuals/heatwave_events.csv | cols=event_id,start,end,group,sites,site_days,kwh_attr,share_pct,cost_gbp,co2_t | round=1}}

- **Sites without CHP: 4.4 % more electricity on heatwave days.** 23 MWh over 370 site-days at 56
  sites, at 4.3 °C above normal; £2,100 at wholesale prices and 3 t CO₂. The median site-day adds
  2.1 %.
- **Sites with CHP: 12 % more grid import.** 180 MWh over 127 site-days at 17 sites, £15,800 and
  25 t CO₂: the heat-led plants turn down when there is no heat demand, so the grid carries the
  hot days. Per site-day the CHP sites cost 22 times what the others do (£125 against £5.60).
- **The days are unusual in shape, too.** The JEPA's latent surprise is 8 % higher on heatwave days
  than on other summer days, at sites with and without CHP alike (p < 0.001 and 0.02).

{{table: code/reports/estate_counterfactuals/heatwave_surprise.csv | round=4}}

## 5. Watching solar go on roofs (`solar_watch`)

A site that puts panels behind its meter imports less on sunny days; its sunshine slope should
break downwards. Daytime demand of April–September days regressed on site × summer × solar
radiation (summers 2023, 2024, 2025) with the usual effects; a break is a fall of more than one
point per kWh/m² beyond 2.5 standard errors.

- **Eight of 107 sites break downwards, and three of them are three of the four sites whose
  returns report new or grown solar.** The breaks date the installations: Church Hill House and
  Potters Bar Community Hospital between the 2023 and 2024 summers (−2 to −23 and −1 to −21
  points), St George's Hospital progressively (−4.5, −7.8, −9.4). Bridlington has no 2023 summer
  in the record and already sits at −25 in 2024.
- **Candidates without reported solar:** Queen Victoria Memorial Hospital (−2 to −18 from 2023 to
  2024) and St Paul's Hemel Hempstead (−2 to −19 in 2025) break as sharply as the reported
  installations. 99 Waverley Road and Hampden House fall to −8 in 2025, the Hospital of St Cross
  to −3. The estates return counts only trust-owned solar; panels installed by a landlord or a
  third party appear here and not there.
- **LED leaves at most a faint opposite trace.** Five sites break upwards, only one of them after
  an LED programme. Sites that went to full LED moved their sunshine slope 0.8 points towards zero
  at the median, the rest 0.1 (regression on LED coverage change, p = 0.1). Before the correction
  of the standard errors this looked like 23 upward breaks driven by LED; it was noise.

{{table: code/reports/solar_watch/breaks.csv | cols=site_code,site_name,sun_2023_pct,sun_2024_pct,sun_2025_pct,change_pct,z,eric_solar_new_or_grown,eric_led_change_pts,break_down,break_up | sort=change_pct | round=2 | head=25}}

## 6. How the CHP plants are dispatched (`chp_dispatch_cold`, `chp_dispatch_price`, `chp_dispatch_red`)

Hospital demand does not follow the power price or the time-of-use band, so what a CHP site's
import does beyond the sites without CHP is the plant. Three tests, each against that placebo:
daily import on cold days (heat-led plants run more, so import falls), hourly import against the
power price within the day (price-following plants run when power is dear), and import in the
working-day red band against weekends (plants run to avoid red-band charges).

- **Pooled, nothing:** the CHP sites' price and red-band coefficients equal the placebo's (+6.4
  against +7.0 % per £100/MWh; +14.2 against +14.2 %). The pooled null hides opposite behaviours.
- **Per site: 10 of 23 heat-led, 6 price-following, 15 red-band avoiding.** The red-band avoiders
  cut import by 3 to 18 % in 16:00–19:00 on working days; four sites (Royal Preston, George Eliot,
  North Devon, Scarborough) import 24 to 41 % *more*, their plant apparently stopping in the
  evening. University Hospital Coventry does all three at a capacity factor of 0.91.
- **For forecasting**, the red-band and price-following plants are predictable from the calendar
  and the day-ahead price, which the forecasting models carry; the heat-led ones from the cold
  dose. The CHP sites' error is the rest: unplanned trips and maintenance.

{{table: code/reports/chp_dispatch/classification.csv | cols=site_code,site_name,cold_vs_no_chp_pct,price_hourly_vs_no_chp_pct,red_band_vs_no_chp_pct,heat_led,price_following,red_band_avoiding,capacity_factor | sort=red_band_vs_no_chp_pct | round=2}}

## Figures

`hourly_pooled`, `hourly_heat_profiles`, `hot_day_peak`, `jepa_cf_check`, `installation_savings`,
`warming_capacity`, `warming_energy`, `heatwave_attribution`, `solar_watch`, `chp_dispatch_cold`,
`chp_dispatch_price`, `chp_dispatch_red`.
