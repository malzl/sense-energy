# Flexibility, design winters, heat pumps in the field and the solar programme

Four new pieces of modelling on data already in the project, 11 November 2026, all run on CPU.
Commands: `flexibility`, `fetch-era5-timeseries`, `build-era5-long`, `design-winter`,
`heat-pump-field`, `solar-programme`. Modules in `code/src/sense_energy/analysis/`
(`flexibility.py`, `design_winter.py`, `heat_pump_field.py`, `solar_programme.py`) and
`data/era5_timeseries.py`; tables in `code/reports/<name>/`; figures from
`code/scripts/flexibility/`.

What the page answers: what a site's load shape costs and what a battery or a solar array is
worth against it; what a load forecast is worth in a decision; whether heat pumps fit under the
connections in a cold winter and not only in 2023/24; what the estates returns say heat pumps
did; and whether the funded hospital solar arrays can be seen from the meter.

New data: ERA5 hourly temperature from 1960 at the panel's 59 grid cells (Copernicus point
time series; contains modified Copernicus Climate Change Service information), the list of NHS
sites funded by Great British Energy (GOV.UK, Open Government Licence v3.0) and the 2025/26
estates return (NHS England, published 15 October 2026).

## 1. What the load shape costs (`bill_shape_premium`)

Every half-hour of 2024/25 priced at the Elexon market index (energy only, no network charges or
levies), for the 124 sites with at least 90 % of the year metered.

- **The panel: 118 GWh, £9.9 million a year at the wholesale index.**
- **The shape costs 4 %.** The median site pays 4.3 % more per kWh than the time-average price,
  because it uses more in the dearer hours: 3.7 % at inpatient sites, 4.8 % at outpatient sites,
  3.8 % for the import of the CHP sites. Five sites pay less than the average price (down to
  −4.7 %); the dearest shape pays 23 % more.
- **The red band holds 10 % of the energy in 8.7 % of the hours** (16:00–19:00 on working days):
  12.4 GWh a year across the panel, so every penny per kWh of red-band charge is £124,000 a year.

{{table: code/reports/flexibility/bill_anatomy.csv | cols=site_code,site_name,group,fy,mwh,wholesale_cost_gbp,paid_p_kwh,shape_premium_pct,red_share_energy | sort=-mwh | round=2 | head=30}}

## 2. What a battery earns (`battery_value`)

A two-hour battery (round-trip efficiency 0.85) cycled each day against the day's prices by a
linear programme, starting and ending empty. Price cases: the market index; the index plus a
red-band unit charge of 2, 5 or 10 p/kWh; and the Octopus Agile tariff of each region, a published
half-hourly retail tariff with its own evening adder. The red-band charges are scenario values:
the real rates differ by network area and voltage and are not in the project. Perfect knowledge
of the day's prices is exact for Agile (published the day before) and an upper bound for the
index (known after delivery). No degradation cost.

{{chart: code/reports/flexibility/battery_unit_annual.csv | kind=bar | x=value_gbp_kwh_year | y=case | color=fy | title=Battery value by price case and year (£ per kWh of capacity per year) | height=520}}

- **£23 per kWh of capacity a year on wholesale spreads** (1.8 cycles a day), £27, £35 and £47
  with a 2, 5 and 10 p red-band charge, £88–96 on Agile. Stable across the three years
  (wholesale £21–24).
- **The tariff sets the value, not the site.** The schedule depends on prices only. The site
  enters through the no-export limit: with a battery of 20 % of its mean demand, 24 of 134 sites
  would push more than 1 % of the discharge past their own load (median 4 %), ten of them CHP
  sites whose import is low when the plant runs.
- **Break-even.** A ten-year simple payback needs an installed cost below £235 per kWh on
  wholesale spreads, £350 with a 5 p red band, about £900 on an Agile-style tariff.
- **Across the panel**, a battery of 20 % of mean demand at every site is 7.5 MWh and earns
  £176,000 a year at the index, £259,000 with a 5 p red band.
- **A load forecast is worth nothing here.** Arbitrage needs the prices, which are published.

## 3. Rooftop solar against the real load (`solar_self_consumption`)

Generation per kWp from ERA5 surface solar radiation at the site (performance ratio 0.8 on the
horizontal irradiance; median yield 825 kWh per kWp a year), against each half-hour of 2024/25 at
the 106 sites without CHP, for arrays sized relative to the site's mean demand.

{{table: code/reports/flexibility/solar_summary.csv | round=3}}

- **Hospitals use everything a roof can make.** Up to 1 kWp per kW of mean demand the site
  consumes all of the generation; at 2 kWp per kW 96 %, at 4 kWp per kW 78 %. Inpatient and
  outpatient sites are alike: the weekend baseload absorbs the weekend sun.
- **The array covers 9 % of demand at 1 kWp per kW**, 18 % at 2, 29 % at 4.
- **Worth £56 per kWp a year at the wholesale index.** Solar hours are 15 % cheaper than the
  average hour (capture ratio 0.85). The saving on the bill is larger, since a kWh consumed on
  site also avoids network charges and levies; that part needs the real tariffs.
- **For the programme**: exports are not the constraint at hospital sites; roof area is.

## 4. What a forecast is worth: peak shaving (`forecast_value`, `forecast_value_quantile`)

Peak shaving is the use where the load forecast matters. A battery (20 % of mean demand, two
hours) starts the day full. A threshold is planned from the day-ahead forecast as the lowest
import ceiling the battery could hold through the forecast day; the battery then defends that
threshold in real time on the actual load. If the forecast is too low the store empties before
the peak and the peak comes through; if too high the cut is smaller than it could be. Scored on
the 91 sites and 69 test days of the proof of concept (6,107 site-days), for the saved forecasts
of each model, against the cut under perfect foresight (11.8 % of the day's peak on average).

{{table: code/reports/flexibility/forecast_value.csv | cols=model,plan,nmae_pct,captured_share,peak_cut_pct,perfect_cut_pct,failed_share | round=3}}

- **The best forecast captures just over half of what is achievable.** Seasonal naive 40 %,
  LightGBM + IFS ENS 48 %, the foundation models 50–53 % (Chronos-2 + IFS ENS 53 %).
- **Six points of nMAE are thirteen points of value**: from seasonal naive (17.9 % nMAE) to
  Chronos-2 + IFS ENS (11.2 %).
- **Plan on the 60–70 % quantile, not the median.** Missing the peak costs the whole cut; aiming
  high costs a part of it. The 70 % quantile adds about 2 to 3 points for the foundation models
  (Chronos-2 + IFS ENS 55 %); the 90 % quantile gives up a quarter to a third of the value. That is what the quantile
  forecasts are for, and it is a modest gain.
- **On a fifth to a quarter of site-days the battery achieves next to nothing** (under 5 % of its
  potential) even with the best model: the days whose peak the forecast misses.

## 5. Design winters (`design_winter_sites`, `design_winter_utilisation`, `design_winter_budget`)

The headroom result (13 of 31 sites above their connection with heat pumps) used the weather of
2023/24. Here the same year's demand is re-weathered with each of 66 weather years, April 1960 to
March 2026: today's electricity through each site's hourly cold and heat response, the heat-pump
load rebuilt from that year's hourly temperature with the headroom model (degree-hours, base
share, air-source COP). Two checks: the long temperature record equals the gridded record on
their overlap (4,371 site-days, difference 0.0 °C), and the reference year reproduces the
headroom peaks to eleven decimal places.

{{table: code/reports/design_winter/years.csv | cols=label,t_winter_mean,t_coldest_day_median,hp_sites_above_capacity,util_hp_median,hp_energy_vs_ref_pct,peak_now_change_median_pct,energy_change_median_pct | sort=t_winter_mean | round=2 | head=12}}

- **The count barely depends on the winter.** 2023/24 was the second-mildest of the last 30
  winters (6.8 °C) and puts 13 sites above capacity; the last 30 winters give 10 to 15, the
  one-in-twenty winter 14, the winter of 1962/63 19. The peak follows the coldest hours of a
  winter, not its mean, and the sites that exceed their connection do so by a wide margin.
- **The margin moves.** Peak over capacity at the median site: 0.88 in 2023/24, 0.95 in the
  one-in-twenty winter, 1.06 in the worst winter on record.
- **Today's estate is safe in a cold winter.** Without heat pumps the one-in-twenty winter raises
  the peak by 4 % at the median site and no site passes its capacity.
- **Electrification makes the budget weather-sensitive.** Across the last 30 weather years
  today's annual electricity varies by 0.3 % (standard deviation, median site); with heat pumps
  by 3.0 %. The heat pumps' own electricity ranges from −1 % to +27 % against 2023/24, and +50 %
  in the weather of 1962/63.

Limits: on the coldest days of the old winters the cold response is extrapolated beyond the
temperatures of the demand record (coldest day −3.2 °C at the median site; up to 49 days a year
in 1962/63, a handful in recent years), and the old winters are colder than today's climate,
which is why the design statistics use the last 30 years.

## 6. Storage against reinforcement (`design_winter_storage`)

For each over-capacity site, the store that would hold it under its connection: power equal to
the largest excess, energy equal to the largest run of excess net of recharging below capacity.
Classes by the hours of the connection's capacity the store holds (scenario bounds: 2 h and 24 h).

{{table: code/reports/design_winter/design_sites.csv | cols=site_code,capacity_kw,util_hp_ref,util_hp_design,util_hp_worst,worst_year,store_kw_design,store_kwh_design,store_hours_design,hours_over_design,store_class_design | sort=-util_hp_design | round=2 | head=16}}

- **A third can be held with a battery.** Of the 14 sites above capacity in the one-in-twenty
  winter, five need 4 to 50 kW and 2 to 126 kWh; two need a store of 4 and 12 hours of capacity,
  the scale of a thermal store; seven are beyond storage, above capacity for 700 to 4,500 hours
  of the year.
- **So flexibility closes the gap at the margin and not beyond.** The seven need a larger
  connection or less heat to electrify: fabric, lower flow temperatures, or a hybrid system that
  keeps gas for the coldest days.

## 7. Heat pumps in the field (`heat_pump_field`)

The three estates returns (2023/24, 2024/25, 2025/26) give each site's gas, electricity and count
of heat pumps. A site that reports its first heat pump in 2024/25 has a full year without it
(2023/24) and a full year with it (2025/26); 48 English sites do, 23 of them usable (no CHP, an
unchanged floor area, a plausible electricity change, gas in the first year). Sites that never
report a heat pump are the comparison.

{{table: code/reports/heat_pump_field/did.csv | cols=pair,status,sites,gas_median_pct,elec_median_pct,gas_did_pct,gas_did_lo,gas_did_hi,elec_did_pct,elec_did_lo,elec_did_hi,gas_fell_fifth_share | round=1}}

- **A full year later the heat pumps show.** Against sites without, the 23 sites use 8.3 points
  more electricity (interval +3.5 to +14.6) and 9.4 points less gas (−21.0 to +7.3, not
  significant); gas fell by a fifth or more at 35 % of them, against 22 %.
- **The installation year hides it.** The same question asked within the installation year gives
  +4.0 points of electricity and −2.9 of gas for 2024/25 (33 sites, neither significant), because
  the pumps ran for part of the year. The 2025/26 installers (15 sites) already show gas down 12.0
  points (−35.1 to −0.5).
- **It is partial electrification.** A tenth of the gas, not all of it: the typical site added
  heat pumps to a gas system. With gas at 2.2 times electricity at these sites, a 9 % fall in gas
  and an 8 % rise in electricity are what a COP near 2 would give; the gas estimate is too
  uncertain to make that a measurement.
- **Thirteen clean switches: a median COP of 2.8.** Across the three comparisons 13 sites that
  report heat pumps cut gas by a quarter or more on an unchanged floor area while electricity
  rose. Their implied COPs spread from 1.3 to 7.4 (quartiles 1.9 and 6.0), because annual returns
  are coarse; in the full-year comparison four sit between 2.7 and 3.3 (Horton Hospital, Regent
  Point, Austen House, Taunton Ambulance Station). The headroom work assumed 2.55.
- **Correction to the installation-savings reading.** The three panel sites that report heat
  pumps use 32 % more electricity, but none shows gas being replaced: one has no gas, one kept
  its gas (−2 %), one changed its reporting sixfold. The 32 % is added electric load, not
  measured electrification.

{{table: code/reports/heat_pump_field/switchers.csv | cols=pair,site_name,site_type,status,heat_pumps_b,gas_change_pct,elec_change_pct,implied_cop,reports_heat_pump | sort=gas_change_pct | round=1 | head=30}}

The 2025/26 return (published 15 October 2026, site table in its second version) was downloaded
by hand, because its page refuses scripted requests, and is loaded with
`sense-energy build-eric --year 2025_26`.

## 8. The solar programme from the meter and the returns (`solar_programme_spring`, `solar_programme_returns`, `solar_metered_vs_reported`)

Great British Energy's first project funds rooftop solar at NHS sites: £100 million for 181 site
postcodes in 78 trusts, announced on 21 March 2025. Two views of it: the estates returns, for
every funded site in England, and the meter, for the ten panel sites on the list.

**In the returns.** 142 of the funded postcodes match a site in both the 2024/25 and 2025/26
returns (217 sites).

- **Funded sites report 84 % more solar generation in 2025/26**, 3.0 to 5.5 GWh, against 39 %
  at all other sites (22.7 to 31.7 GWh). They are 8 % of the sites and 22 % of the national
  increase.
- **22 of the 142 funded postcodes show new or grown generation** by March 2026; 44 report any
  solar, 35 the year before. Most of the programme is still to appear: the return's year ends a
  year after the announcement, and an array switched on during the year reports a part-year.

{{table: code/reports/solar_programme/returns_check.csv | cols=site_name,trust_code,site_type,funded,solar_generated_kwh_a,solar_generated_kwh_b,added_kwh,new,grew | sort=-added_kwh | round=0 | head=25}}

**At the meter.** The sunshine slope is estimated on a spring window (1 March to 8 May) for 2023
to 2026, and the spring of 2026 is compared with the spring of 2025 and with the spring of 2024,
the last one before the programme (an array switched on during 2025 already shows in spring
2025). Eight of the ten funded panel sites are testable.

{{table: code/reports/solar_programme/funded_sites.csv | cols=site_code,site_name,group,trust_funding_gbp,sun_2024_pct,sun_2025_pct,sun_2026_pct,change_since_baseline_pct,z_since_baseline,break_any,day_kw,implied_kwp_since_baseline,eric_solar_mwh_before,eric_solar_mwh_after | round=1}}

- **Three funded sites show their panels**: The Stewarts (slope +2.0 to −15.9 % per kWh/m²,
  z = −4.3, 13 kWp of self-consumed generation), Winchester and Eastleigh Resource Centre (−0.8 to
  −5.4, z = −2.9) and Wythenshawe Hospital (−6.4 to −8.7, z = −2.7, 62 kWp).
- **The returns agree where both speak.** Wythenshawe reports 61 MWh more generation, about
  70 kWp, against 62 kWp from the meter; Winchester and Eastleigh reports its first 54 MWh. The
  three testable funded sites that report no solar (Chesterfield Royal, Tolworth, Didcot) show
  nothing at the meter either. Chesterfield Royal has the largest award (£6.6 million); an array
  of 0.9 MWp would be visible there, and by 8 May 2026 none is.
- **The meter measures the solar a site consumes.** Robert Jones and Agnes Hunt doubled its
  reported generation (843 to 1,729 MWh) and the meter shows no change: a CHP site whose import
  is already near zero on sunny days exports the rest. A break of the slope is a size: panels of
  k kWp lower daytime demand by k × 0.8 / 12 kW per kWh/m². At St George's Hospital the meter
  gives 215 kWp against 230 kWp implied by the generation the return reports; where the array is
  larger than the daytime load the meter sees only the part consumed on site: 52 of 93 kWp at
  Potters Bar, 349 of 684 kWp at Bridlington, 8 of 61 kWp at Church Hill House.
- **Sites off the list break as well**: St Paul's Hemel Hempstead, Mardon House, 99 Waverley
  Road, Royal Leamington Spa, London House and Little Plumstead, sized at 15 to 200 kWp.
- **Short windows are noisy.** Against spring 2024, nine sites break downwards and eleven
  upwards, the upward ones up to z = 3.7. The five downward breaks at z = −4 or beyond stand on
  their own; Wythenshawe and Winchester are within the noise and rest on the returns'
  confirmation.

{{table: code/reports/solar_programme/summer_measured.csv | cols=site_code,site_name,change_pct,day_kw,implied_kwp,implied_kwp_level,eric_solar_mwh_2024_25,eric_total_kwp,metered_share_of_reported | round=1}}

## 9. What this changes

- The headroom finding stands in a cold winter and gets a second half: a battery solves it at a
  third of the over-capacity sites, reinforcement or less heat at the rest.
- Forecast accuracy has a price tag now: the share of an achievable peak cut it captures. The
  quantile forecasts earn 2 to 3 points of it.
- The value of a battery is a question about the tariff. The next step is the real distribution
  charges per network area and voltage.
- The estates returns do measure heat pumps once a full year is compared with a full year: 8
  points more electricity, about a tenth less gas. They show partial electrification, which is a
  smaller load than the headroom scenario of replacing all gas heat.
- The solar programme can be followed two ways: the returns for every funded site, the meter
  for the date and the self-consumed size. By March 2026 a sixth of the funded postcodes show it.

## Figures

`bill_shape_premium`, `battery_value`, `solar_self_consumption`, `forecast_value`,
`forecast_value_quantile`, `design_winter_sites`, `design_winter_utilisation`,
`design_winter_budget`, `design_winter_storage`, `heat_pump_field`, `solar_programme_spring`,
`solar_programme_returns`, `solar_metered_vs_reported`.
