# Heat waves and cold spells at the sites: the reaction, when, where, and what explains it

Run on 15 November 2026, on CPU. Command: `extreme-events`. Module:
`code/src/sense_energy/analysis/extreme_events.py`. Tables in `code/reports/extreme_events/`,
figures from `code/scripts/extreme_events/`. The kickoff deck (`doc/kickoff_deck/`) is built on
this page.

The earlier weather work ([weather_and_cooling.md](weather_and_cooling.md)) estimates what a
degree does to a site's electricity. This page asks a different question: what happens **in an
event**. It takes the catalogue of heat waves and cold spells, measures each site's reaction in
each event, and sets the reactions against time (day of the event, hour of the day, year),
space (neighbours, regions, trusts) and the information that exists about a site (on-site
generation, the Display Energy Certificate, the estates return).

## The answers in short

- **Cold acts at once and all day. Heat builds, and comes in the evening.** In a cold spell the
  sites without on-site generation use 11 % more electricity, at every hour, from the first day.
  In a heat wave they use 6 % more, but almost nothing on the first day and in the morning: per
  degree, the reaction from the third day on is 2.4 times that of the first day (0.31 to 0.76 %
  per °C; difference +0.44, 95 % interval 0.17 to 0.72), and it sits between 16:00 and 22:00
  (+9 %; 08:00 to 12:00: +1 %).
- **The weather is regional, the reaction is not.** The NHS region accounts for 56 % of the
  between-site differences in heat-wave temperature and 37 % in cold-spell temperature, and for
  9 % and 5 % of the differences in reaction, which is what chance gives (8 % and 7 %).
  Neighbouring sites do not react alike (Moran's I −0.02 and 0.01; the temperature: 0.54 and
  0.80). Sites of one trust do not either.
- **The reaction belongs to the site.** A site's reaction in half of its events explains 37 %
  (heat) and 56 % (cold) of its reaction in the other half; the reliability of the full measure
  is 0.70 and 0.89.
- **On-site generation turns the reaction around.** At the sites with CHP the grid import rises
  by 25 % in heat waves (95 % interval 9 to 42; 11 of 12 sites rise; +41 % from 16:00 to 22:00)
  against 6 % elsewhere, and by 2 % in cold spells where the plant follows the heat demand
  (−4 to +8; 8 sites) against 11 % elsewhere. For a site with a plant, the heat wave and not
  the cold spell is the event at the grid connection.
- **The certificate explains little.** At the sites without on-site generation, the building
  environment, the operational rating, the electricity per m² and the electric heating share of
  the Display Energy Certificate each explain 0 to 2 % of the differences in reaction. The floor
  share with any air conditioning explains 9 % (heat) and 6 % (cold). The best single
  descriptor is the gas-to-electricity ratio of the estates return: 11 % (heat) and 18 %
  (cold).
- **No trend over the record.** The reaction per degree is the same in each winter (1.4 to
  1.7 % per °C) and in the summers of 2023 and 2025 (0.6 and 0.8).

## 1. Events and the measure

The events are those of `weather_events` (section 3 of the weather page): a **heat wave** at a
site is three or more days with a daily maximum at or above the Met Office county threshold
(25 to 28 °C); a **cold spell** is two or more days with a daily mean at or below 2 °C (the
UKHSA cold-alert criterion). A national event is a run of days with at least 5 % of the sites in
a site event. The electricity record (7 December 2022 to 8 May 2026) holds 7 national heat waves
and 14 national cold spells; 13 of the cold spells have scored site-days.

**Reaction.** Each site-day gets a reference: the median demand of the same weekday one, two
and three weeks before and after, without the days that are themselves in a heat wave or cold
spell at the site, bank holidays and the days from 21 December to 3 January. At least two
reference days are needed. The reaction of a set of days is the summed demand over the summed
reference, in per cent (100 × log ratio). Sums are used, not the mean of daily ratios, because
the grid import of a site with a plant can be near zero on single days. The same difference for
the temperature (daily maximum for heat waves, daily mean for cold spells) is the **size of the
event at the site** in degrees: 7.9 °C on average in heat waves, 6.7 °C in cold spells.

**Sites.** 134 sites have half-hourly electricity. 42 are left out of every site-level result
because their meter set is incomplete or changes (`meter_gaps`, `partial_supply`,
`aggregate_supply` in the coverage flags); 92 remain, 14 of them with on-site generation (the
CHP sample of `chp_dispatch`), 76 with a Display Energy Certificate. A site needs three scored
event days to get a reaction: 88 sites for cold spells, 76 for heat waves.

**What the measure is and is not.** It is a comparison with nearby ordinary days of the same
site, so it carries no national day shocks only as far as the reference weeks are ordinary; the
fixed-effect estimates of the weather page (section 5) remove them fully and give 1.1 to 1.4 %
per °C for cold and 1.5 to 2 % per °C for heat. The reactions here, per degree, are of that size
for cold (1.4 to 1.9) and lower for heat (0.3 to 0.8 per °C of daily maximum against the
reference days), because the heat-wave dose is measured on the daily maximum. The reference
days of a heat wave are themselves warm days; the measure is the reaction to the event, not to
summer.

{{figure: event_timeline | caption=Every national event in the record: mean reaction of the sites without on-site generation, with the 95 % interval across sites.}}

{{table: code/reports/extreme_events/table_events.csv | round=1}}

Every cold spell with at least ten sites raises demand at the sites without on-site generation
(5 to 14 %); every heat wave but the small one of July 2024 does (3 to 7 %). The columns for
the sites with a plant show what section 4 is about: their import rises by 20 to 30 % in the
large heat waves and stays below that of the other sites in most cold spells. In two short cold spells of early
2025 the summed import of the few plant sites in the event fell to a third: a plant came back
during the event.

## 2. Time

### 2.1 Day by day (`event_time`, `event_temperature`, `build_up`)

{{figure: event_time | caption=Reaction by day from the first day of the event at the site. Sites without on-site generation, mean and 95 % interval across sites.}}

{{figure: event_temperature | caption=The size of the event by day: degrees warmer (heat wave, daily maximum) or colder (cold spell, daily mean) than the reference days.}}

The temperature steps on day 0 in both kinds of event (6 to 7 °C beyond the reference days).
The demand of a cold spell steps with it: +9 % on day 0, +10 % on day 1. The demand of a heat
wave does not: +2 % on day 0, +5 % on day 1, +7 % on days 2 and 3, with the temperature
unchanged. After the event both fall back within a week.

{{figure: build_up | caption=Reaction per degree on the first day of an event and from the third day on, at the sites that have both.}}

{{table: code/reports/extreme_events/build_up.csv | round=2}}

The test is paired within sites: the reaction per degree from the third day on minus that of
the first day. Heat waves: 0.31 to 0.76 % per °C, +0.44 (0.17 to 0.72), 50 sites. Cold spells:
1.45 to 1.86, +0.42 (0.10 to 0.73), 75 sites. Both build by the same absolute amount; for heat
that is a factor of 2.4, for cold a factor of 1.3. A model that takes the temperature of the
day alone overstates the first day of a heat wave and understates the third.

### 2.2 Hour by hour (`hour_profile_heatwave`, `hour_profile_cold_spell`)

{{figure: hour_profile_heatwave | caption=Heat-wave days: reaction by half-hour of the day, sites with and without on-site generation.}}

{{figure: hour_profile_cold_spell | caption=Cold-spell days: reaction by half-hour of the day.}}

At the sites without on-site generation a cold spell lifts every hour by 9 to 13 %. A heat wave
lifts the hours from 08:00 to 12:00 by 1 % and the hours from 16:00 to 22:00 by 9 %; the
reaction stays above 5 % until after midnight. The load of a heat wave falls on the evening,
when the national system has its peak, and on the night, when the building gives its heat back.
At the sites with a plant the import in a heat wave rises by 41 % from 16:00 to 22:00.

### 2.3 Year by year (`seasons`)

{{figure: seasons | caption=Reaction per degree in each winter and summer of the record, sites without on-site generation (95 % interval from resampling sites). Summer 2024 had one small heat wave and is left out.}}

{{table: code/reports/extreme_events/seasons.csv | round=2}}

No trend: 1.4 to 1.7 % per °C in each of the four winters, 0.6 and 0.8 in the summers of 2023
and 2025, with overlapping intervals. Three summers and four winters cannot show a slow change.

## 3. Space (`map_reaction_heatwave`, `map_reaction_cold_spell`, `region_share`)

{{figure: map_reaction_heatwave | caption=Each site's reaction in heat waves. Black edge: on-site generation.}}

{{figure: map_reaction_cold_spell | caption=Each site's reaction in cold spells.}}

{{figure: region_share | caption=Share of the between-site variance that the seven NHS regions account for. Black ticks: the share that regions get by chance.}}

{{table: code/reports/extreme_events/spatial.csv | round=3}}

Two tests, each for the size of the event at the site and for the reaction of the site:

- **Neighbours** (Moran's I on the eight nearest sites, median distance to the eighth 38 km,
  999 permutations). The size of the event is alike among neighbours (0.54 for heat waves, 0.80
  for cold spells, p = 0.001). The reaction is not (−0.02, p = 0.51; 0.01, p = 0.28).
- **Regions and trusts** (share of the between-site variance, permutation test). The region
  accounts for 56 % and 37 % of the size of the event and for 9 % and 5 % of the reaction
  (chance: 8 % and 7 %; p = 0.28 and 0.64). The trust (13 and 16 trusts with two or more sites)
  accounts for 69 % and 87 % of the size of the event, because the sites of a trust are close
  together, and for 17 % and 15 % of the reaction (chance: 17 % and 19 %; p = 0.39 and 0.68).

The same holds without the sites with on-site generation. An event is a regional fact; what a
site does in it is not, and it is not a trust fact either.

**The reaction is a property of the site** (`reliability.csv`). Splitting each site's events
into odd and even ones in date order, the reaction in one half explains 37 % (heat, 59 sites)
and 56 % (cold, 88 sites) of the variance of the reaction in the other half; stepped up to all
events the reliability is 0.70 and 0.89.

{{table: code/reports/extreme_events/reliability.csv | round=3}}

## 4. Site information

### 4.1 On-site generation (`plant_reaction`)

{{figure: plant_reaction | caption=Mean site reaction with and without on-site generation, 95 % interval across sites. With a plant the meter records grid import.}}

{{table: code/reports/extreme_events/group_means.csv | cols=type,descriptor,level,sites,reaction_pct,lo,hi,median_pct,share_rising,extra_kw_per_site,mean_kw | round=1}}

- **Heat waves.** Import at the 12 sites with a plant rises by 25 % (9 to 42; median 19 %; 11
  of 12 rise), against 6 % (3 to 9) at the 64 other sites: 19 points more with the size of the
  event held fixed (4 to 34, p = 0.01). In kilowatts: 207 kW more import per site on a
  heat-wave day (mean import 382 kW), against 5 kW at the other sites (mean demand 66 kW). At
  the seven sites whose plant follows the heat demand (`chp_dispatch`), every one rises (24 %,
  11 to 36).
- **Cold spells.** Import at the 13 sites with a plant rises by 8 % (−1 to 17; median 4 %),
  against 11 % (9 to 13) elsewhere. At the eight heat-led sites it rises by 2 % (−4 to 8); the
  five others are wide (17 %, −9 to 43).

The plausible mechanism for both: a plant that follows heat runs flat out in a cold spell and
covers the extra electricity, and is turned down in a heat wave when the site needs little
heat, so the import rises by more than the cooling load. The meters cannot confirm it: they
show import only. Plant output data would.

### 4.2 Certificates and the estates return (`descriptor_share`)

{{figure: descriptor_share | caption=Share of the between-site variance of the reaction that each descriptor adds to the size of the event (adjusted R², one descriptor at a time), sites without on-site generation.}}

{{table: code/reports/extreme_events/alignment_fits.csv | cols=type,descriptor,sites,joint_p,adj_r2_gain | round=3}}

{{table: code/reports/extreme_events/alignment_terms.csv | cols=type,descriptor,level,sites,mean_reaction_pct,difference_pts,lo,hi,p | round=2}}

One regression per descriptor of the site reaction on the descriptor with the size of the event
held fixed, robust standard errors, on the sites without on-site generation (the plant would
otherwise stand in for every descriptor that goes with large acute hospitals: with the plant
sites in, building use, size and the gas ratio all "explain" the heat reaction).

- **Display Energy Certificate.** Building environment (natural ventilation, mechanical
  ventilation, mixed mode or air conditioning): joint p = 0.42 (heat) and 0.26 (cold), 1 % of
  the variance. Operational rating: p = 0.31 and 0.16, 0 %. Electricity per m²: 2 % and 1 %.
  Floor share heated with electricity: 0 %. Floor share with any air conditioning: +3.5 points
  per standard deviation in heat waves (0.7 to 6.4, p = 0.02; 9 %) and +2.5 in cold spells (0.4
  to 4.7, p = 0.02; 6 %): the sites with more conditioned floor react more to both, which reads
  as electrically serviced buildings and not as cooling.
- **Estates return.** Gas-to-electricity ratio: −4.1 points per standard deviation in heat
  waves (−7.8 to −0.5, p = 0.03; 11 %) and −3.9 in cold spells (−6.6 to −1.1, p = 0.01; 18 %):
  the more a site runs on gas, the less its electricity reacts, to cold as the weather page
  found and to heat as well. Building use: mental-health and non-inpatient sites react 4 points
  more to cold than the three acute hospitals without a plant (p = 0.03 and 0.04; 0 % of the
  variance after the degrees of freedom). Mean build year: +2.0 points per standard deviation
  for cold (p = 0.02; 4 %).
- Twenty tests were run and five have p below 0.05, where chance gives one; the weaker ones
  should not be read alone. The gas ratio (p = 0.008 for cold) is the one that also has a mechanism
  and an earlier, independent estimate.

The comparison that matters: the best descriptor explains 11 to 18 %, the fields of the
certificate 0 to 9 %, and the site's own other events 37 to 56 %. What a site does in an event
is stable and is not in the registers.

## 5. What follows

- For load on the network, the unit of analysis is the site, not the region, and for sites
  with a plant the season of concern is the summer.
- A forecast or a planning model needs the heat of the last days, not the temperature of the
  day, and it needs the hour: the evening of the third day is where a heat wave lands. The
  forecast thread found the consequence from the other side: on heat-wave days the day-ahead
  forecast secures 33 % of the achievable peak cut against 50 % on the other days of the same
  site and month ([forecast_error_anatomy.md](forecast_error_anatomy.md)).
- Two data requests would turn the open readings into tests: plant output (does the plant turn
  down in heat waves?) and the size of cooling plant (recorded on 19 % of recent certificates).

## Limits

- 92 sites, 14 with a plant, 7 heat waves: the plant results rest on 12 and 13 sites and have
  wide intervals; the direction is consistent across events.
- The reference days are close in the calendar, not matched on anything else. A change at a
  site within three weeks of an event (a meter change, a ward closure) enters its reaction.
- The heat-wave definition is the Met Office's by county; the temperatures are ERA5 at the site
  (the weather page, section 6, compares it with ERA5-Land and stations).
- Reactions are in per cent of the site's own demand, so small sites count as much as large
  ones in the means; the summed columns of the event table give the kilowatt view.
- Gas is not in this page.
