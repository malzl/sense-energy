# Kickoff deck: the project in two slides

Two slides for the SENSE healthcare kickoff, kept broad on purpose: the project is at an early
stage (it started in November 2026), so the slides give the set-up and the questions, not
findings.

| Slide | Content | Visual |
|---|---|---|
| 1 | NHS hospital energy demand as a forecasting problem: the data (electricity, weather, mobility, registers), the problem, why a forecast | one site and eight days: temperature, visitors per day, demand and the forecast of the next day; the forecast as a formula |
| 2 | The policy angle: which hospitals benefit from which retrofit, and four questions | the demand as base load plus heating, cooling and lighting terms; three maps with first estimates of how much each site follows cold, heat and daylight |

The text follows ASD-STE100 Simplified Technical English. The look follows the Oxford / SENSE
slide template: Oxford blue, serif titles (Charter), Helvetica text (TeX Gyre Heros), the
template's chart palette. The maps on slide 2 are an illustration of the approach; they use the
site slopes of `doc/weather_and_cooling.md` section 8 and are not presented as a result.

| File | Content |
|---|---|
| `kickoff_deck.pdf` | the deck, 16:9 |
| `kickoff_deck.tex` | slide source (pdfLaTeX, Beamer); every element is placed in millimetres |
| `figures/` | one PDF, PNG and data CSV per visual |
| `build.sh` | renders the figures (`code/scripts/deck/`) and compiles the deck |

```
bash doc/kickoff_deck/build.sh               # figures and deck
bash doc/kickoff_deck/build.sh --no-figures  # deck only
```

- **Earlier versions.** They follow the two slides as backup slides when `\backuptrue` is set
  at the top of `kickoff_deck.tex` (17 slides): the seven-slide version that follows the findings
  map (`doc/findings_map.md`, commit 2fd9f9a), then the detail slides on heat and cold events
  (270e3e8), the forecast (5f8ee7c) and the first selection (ac87b39).
- **Forecast numbers (backup slides).** The numbers of the forecast slide and of the overview are the macros `\FCerr`,
  `\FCnaive`, `\FCplant`, `\FCwave` and `\FCwaveci` at the top of `kickoff_deck.tex`; set them
  from `code/reports/forecast_record/` after a new run.
- **Logos.** The title slide of the backup version has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Date.** The two slides carry no date; the backup title slide takes `\deckdate` and `\deckevent`.
- **Figures** are drawn at their exact size on the 160 × 90 mm slide, so change sizes in the
  figure script and not in LaTeX.
- **Numbers** come from `code/reports/` (event_audit, extreme_events, forecast_record,
  forecast_anatomy, estates, design_winter; the backup slides also use poc, flexibility,
  heat_drivers, solar_watch, solar_programme). No site is named on
  any slide.
