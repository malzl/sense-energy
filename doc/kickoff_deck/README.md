# Kickoff deck: what we found so far

Seven slides for the SENSE healthcare kickoff. They follow the findings map
(`doc/findings_map.md`): a title, an overview of five questions with a grade for each finding
(solid, indicative, scenario), one slide each for the drivers of demand, the differences
between sites, the forecast and the future estate, and a discussion slide. Each question slide
has a text column with the implication, how sure we are (the checks) and what is open. The text
follows ASD-STE100 Simplified Technical English. The look follows the Oxford / SENSE slide
template: Oxford blue, serif titles (Charter), Helvetica text (TeX Gyre Heros), the template's
chart palette.

| Slide | Content | Source |
|---|---|---|
| 2 | Five questions, and how sure we are of each answer | `doc/findings_map.md` |
| 3 | The weather changes the demand. Patient activity does not | findings 2.1 to 2.6 |
| 4 | The reaction belongs to the site, not to the region | findings 3.2 to 3.8 |
| 5 | The forecast is stable for 4 years, but loses value in heat waves | findings 4.1 to 4.6 |
| 6 | Scenario: heat pumps put half of the sites above capacity | findings 5.1 to 5.3 |

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

- **Earlier versions.** Version 1 (commit ac87b39): heat pumps against grid connections, the
  response to hot days, solar and heat pumps seen from the meter. Version 2 (5f8ee7c): forecast
  value, the error at sites with on-site generation, strikes and visitor numbers. Version 3
  (270e3e8): heat waves and cold spells in time, in space and against site information. Their
  ten detail slides are kept as backup slides: set `\backuptrue` at the top of
  `kickoff_deck.tex` and they follow the discussion slide. Two statements on those slides did
  not survive the checks of the findings map (the cause of the import rise at sites with a
  plant, and the loss of forecast value in cold spells).
- **Forecast numbers.** The numbers of slide 5 and of the overview are the macros `\FCerr`,
  `\FCnaive`, `\FCplant`, `\FCwave` and `\FCwaveci` at the top of `kickoff_deck.tex`; set them
  from `code/reports/forecast_record/` after a new run.
- **Logos.** The title slide has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Date.** Set `\deckdate` at the top of `kickoff_deck.tex`; the event name is `\deckevent`.
- **Figures** are drawn at their exact size on the 160 × 90 mm slide, so change sizes in the
  figure script and not in LaTeX.
- **Numbers** come from `code/reports/` (event_audit, extreme_events, forecast_record,
  forecast_anatomy, estates, design_winter; the backup slides also use poc, flexibility,
  heat_drivers, solar_watch, solar_programme). No site is named on
  any slide.
