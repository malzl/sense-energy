# Kickoff deck: four exploratory findings

Six slides for the SENSE healthcare kickoff: title, four findings on NHS sites in heat waves and
cold spells (each with its implication, a research avenue and two hypotheses) and a discussion
slide. The text follows ASD-STE100 Simplified Technical English. The look follows the Oxford /
SENSE slide template: Oxford blue, serif titles (Charter), Helvetica text (TeX Gyre Heros), the
template's chart palette.

| Slide | Finding | Source |
|---|---|---|
| 2 | Cold acts at once, all day. Heat builds for days, in the evening | `doc/extreme_events.md` section 2 |
| 3 | The weather is regional, but the reaction of a site is not | section 3 |
| 4 | A plant on site changes the reaction. Certificates explain little | section 4 |
| 5 | On heat-wave days the forecast loses a third of its value | `doc/forecast_error_anatomy.md` section 7 |

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

- **Earlier findings.** Version 1 (commit ac87b39): heat pumps against grid connections, the
  response to hot days, solar and heat pumps seen from the meter. Version 2 (commit 5f8ee7c):
  forecast value, the error at sites with on-site generation, strikes and visitor numbers.
  Their six slides are kept as backup slides: set `\backuptrue` at the top of
  `kickoff_deck.tex` and they follow the discussion slide.
- **Logos.** The title slide has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Date.** Set `\deckdate` at the top of `kickoff_deck.tex`; the event name is `\deckevent`.
- **Figures** are drawn at their exact size on the 160 × 90 mm slide, so change sizes in the
  figure script and not in LaTeX.
- **Numbers** come from `code/reports/` (extreme_events, forecast_anatomy; the backup slides
  use poc, flexibility, estates, design_winter, heat_drivers, solar_watch, solar_programme). No site is named on
  any slide.
