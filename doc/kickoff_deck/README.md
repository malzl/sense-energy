# Kickoff deck: four exploratory findings

Six slides for the SENSE healthcare kickoff: title, four findings on the day-ahead forecast
(each with its implication, a research avenue and two hypotheses) and a discussion slide. The
text follows ASD-STE100 Simplified Technical English. The look follows the Oxford / SENSE slide
template: Oxford blue, serif titles (Charter), Helvetica text (TeX Gyre Heros), the template's
chart palette.

| Slide | Finding | Source |
|---|---|---|
| 2 | Zero-shot models lead, but get only half of the possible value | `doc/poc_results.md`, `doc/forecast_error_anatomy.md` sections 1, 2, 8 |
| 3 | Sites with on-site generation have 79 % of the forecast error | section 3 |
| 4 | Strikes and visitor numbers do not change electricity demand | sections 4, 5 |
| 5 | On heat-wave days the forecast loses a third of its value | section 7, `doc/heat_cold_wave_paper_check.md` |

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

- **Earlier findings.** The first version (commit ac87b39) had three other findings: heat pumps
  against grid connections, the response to hot days, and solar and heat pumps seen from the
  meter. Their slides are kept as backup slides: set `\backuptrue` at the top of
  `kickoff_deck.tex` and they follow the discussion slide.
- **Logos.** The title slide has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Date.** Set `\deckdate` at the top of `kickoff_deck.tex`; the event name is `\deckevent`.
- **Figures** are drawn at their exact size on the 160 × 90 mm slide, so change sizes in the
  figure script and not in LaTeX.
- **Numbers** come from `code/reports/` (poc, flexibility, forecast_anatomy; the backup slides
  use estates, design_winter, heat_drivers, solar_watch, solar_programme). No site is named on
  any slide.
