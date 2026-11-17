# Kickoff deck: the project in two slides

Two slides for the SENSE healthcare kickoff, kept broad on purpose: the project is at an early
stage (it started in November 2026), so the slides give the set-up and the questions, not
findings.

| Slide | Content | Visual |
|---|---|---|
| 1 | The energy demand of NHS hospitals as a forecast problem | one site and eight days across the full width: air temperature, visitors each day, demand and the forecast of the next day; below it the forecast as a formula and two numbers (134 sites, 48 half-hours) |
| 2 | The correct retrofit for each hospital | a strip with the demand of one site above its base load through 2025, divided into heating, lighting and cooling; three maps with initial data, each with its label at its left |

The slides have very little text on purpose: a title, one line under it, the visuals and one
footer line. No bullets, no questions, no hypotheses.

The rules are also in `/home/malzn/taste.MD`, a summary for all agents with links to the
skills.

**Design rules** (from the two skills the slides are checked against,
`Leonxlnx/taste-skill` and `nextlevelbuilder/ui-ux-pro-max-skill`, and from a checklist of signs
of machine-made slides):

- no caption under a visual: legends are part of the visual, credits are in the one footer line;
- the footer is the same text at the same place on both slides;
- equal left and right edges, no half-empty area, visuals across the full width;
- text and thin lines have a contrast of at least 4.5:1 (the darker shades `TEXT_ORANGE`,
  `TEXT_TEAL`, `TEXT_GOLD` in `code/scripts/deck/_deck.py`; the brighter palette is for fills);
- a series is never told apart by colour alone: the forecast line is dashed, its range is named,
  the yellow bars and circles have an edge, each map has its name next to it;
- no bullet lists with the same rhythm, no stock icons, no stock phrases, two titles of different
  shape, real data in every visual, specific numbers.

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
