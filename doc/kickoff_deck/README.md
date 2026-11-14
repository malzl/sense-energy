# Kickoff deck: four exploratory findings

Six slides for the SENSE healthcare kickoff: title, four ranked findings (each with its
implication, a research avenue and two hypotheses) and a discussion slide. The text follows
ASD-STE100 Simplified Technical English. The look follows the Oxford / SENSE slide template:
Oxford blue, serif titles (Charter), Helvetica text (TeX Gyre Heros), the template's chart
palette.

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

- **Logos.** The title slide has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Date.** Set `\deckdate` at the top of `kickoff_deck.tex`; the event name is `\deckevent`.
- **Figures** are drawn at their exact size on the 160 × 90 mm slide, so change sizes in the
  figure script and not in LaTeX.
- **Numbers** come from `code/reports/` (estates, design_winter, heat_drivers, poc, flexibility,
  solar_watch, solar_programme). No site is named on any slide.
