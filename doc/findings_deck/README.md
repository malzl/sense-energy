# Findings deck: what we found to date

Fourteen slides with all findings of the project to 22 November 2026, in the look of the kickoff
deck (`doc/kickoff_deck/`). The deck follows the findings map (`doc/findings_map.md`): one
question for each slide, a visual across the full width, the method in a footnote, and a grade
for each finding.

| Slide | Content | Visual |
|---|---|---|
| 1 | Cover | the 134 sites on Oxford blue; gold: on-site generation |
| 2 | Five questions, and how sure we are of each answer | one mark for each of the 45 findings, by grade |
| 3 | Question 1: the meters agree with the estates returns | meter against return; import and plant output at 20 plant sites; the 134 sites by data check |
| 4 | Question 2: weather changes the demand, patient activity does not | change by day of the event and by hour; cold, heat, placebo and strikes |
| 5 | Question 2: three checks of the weather effects | 90 sets of analysis choices with placebo runs; what is left of day 3 with four days of temperature; bounds for a hidden factor |
| 6 | Question 3: each site has its own reaction | two maps of the tested site reactions; what explains the differences between sites |
| 7 | Question 4: a model with no training is the best forecast | error in each quarter against LightGBM trained once and each quarter; error against history length |
| 8 | Question 4: heat-wave days cost value, not accuracy | peak cut by battery storage; heat-wave days against other days; the two differences |
| 9 | Question 4: twelve sites with a plant have most of the error | map of the error of each site; their share of demand and error; the forecast with meter data of the same day |
| 10 | Question 4: two faults and two corrections | the rule for special days on all, normal, special and proximity days; where the 80 % range misses, before and after online tracking |
| 11 | Question 5: scenario, heat pumps and the grid connection | map of 631 sites; panel sites in 66 winters; field data of 23 sites with new heat pumps |
| 12 | What the checks changed | eight earlier statements and the result after the check |
| 13, 14 | All findings | one line for each of the 45 findings with its grade and number |

**Grades** (as in the findings map): solid, an interval that excludes zero and a second,
independent check; solid in part; indicative, one method, one period or few sites; open,
tested without a clear answer; scenario, a calculation; withdrawn.

**Design rules** are those of the kickoff deck and `/home/malzn/taste.MD`: a title, one line
under it, the visual and one footnote; no bullets; no caption under a visual (each panel has a
short name at its top left); equal edges; text colours with a contrast of at least 4.5:1; a
series is not told apart by colour alone (line styles, hatching, direct labels); the same
footer at the right of each slide; no site is named. Colours keep one meaning across the deck:
cold blue, heat orange, sites with a plant gold, the zero-shot forecast blue, trained models
and corrections teal, reference forecasts grey. The text follows ASD-STE100 Simplified Technical
English.

| File | Content |
|---|---|
| `findings_deck.pdf` | the deck, 16:9 |
| `findings_deck.tex` | slide source (pdfLaTeX, Beamer); every element is placed in millimetres |
| `figures/` | one PDF, PNG and data CSV for each visual, drawn at its size on the slide |
| `build.sh` | renders the figures (`code/scripts/findings_deck/`) and compiles the deck |

```
bash doc/findings_deck/build.sh               # figures and deck
bash doc/findings_deck/build.sh --no-figures  # deck only
```

- **Numbers** come from `code/reports/` (estates, chp, extreme_events, event_audit,
  causal_sensitivity, forecast_record, forecast_anatomy, forecast_extensions, forecast_range,
  special_days, design_winter, heat_pump_field). The numbers in the slide text and in the
  reference slides are typed in the .tex: update them with the findings map after a new run.
- **Logos.** The cover has three logo slots. Put `oxford_logo`, `sbs_logo` or `sense_logo`
  (`.pdf` or `.png`) into `logos/` and rebuild; a slot without a file stays empty.
- **Visitor data.** Finding 2.6 (slide 13) and 4.8 (slide 14) use the visitor counts of The
  Oxford Partnership. Its licence asks for an attribution where the data are used; the kickoff
  deck has none, at the owner's request.
