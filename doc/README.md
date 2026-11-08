# Documentation

| Document | Purpose |
|---|---|
| [project_brief.md](project_brief.md) | Problem, scope, stakeholders, success criteria |
| [data_dictionary.md](data_dictionary.md) | Every field in every source, with units and provenance |
| [methodology.md](methodology.md) | Modelling approach, validation strategy, metrics |
| [data_governance.md](data_governance.md) | Information governance, sharing, retention |
| [adr/](adr/) | Architecture decision records — one file per significant decision |
| [figures/](figures/) | Diagrams used in these documents |
| [references/](references/) | Papers, standards, external reports |

Documentation here is for humans reasoning about the project. API-level detail
belongs in docstrings in `code/src/sense_energy/`.

## HTML pages

The write-ups are published as HTML pages in `doc/html/` with the result tables, figures and
interactive charts embedded next to the prose (`sense-energy build-docs`, `code/src/sense_energy/docs/build.py`).
Open `doc/html/index.html`. The markdown files in this folder are the page sources; they carry
shortcodes for tables (`{{table: path.csv | cols=... | round=2}}`), charts (`{{chart: path.csv | kind=bar | x=... | y=... | color=...}}`)
and figures (`{{figure: name | caption=...}}`), and any heading or `## Figures` list that names
figures in backticks gets them embedded. Rebuild the pages after an analysis run.
