# sense-energy

Energy demand forecasting for NHS hospitals.

Forecasting site-level electricity (and gas/heat) demand for NHS acute and community
sites, to support procurement, flexibility/DSR participation, and net-zero reporting.

## Repository layout

```
sense-energy/
├── doc/                      # Project documentation (written for humans)
│   ├── data_dictionary.md    # Every field in every source, with units
│   ├── methodology.md        # Modelling approach, validation strategy
│   ├── data_governance.md    # Data sharing, IG, retention
│   ├── adr/                  # Architecture decision records
│   ├── figures/              # Diagrams used in the docs
│   └── references/           # Papers, standards, external reports
│
├── code/                     # Everything executable
│   ├── data/                 # Data tree (git-ignored, see code/data/README.md)
│   │   ├── raw/              # Immutable source data, exactly as received
│   │   ├── external/         # Third-party data (weather, calendars, tariffs)
│   │   ├── interim/          # Cleaned/reshaped intermediates
│   │   └── processed/        # Model-ready tables
│   ├── src/sense_energy/     # The installable Python package
│   │   ├── data/             # Loading, validation, cleaning
│   │   ├── features/         # Feature engineering
│   │   ├── models/           # Baselines + ML models, train/predict
│   │   ├── evaluation/       # Metrics, backtesting, error analysis
│   │   └── visualization/    # Plotting helpers
│   ├── configs/              # YAML configs (data, features, models)
│   ├── notebooks/            # Exploration only - logic graduates into src/
│   ├── scripts/              # One-off / operational scripts
│   ├── tests/                # pytest suite
│   ├── models/               # Serialised trained models (git-ignored)
│   └── reports/              # Generated outputs, metrics, figures
│
├── pyproject.toml            # Package metadata, deps, tool config
├── Makefile                  # Common commands
└── .github/workflows/        # CI
```

## Getting started

```bash
git clone <repo-url> sense-energy
cd sense-energy
make setup                 # venv + editable install + pre-commit hooks
source .venv/bin/activate
cp .env.example .env       # then fill in any API keys
pytest
```

## Foundation models

Chronos-2, TimesFM 3.0 and TabPFN v3 (incl. its time-series checkpoint) are
optional: `pip install -e ".[models]"` after installing a torch build that
matches the GPU driver (see the comment in `pyproject.toml`). Weights live in
the HuggingFace cache (`~/.cache/huggingface/hub`), not in the repo. Use the
idle card: `CUDA_VISIBLE_DEVICES=1`.

## Workflow

```bash
make data        # raw -> interim -> processed
make features    # processed -> model-ready feature table
make train       # fit a model, write to code/models/
make evaluate    # backtest, write metrics/figures to code/reports/
```

Each step is also available directly:

```bash
python -m sense_energy.cli build-dataset --config code/configs/data.yaml
```

## Data

**No data is committed to this repository.** `code/data/` is git-ignored end to end.
Drop source files into `code/data/raw/` and record their provenance in
[doc/data_dictionary.md](doc/data_dictionary.md). See
[doc/data_governance.md](doc/data_governance.md) before sharing anything derived
from NHS site data.

## Conventions

- Logic lives in `code/src/sense_energy/`; notebooks import from it, never the reverse.
- Every stage is driven by a YAML config in `code/configs/` — no hard-coded paths.
- Time series validation is **always** forward-chaining; never random k-fold.
- All timestamps are stored UTC, tz-aware; local time (Europe/London) is a display concern.
