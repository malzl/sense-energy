# Notebooks

Exploration only. Anything that gets reused belongs in `code/src/sense_energy/`.

## Naming

`<order>-<initials>-<short-description>.ipynb`, e.g. `01-nm-data-quality.ipynb`.

## Suggested sequence

| Notebook                          | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `01-*-data-quality.ipynb`         | Coverage, gaps, duplicates, outliers per site        |
| `02-*-eda-load-profiles.ipynb`    | Daily/weekly/seasonal profiles, site comparison      |
| `03-*-weather-sensitivity.ipynb`  | Demand vs. temperature, degree-day base selection    |
| `04-*-baseline-models.ipynb`      | Seasonal naive and profile-mean benchmarks           |
| `05-*-model-comparison.ipynb`     | Backtest results across models                       |

## Setup cell

```python
%load_ext autoreload
%autoreload 2

from sense_energy.config import PROCESSED_DIR
from sense_energy.data.make_dataset import load_processed
from sense_energy.visualization import plots

df = load_processed()
```

Outputs are stripped on commit by `nbstripout` — deliberately, since outputs can
contain site-level data.
