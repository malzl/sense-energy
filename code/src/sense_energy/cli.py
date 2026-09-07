"""Command line interface.

sense-energy build-dataset --config code/configs/data.yaml
sense-energy build-features --config code/configs/features.yaml
sense-energy train --config code/configs/model_baseline.yaml
sense-energy evaluate --config code/configs/model_baseline.yaml
"""

from __future__ import annotations

import click
import pandas as pd

from .config import PROCESSED_DIR, REPORTS_DIR, load_config
from .logging_utils import setup_logging


@click.group()
@click.option("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
def cli(log_level: str | None) -> None:
    """Energy demand forecasting for NHS hospitals."""
    setup_logging(log_level)


@cli.command("build-interim")
@click.option("--config", "config_path", default="code/configs/data.yaml", show_default=True)
def build_interim_cmd(config_path: str) -> None:
    """Clean the raw extract into the interim tables."""
    from .data.make_dataset import build_interim

    paths = build_interim(load_config(config_path))
    for name, path in paths.items():
        click.echo(f"{name:<12} {path}")


@cli.command("fetch-geo")
@click.option("--overwrite", is_flag=True, help="Re-download even if files exist.")
@click.option("--shapefile", is_flag=True, help="Also write ESRI Shapefiles alongside the GeoJSON.")
def fetch_geo_cmd(overwrite: bool, shapefile: bool) -> None:
    """Download boundary layers and geocode site postcodes into code/data/geo/."""
    from .data import geo
    from .data.make_dataset import load_sites

    for key, path in geo.fetch_all_boundaries(overwrite=overwrite).items():
        click.echo(f"{key:<16} {path}")
        if shapefile:
            click.echo(f"{'':<16} {geo.to_shapefile(path)}")

    click.echo(f"{'sites_geo':<16} {geo.build_site_geometry(load_sites(), overwrite=overwrite)}")


@cli.command("build-features")
@click.option("--config", "config_path", default="code/configs/features.yaml", show_default=True)
def build_features_cmd(config_path: str) -> None:
    """Build the model-ready feature table."""
    from .features.build_features import run

    path = run(load_config(config_path))
    click.echo(f"Wrote {path}")


@cli.command("train")
@click.option(
    "--config", "config_path", default="code/configs/model_baseline.yaml", show_default=True
)
def train_cmd(config_path: str) -> None:
    """Train a model and persist it."""
    from .models.train import train

    path = train(load_config(config_path))
    click.echo(f"Wrote {path}")


@cli.command("evaluate")
@click.option(
    "--config", "config_path", default="code/configs/model_baseline.yaml", show_default=True
)
def evaluate_cmd(config_path: str) -> None:
    """Backtest the configured model and write a metrics CSV."""
    from .evaluation.backtest import backtest
    from .models.train import build_model

    config = load_config(config_path)
    df = pd.read_parquet(PROCESSED_DIR / config.get("input_filename", "features.parquet"))

    results = backtest(
        model_factory=lambda: build_model(config),
        df=df,
        features=config["features"],
        target=config.get("target", "value"),
        n_splits=config["backtest"]["n_splits"],
        test_size=config["backtest"]["test_size"],
        gap=config["backtest"].get("gap", 0),
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"backtest_{config.get('run_name', 'model')}.csv"
    results.to_csv(out_path, index=False)
    click.echo(results.to_string(index=False))
    click.echo(f"\nWrote {out_path}")


if __name__ == "__main__":
    cli()
