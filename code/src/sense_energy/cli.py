"""Command line interface.

sense-energy build-dataset --config code/configs/data.yaml
sense-energy build-features --config code/configs/features.yaml
sense-energy train --config code/configs/model_baseline.yaml
sense-energy evaluate --config code/configs/model_baseline.yaml
"""

from __future__ import annotations

import click
import pandas as pd

from .config import GEO_DIR, PROCESSED_DIR, REPORTS_DIR, load_config
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


@cli.command("fetch-weather")
@click.option("--config", "config_path", default="code/configs/weather.yaml", show_default=True)
@click.option(
    "--product", "products", multiple=True, help="Limit to product type(s); default all in config."
)
def fetch_weather_cmd(config_path: str, products: tuple[str, ...]) -> None:
    """Pull ERA5 weather from CDS into code/data/external/weather/. Resumable."""
    from .data.weather import fetch_all

    paths = fetch_all(load_config(config_path), list(products) or None)
    click.echo(f"{len(paths)} product-months present")


@cli.command("build-weather")
@click.option("--config", "config_path", default="code/configs/weather.yaml", show_default=True)
@click.option("--product", "product", default="reanalysis", show_default=True)
def build_weather_cmd(config_path: str, product: str) -> None:
    """Extract nearest-grid-point weather for every geocoded site into interim/."""
    from .data.weather import build_weather_sites

    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    click.echo(build_weather_sites(load_config(config_path), sites, product_type=product))


@cli.command("harvest-forecasts")
@click.option("--config", "config_path", default="code/configs/forecasts.yaml", show_default=True)
def harvest_forecasts_cmd(config_path: str) -> None:
    """Harvest every AIFS-ENS run in ECMWF open data not yet on disk (run daily)."""
    from .data.forecasts import harvest_available

    paths = harvest_available(load_config(config_path))
    click.echo(f"{len(paths)} runs on disk")


@cli.command("build-forecasts")
@click.option("--config", "config_path", default="code/configs/forecasts.yaml", show_default=True)
def build_forecasts_cmd(config_path: str) -> None:
    """Extract nearest-grid-point AIFS-ENS forecasts for every geocoded site into interim/."""
    from .data.forecasts import build_forecast_sites

    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    click.echo(build_forecast_sites(load_config(config_path), sites))


@cli.command("fetch-prices")
@click.option("--config", "config_path", default="code/configs/prices.yaml", show_default=True)
def fetch_prices_cmd(config_path: str) -> None:
    """Pull Octopus Agile (day-ahead-known) and Elexon MID prices; map sites to GSP groups."""
    from .data.prices import build_prices, map_sites_to_gsp

    for name, path in build_prices(load_config(config_path)).items():
        click.echo(f"{name:<12} {path}")

    geo_path = GEO_DIR / "sites_geo.parquet"
    if geo_path.exists():
        sites = pd.read_parquet(geo_path)
        if "gsp_group" not in sites.columns:
            map_sites_to_gsp(sites).to_parquet(geo_path, index=False)
            click.echo(f"{'gsp_group':<12} added to {geo_path}")


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
