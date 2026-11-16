"""Command line interface.

sense-energy build-dataset --config code/configs/data.yaml
sense-energy build-features --config code/configs/features.yaml
sense-energy train --config code/configs/model_baseline.yaml
sense-energy evaluate --config code/configs/model_baseline.yaml
"""

from __future__ import annotations

import json

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
    """Harvest every AIFS-ENS run in ECMWF open data not yet on disk (cron harvest retired 2026-11-09)."""
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


@cli.command("fetch-tigge")
@click.option("--config", "config_path", default="code/configs/tigge.yaml", show_default=True)
@click.option(
    "--type",
    "types",
    multiple=True,
    help="control_forecast and/or perturbed_forecast; default both.",
)
@click.option(
    "--until-complete",
    is_flag=True,
    help="Keep passing (with a pause) until every month is extracted.",
)
def fetch_tigge_cmd(config_path: str, types: tuple[str, ...], until_complete: bool) -> None:
    """Pull IFS ENS historical forecasts from TIGGE via the ECMWF Data Store. Resumable."""
    from .data.tigge import fetch_all, fetch_until_complete

    run = fetch_until_complete if until_complete else fetch_all
    paths = run(load_config(config_path), list(types) or None)
    click.echo(f"{len(paths)} type-months present")


@cli.command("build-tigge")
@click.option("--config", "config_path", default="code/configs/tigge.yaml", show_default=True)
def build_tigge_cmd(config_path: str) -> None:
    """Extract nearest-grid-point TIGGE forecasts for every geocoded site into interim/."""
    from .data.tigge import build_forecast_sites

    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet")
    click.echo(build_forecast_sites(load_config(config_path), sites))


@cli.command("fetch-neso")
@click.option("--config", "config_path", default="code/configs/neso.yaml", show_default=True)
@click.option("--overwrite", is_flag=True)
def fetch_neso_cmd(config_path: str, overwrite: bool) -> None:
    """Download FES tables, GSP boundaries and national demand from the NESO data portal."""
    from .data.neso import fetch

    click.echo(f"{len(fetch(load_config(config_path), overwrite=overwrite))} files present")


@cli.command("build-neso")
@click.option("--config", "config_path", default="code/configs/neso.yaml", show_default=True)
def build_neso_cmd(config_path: str) -> None:
    """Tidy the NESO tables into interim/ and stamp sites with their GSP region."""
    from .data.neso import build

    for name, path in build(load_config(config_path)).items():
        click.echo(f"{name:<40} {path}")


@cli.command("build-processed")
@click.option("--config", "config_path", default="code/configs/processed.yaml", show_default=True)
def build_processed_cmd(config_path: str) -> None:
    """Write clean analysis-ready datasets (elec/gas x meter/site/trust, raw + imputed) and the data manifest."""
    from .data.processed import build

    record = build(load_config(config_path))
    click.echo(f"{len(record.files)} files written; manifest at doc/data_manifest.md")


@cli.command("build-neso-covariates")
@click.option(
    "--config", "config_path", default="code/configs/neso_covariates.yaml", show_default=True
)
def build_neso_covariates_cmd(config_path: str) -> None:
    """Half-hourly NESO covariates (actuals and day-ahead-known) and static FES site attributes."""
    from .data.neso_covariates import build

    out = build(load_config(config_path))
    click.echo(f"{out['halfhourly']}\n{out['site']}\n{len(out['columns'])} columns")


@cli.command("build-mobility")
def build_mobility_cmd() -> None:
    """Tidy the ESC mobile-visitation dataset (one trust, 2024) into interim/."""
    from .data.mobility import build

    for name, path in build().items():
        click.echo(f"{name:<10} {path}")


@cli.command("fetch-isd")
@click.option("--year", "years", multiple=True, type=int, help="Years to pull; default 2022-2026.")
def fetch_isd_cmd(years: tuple[int, ...]) -> None:
    """NOAA ISD hourly station files for the stations nearest the sites (open data, no key)."""
    from .data import isd

    sites = pd.read_parquet(GEO_DIR / "sites_geo.parquet").dropna(subset=["latitude", "longitude"])
    cand = isd.candidates(sites, isd.station_table())
    ids = sorted(cand["station_id"].unique())
    click.echo(f"{len(ids)} stations serve {cand['site_code'].nunique()} sites")
    click.echo(str(isd.fetch(list(years) or list(range(2022, 2027)), ids)))


@cli.command("build-isd")
def build_isd_cmd() -> None:
    """Station hourly table, nearest-station hourly series per site and the ISD daily table."""
    from .data.isd import build

    out = build()
    click.echo(f"{len(out):,} site-days, {out['site_code'].nunique()} sites")


@cli.command("build-weather-daily")
@click.option(
    "--product", default="reanalysis", show_default=True, help="interim/weather_<product>.parquet"
)
@click.option("--suffix", default="", help="Output interim/weather_daily<suffix>.parquet")
def build_weather_daily_cmd(product: str, suffix: str) -> None:
    """Daily degree days, hot hours, humidity, solar, wind, rain and snow per site from ERA5."""
    from .data.weather_daily import build

    out = build(product=product, suffix=suffix)
    click.echo(f"{len(out):,} site-days, {out['site_code'].nunique()} sites")


@cli.command("gpu-check")
@click.option("--n-series", default=256, show_default=True)
def gpu_check_cmd(n_series: int) -> None:
    """Show CUDA devices and benchmark each foundation model on the selected GPU."""
    from .compute import gpu_inventory, select_device
    from .models.foundation import benchmark

    for d in gpu_inventory():
        click.echo(f"{d.device}: {d.name}  free {d.free_gb:.1f}/{d.total_gb:.1f} GB  bf16={d.bf16}")
    device = select_device()
    for r in benchmark(n_series=n_series, device=device):
        click.echo(
            f"{r.model:<26} {r.device:<8} {r.n_series:>4} series x {r.context} ctx -> {r.horizon} steps: {r.seconds:6.2f} s  ({r.series_per_second:7.1f} series/s)"
        )


@cli.command("harvest-ifs-ens")
@click.option("--config", "config_path", default="code/configs/ifs_ens.yaml", show_default=True)
@click.option(
    "--until-complete", is_flag=True, help="Keep re-passing until every run is extracted."
)
@click.option(
    "--start",
    default=None,
    help="Override the config start date (YYYY-MM-DD) so several processes can split the range.",
)
@click.option("--end", default=None, help="Override the config end date (YYYY-MM-DD).")
def harvest_ifs_ens_cmd(
    config_path: str, until_complete: bool, start: str | None, end: str | None
) -> None:
    """Harvest IFS ENS (51 members) from the AWS mirror of ECMWF open data, by byte range. Resumable."""
    from .data.ifs_ens import harvest

    config = load_config(config_path)
    if start:
        config["start"] = start
    if end:
        config["end"] = end
    click.echo(f"{harvest(config, until_complete=until_complete)} runs skipped or failed")


@cli.command("build-ifs-ens")
@click.option("--config", "config_path", default="code/configs/ifs_ens.yaml", show_default=True)
def build_ifs_ens_cmd(config_path: str) -> None:
    """Concatenate IFS ENS run extracts into monthly interim parquet files."""
    from .data.ifs_ens import build_forecast_sites

    for path in build_forecast_sites(load_config(config_path)):
        click.echo(path)


@cli.command("run-poc")
@click.option(
    "--config", "config_path", default="code/configs/experiments/poc.yaml", show_default=True
)
@click.option(
    "--model", "models", multiple=True, help="Model(s) to run; default: all in the config."
)
@click.option("--score-only", is_flag=True, help="Only re-score the forecasts already on disk.")
@click.option(
    "--level",
    default=None,
    type=click.Choice(["meter", "site", "trust", "region", "total"]),
    help="Hierarchy level (default: the config's); outputs go to <outputs_dir>_<level>.",
)
def run_poc_cmd(
    config_path: str, models: tuple[str, ...], score_only: bool, level: str | None
) -> None:
    """Run the day-ahead proof of concept and print pooled scores."""
    from .experiments.runner import run_models, score_all

    config = load_config(config_path)
    if level:
        config["level"] = level
    if not score_only:
        run_models(config, list(models) or list(config["models"]))
    click.echo(score_all(config).round(3).to_string(index=False))


@cli.command("poc-reconcile")
@click.option(
    "--config", "config_path", default="code/configs/experiments/poc.yaml", show_default=True
)
@click.option("--model", "models", multiple=True, required=True, help="Base-forecast model(s).")
@click.option("--method", "methods", multiple=True, help="Reconciliation method(s); default all.")
def poc_reconcile_cmd(config_path: str, models: tuple[str, ...], methods: tuple[str, ...]) -> None:
    """Reconcile the per-level base forecasts of a model and score every level."""
    from .experiments.reconciliation import run

    out = run(load_config(config_path), list(models), list(methods) or None)
    out = out[out["model"].isin(models)]
    table = out.pivot_table(index=["model", "method"], columns="level", values="nmae")
    click.echo((table[["meter", "site", "trust", "region", "total"]] * 100).round(2).to_string())


@cli.command("fetch-nhs-activity")
def fetch_nhs_activity_cmd() -> None:
    """Public NHS activity statistics (A&E monthly, AmbSYS, KH03 beds) as trust covariates."""
    from .data.nhs_activity import fetch_all

    out = fetch_all()
    click.echo(
        out[
            ["organisation_name", "ods_code", "n_sites", "in_ae", "in_ambsys", "in_kh03"]
        ].to_string(index=False)
    )


@cli.command("estates-analysis")
def estates_analysis_cmd() -> None:
    """ERIC 2023/24 and KH03 beds: meter coverage flags, heat-pump headroom, intensity."""
    from .analysis.estates import run

    out = run()
    click.echo(out["flags"]["coverage_flag"].value_counts().to_string())
    h, e = out["headroom_main"], out["england"]
    click.echo(out["headroom"]["headroom_status"].value_counts().to_string())
    click.echo(
        f"heat-pump headroom: {int((h['util_hp'] > 1).sum())} of {len(h)} panel sites exceed "
        f"capacity; England {float((e['util_hp'] > 1).mean()):.0%} of {len(e)} sites"
    )


@cli.command("cooling-analysis")
def cooling_analysis_cmd() -> None:
    """Per-site electricity response to cooling degree days (month fixed effects, HAC)."""
    from .analysis.cooling import run

    s = run()["summary"]
    for group in ("no_chp", "chp"):
        g = s[group]
        click.echo(
            f"{group}: {g['sites']} sites, median {g['cdd_pct_q10_q50_q90'][1]} % per degC day, "
            f"{g['significant_positive']} significantly positive"
        )


@cli.command("weather-events")
def weather_events_cmd() -> None:
    """Catalogue heatwaves, cold spells, high wind, heavy rain and snow at the sites (ERA5)."""
    from .analysis.weather_events import run

    s = run()["summary"]
    click.echo(f"national events: {s['national_events']}")
    click.echo(f"site events: {s['site_events']}")


@cli.command("weather-response")
def weather_response_cmd() -> None:
    """Nonlinear demand response to temperature (binned, fixed effects), by hour, persistence."""
    from .analysis.weather_response import run

    s = run()["summary"]
    click.echo(f"{s['site_days']:,} site-days; sites {s['sites']}")
    for k, v in s["pct_ci"].items():
        if "daily mean" in k:
            click.echo(f"{k}: {v[0]:+.1f} % [{v[1]:+.1f}, {v[2]:+.1f}]")


@cli.command("chp-profile")
def chp_profile_cmd() -> None:
    """How sites with fossil-fuel CHP differ: import volatility, jump days, forecast error."""
    from .analysis.chp import run

    click.echo(run()["summary"].round(3).to_string(index=False))


@cli.command("weather-causal")
@click.option("--n-perm", default=200, show_default=True, help="Randomisation-inference draws.")
def weather_causal_cmd(n_perm: int) -> None:
    """Quasi-experimental weather effects: nested fixed effects, stacked event studies, placebos."""
    from .analysis.weather_causal import run

    s = run(n_perm=n_perm)["summary"]
    for k, v in s["twfe_pct_per_degree"].items():
        if k.startswith("no CHP"):
            click.echo(f"{k}: {v[0]:+.2f} % per degC [{v[1]:+.2f}, {v[2]:+.2f}]")
    for r in s["event_per_degree"]:
        click.echo(
            f"{r['type']} | {r['group']}: ATT {r['demand_att_pct']:+.1f} %, {r['pct_per_c']:+.2f} % per degC"
        )
    click.echo(f"randomisation p: { {k: v['p_two_sided'] for k, v in s['randomisation'].items()} }")


@cli.command("thermometers")
def thermometers_cmd() -> None:
    """Compare site thermometers (ERA5, ERA5-Land, stations) and re-estimate the weather effects."""
    from .analysis.thermometers import run

    s = run()["summary"]
    click.echo(f"{s['common_site_days']:,} common site-days; agreement {s['agreement_medians']}")
    for k, v in s["effects_pct_per_c"].items():
        if "| D |" in k:
            click.echo(f"{k}: {v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]")


@cli.command("weather-heterogeneity")
def weather_heterogeneity_cmd() -> None:
    """Site-level cold and heat slopes, their true spread, moderators and hour-of-day by use."""
    from .analysis.weather_heterogeneity import run

    s = run()["summary"]
    for dose, r in s["random_effects"].items():
        click.echo(
            f"{dose}: pooled {r['pooled_pct']:+.2f} % per degC, true SD between sites {r['tau_pct']:.2f}, "
            f"I2 {r['i2']:.2f}; {r['sites_significant_positive']} sites > 0, {r['sites_significant_negative']} < 0"
        )
    click.echo(f"moderator joint p: {s['moderator_joint_p']}")


@cli.command("build-site-context")
def build_site_context_cmd() -> None:
    """Rural-urban class, LSOA population density and REPD solar schemes for every site."""
    from .data.site_context import build

    out = build()
    click.echo(out["urban"].value_counts(dropna=False).to_string())


@cli.command("build-eric")
@click.option("--year", default="2024_25", show_default=True, help="2023_24, 2024_25 or 2025_26")
def build_eric_cmd(year: str) -> None:
    """Load an ERIC collection (site and trust tables) into interim/eric_<year>_*.parquet."""
    from .data.eric import build

    site, trust = build(year=year)
    click.echo(f"ERIC {year}: {len(site)} sites, {len(trust)} trusts")


@cli.command("build-dec")
@click.option(
    "--no-fetch", is_flag=True, help="Rebuild the tables from the cached certificates only."
)
def build_dec_cmd(no_fetch: bool) -> None:
    """Display Energy Certificates at every site postcode: fetch (cached), flatten, aggregate."""
    from .data.dec import build

    certs, sites = build(fetch=not no_fetch)
    click.echo(
        f"{len(certs)} certificates; {int((sites['dec_buildings'] > 0).sum())} sites with a DEC"
    )


@cli.command("build-epc")
@click.option("--no-fetch", is_flag=True, help="Rebuild from the cached certificates only.")
def build_epc_cmd(no_fetch: bool) -> None:
    """Non-domestic EPCs at every site postcode: fetch (cached), flatten, aggregate."""
    from .data.dec import build_epc

    certs, sites = build_epc(fetch=not no_fetch)
    click.echo(f"{len(certs)} EPCs; {int((sites['epc_buildings'] > 0).sum())} sites with an EPC")


@cli.command("heat-causal-ml")
def heat_causal_ml_cmd() -> None:
    """R-learner and causal forest on the heat response, cross-fitted by site."""
    from .analysis.heat_causal_ml import run

    s = run()["summary"]
    for k in ("observables", "observables + signature"):
        click.echo(f"{k}: forest {s[k]['forest']}")


@cli.command("train-jepa")
@click.option("--steps", default=4000, show_default=True)
@click.option("--device", default=None, help="cuda:0, cuda:1 or cpu (default: first GPU)")
@click.option(
    "--weather",
    is_flag=True,
    help="Condition the day tokens and the predictor on the day's weather.",
)
def train_jepa_cmd(steps: int, device: str | None, weather: bool) -> None:
    """Self-supervised day/site embeddings of the demand panel (JEPA); embeddings and surprise."""
    from .experiments.jepa import run

    s = run({"steps": steps, "weather": weather}, device)["summary"]
    click.echo(
        f"JEPA: {s['sites']} sites, {s['days_embedded']} days embedded, final loss {s['final_loss']:.4f}"
    )


@cli.command("jepa-probe")
@click.option("--device", default=None)
def jepa_probe_cmd(device: str | None) -> None:
    """Probe the JEPA, Chronos-2 and descriptor representations of each site for the heat slope."""
    from .analysis.jepa_probe import run

    s = run(device)["summary"]
    for k, v in s["heat_share_true_variance"].items():
        click.echo(f"{k}: share of true heat variance {v:.2f}")


@cli.command("build-docs")
def build_docs_cmd() -> None:
    """Build doc/html/ from the write-ups with tables, figures and charts embedded."""
    from .docs.build import build

    pages = build()
    click.echo(f"{len(pages)} pages written to doc/html/")


@cli.command("train-latent-pfn")
@click.option("--steps", default=3000, show_default=True)
@click.option("--device", default=None)
@click.option("--no-train", is_flag=True, help="Forecast and score with the saved models.")
def train_latent_pfn_cmd(steps: int, device: str | None, no_train: bool) -> None:
    """Latent PFN on the weather-conditioned JEPA: in-context latent forecasting, decoded and scored day-ahead."""
    from .experiments.latent_pfn import run

    s = run({"steps": steps}, device, retrain=not no_train)["summary"]
    click.echo(f"nMAE on 28 C+ days: {s['nmae_28C_plus']}")


@cli.command("hourly-response")
@click.option("--no-jepa", is_flag=True, help="Skip the JEPA counterfactual check.")
@click.option("--device", default=None)
def hourly_response_cmd(no_jepa: bool, device: str | None) -> None:
    """Per-site 24-hour heat, cold and sunshine response, and the JEPA counterfactual check."""
    from .analysis.hourly_response import run

    s = run(with_jepa=not no_jepa, device=device)["summary"]
    click.echo(
        f"{s['sites']} sites; median red-band share of the heat response {s['median_redband_share']}"
    )


@cli.command("installation-savings")
def installation_savings_cmd() -> None:
    """Weather-normalised change in electricity at the sites whose estates returns report changes."""
    from .analysis.installation_savings import run

    s = run()["summary"]
    click.echo(f"{s['sites_with_nac_both_years']} sites; LED per 10 points: {s['led_per_10pts']}")


@cli.command("estate-counterfactuals")
def estate_counterfactuals_cmd() -> None:
    """Warming stress test against capacity and the attribution of the 2025 heatwaves."""
    from .analysis.estate_counterfactuals import run

    s = run()["summary"]
    click.echo(json.dumps(s["heatwaves_2025"], indent=1))


@cli.command("fetch-era5-timeseries")
@click.option("--start", default="1960-01-01", show_default=True)
@click.option("--end", default="2022-12-31", show_default=True)
@click.option("--workers", default=4, show_default=True)
def fetch_era5_timeseries_cmd(start: str, end: str, workers: int) -> None:
    """Hourly ERA5 temperature since 1960 at the panel's grid cells (point time series, resumable)."""
    from .data.era5_timeseries import fetch_all

    click.echo(f"{fetch_all(start, end, workers)} cells failed")


@cli.command("build-era5-long")
def build_era5_long_cmd() -> None:
    """Long hourly and daily temperature tables per site from the ERA5 point time series."""
    from .data.era5_timeseries import build

    click.echo(json.dumps(build(), indent=1))


@cli.command("flexibility")
def flexibility_cmd() -> None:
    """Bill anatomy, battery arbitrage, rooftop solar against the load and the value of a forecast."""
    from .analysis.flexibility import run

    s = run()["summary"]
    click.echo(
        json.dumps(
            {k: s[k] for k in ("bill_2024_25", "battery_gbp_per_kwh_year_2024_25")}, indent=1
        )
    )


@cli.command("heat-pump-field")
def heat_pump_field_cmd() -> None:
    """Gas and electricity changes at sites that report new heat pumps, and the implied COP."""
    from .analysis.heat_pump_field import run

    s = run()["summary"]
    click.echo(
        json.dumps(
            {k: s[k] for k in ("new_heat_pump_sites", "switchers", "implied_cop_all")}, indent=1
        )
    )


@cli.command("design-winter")
def design_winter_cmd() -> None:
    """Demand and heat-pump headroom under every winter since 1960, with the storage it needs."""
    from .analysis.design_winter import run

    s = run()["summary"]
    click.echo(json.dumps({k: s[k] for k in ("check", "heat_pump_sites_above_capacity")}, indent=1))


@cli.command("solar-programme")
def solar_programme_cmd() -> None:
    """The funded hospital solar sites seen from the meter: spring breaks sized in kWp."""
    from .analysis.solar_programme import run

    click.echo(json.dumps(run()["summary"], indent=1))


@cli.command("extreme-events")
def extreme_events_cmd() -> None:
    """Heat waves and cold spells at the sites: the reaction in time and space, and what the
    plant, the certificates and the estates return say about it."""
    from .analysis.extreme_events import run

    click.echo(json.dumps(run()["summary"], indent=1, default=float))


@cli.command("event-audit")
@click.option("--tables-only", is_flag=True, help="Reuse the stored specification grid.")
def event_audit_cmd(tables_only: bool) -> None:
    """Audit of the event findings: 90 sets of analysis choices, placebo events, prediction of
    a site's next event, the temperatures of the days before, and what the plant does."""
    from .analysis.event_audit import run

    out = run(grid=not tables_only)
    click.echo(json.dumps(out["summary"]["plant"], indent=1, default=float))


@cli.command("forecast-record")
def forecast_record_cmd() -> None:
    """Score the whole-record forecasts: by year, on event days, by battery, and the paired
    differences between the stored models."""
    from .analysis.forecast_record import run

    click.echo(json.dumps(run()["summary"], indent=1, default=float))


@cli.command("forecast-probes")
@click.option(
    "--which",
    type=click.Choice(["daily", "context", "mobility", "intraday", "plant", "record"]),
    required=True,
    help="daily: every test day; context: shorter histories; mobility: footfall in context; "
    "intraday: issued on the target day; plant: a flag for on-site generation and the plant "
    "state; record: every day of the meter record, without inputs and with the weather forecast "
    "(uses a GPU if one is visible).",
)
@click.option("--threads", default=12, show_default=True)
@click.option("--limit", default=None, type=int, help="Debug: only the first n origins.")
def forecast_probes_cmd(which: str, threads: int, limit: int | None) -> None:
    """Chronos-2 on the CPU: every test day, shorter histories, or footfall given in context."""
    from .experiments import forecast_probe as fp

    run = {
        "daily": fp.run_daily,
        "context": fp.run_context,
        "mobility": fp.run_mobility,
        "intraday": fp.run_intraday,
        "plant": fp.run_plant,
    }.get(which)
    if which == "record":
        for weather in ("none", "ifs"):
            out = fp.run_record(weather)
            click.echo(f"record, {weather}: {len(out):,} forecast rows")
        return
    out = run(threads, limit)
    click.echo(f"{which}: {len(out):,} forecast rows, models {sorted(out['model'].unique())}")


@cli.command("forecast-anatomy")
def forecast_anatomy_cmd() -> None:
    """Where the day-ahead error sits: site group, kind of day, level and shape, strikes, footfall."""
    from .analysis.forecast_anatomy import run

    s = run()["summary"]
    click.echo(json.dumps({k: s[k] for k in ("stored_best_model", "strike") if k in s}, indent=1))


@cli.command("esdf-check")
@click.option("--chronos/--no-chronos", default=True, help="Run Chronos-2 zero-shot (CPU).")
@click.option("--threads", default=8, show_default=True)
@click.option("--limit", default=None, type=int, help="First n series per region (smoke test).")
@click.option("--tables-only", is_flag=True, help="Rebuild the write-up tables from saved scores.")
def esdf_check_cmd(chronos: bool, threads: int, limit: int | None, tables_only: bool) -> None:
    """Check the heat and cold wave paper (Lu et al. 2026) on its own data and saved forecasts."""
    from .experiments.esdf_check import run, tables

    if tables_only:
        for name, table in tables().items():
            click.echo(f"{name}\n{table.round(2).to_string(index=False)}")
        return
    click.echo(json.dumps(run(chronos=chronos, threads=threads, limit=limit), indent=1))


@cli.command("solar-watch")
def solar_watch_cmd() -> None:
    """Breaks in each site's sunshine response, summer by summer (new solar behind the meter)."""
    from .analysis.solar_watch import run

    s = run()["summary"]
    click.echo(
        f"{s['breaks_down']} downward breaks; {s['eric_solar_sites_flagged']} of {s['eric_solar_sites_in_sample']} reported solar sites flagged"
    )


@cli.command("chp-dispatch")
def chp_dispatch_cmd() -> None:
    """How each CHP plant is dispatched: heat-led, price-following, red-band avoiding."""
    from .analysis.chp_dispatch import run

    s = run()["summary"]
    click.echo(
        f"heat-led {s['heat_led_sites']}, price-following {s['price_following_sites']}, red-band avoiding {s['red_band_avoiding_sites']} of {s['chp_sites']}"
    )


@cli.command("heat-drivers")
def heat_drivers_cmd() -> None:
    """What moves the heat slope: sunshine, time of day, the week, moderators, a typology."""
    from .analysis.heat_drivers import run

    s = run()["summary"]
    tau = s["tau_pct"]
    click.echo(
        f"heat slope: true SD between sites {tau['plain']:.2f} -> "
        f"{tau['sunshine_controlled']:.2f} points with site sunshine terms"
    )
    click.echo(f"types: {s['types']}")
    click.echo(f"variance explained: {s['variance_explained']}")


@cli.command("cluster-demand")
@click.option("--energy", "energies", multiple=True, default=("elec", "gas"), show_default=True)
@click.option(
    "--level", "levels", multiple=True, default=("meter", "site", "trust"), show_default=True
)
@click.option("--k-max", default=8, show_default=True)
def cluster_demand_cmd(energies: tuple[str, ...], levels: tuple[str, ...], k_max: int) -> None:
    """Shape-based clustering of demand series per energy type and level."""
    from .analysis.clustering import run

    for energy in energies:
        for level in levels:
            out = run(energy, level, {"k_max": k_max})
            if out:
                click.echo(f"{energy}/{level}: {len(out['table'])} series, k={out['best_k']}")


@cli.command("poc-hierarchy")
@click.option(
    "--config", "config_path", default="code/configs/experiments/poc.yaml", show_default=True
)
@click.option(
    "--level",
    "levels",
    multiple=True,
    type=click.Choice(["site", "trust", "region", "total"]),
    help="Levels to build bottom-up forecasts for (default: all with lower-level forecasts).",
)
def poc_hierarchy_cmd(config_path: str, levels: tuple[str, ...]) -> None:
    """Bottom-up forecasts from the level below and a summary table across all levels."""
    from .experiments.hierarchy import bottom_up, summary

    config = load_config(config_path)
    for lvl in levels or ("site", "trust", "region", "total"):
        bottom_up(config, lvl)
    out = summary(config)
    cols = ["level", "kind", "model", "sites", "nmae", "crps_q", "coverage_80"]
    click.echo(out[[c for c in cols if c in out.columns]].round(3).to_string(index=False))


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
