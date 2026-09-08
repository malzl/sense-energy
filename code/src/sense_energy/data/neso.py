"""NESO data portal: FES scenario tables, GSP boundaries, national demand.

Everything comes from the CKAN API at ``api.neso.energy`` and is fetched by
resource *name* within a dataset, so a re-run picks up re-issued files. Local
names are clean and stable; the manifest below is the only place they live.

What is here and why:

* **FES tables** - the published Future Energy Scenarios workbook, sheet by
  sheet: ED1 (GB demand summary with annual/peak/minimum by pathway), ES1
  (supply capacity by connection type - "Distributed" is the embedded fleet),
  FLX1 (flexibility), the building blocks (per-GSP counts and capacities), and
  the regional breakdown (peak/AM/PM demand and distributed generation per GSP).
* **GSP boundaries** - Grid Supply Point regions, to place each hospital in its
  GSP and GSP group and so join it to the regional FES pathways.
* **National demand** - NESO's half-hourly historic demand with embedded wind
  and solar, and the archive of its day-ahead embedded generation forecasts,
  both of which are known-beforehand covariates.

Attribution: "Contains NESO open data".
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import EXTERNAL_DIR, GEO_DIR, INTERIM_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

CKAN = "https://api.neso.energy/api/3/action"
TABLES_DIR = EXTERNAL_DIR / "neso"
BOUNDARIES_DIR = GEO_DIR / "neso"

#: local filename -> (dataset id, regex on resource name, target dir). Regexes
#: are matched case-insensitively against the CKAN resource name; ``{y}`` is
#: substituted with a year for the yearly families below.
MANIFEST: dict[str, tuple[str, str, Path]] = {
    "fes_{y}_ed1_demand_summary.csv": (
        "fes-electricity-demand-summary-data-table-ed1",
        r"ED1.*{y}|{y}.*ED1",
        TABLES_DIR,
    ),
    "fes_{y}_es1_supply.csv": (
        "future-energy-scenario-electricity-supply-data-table-es1",
        r"\(ES1\) {y}",
        TABLES_DIR,
    ),
    "fes_{y}_flx1_flexibility.csv": (
        "fes-flexibility-data-table-data-table-flx1",
        r"\(FLX1\) {y}",
        TABLES_DIR,
    ),
    "fes_{y}_building_blocks.csv": (
        "future-energy-scenario-fes-building-block-data",
        r"^FES {y} Building Blocks",
        TABLES_DIR,
    ),
    "fes_{y}_building_block_definitions.csv": (
        "future-energy-scenario-fes-building-block-data",
        r"^Building Block Definitions {y}",
        TABLES_DIR,
    ),
    "fes_{y}_regional_demand.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"Regional Breakdown of {y} FES: Demand \(Active Power\)",
        TABLES_DIR,
    ),
    "fes_{y}_regional_dg_ge_1mw.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"Regional Breakdown of {y} FES: Distributed generation great",
        TABLES_DIR,
    ),
    "fes_{y}_regional_dg_lt_1mw.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"Regional Breakdown of {y} FES: Distributed generation less",
        TABLES_DIR,
    ),
    "fes_{y}_regional_dsr.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"Regional Breakdown of {y} FES: Demand Side Response",
        TABLES_DIR,
    ),
    "fes_{y}_regional_storage.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"Regional Breakdown of {y} FES: Demands? from distributed stor",
        TABLES_DIR,
    ),
    "fes_{y}_gsp_info.csv": (
        "regional-breakdown-of-fes-data-electricity",
        r"^FES {y} Grid Supply Point Info",
        TABLES_DIR,
    ),
    "historic_demand_{y}.csv": (
        "historic-demand-data",
        r"Historic Demand Data {y}$|demanddata_{y}",
        TABLES_DIR,
    ),
    "embedded_forecast_archive_{y}.csv": (
        "embedded-wind-and-solar-forecasts",
        r"Embedded Solar and Wind Forecast Archive {y}",
        TABLES_DIR,
    ),
    "gsp_regions_20260209.zip": (
        "gis-boundaries-for-gb-grid-supply-points",
        r"^GSP Regions 20260209$",
        BOUNDARIES_DIR,
    ),
    "gsp_regions_20220314.geojson": (
        "gis-boundaries-for-gb-grid-supply-points",
        r"^GSP Regions 20220314 \(GeoJSON\)$",
        BOUNDARIES_DIR,
    ),
    "gsp_gnode_lookup_20181031.csv": (
        "gis-boundaries-for-gb-grid-supply-points",
        r"Region Lookup 20181031",
        BOUNDARIES_DIR,
    ),
    "gsp_regions_changelog.txt": (
        "gis-boundaries-for-gb-grid-supply-points",
        r"^GSP Regions changelog$",
        BOUNDARIES_DIR,
    ),
    "tresp_gsp_areas_2025.gpkg": (
        "tresp-demand-pathways",
        r"tRESP Grid Supply Point \(GSP\) Areas",
        BOUNDARIES_DIR,
    ),
}

YEARLY_FAMILIES = {
    "fes": [
        "fes_{y}_ed1_demand_summary.csv",
        "fes_{y}_es1_supply.csv",
        "fes_{y}_flx1_flexibility.csv",
        "fes_{y}_building_blocks.csv",
        "fes_{y}_building_block_definitions.csv",
        "fes_{y}_regional_demand.csv",
        "fes_{y}_regional_dg_ge_1mw.csv",
        "fes_{y}_regional_dg_lt_1mw.csv",
        "fes_{y}_regional_dsr.csv",
        "fes_{y}_regional_storage.csv",
        "fes_{y}_gsp_info.csv",
    ],
    "history": ["historic_demand_{y}.csv"],
    "forecast_archive": ["embedded_forecast_archive_{y}.csv"],
}

#: Scenario code -> FES pathway name, as used in the regional files.
SCENARIO_NAMES = {
    "HE": "Hydrogen Evolution",
    "EE": "Electric Engagement",
    "HT": "Holistic Transition",
    "CF": "Counterfactual",
    "FB": "Falling Behind",
    "LW": "Leading the Way",
    "CT": "Consumer Transformation",
    "ST": "System Transformation",
    "FS": "Falling Short",
}


# --------------------------------------------------------------------------- #
# Fetch
# --------------------------------------------------------------------------- #


def _ckan_json(action: str, **params: Any) -> dict[str, Any]:
    url = f"{CKAN}/{action}?{urllib.parse.urlencode(params)}"
    with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310
        payload = json.load(r)
    if not payload.get("success"):
        raise RuntimeError(f"CKAN {action} failed: {payload.get('error')}")
    return payload["result"]


def resources(dataset: str) -> list[dict[str, Any]]:
    return _ckan_json("package_show", id=dataset)["resources"]


def planned_files(config: dict[str, Any]) -> list[tuple[str, str, str, Path]]:
    """Expand the manifest for the configured years -> (local name, dataset, name regex, dir)."""
    plan = []
    for template, (dataset, pattern, target_dir) in MANIFEST.items():
        if "{y}" not in template:
            plan.append((template, dataset, pattern, target_dir))
            continue
        family = next(f for f, members in YEARLY_FAMILIES.items() if template in members)
        years = {
            "fes": config["fes_years"],
            "history": config["history_years"],
            "forecast_archive": config["forecast_archive_years"],
        }[family]
        for y in years:
            plan.append((template.format(y=y), dataset, pattern.format(y=y), target_dir))
    return plan


def fetch(config: dict[str, Any], overwrite: bool = False) -> list[Path]:
    """Download every planned file not yet present. Returns the paths present afterwards."""
    catalogue: dict[str, list[dict[str, Any]]] = {}
    present: list[Path] = []
    for local, dataset, pattern, target_dir in planned_files(config):
        path = target_dir / local
        if path.exists() and path.stat().st_size > 0 and not overwrite:
            present.append(path)
            continue
        catalogue.setdefault(dataset, resources(dataset))
        match = [
            r for r in catalogue[dataset] if re.search(pattern, r.get("name", "").strip(), re.I)
        ]
        if not match:
            logger.warning("No resource in %s matches /%s/ - skipping %s", dataset, pattern, local)
            continue
        if len(match) > 1:
            match.sort(key=lambda r: r.get("last_modified") or "", reverse=True)
        url = match[0]["url"]
        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s <- %s", local, match[0].get("name"))
        request = urllib.request.Request(url, headers={"User-Agent": "sense-energy/0.1"})
        with urllib.request.urlopen(request, timeout=600) as r, path.open("wb") as fh:  # noqa: S310
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        present.append(path)
    logger.info("%d NESO files present", len(present))
    return present


# --------------------------------------------------------------------------- #
# FES tables -> tidy
# --------------------------------------------------------------------------- #


def _year_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if re.fullmatch(r"\d{4}", str(c).strip())]


def _normalise_headers(frame: pd.DataFrame) -> pd.DataFrame:
    """FES 2023 files say 'Scenario' where 2024+ say 'Pathway'; unify to 'Pathway'."""
    frame.columns = [c.strip() for c in frame.columns]
    renames = {
        c: "Pathway" for c in frame.columns if c in ("Scenario", "FES Scenario", "FES Pathway")
    }
    return frame.rename(columns=renames)


def _melt_years(frame: pd.DataFrame, id_vars: list[str]) -> pd.DataFrame:
    years = _year_columns(frame)
    out = frame.melt(id_vars=id_vars, value_vars=years, var_name="year", value_name="value")
    out["year"] = out["year"].astype(int)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    return out


def load_ed1(fes_year: int) -> pd.DataFrame:
    """Demand summary: GB annual, peak and minimum electricity demand by pathway."""
    frame = _normalise_headers(
        pd.read_csv(TABLES_DIR / f"fes_{fes_year}_ed1_demand_summary.csv", encoding="utf-8-sig")
    )
    frame = frame.rename(
        columns={
            "Aggregation Level": "aggregation_level",
            "Data item": "data_item",
            "Unit": "unit",
            "Pathway": "pathway",
            "Fuel": "fuel",
            "Peak/ Annual/ Minimum": "statistic",
        }
    )
    out = _melt_years(
        frame, ["aggregation_level", "data_item", "unit", "pathway", "fuel", "statistic"]
    )
    out.insert(0, "fes_year", fes_year)
    return out


def peak_demand(ed1: pd.DataFrame) -> pd.DataFrame:
    """Rows of the ED1 table that describe peak demand (winter peak, GW)."""
    mask = ed1["statistic"].astype(str).str.contains("peak", case=False, na=False)
    return ed1[mask].reset_index(drop=True)


def load_es1(fes_year: int) -> pd.DataFrame:
    """Supply: capacity and output by connection type, pathway and technology.

    ``connection == "Distributed"`` is the embedded (distribution-connected) fleet.
    """
    frame = _normalise_headers(
        pd.read_csv(TABLES_DIR / f"fes_{fes_year}_es1_supply.csv", encoding="utf-8-sig")
    )
    frame = frame.rename(
        columns={
            "Connection": "connection",
            "Pathway": "pathway",
            "Variable": "variable",
            "Category": "category",
            "Type": "type",
            "SubType": "subtype",
        }
    )
    out = _melt_years(frame, ["connection", "pathway", "variable", "category", "type", "subtype"])
    out.insert(0, "fes_year", fes_year)
    return out


def load_flx1(fes_year: int) -> pd.DataFrame:
    frame = _normalise_headers(
        pd.read_csv(TABLES_DIR / f"fes_{fes_year}_flx1_flexibility.csv", encoding="utf-8-sig")
    )
    frame = frame.rename(
        columns={
            "Flexibility type": "flexibility_type",
            "Flexibility sub-type": "flexibility_subtype",
            "Data item": "data_item",
            "Unit": "unit",
            "Pathway": "pathway",
            "Fuel": "fuel",
            "Detail": "detail",
        }
    )
    out = _melt_years(
        frame,
        [
            "flexibility_type",
            "flexibility_subtype",
            "data_item",
            "unit",
            "pathway",
            "fuel",
            "detail",
        ],
    )
    out.insert(0, "fes_year", fes_year)
    return out


def load_building_blocks(fes_year: int) -> pd.DataFrame:
    """Per-GSP building blocks (counts, capacities) joined to their definitions."""
    frame = _normalise_headers(
        pd.read_csv(
            TABLES_DIR / f"fes_{fes_year}_building_blocks.csv",
            encoding="utf-8-sig",
            low_memory=False,
        )
    )
    frame = frame.rename(
        columns={
            "Pathway": "pathway",
            "Building Block ID Number": "building_block_id",
            "Unit": "unit",
            "DNO License Area": "dno_licence_area",
            "GSP": "gsp_name",
            "Share of GSP": "share_of_gsp",
            "Comment": "comment",
        }
    )
    out = _melt_years(
        frame,
        ["pathway", "building_block_id", "unit", "dno_licence_area", "gsp_name", "share_of_gsp"],
    )

    defs_path = TABLES_DIR / f"fes_{fes_year}_building_block_definitions.csv"
    if defs_path.exists():
        defs = pd.read_csv(defs_path, encoding="utf-8-sig")
        defs.columns = [c.strip() for c in defs.columns]
        defs = defs.rename(
            columns={
                "Template": "template",
                "Technology": "technology",
                "Building Block ID Number": "building_block_id",
                "Technology Detail": "technology_detail",
                "Units": "def_units",
                "Detail": "detail",
            }
        )
        defs["template"] = defs["template"].astype(str).str.strip()
        out = out.merge(
            defs[["building_block_id", "template", "technology", "technology_detail", "detail"]],
            on="building_block_id",
            how="left",
        )
    out.insert(0, "fes_year", fes_year)
    return out


def load_regional_demand(fes_year: int) -> pd.DataFrame:
    """Peak / AM / PM demand per GSP and scenario (MW). Years arrive as 2 digits."""
    frame = pd.read_csv(TABLES_DIR / f"fes_{fes_year}_regional_demand.csv", encoding="utf-8-sig")
    frame = frame.rename(
        columns={
            "scenario": "scenario",
            "GSP": "gsp_id",
            "DemandPk": "demand_peak_mw",
            "DemandAM": "demand_am_mw",
            "DemandPM": "demand_pm_mw",
            "type": "type",
            "year": "year",
        }
    )
    frame["year"] = frame["year"].astype(int).map(lambda y: y + 2000 if y < 100 else y)
    frame["pathway"] = frame["scenario"].map(SCENARIO_NAMES).fillna(frame["scenario"])
    frame.insert(0, "fes_year", fes_year)
    return frame


def load_regional_distributed_generation(fes_year: int) -> pd.DataFrame:
    """Distributed generation capacity per GSP, technology and scenario, both size bands."""
    frames = []
    for band, name in (
        ("ge_1mw", "fes_{y}_regional_dg_ge_1mw.csv"),
        ("lt_1mw", "fes_{y}_regional_dg_lt_1mw.csv"),
    ):
        path = TABLES_DIR / name.format(y=fes_year)
        if not path.exists():
            continue
        f = pd.read_csv(path, encoding="utf-8-sig")
        f = f.rename(
            columns={
                "etys_location": "gsp_id",
                "capacity": "capacity_mw",
                "wintpk": "winter_peak_mw",
                "summam": "summer_am_mw",
                "summpm": "summer_pm_mw",
            }
        )
        f["size_band"] = band
        frames.append(f)
    if not frames:
        raise FileNotFoundError(f"no regional distributed-generation files for FES {fes_year}")
    out = pd.concat(frames, ignore_index=True)
    out["year"] = out["year"].astype(int).map(lambda y: y + 2000 if y < 100 else y)
    out["pathway"] = out["scenario"].map(SCENARIO_NAMES).fillna(out["scenario"])
    out.insert(0, "fes_year", fes_year)
    return out


def load_gsp_info(fes_year: int) -> pd.DataFrame:
    frame = pd.read_csv(TABLES_DIR / f"fes_{fes_year}_gsp_info.csv", encoding="utf-8-sig")
    frame.columns = [c.strip() for c in frame.columns]
    frame = frame.rename(
        columns={
            "GSP ID": "gsp_id",
            "GSP Group": "gsp_group",
            "Minor FLOP": "minor_flop",
            "Name": "gsp_name",
            "Latitude": "latitude",
            "Longitude": "longitude",
            "Comments": "comments",
        }
    )
    frame["gsp_group"] = frame["gsp_group"].astype(str).str.lstrip("_")
    frame.insert(0, "fes_year", fes_year)
    return frame


# --------------------------------------------------------------------------- #
# National demand and embedded forecasts
# --------------------------------------------------------------------------- #


def _settlement_to_utc(date: pd.Series, period: pd.Series) -> pd.Series:
    """Settlement date + period -> UTC period start.

    Periods count half hours from local midnight (Europe/London), so days with
    clock changes have 46 or 50 periods; the arithmetic below is exact for them.
    """
    local_midnight = pd.to_datetime(date).dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="shift_forward"
    )
    utc_midnight = local_midnight.dt.tz_convert("UTC")
    return utc_midnight + pd.to_timedelta((period.astype(int) - 1) * 30, unit="min")


def load_historic_demand(years: list[int]) -> pd.DataFrame:
    """NESO half-hourly national demand with embedded wind/solar, in UTC."""
    frames = []
    for y in years:
        path = TABLES_DIR / f"historic_demand_{y}.csv"
        if not path.exists():
            logger.warning("historic_demand_%d.csv missing", y)
            continue
        f = pd.read_csv(path)
        f.columns = [c.strip().lower() for c in f.columns]
        f["datetime"] = _settlement_to_utc(
            pd.to_datetime(f["settlement_date"], format="mixed", dayfirst=True),
            f["settlement_period"],
        )
        frames.append(f)
    out = pd.concat(frames, ignore_index=True)
    out = (
        out.dropna(subset=["datetime"])
        .drop_duplicates("datetime")
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    keep = [
        "datetime",
        "settlement_date",
        "settlement_period",
        "nd",
        "tsd",
        "england_wales_demand",
        "embedded_wind_generation",
        "embedded_wind_capacity",
        "embedded_solar_generation",
        "embedded_solar_capacity",
        "non_bm_stor",
        "pump_storage_pumping",
    ]
    return out[
        [c for c in keep if c in out.columns] + [c for c in out.columns if c.endswith("_flow")]
    ]


def load_embedded_forecast_archive(years: list[int]) -> pd.DataFrame:
    """Every NESO embedded wind/solar forecast issued, keyed by issue and target time.

    Files are large (~500 MB per year); pyarrow does the parsing.
    """
    import pyarrow.csv as pacsv

    frames = []
    for y in years:
        path = TABLES_DIR / f"embedded_forecast_archive_{y}.csv"
        if not path.exists():
            logger.warning("embedded_forecast_archive_%d.csv missing", y)
            continue
        logger.info("Parsing %s", path.name)
        table = pacsv.read_csv(path)
        f = table.to_pandas()
        f.columns = [c.strip().lower() for c in f.columns]
        f["datetime"] = _settlement_to_utc(
            pd.to_datetime(f["settlement_date"], utc=True).dt.tz_convert(None),
            f["settlement_period"],
        )
        f["issued_at"] = pd.to_datetime(f["forecast_datetime"], utc=True)
        frames.append(
            f[
                [
                    "issued_at",
                    "datetime",
                    "embedded_wind_forecast",
                    "embedded_wind_capacity",
                    "embedded_solar_forecast",
                    "embedded_solar_capacity",
                ]
            ]
        )
    out = pd.concat(frames, ignore_index=True)
    out["lead_hours"] = (out["datetime"] - out["issued_at"]) / pd.Timedelta(hours=1)
    return out.sort_values(["datetime", "issued_at"]).reset_index(drop=True)


def latest_forecast_before(archive: pd.DataFrame, min_lead_hours: float) -> pd.DataFrame:
    """For each target period, the most recent forecast issued at least ``min_lead_hours`` ahead.

    This is how a day-ahead feature must be built: a forecast issued *after*
    the model's own issue time is future information.
    """
    eligible = archive[archive["lead_hours"] >= min_lead_hours]
    return (
        eligible.sort_values("issued_at")
        .drop_duplicates("datetime", keep="last")
        .sort_values("datetime")
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------- #
# GSP regions -> sites
# --------------------------------------------------------------------------- #


def load_gsp_regions(layer: str = "gsp_regions_20260209"):
    """GSP region polygons in EPSG:4326 with ``gsp_region`` and ``gsp_group``."""
    import geopandas as gpd

    if layer == "gsp_regions_20260209":
        zip_path = BOUNDARIES_DIR / "gsp_regions_20260209.zip"
        extract = BOUNDARIES_DIR / "gsp_regions_20260209"
        if not extract.exists():
            import zipfile

            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract)
        g = gpd.read_file(extract / "Proj_4326" / "GSP_regions_4326_20260209.gpkg")
        g = g.rename(columns={"GSPs": "gsp_region", "GSPGroup": "gsp_group"})
    elif layer == "gsp_regions_20220314":
        g = gpd.read_file(BOUNDARIES_DIR / "gsp_regions_20220314.geojson").to_crs(4326)
        g = g.rename(columns={"GSPs": "gsp_region", "GSPGroup": "gsp_group"})
    else:
        raise KeyError(layer)
    g["gsp_group"] = g["gsp_group"].astype(str).str.lstrip("_")
    return g[["gsp_region", "gsp_group", "geometry"]]


def assign_gsp_regions(
    sites_geo: pd.DataFrame, layer: str = "gsp_regions_20260209"
) -> pd.DataFrame:
    """Point-in-polygon: add ``gsp_region`` and ``gsp_group_neso`` to located sites.

    Sites without coordinates keep NaN. Coastal points that fall just outside
    every polygon are snapped to the nearest region within 5 km.
    """
    import geopandas as gpd

    regions = load_gsp_regions(layer)
    located = sites_geo[sites_geo["latitude"].notna()].copy()
    points = gpd.GeoDataFrame(
        located[["site_code"]],
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs=4326,
    )

    joined = gpd.sjoin(points, regions, how="left", predicate="within").drop(columns="index_right")
    missing = joined["gsp_region"].isna()
    if missing.any():
        near = gpd.sjoin_nearest(
            points[missing].to_crs(27700), regions.to_crs(27700), how="left", max_distance=5000
        )
        joined.loc[missing, ["gsp_region", "gsp_group"]] = near[
            ["gsp_region", "gsp_group"]
        ].to_numpy()
    joined = joined.drop_duplicates("site_code")

    out = sites_geo.merge(
        joined[["site_code", "gsp_region", "gsp_group"]].rename(
            columns={"gsp_group": "gsp_group_neso"}
        ),
        on="site_code",
        how="left",
    )
    logger.info(
        "GSP regions assigned to %d of %d located sites",
        int(out["gsp_region"].notna().sum()),
        len(located),
    )
    if "gsp_group" in out.columns:
        both = out[["gsp_group", "gsp_group_neso"]].dropna()
        disagree = int((both["gsp_group"] != both["gsp_group_neso"]).sum())
        if disagree:
            logger.warning("%d sites: Octopus GSP group differs from NESO boundary group", disagree)
    return out


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #


def build(config: dict[str, Any]) -> dict[str, Path]:
    """Write tidy interim parquets for everything fetched, and stamp sites with GSP regions."""
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    def save(name: str, frame: pd.DataFrame) -> None:
        path = INTERIM_DIR / f"{name}.parquet"
        frame.to_parquet(path, index=False, compression="zstd")
        written[name] = path
        logger.info("Wrote %s (%s rows)", path.name, f"{len(frame):,}")

    years = config["fes_years"]
    for name, loader in [
        ("fes_ed1_demand_summary", load_ed1),
        ("fes_es1_supply", load_es1),
        ("fes_flx1_flexibility", load_flx1),
        ("fes_building_blocks", load_building_blocks),
        ("fes_regional_demand", load_regional_demand),
        ("fes_regional_distributed_generation", load_regional_distributed_generation),
        ("fes_gsp_info", load_gsp_info),
    ]:
        frames = []
        for y in years:
            try:
                frames.append(loader(y))
            except FileNotFoundError as exc:
                logger.warning("%s %d: %s", name, y, exc)
        if frames:
            save(name, pd.concat(frames, ignore_index=True))

    save("national_demand_halfhourly", load_historic_demand(config["history_years"]))
    save(
        "embedded_forecast_archive",
        load_embedded_forecast_archive(config["forecast_archive_years"]),
    )

    geo_path = GEO_DIR / "sites_geo.parquet"
    if geo_path.exists():
        sites = pd.read_parquet(geo_path)
        sites = sites.drop(
            columns=[c for c in ("gsp_region", "gsp_group_neso") if c in sites.columns]
        )
        assign_gsp_regions(
            sites, config.get("gsp_regions_layer", "gsp_regions_20260209")
        ).to_parquet(geo_path, index=False)
        written["sites_geo"] = geo_path
    return written
