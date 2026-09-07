"""Geography: boundary layers and postcode geocoding.

The NHS extract carries **no coordinates** — ``postcode`` is the only direct
locator, present for 275 of 284 sites (141 of the 146 with half-hourly data).
Mapping therefore needs two external inputs, both fetched here:

**Boundaries** from the ONS Open Geography Portal (Open Government Licence v3,
"Contains OS data © Crown copyright and database right"). Generalised-clipped
(``BGC``) versions are used: full-resolution coastlines are ~100x larger and
indistinguishable at national scale.

**Postcode centroids** from postcodes.io, a free service over ONS open data.
Only postcodes are sent — never consumption. Hospital postcodes are public
information, but see ``doc/data_governance.md`` before extending this to
anything site-level.

Everything written here lands in ``code/data/geo/``, which is git-ignored.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import GEO_DIR
from ..logging_utils import get_logger

logger = get_logger(__name__)

ARCGIS_ROOT = "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services"
POSTCODES_IO_BULK = "https://api.postcodes.io/postcodes"
POSTCODES_IO_TERMINATED = "https://api.postcodes.io/terminated_postcodes"

#: Boundary layers to fetch. BGC = generalised (20 m), clipped to coastline.
BOUNDARY_LAYERS: dict[str, dict[str, str]] = {
    "countries_uk": {
        "service": "Countries_December_2024_Boundaries_UK_BGC",
        "description": "UK country outlines - map background",
        "name_field": "CTRY24NM",
    },
    "nhs_regions_en": {
        "service": "NHS_England_Regions_January_2024_EN_BGC",
        "description": "NHS England regions - matches sites.commissioning_region",
        "name_field": "NHSER24NM",
    },
    "icb_en": {
        "service": "Integrated_Care_Boards_April_2023_EN_BGC",
        "description": "Integrated Care Boards - matches sites.integrated_care_board",
        "name_field": "ICB23NM",
    },
}

#: postcodes.io accepts at most 100 postcodes per bulk request.
BULK_CHUNK = 100


def _get_json(url: str, timeout: int = 120) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def _post_json(url: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def fetch_boundary(key: str, overwrite: bool = False) -> Path:
    """Download one boundary layer as GeoJSON (EPSG:4326) into ``code/data/geo/``.

    Returns the written path. Existing files are reused unless ``overwrite``.
    """
    if key not in BOUNDARY_LAYERS:
        raise KeyError(f"Unknown layer '{key}'. Available: {sorted(BOUNDARY_LAYERS)}")

    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEO_DIR / f"{key}.geojson"
    if out_path.exists() and not overwrite:
        logger.info("%s already present, skipping download", out_path.name)
        return out_path

    service = BOUNDARY_LAYERS[key]["service"]
    url = (
        f"{ARCGIS_ROOT}/{service}/FeatureServer/0/query"
        "?where=1%3D1&outFields=*&outSR=4326&f=geojson"
    )
    logger.info("Fetching %s", service)
    data = _get_json(url)

    features = data.get("features", [])
    if not features:
        raise RuntimeError(f"Layer '{key}' returned no features")
    if data.get("exceededTransferLimit"):
        logger.warning("%s exceeded the transfer limit; some features may be missing", key)
    if all(f.get("geometry") is None for f in features):
        raise RuntimeError(
            f"Layer '{key}' returned no geometry - the service is probably an "
            "'_NC' (no coordinates) lookup table rather than a boundary layer."
        )

    out_path.write_text(json.dumps(data))
    logger.info(
        "Wrote %s (%d features, %.1f MB)",
        out_path.name,
        len(features),
        out_path.stat().st_size / 1e6,
    )
    return out_path


def fetch_all_boundaries(overwrite: bool = False) -> dict[str, Path]:
    """Download every layer in :data:`BOUNDARY_LAYERS`."""
    return {key: fetch_boundary(key, overwrite=overwrite) for key in BOUNDARY_LAYERS}


def _lookup_terminated(postcode: str | None) -> dict[str, Any] | None:
    """Resolve a retired postcode via the terminated-postcodes endpoint."""
    if not postcode:
        return None
    encoded = urllib.parse.quote(postcode)
    try:
        response = _get_json(f"{POSTCODES_IO_TERMINATED}/{encoded}", timeout=30)
    except (urllib.error.URLError, urllib.error.HTTPError):
        return None
    result = response.get("result")
    if result:
        logger.info(
            "%s is a terminated postcode (%s-%s); using its centroid",
            postcode,
            result.get("year_terminated"),
            result.get("month_terminated"),
        )
        result["is_terminated"] = True
    return result


def geocode_postcodes(postcodes: list[str], pause: float = 0.2) -> pd.DataFrame:
    """Look up postcode centroids via postcodes.io, in chunks of 100.

    Returns one row per input postcode with ``latitude``/``longitude`` (NaN when
    the postcode is unknown or terminated), plus the administrative geographies
    the service resolves, which are useful for cross-checking the site table.
    """
    cleaned = [p.strip().upper() for p in postcodes if isinstance(p, str) and p.strip()]
    unique = sorted(set(cleaned))
    logger.info("Geocoding %d unique postcodes", len(unique))

    rows: list[dict[str, Any]] = []
    for start in range(0, len(unique), BULK_CHUNK):
        chunk = unique[start : start + BULK_CHUNK]
        try:
            response = _post_json(POSTCODES_IO_BULK, {"postcodes": chunk})
        except urllib.error.URLError as exc:
            raise RuntimeError(f"postcodes.io request failed: {exc}") from exc

        for item in response.get("result", []):
            query = item.get("query")
            result = item.get("result")
            if result is None:
                # NHS estates data outlives the postcode register: several sites
                # sit on postcodes retired years ago, which still have centroids.
                result = _lookup_terminated(query)
            if result is None:
                rows.append({"postcode": query, "latitude": None, "longitude": None})
                continue
            rows.append(
                {
                    "postcode": query,
                    "postcode_canonical": result.get("postcode"),
                    "latitude": result.get("latitude"),
                    "longitude": result.get("longitude"),
                    "eastings": result.get("eastings"),
                    "northings": result.get("northings"),
                    "country": result.get("country"),
                    "region_ons": result.get("region"),
                    "admin_district": result.get("admin_district"),
                    "postcode_is_terminated": bool(result.get("is_terminated", False)),
                }
            )
        if start + BULK_CHUNK < len(unique):
            time.sleep(pause)

    out = pd.DataFrame(rows)
    n_found = out["latitude"].notna().sum()
    logger.info("Resolved %d of %d postcodes", n_found, len(out))
    return out


def build_site_geometry(sites: pd.DataFrame, overwrite: bool = False) -> Path:
    """Geocode the site dimension and write ``code/data/geo/sites_geo.parquet``.

    Result is one row per site with ``latitude``/``longitude`` added. Cached:
    geocoding is skipped entirely if the file already exists.
    """
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GEO_DIR / "sites_geo.parquet"
    if out_path.exists() and not overwrite:
        logger.info("%s already present, skipping geocoding", out_path.name)
        return out_path

    sites = sites.copy()
    sites["postcode_key"] = sites["postcode"].str.strip().str.upper()
    lookup = geocode_postcodes(sites["postcode_key"].dropna().tolist())

    merged = sites.merge(
        lookup.rename(columns={"postcode": "postcode_key"}), on="postcode_key", how="left"
    ).drop(columns=["postcode_key"])

    missing = merged["latitude"].isna().sum()
    if missing:
        logger.warning("%d of %d sites could not be geocoded", missing, len(merged))

    merged.to_parquet(out_path, index=False)
    logger.info(
        "Wrote %s (%d sites, %d geocoded)",
        out_path.name,
        len(merged),
        int(merged["latitude"].notna().sum()),
    )
    return out_path


def to_shapefile(geojson_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Convert a downloaded GeoJSON layer to an ESRI Shapefile.

    GeoJSON is the native download format and the better archival choice — one
    file, explicit CRS, no 10-character field-name truncation. Shapefiles are
    written for tools that require them.
    """
    import geopandas as gpd

    geojson_path = Path(geojson_path)
    out_path = Path(out_path) if out_path else geojson_path.with_suffix(".shp")
    gdf = gpd.read_file(geojson_path)
    gdf.to_file(out_path, driver="ESRI Shapefile")
    logger.info("Wrote %s (%d features)", out_path.name, len(gdf))
    return out_path


def load_boundary(key: str):
    """Read a downloaded boundary layer as a GeoDataFrame."""
    import geopandas as gpd

    path = GEO_DIR / f"{key}.geojson"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: sense-energy fetch-geo")
    return gpd.read_file(path)


def load_sites_geo():
    """Read the geocoded site table as a GeoDataFrame in EPSG:4326."""
    import geopandas as gpd

    path = GEO_DIR / "sites_geo.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: sense-energy fetch-geo")
    df = pd.read_parquet(path)
    located = df[df["latitude"].notna()]
    return gpd.GeoDataFrame(
        located,
        geometry=gpd.points_from_xy(located["longitude"], located["latitude"]),
        crs="EPSG:4326",
    )
