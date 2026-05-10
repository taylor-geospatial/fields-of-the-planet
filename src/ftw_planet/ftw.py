"""Indexing helpers for the Fields of the World (FTW) dataset.

After ``ftw data download --countries <c> --out <root>`` the layout looks like:

    <root>/<country>/
        s2_images/window_a/<id>.tif
        s2_images/window_b/<id>.tif
        label_masks/semantic_2class/<id>.tif
        label_masks/semantic_3class/<id>.tif
        label_masks/instance/<id>.tif
        chips_<country>.parquet
        data_config_<country>.json    # season windows live here in v2

FTW v2 changes vs v1:
- chips parquet only carries (aoi_id, geometry, split) — no per-chip dates.
  Dates come from `data_config_<country>.json` as season ranges (start/end).
- S2 chips are stored in EPSG:4326. We pick a UTM zone per chip centroid as
  the target projection for the matched PlanetScope output, so downstream
  training has a meter-aligned grid at ~3 m.
"""

import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import geopandas as gpd
import rasterio
from shapely.geometry import box, mapping

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry


@dataclass
class Patch:
    """One FTW patch (one AOI / chip) with both seasonal windows."""

    id: str
    country: str
    win_a_path: Path
    win_b_path: Path
    mask_path: Path | None
    # Target acquisition date — midpoint of the FTW season window. Real S2
    # scene-level dates are not exposed by FTW v2 per-chip.
    win_a_date: str
    win_b_date: str
    win_a_range: tuple[str, str]  # (start, end) season window from data_config
    win_b_range: tuple[str, str]
    crs: str  # FTW chip CRS — typically "EPSG:4326"
    target_crs: str  # UTM zone for PlanetScope output, e.g. "EPSG:32735"
    transform: tuple[float, float, float, float, float, float]  # rasterio affine, flat
    width: int
    height: int
    bounds: tuple[float, float, float, float]  # in image CRS
    bounds_4326: tuple[float, float, float, float]  # lon/lat
    geometry_4326: dict  # GeoJSON-ish dict; ready to send to Planet Data API

    def to_record(self) -> dict:
        return {
            "id": self.id,
            "country": self.country,
            "win_a_path": str(self.win_a_path),
            "win_b_path": str(self.win_b_path),
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "win_a_date": self.win_a_date,
            "win_b_date": self.win_b_date,
            "win_a_range": list(self.win_a_range),
            "win_b_range": list(self.win_b_range),
            "crs": self.crs,
            "target_crs": self.target_crs,
            "transform": list(self.transform),
            "width": self.width,
            "height": self.height,
            "bounds": list(self.bounds),
            "bounds_4326": list(self.bounds_4326),
            "geometry_4326": self.geometry_4326,
        }


@dataclass
class FTWCountry:
    """Lazy index of a downloaded FTW country."""

    country: str
    root: Path
    parquet_path: Path = field(init=False)
    config_path: Path = field(init=False)

    def __post_init__(self) -> None:
        country_dir = self.root / self.country
        candidates = list(country_dir.glob("chips_*.parquet"))
        if not candidates:
            raise FileNotFoundError(
                f"No chips_*.parquet under {country_dir} — did you run "
                f"`ftw data download --countries {self.country} --out {self.root.parent}`?"
            )
        self.parquet_path = candidates[0]
        cfgs = list(country_dir.glob("data_config_*.json"))
        if not cfgs:
            raise FileNotFoundError(f"No data_config_*.json under {country_dir}")
        self.config_path = cfgs[0]

    def patches(self) -> list[Patch]:
        gdf = gpd.read_parquet(self.parquet_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)

        country_dir = self.root / self.country
        s2_dir = country_dir / "s2_images"
        mask_dir = country_dir / "label_masks"

        with self.config_path.open() as f:
            cfg = json.load(f)
        a_range, b_range = _country_season_ranges(cfg)
        a_mid = _midpoint_iso(*a_range)
        b_mid = _midpoint_iso(*b_range)

        id_col = _pick_col(gdf, ["aoi_id", "id"])

        patches: list[Patch] = []
        for _, row in gdf.iterrows():
            chip_id = str(row[id_col])
            win_a = _resolve_image(s2_dir, "window_a", chip_id)
            win_b = _resolve_image(s2_dir, "window_b", chip_id)
            if win_a is None or win_b is None:
                continue  # missing one window — skip
            mask = _resolve_mask(mask_dir, chip_id)

            with rasterio.open(win_a) as src:
                crs = src.crs.to_string()
                transform = tuple(src.transform)[:6]
                width, height = src.width, src.height
                bounds = tuple(src.bounds)  # (minx, miny, maxx, maxy) in image CRS

            geom_4326: BaseGeometry = row.geometry
            if geom_4326 is None or geom_4326.is_empty:
                geom_4326 = box(*bounds)
            bounds_4326 = tuple(geom_4326.bounds)
            target_crs = "EPSG:3857"  # global meter-grid for all patches

            patches.append(
                Patch(
                    id=chip_id,
                    country=self.country,
                    win_a_path=win_a,
                    win_b_path=win_b,
                    mask_path=mask,
                    win_a_date=a_mid,
                    win_b_date=b_mid,
                    win_a_range=a_range,
                    win_b_range=b_range,
                    crs=crs,
                    target_crs=target_crs,
                    transform=transform,
                    width=width,
                    height=height,
                    bounds=bounds,
                    bounds_4326=bounds_4326,
                    geometry_4326=mapping(geom_4326),
                )
            )
        return patches


def _country_season_ranges(cfg: dict) -> tuple[tuple[str, str], tuple[str, str]]:
    """Return (window_a_range, window_b_range) for a country.

    FTW v2 ``data_config_<country>.json`` ships ``seasons`` as either:
      - dict ``{"window_a": {start,end}, "window_b": {...}}`` (one country-wide schedule)
      - list of per-grid dicts ``[{"window_a": {...}, "window_b": {...}}, ...]``

    For the list form we take the **outer envelope** (earliest start, latest end)
    across all grids, so the search window covers every patch in the country
    regardless of which grid it belongs to.
    """
    seasons = cfg["seasons"]
    if isinstance(seasons, dict):
        items: list[dict] = [seasons]
    else:
        items = list(seasons)
    a_starts = [it["window_a"]["start"] for it in items]
    a_ends = [it["window_a"]["end"] for it in items]
    b_starts = [it["window_b"]["start"] for it in items]
    b_ends = [it["window_b"]["end"] for it in items]
    return (min(a_starts), max(a_ends)), (min(b_starts), max(b_ends))


def _midpoint_iso(start: str, end: str) -> str:
    s = date.fromisoformat(start)
    e = date.fromisoformat(end)
    mid_days = (e - s).days // 2
    return (s + timedelta(days=mid_days)).isoformat()


def _pick_col(gdf: gpd.GeoDataFrame, names: list[str]) -> str:
    for n in names:
        if n in gdf.columns:
            return n
    raise KeyError(f"None of {names} found in {list(gdf.columns)}")


def _resolve_image(s2_dir: Path, window: str, chip_id: str) -> Path | None:
    for cand in (
        s2_dir / window / f"{chip_id}.tif",
        s2_dir / f"{chip_id}_{window[-1]}.tif",  # window_a -> _a
    ):
        if cand.exists():
            return cand
    return None


def _resolve_mask(mask_dir: Path, chip_id: str) -> Path | None:
    for sub in ("semantic_2class", "2-class", "semantic_3class", "3-class", "instance"):
        cand = mask_dir / sub / f"{chip_id}.tif"
        if cand.exists():
            return cand
    return None


def write_index(patches: list[Patch], path: Path) -> None:
    """Write a JSONL index — one patch per line, easy to stream in match_planet.py."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for p in patches:
            f.write(json.dumps(p.to_record()) + "\n")


def read_index(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]
