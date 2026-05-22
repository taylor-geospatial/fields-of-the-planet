"""FTW-Planet dataset.

Paired PlanetScope SR (window A + window B) + FTW-aligned 3-class label.
Reads from the GeoParquet index at ``<root>/planet/index.parquet`` and
derives train/val/test splits by joining ``patch_id`` against the official
FTW Sentinel-2 ``chips_<country>.parquet`` (aoi_id) so model evaluations
stay comparable across the two modalities.

``FTWPlanetAlignedDataset`` is a sibling dataset that reprojects each Planet
window on-the-fly to the corresponding S2 chip's CRS and bounding box,
reorders bands from PSScene BGR(N) to RGB(N) (matching S2 channel semantics),
and uses the S2 3-class label as the shared mask.  This makes the joint
Planet+S2 training dataset truly spatially aligned per patch_id.
"""

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
import torch
from torch import Tensor
from torchgeo.datasets import NonGeoDataset

PLANET_SR_SCALE = 10000.0  # PlanetScope SR DN -> reflectance


class FTWPlanet(NonGeoDataset):
    """PlanetScope variant of the FTW dataset."""

    valid_splits = ("train", "val", "test")

    def __init__(
        self,
        root: str = "data",
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        load_boundaries: bool = True,
        usable_only: bool = True,
        boundary_dilate_px: int = 0,
        return_sdf: bool = False,
        sdf_clip_px: float = 20.0,
        swap_order: bool = False,
    ) -> None:
        if countries is None:
            raise ValueError("Specify countries to load")
        if isinstance(countries, str):
            countries = [countries]
        countries = [c.lower() for c in countries]
        assert split in self.valid_splits, f"bad split {split}"

        self.root = root
        self.split = split
        self.transforms = transforms
        self.load_boundaries = load_boundaries
        self.boundary_dilate_px = int(boundary_dilate_px)
        self.return_sdf = bool(return_sdf)
        self.sdf_clip_px = float(sdf_clip_px)
        self.swap_order = bool(swap_order)

        planet_root = os.path.join(root, "planet")
        ftw_root = os.path.join(root, "ftw")

        idx = gpd.read_parquet(os.path.join(planet_root, "index.parquet"))
        idx = idx[idx["country"].isin(countries)].copy()
        if usable_only:
            idx = idx[idx["usable_pair"] == True]  # noqa: E712
        idx["patch_id"] = idx["patch_id"].astype(str)

        # Join split assignment from official FTW chips parquet. Some countries
        # store ``aoi_id`` as int and others as str, so coerce both sides.
        import pandas as pd

        sub_frames = []
        for country in countries:
            chips = gpd.read_parquet(os.path.join(ftw_root, country, f"chips_{country}.parquet"))
            chips = chips[chips["split"] == split][["aoi_id"]].rename(
                columns={"aoi_id": "patch_id"}
            )
            chips["patch_id"] = chips["patch_id"].astype(str)
            chips["country"] = country
            sub_frames.append(idx.merge(chips, on=["country", "patch_id"], how="inner"))
        merged = pd.concat(sub_frames, ignore_index=True) if sub_frames else idx.iloc[0:0]

        self.records: list[dict[str, str]] = [
            {
                "country": r["country"],
                "patch_id": r["patch_id"],
                "window_a": os.path.join(planet_root, r["image_a_path"]),
                "window_b": os.path.join(planet_root, r["image_b_path"]),
                "label": os.path.join(planet_root, r["label_path"]),
            }
            for r in merged.to_dict(orient="records")
        ]
        if not self.records:
            raise RuntimeError(f"no samples for split={split} countries={countries}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        r = self.records[index]
        with rasterio.open(r["window_a"]) as src:
            a = src.read().astype(np.float32)
        with rasterio.open(r["window_b"]) as src:
            b = src.read().astype(np.float32)
        with rasterio.open(r["label"]) as src:
            lbl = src.read(1).astype(np.int64)

        # SDF target: distance from each pixel to the nearest boundary pixel.
        # Computed on the ORIGINAL (un-dilated) boundary so it stays a clean
        # geometric quantity. Clipped to ``sdf_clip_px`` so the L1 loss isn't
        # dominated by faraway pixels.
        sdf_target = None
        if self.return_sdf:
            from scipy.ndimage import distance_transform_edt

            sdf_target = distance_transform_edt(lbl != 2).astype(np.float32)
            np.clip(sdf_target, 0.0, self.sdf_clip_px, out=sdf_target)

        # Optional dilation of the boundary class to widen the supervision
        # band. Helps training because a 1-px GT under all_touched
        # rasterization gives near-zero overlap signal when predictions are
        # off by 1 px. We do this here (in the dataset) so the broader
        # boundary is treated as the target everywhere — interior pixels
        # inside the dilated ring become class 2, eating into class 1.
        if self.boundary_dilate_px > 0:
            from scipy.ndimage import binary_dilation

            boundary = lbl == 2
            k = 2 * self.boundary_dilate_px + 1
            struct = np.ones((k, k), dtype=bool)
            dilated = binary_dilation(boundary, structure=struct)
            lbl = lbl.copy()
            lbl[dilated] = 2

        # Source tifs occasionally differ by 1 px in H/W due to reprojection
        # rounding; clip to the common min so the stack lines up.
        h = min(a.shape[1], b.shape[1], lbl.shape[0])
        w = min(a.shape[2], b.shape[2], lbl.shape[1])
        a = a[:, :h, :w]
        b = b[:, :h, :w]
        lbl = lbl[:h, :w]

        # Window B first then A — matches stock FTW datamodule channel order.
        # If swap_order, randomly use [A, B] instead (per-sample, p=0.5).
        if self.swap_order and np.random.rand() < 0.5:
            image = np.concatenate([a, b], axis=0)
        else:
            image = np.concatenate([b, a], axis=0)
        sample: dict[str, Any] = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(lbl if self.load_boundaries else (lbl > 0).astype(np.int64)),
            "country": r["country"],
        }
        if sdf_target is not None:
            sample["sdf"] = torch.from_numpy(sdf_target)
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample


# ---------------------------------------------------------------------------
# Spatially-aligned joint dataset
# ---------------------------------------------------------------------------

# Planet PSScene native band order: [B, G, R, NIR].
# FTW S2 band order:                [R, G, B, NIR].
# Reindex Planet 4-band array so both share [R, G, B, NIR] semantics.
_PLANET_BGR_TO_RGB: list[int] = [2, 1, 0, 3]


class FTWPlanetAlignedDataset:
    """Planet imagery reprojected on-the-fly onto the S2 chip's grid.

    For each ``patch_id`` that has both a Planet SR pair and a corresponding
    S2 chip, this dataset:

    1. Opens S2 window-B to obtain the target CRS and bounding box.
    2. Reprojects each of the two Planet windows (PSScene native UTM) into
       that CRS+bbox at Planet's native GSD (~853 px for a 2560 m S2 chip).
    3. Reorders bands from PSScene BGR(NIR) to RGB(NIR) so the channel
       semantics match the S2 dataset.
    4. Uses the S2 3-class label as the mask (resized to Planet's native pixel
       grid with nearest-neighbour interpolation so class IDs stay exact).

    The output dict mirrors :class:`FTWPlanet`:
    ``{image: Tensor(8, T, T), mask: Tensor(T, T), country: str}``.

    ``S2 layout`` (ftw_tools convention)::

        <s2_root>/<country>/s2_images/window_a/<patch_id>.tif
        <s2_root>/<country>/s2_images/window_b/<patch_id>.tif
        <s2_root>/<country>/label_masks/semantic_3class/<patch_id>.tif
    """

    valid_splits = ("train", "val", "test")

    def __init__(
        self,
        planet_root: str = "data",
        s2_root: str = "data/ftw",
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        load_boundaries: bool = True,
        usable_only: bool = True,
        swap_order: bool = False,
    ) -> None:
        if countries is None:
            raise ValueError("Specify countries to load")
        if isinstance(countries, str):
            countries = [countries]
        countries = [c.lower() for c in countries]
        assert split in self.valid_splits, f"bad split {split}"

        self.transforms = transforms
        self.load_boundaries = load_boundaries
        self.swap_order = bool(swap_order)

        pl_root = Path(planet_root) / "planet"
        ftw_root = Path(s2_root)

        idx = __import__("geopandas").read_parquet(pl_root / "index.parquet")
        idx = idx[idx["country"].isin(countries)].copy()
        if usable_only:
            idx = idx[idx["usable_pair"] == True]  # noqa: E712
        idx["patch_id"] = idx["patch_id"].astype(str)

        records = []
        for country in countries:
            chips_path = ftw_root / country / f"chips_{country}.parquet"
            if not chips_path.exists():
                continue
            chips = __import__("geopandas").read_parquet(chips_path)
            chips = chips[chips["split"] == split][["aoi_id"]].rename(
                columns={"aoi_id": "patch_id"}
            )
            chips["patch_id"] = chips["patch_id"].astype(str)
            chips["country"] = country
            sub = idx.merge(chips, on=["country", "patch_id"], how="inner")
            for r in sub.to_dict(orient="records"):
                pid = str(r["patch_id"])
                s2_a = ftw_root / country / "s2_images" / "window_a" / f"{pid}.tif"
                s2_b = ftw_root / country / "s2_images" / "window_b" / f"{pid}.tif"
                s2_lbl = ftw_root / country / "label_masks" / "semantic_3class" / f"{pid}.tif"
                # Only keep patches where all S2 files exist.
                if not (s2_a.exists() and s2_b.exists() and s2_lbl.exists()):
                    continue
                records.append(
                    {
                        "country": country,
                        "patch_id": pid,
                        "planet_window_a": str(pl_root / r["image_a_path"]),
                        "planet_window_b": str(pl_root / r["image_b_path"]),
                        "s2_window_b": str(s2_b),
                        "s2_label": str(s2_lbl),
                    }
                )
        if not records:
            raise RuntimeError(
                f"FTWPlanetAlignedDataset: no samples for split={split} "
                f"countries={countries} (check S2 paths exist under {s2_root})"
            )
        self.records: list[dict[str, str]] = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        import rasterio
        from rasterio.transform import from_bounds as _from_bounds
        from rasterio.warp import Resampling, calculate_default_transform, reproject

        r = self.records[index]

        # Reference S2 chip: provides target CRS and bounding box.
        with rasterio.open(r["s2_window_b"]) as s2:
            s2_crs = s2.crs
            s2_bounds = s2.bounds  # (left, bottom, right, top) in s2_crs

        # Compute Planet's native GSD projected into s2_crs, then derive the
        # output pixel grid that covers the S2 extent at Planet resolution.
        # This avoids any downsampling: ~853 px for a 2560 m S2 patch at 3 m.
        with rasterio.open(r["planet_window_b"]) as pl_ref:
            native_xform, _, _ = calculate_default_transform(
                pl_ref.crs,
                s2_crs,
                pl_ref.width,
                pl_ref.height,
                *pl_ref.bounds,
            )
        native_gsd = abs(native_xform.a)  # pixel width in s2_crs units
        out_w = max(1, round((s2_bounds.right - s2_bounds.left) / native_gsd))
        out_h = max(1, round((s2_bounds.top - s2_bounds.bottom) / native_gsd))
        out_transform = _from_bounds(
            s2_bounds.left, s2_bounds.bottom, s2_bounds.right, s2_bounds.top, out_w, out_h
        )

        # S2 3-class label (source of truth for both modalities).
        # Resize to match Planet's native pixel grid (NN so class ids stay exact).
        with rasterio.open(r["s2_label"]) as lbl_src:
            lbl = lbl_src.read(1).astype(np.int64)
        if lbl.shape[0] != out_h or lbl.shape[1] != out_w:
            lbl_t = torch.from_numpy(lbl.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl = (
                torch.nn.functional.interpolate(lbl_t, size=(out_h, out_w), mode="nearest")
                .squeeze(0)
                .squeeze(0)
                .numpy()
                .astype(np.int64)
            )

        def _warp_planet(path: str) -> np.ndarray:
            """Reproject one 4-band Planet window → (4, out_h, out_w) float32."""
            with rasterio.open(path) as src:
                out = np.zeros((src.count, out_h, out_w), dtype=np.float32)
                for band_i in range(src.count):
                    reproject(
                        source=rasterio.band(src, band_i + 1),
                        destination=out[band_i],
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=out_transform,
                        dst_crs=s2_crs,
                        resampling=Resampling.bilinear,
                    )
            return out

        a = _warp_planet(r["planet_window_a"])[_PLANET_BGR_TO_RGB]  # RGB(N)
        b = _warp_planet(r["planet_window_b"])[_PLANET_BGR_TO_RGB]  # RGB(N)

        # [window_b, window_a] default; swap_order randomly flips (p=0.5).
        if self.swap_order and np.random.rand() < 0.5:
            image = np.concatenate([a, b], axis=0)
        else:
            image = np.concatenate([b, a], axis=0)

        sample: dict[str, Any] = {
            "image": torch.from_numpy(image),
            "mask": torch.from_numpy(lbl if self.load_boundaries else (lbl > 0).astype(np.int64)),
            "country": r["country"],
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample
