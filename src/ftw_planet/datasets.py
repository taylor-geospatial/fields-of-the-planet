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
    """Planet imagery pre-aligned to the corresponding S2 chip's grid.

    Reads pre-computed TIFs produced by ``scripts/precompute_aligned_planet.py``
    — no on-the-fly reprojection.  Each TIF is already in S2 CRS+bounds at
    Planet's native ~3 m GSD (~853 px) with bands reordered to RGB(NIR).

    Pre-computed layout::

        <planet_root>/aligned_window_a/<country>/<patch_id>.tif
        <planet_root>/aligned_window_b/<country>/<patch_id>.tif

    S2 label layout (ftw_tools convention)::

        <s2_root>/<country>/label_masks/semantic_3class/<patch_id>.tif

    The output dict mirrors :class:`FTWPlanet`:
    ``{image: Tensor(8, H, W), mask: Tensor(H, W), country: str}``.
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
        aligned_a_root = pl_root / "aligned_window_a"
        aligned_b_root = pl_root / "aligned_window_b"

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
                al_a = aligned_a_root / country / f"{pid}.tif"
                al_b = aligned_b_root / country / f"{pid}.tif"
                s2_lbl = ftw_root / country / "label_masks" / "semantic_3class" / f"{pid}.tif"
                if not (al_a.exists() and al_b.exists() and s2_lbl.exists()):
                    continue
                records.append(
                    {
                        "country": country,
                        "patch_id": pid,
                        "window_a": str(al_a),
                        "window_b": str(al_b),
                        "label": str(s2_lbl),
                    }
                )
        if not records:
            raise RuntimeError(
                f"FTWPlanetAlignedDataset: no samples for split={split} "
                f"countries={countries}. Run scripts/precompute_aligned_planet.py first."
            )
        self.records: list[dict[str, str]] = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.records[index]

        # Pre-computed TIFs: already in S2 CRS+bounds, RGB(NIR) band order,
        # float32 DN (not yet divided by PLANET_SR_SCALE — normalisation is
        # applied by _NormWrapper in the datamodule).
        with rasterio.open(r["window_a"]) as src:
            a = src.read().astype(np.float32)  # (4, H, W) RGB(N)
        with rasterio.open(r["window_b"]) as src:
            b = src.read().astype(np.float32)  # (4, H, W) RGB(N)
        with rasterio.open(r["label"]) as src:
            lbl = src.read(1).astype(np.int64)

        # Windows may differ by 1 px due to independent reprojection rounding;
        # clip to the common min so the concatenation always succeeds.
        h = min(a.shape[1], b.shape[1])
        w = min(a.shape[2], b.shape[2])
        a = a[:, :h, :w]
        b = b[:, :h, :w]

        # Label is S2 native 256x256; resize to match Planet's pixel grid (NN).
        if lbl.shape != (h, w):
            lbl_t = torch.from_numpy(lbl.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl = (
                torch.nn.functional.interpolate(
                    lbl_t, size=(h, w), mode="nearest"
                )
                .squeeze(0)
                .squeeze(0)
                .numpy()
                .astype(np.int64)
            )

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


# ---------------------------------------------------------------------------
# Paired Planet + S2 dataset (same patch_id, both modalities)
# ---------------------------------------------------------------------------


class FTWPairedDataset:
    """Paired PlanetScope + Sentinel-2 samples for the same patch_id.

    For each patch, returns:
    * ``planet_image``: (8, H_pl, W_pl) pre-aligned Planet DN, RGB(N)x2 windows.
    * ``planet_mask``:  (H_pl, W_pl) 3-class label upsampled to Planet grid (NN).
    * ``s2_image``:     (8, 256, 256) Sentinel-2 DN, RGB(N)x2 windows.
    * ``s2_mask``:      (256, 256) 3-class label at native S2 resolution.
    * ``country``:      str.

    Both modalities share the same label geometry (same AOI); the planet mask
    is the S2 3-class label NN-resized to the Planet pixel grid so the
    segmentation loss sees the same semantics at both resolutions.

    ``planet_transforms`` / ``s2_transforms`` are called with
    ``{"image": ..., "mask": ...}`` sub-dicts and should return the same keys.
    """

    valid_splits = ("train", "val", "test")

    def __init__(
        self,
        planet_root: str = "data",
        s2_root: str = "data/ftw",
        countries: Sequence[str] | str | None = None,
        split: str = "train",
        planet_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        s2_transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
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

        self.planet_transforms = planet_transforms
        self.s2_transforms = s2_transforms
        self.load_boundaries = load_boundaries
        self.swap_order = bool(swap_order)

        pl_root = Path(planet_root) / "planet"
        ftw_root = Path(s2_root)
        aligned_a_root = pl_root / "aligned_window_a"
        aligned_b_root = pl_root / "aligned_window_b"

        idx = __import__("geopandas").read_parquet(pl_root / "index.parquet")
        idx = idx[idx["country"].isin(countries)].copy()
        if usable_only:
            idx = idx[idx["usable_pair"] == True]  # noqa: E712
        idx["patch_id"] = idx["patch_id"].astype(str)

        records: list[dict[str, str]] = []
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
                pl_a = aligned_a_root / country / f"{pid}.tif"
                pl_b = aligned_b_root / country / f"{pid}.tif"
                s2_a = ftw_root / country / "s2_images" / "window_a" / f"{pid}.tif"
                s2_b = ftw_root / country / "s2_images" / "window_b" / f"{pid}.tif"
                lbl = ftw_root / country / "label_masks" / "semantic_3class" / f"{pid}.tif"
                if not (pl_a.exists() and pl_b.exists() and s2_a.exists() and s2_b.exists() and lbl.exists()):
                    continue
                records.append(
                    {
                        "country": country,
                        "patch_id": pid,
                        "planet_a": str(pl_a),
                        "planet_b": str(pl_b),
                        "s2_a": str(s2_a),
                        "s2_b": str(s2_b),
                        "label": str(lbl),
                    }
                )
        if not records:
            raise RuntimeError(
                f"FTWPairedDataset: no samples for split={split} "
                f"countries={countries}. Ensure precompute_aligned_planet.py has run."
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.records[index]

        # --- Planet (pre-aligned, already RGB(N)) ---
        with rasterio.open(r["planet_a"]) as src:
            pl_a = src.read().astype(np.float32)  # (4, H_pl, W_pl)
        with rasterio.open(r["planet_b"]) as src:
            pl_b = src.read().astype(np.float32)

        # --- S2 (native 256x256, RGB(N) per window) ---
        with rasterio.open(r["s2_a"]) as src:
            s2_a = src.read().astype(np.float32)  # (4, 256, 256)
        with rasterio.open(r["s2_b"]) as src:
            s2_b = src.read().astype(np.float32)

        # --- Shared S2 3-class label ---
        with rasterio.open(r["label"]) as src:
            lbl_s2 = src.read(1).astype(np.int64)  # (256, 256)

        # Clip planet to min H/W (independent reprojection can differ by 1 px)
        h_pl = min(pl_a.shape[1], pl_b.shape[1])
        w_pl = min(pl_a.shape[2], pl_b.shape[2])
        pl_a = pl_a[:, :h_pl, :w_pl]
        pl_b = pl_b[:, :h_pl, :w_pl]

        # Planet label: NN-upsample the S2 label to the planet pixel grid
        if lbl_s2.shape != (h_pl, w_pl):
            lbl_t = torch.from_numpy(lbl_s2.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            lbl_pl = (
                torch.nn.functional.interpolate(lbl_t, size=(h_pl, w_pl), mode="nearest")
                .squeeze(0)
                .squeeze(0)
                .numpy()
                .astype(np.int64)
            )
        else:
            lbl_pl = lbl_s2.copy()

        # Temporal window ordering: window_b first by default.
        # Apply the SAME swap decision to both modalities so temporal order is
        # consistent across Planet and S2 for the same sample.
        swap = self.swap_order and np.random.rand() < 0.5
        planet_image = np.concatenate([pl_a, pl_b] if swap else [pl_b, pl_a], axis=0)
        s2_image = np.concatenate([s2_a, s2_b] if swap else [s2_b, s2_a], axis=0)

        if not self.load_boundaries:
            lbl_pl = (lbl_pl > 0).astype(np.int64)
            lbl_s2 = (lbl_s2 > 0).astype(np.int64)

        sample: dict[str, Any] = {
            "planet_image": torch.from_numpy(planet_image),
            "planet_mask": torch.from_numpy(lbl_pl),
            "s2_image": torch.from_numpy(s2_image),
            "s2_mask": torch.from_numpy(lbl_s2),
            "country": r["country"],
        }

        if self.planet_transforms is not None:
            pl_sub = self.planet_transforms(
                {"image": sample["planet_image"], "mask": sample["planet_mask"]}
            )
            sample["planet_image"] = pl_sub["image"]
            sample["planet_mask"] = pl_sub["mask"]

        if self.s2_transforms is not None:
            s2_sub = self.s2_transforms(
                {"image": sample["s2_image"], "mask": sample["s2_mask"]}
            )
            sample["s2_image"] = s2_sub["image"]
            sample["s2_mask"] = s2_sub["mask"]

        return sample
