"""FTW-Planet dataset: paired PlanetScope SR windows + FTW 3-class label.

Reads the GeoParquet index at ``<root>/planet/index.parquet`` and derives
train/val/test splits by joining ``patch_id`` against FTW's
``chips_<country>.parquet``.
"""

import os
from collections.abc import Callable, Sequence
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
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
        self.swap_order = swap_order

        planet_root = os.path.join(root, "planet")
        ftw_root = os.path.join(root, "ftw")

        idx = gpd.read_parquet(os.path.join(planet_root, "index.parquet"))
        idx = idx[idx["country"].isin(countries)].copy()
        if usable_only:
            idx = idx[idx["usable_pair"] == True]  # noqa: E712
        idx["patch_id"] = idx["patch_id"].astype(str)

        # Join split assignment from the official FTW chips parquet. aoi_id is
        # int for some countries and str for others, so coerce both sides.
        frames = []
        for country in countries:
            chips = gpd.read_parquet(os.path.join(ftw_root, country, f"chips_{country}.parquet"))
            chips = chips[chips["split"] == split][["aoi_id"]].rename(
                columns={"aoi_id": "patch_id"}
            )
            chips["patch_id"] = chips["patch_id"].astype(str)
            chips["country"] = country
            frames.append(idx.merge(chips, on=["country", "patch_id"], how="inner"))
        merged = pd.concat(frames, ignore_index=True) if frames else idx.iloc[0:0]

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

    def __getitem__(self, index: int) -> dict[str, Any]:
        r = self.records[index]
        with rasterio.open(r["window_a"]) as src:
            a = src.read().astype(np.float32)
        with rasterio.open(r["window_b"]) as src:
            b = src.read().astype(np.float32)
        with rasterio.open(r["label"]) as src:
            lbl = src.read(1).astype(np.int64)

        # Source tifs occasionally differ by 1 px in H/W from reprojection
        # rounding; clip to the common min so the stack lines up.
        h = min(a.shape[1], b.shape[1], lbl.shape[0])
        w = min(a.shape[2], b.shape[2], lbl.shape[1])
        a, b, lbl = a[:, :h, :w], b[:, :h, :w], lbl[:h, :w]

        # Window B first then A, matching the stock FTW channel order. With
        # swap_order, use [A, B] instead per-sample (p=0.5).
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
