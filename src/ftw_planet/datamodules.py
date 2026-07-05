"""FTW-Planet LightningDataModule.

Reads the PlanetScope layout via :class:`FTWPlanet`, or stock FTW Sentinel-2
via ``ftw_tools`` when ``dataset_backend="s2"``. Patches are non-uniform in
size, so we crop to a fixed ``crop_size`` before augmenting.
"""

from collections.abc import Callable
from typing import Any

import kornia
import kornia.augmentation as K
import torch
from lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

S2_SCALE = 3000.0
PAD_IGNORE_INDEX = 3  # mask pad value; matches trainer ignore_index=3


def ftw_s2_dataset(
    root: str,
    countries: list[str],
    split: str,
    transforms: Callable | None,
    load_boundaries: bool,
    swap_order: bool = False,
) -> Any:
    """FTW Sentinel-2 dataset from ``ftw_tools``, with ``country`` injected per
    sample for per-country validation metrics."""
    from ftw_tools.training.datasets import FTW

    base = FTW(
        root=root,
        countries=countries,
        split=split,
        transforms=transforms,
        load_boundaries=load_boundaries,
        temporal_options="stacked",
        swap_order=swap_order,
    )
    # data/ftw/<country>/s2_images/window_b/<id>.tif -> <country>
    country_for = [str(p["window_b"]).split("/ftw/")[-1].split("/")[0] for p in base.filenames]

    class Wrapped:
        def __init__(self, base: Any, country_for: list[str]) -> None:
            self.base = base
            self.country_for = country_for
            self.filenames = base.filenames

        def __len__(self) -> int:
            return len(self.base)

        def __getitem__(self, i: int) -> dict:
            sample = self.base[i]
            sample["country"] = self.country_for[i]
            return sample

    return Wrapped(base, country_for)


def pad_to(x: Tensor, size: int, value: float = 0.0) -> Tensor:
    """Zero/constant-pad bottom and right to reach ``size`` in H/W."""
    h, w = x.shape[-2], x.shape[-1]
    ph, pw = max(size - h, 0), max(size - w, 0)
    if ph == 0 and pw == 0:
        return x
    return torch.nn.functional.pad(x, (0, pw, 0, ph), value=value)  # (left, right, top, bottom)


def make_crop_transform(size: int, train: bool) -> Callable[[dict], dict]:
    """Crop image and mask together: random crop for train, center crop
    otherwise. Mask pads with ignore_index so loss/metrics skip it."""

    def transform(sample: dict) -> dict:
        img = pad_to(sample["image"], size, value=0.0)
        msk = pad_to(sample["mask"], size, value=PAD_IGNORE_INDEX)
        h, w = img.shape[-2], img.shape[-1]
        if train:
            top = torch.randint(0, h - size + 1, (1,)).item()
            left = torch.randint(0, w - size + 1, (1,)).item()
        else:
            top, left = (h - size) // 2, (w - size) // 2
        sample["image"] = img[..., top : top + size, left : left + size]
        sample["mask"] = msk[..., top : top + size, left : left + size]
        return sample

    return transform


class FTWPlanetDataModule(LightningDataModule):
    """LightningDataModule for the PlanetScope ftw-planet dataset."""

    def __init__(
        self,
        root: str = "data",
        batch_size: int = 64,
        num_workers: int = 8,
        train_countries: tuple[str, ...] | list[str] = ("austria",),
        val_countries: tuple[str, ...] | list[str] = ("austria",),
        test_countries: tuple[str, ...] | list[str] = ("austria",),
        crop_size: int = 256,
        load_boundaries: bool = True,
        usable_only: bool = True,
        swap_order: bool = False,
        preprocess_aug: bool = False,
        resize_aug: bool = False,
        per_band_gamma_aug: bool = False,
        per_band_gamma_range: tuple[float, float] = (0.8, 1.2),
        per_band_gamma_p: float = 0.3,
        per_band_affine_aug: bool = False,
        per_band_affine_a_range: tuple[float, float] = (0.9, 1.1),
        per_band_affine_b_range: tuple[float, float] = (-0.02, 0.02),
        per_band_affine_p: float = 0.3,
        gaussian_blur_aug: bool = False,
        gaussian_blur_sigma: tuple[float, float] = (0.5, 1.5),
        gaussian_blur_p: float = 0.3,
        gaussian_noise_aug: bool = False,
        gaussian_noise_std: float = 0.015,
        gaussian_noise_p: float = 0.3,
        single_window_dropout_p: float = 0.0,
        small_angle_rotation_aug: bool = False,
        small_angle_rotation_degrees: float = 30.0,
        small_angle_rotation_p: float = 0.5,
        shear_aug: bool = False,
        shear_degrees: float = 5.0,
        shear_p: float = 0.3,
        boundary_jitter_aug: bool = False,
        boundary_jitter_max_px: int = 2,
        boundary_jitter_p: float = 0.5,
        dataset_backend: str = "planet",
        **_: Any,
    ) -> None:
        super().__init__()
        if dataset_backend not in ("planet", "s2"):
            raise ValueError(f"dataset_backend must be 'planet' or 's2', got {dataset_backend!r}")
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_countries = list(train_countries)
        self.val_countries = list(val_countries)
        self.test_countries = list(test_countries)
        self.crop_size = crop_size
        self.load_boundaries = load_boundaries
        self.usable_only = usable_only
        self.swap_order = swap_order
        self.dataset_backend = dataset_backend

        self.preprocess_aug = preprocess_aug
        self.resize_aug = resize_aug
        self.per_band_gamma_aug = per_band_gamma_aug
        self.per_band_gamma_range = per_band_gamma_range
        self.per_band_gamma_p = per_band_gamma_p
        self.per_band_affine_aug = per_band_affine_aug
        self.per_band_affine_a_range = per_band_affine_a_range
        self.per_band_affine_b_range = per_band_affine_b_range
        self.per_band_affine_p = per_band_affine_p
        self.gaussian_blur_aug = gaussian_blur_aug
        self.gaussian_blur_sigma = gaussian_blur_sigma
        self.gaussian_blur_p = gaussian_blur_p
        self.gaussian_noise_aug = gaussian_noise_aug
        self.gaussian_noise_std = gaussian_noise_std
        self.gaussian_noise_p = gaussian_noise_p
        self.single_window_dropout_p = single_window_dropout_p
        self.small_angle_rotation_aug = small_angle_rotation_aug
        self.small_angle_rotation_degrees = small_angle_rotation_degrees
        self.small_angle_rotation_p = small_angle_rotation_p
        self.shear_aug = shear_aug
        self.shear_degrees = shear_degrees
        self.shear_p = shear_p
        self.boundary_jitter_aug = boundary_jitter_aug
        self.boundary_jitter_max_px = boundary_jitter_max_px
        self.boundary_jitter_p = boundary_jitter_p

        # 8 channels = window B (4) + window A (4). Divisor depends on source:
        # /10000 for PlanetScope SR, /3000 for Sentinel-2.
        scale = PLANET_SR_SCALE if dataset_backend == "planet" else S2_SCALE
        self.mean = torch.zeros(8)
        self.std = torch.full((8,), scale)
        self.train_aug = self._build_train_aug(scale)
        self.aug = K.AugmentationSequential(K.Normalize(self.mean, self.std), data_keys=None)

    def _build_train_aug(self, scale: float) -> K.AugmentationSequential:
        # preprocess_aug: normalize with a random divisor in [0.5x, 1.5x] of
        # scale instead of a fixed one, simulating atmospheric-correction drift.
        def random_divisor_normalize(x: Tensor) -> Tensor:
            if x.dim() != 4:
                return x
            divisors = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                scale * 0.5, scale * 1.5
            )
            return x / divisors

        gamma_lo, gamma_hi = self.per_band_gamma_range
        gamma_p = self.per_band_gamma_p

        def per_band_gamma(x: Tensor) -> Tensor:
            if x.dim() != 4 or torch.rand(1).item() > gamma_p:
                return x
            b, c = x.shape[0], x.shape[1]
            gammas = torch.empty(b, c, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                gamma_lo, gamma_hi
            )
            return x.clamp(min=1e-6) ** gammas

        # Per-band affine y = a*x + b, a generalization of gamma for
        # inter-satellite radiometric offsets.
        aff_a_lo, aff_a_hi = self.per_band_affine_a_range
        aff_b_lo, aff_b_hi = self.per_band_affine_b_range
        aff_p = self.per_band_affine_p

        def per_band_affine(x: Tensor) -> Tensor:
            if x.dim() != 4 or torch.rand(1).item() > aff_p:
                return x
            b, c = x.shape[0], x.shape[1]
            a = torch.empty(b, c, 1, 1, device=x.device, dtype=x.dtype).uniform_(aff_a_lo, aff_a_hi)
            off = torch.empty(b, c, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                aff_b_lo, aff_b_hi
            )
            return a * x + off

        augs: list[Any] = [
            kornia.contrib.Lambda(random_divisor_normalize)
            if self.preprocess_aug
            else K.Normalize(self.mean, self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
        ]
        if self.small_angle_rotation_aug:
            augs.append(
                K.RandomAffine(
                    degrees=self.small_angle_rotation_degrees, p=self.small_angle_rotation_p
                )
            )
        if self.shear_aug:
            augs.append(K.RandomAffine(degrees=0.0, shear=self.shear_degrees, p=self.shear_p))
        if self.per_band_gamma_aug:
            augs.append(kornia.contrib.Lambda(per_band_gamma))
        if self.per_band_affine_aug:
            augs.append(kornia.contrib.Lambda(per_band_affine))
        if self.gaussian_blur_aug:
            augs.append(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3), sigma=self.gaussian_blur_sigma, p=self.gaussian_blur_p
                )
            )
        if self.gaussian_noise_aug:
            augs.append(
                K.RandomGaussianNoise(
                    mean=0.0, std=self.gaussian_noise_std, p=self.gaussian_noise_p
                )
            )
        if self.resize_aug:
            augs.append(
                K.RandomResizedCrop(
                    (self.crop_size, self.crop_size), scale=(0.3, 0.9), ratio=(0.75, 1.33), p=0.5
                )
            )
        return K.AugmentationSequential(*augs, data_keys=None)

    def _dataset(self, countries: list[str], split: str, train: bool) -> Any:
        transforms = make_crop_transform(self.crop_size, train=train)
        if self.dataset_backend == "s2":
            return ftw_s2_dataset(
                root=self.root,
                countries=countries,
                split=split,
                transforms=transforms,
                load_boundaries=self.load_boundaries,
                swap_order=self.swap_order if train else False,
            )
        return FTWPlanet(
            root=self.root,
            countries=countries,
            split=split,
            transforms=transforms,
            load_boundaries=self.load_boundaries,
            usable_only=self.usable_only,
            swap_order=self.swap_order if train else False,
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.train_dataset = self._dataset(self.train_countries, "train", train=True)
            print(f"[ftw-{self.dataset_backend}] train samples: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.val_dataset = self._dataset(self.val_countries, "val", train=False)
            print(f"[ftw-{self.dataset_backend}] val samples: {len(self.val_dataset)}")
        if stage == "test":
            self.test_dataset = self._dataset(self.test_countries, "test", train=False)
            print(f"[ftw-{self.dataset_backend}] test samples: {len(self.test_dataset)}")

    def _loader(self, ds: Any, shuffle: bool) -> DataLoader:
        kw: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            kw["prefetch_factor"] = 4  # keep the GPU from waiting on rasterio
        return DataLoader(ds, **kw)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def _dilate_boundary(self, mask: Tensor, iters: int) -> Tensor:
        """Dilate the boundary class (2) by ``iters`` pixels via max-pooling."""
        squeezed = mask.dim() == 3
        m = mask.unsqueeze(1) if squeezed else mask
        b = (m == 2).float()
        for _ in range(iters):
            b = torch.nn.functional.max_pool2d(b, kernel_size=3, stride=1, padding=1)
        out = m.clone()
        out[(b > 0.5) & (m != PAD_IGNORE_INDEX)] = 2  # don't overwrite padded pixels
        return out.squeeze(1) if squeezed else out

    def _single_window_dropout(self, image: Tensor) -> Tensor:
        """Zero out window B or window A for the whole batch."""
        if self.single_window_dropout_p <= 0 or torch.rand(1).item() > self.single_window_dropout_p:
            return image
        image = image.clone()
        if torch.rand(1).item() < 0.5:
            image[:, 0:4] = 0.0
        else:
            image[:, 4:8] = 0.0
        return image

    def _random_boundary_jitter(self, mask: Tensor) -> Tensor:
        """Randomly dilate the boundary class by 0..boundary_jitter_max_px."""
        if not self.boundary_jitter_aug or torch.rand(1).item() > self.boundary_jitter_p:
            return mask
        iters = int(torch.randint(0, self.boundary_jitter_max_px + 1, (1,)).item())
        return self._dilate_boundary(mask, iters) if iters > 0 else mask

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int = 0) -> dict:
        training = self.trainer is not None and self.trainer.training
        # kornia treats non-tensor keys awkwardly; pop metadata before augs.
        country = batch.pop("country", None)
        batch = (self.train_aug if training else self.aug)(batch)
        if country is not None:
            batch["country"] = country
        if training:
            batch["image"] = self._single_window_dropout(batch["image"])
            batch["mask"] = self._random_boundary_jitter(batch["mask"])
        return batch
