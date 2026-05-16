"""FTW-Planet LightningDataModule.

Mirrors the canonical ``FTWDataModule`` from ftw-baselines but reads from
the PlanetScope ftw-planet layout via :class:`FTWPlanet`.

Differences from the stock S2 datamodule:

* Reflectance scaling is /10000 (PlanetScope SR convention) rather than /3000.
* Patches are non-uniform in spatial size (~510 x ~330 at 3 m); we crop to
  a fixed ``crop_size`` (center crop for val/test, random crop for train)
  before any kornia augmentations.
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

PAD_IGNORE_INDEX = 3  # mask pad value; matches trainer.model.ignore_index=3


def _pad_to(x: Tensor, size: int, value: float = 0.0) -> Tensor:
    h, w = x.shape[-2], x.shape[-1]
    ph = max(size - h, 0)
    pw = max(size - w, 0)
    if ph == 0 and pw == 0:
        return x
    # F.pad order: (left, right, top, bottom)
    return torch.nn.functional.pad(x, (0, pw, 0, ph), value=value)


def _center_crop(image: Tensor, mask: Tensor, size: int) -> tuple[Tensor, Tensor]:
    image = _pad_to(image, size, value=0.0)
    mask = _pad_to(mask, size, value=PAD_IGNORE_INDEX)
    h, w = image.shape[-2], image.shape[-1]
    top = (h - size) // 2
    left = (w - size) // 2
    return (
        image[..., top : top + size, left : left + size],
        mask[..., top : top + size, left : left + size],
    )


def _random_crop(image: Tensor, mask: Tensor, size: int) -> tuple[Tensor, Tensor]:
    image = _pad_to(image, size, value=0.0)
    mask = _pad_to(mask, size, value=PAD_IGNORE_INDEX)
    h, w = image.shape[-2], image.shape[-1]
    top = torch.randint(0, h - size + 1, (1,)).item()
    left = torch.randint(0, w - size + 1, (1,)).item()
    return (
        image[..., top : top + size, left : left + size],
        mask[..., top : top + size, left : left + size],
    )


def _make_crop_transform(size: int, train: bool) -> Callable[[dict], dict]:
    """Crop image, mask, and (optionally) sdf with identical geometry."""

    def _t(sample: dict) -> dict:
        img = sample["image"]
        msk = sample["mask"]
        # Pad all tensors to ``size``. Image: zeros, mask: ignore_index, sdf:
        # clip value (treated as "far from boundary"). Use a high but finite
        # constant so L1 stays bounded.
        img = _pad_to(img, size, value=0.0)
        msk = _pad_to(msk, size, value=PAD_IGNORE_INDEX)
        sdf = None
        if "sdf" in sample:
            sdf = _pad_to(sample["sdf"].unsqueeze(0), size, value=20.0).squeeze(0)
        h, w = img.shape[-2], img.shape[-1]
        if train:
            top = torch.randint(0, h - size + 1, (1,)).item()
            left = torch.randint(0, w - size + 1, (1,)).item()
        else:
            top = (h - size) // 2
            left = (w - size) // 2
        sample["image"] = img[..., top : top + size, left : left + size]
        sample["mask"] = msk[..., top : top + size, left : left + size]
        if sdf is not None:
            sample["sdf"] = sdf[..., top : top + size, left : left + size]
        return sample

    return _t


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
        random_shuffle: bool = False,
        boundary_dilate_px: int = 0,
        boundary_dilate_schedule: dict | None = None,
        return_sdf: bool = False,
        preprocess_aug: bool = False,
        resize_aug: bool = False,
        swap_order: bool = False,
        per_band_gamma_aug: bool = False,
        per_band_gamma_range: tuple[float, float] = (0.8, 1.2),
        per_band_gamma_p: float = 0.3,
        cutmix_aug: bool = False,
        cutmix_p: float = 0.5,
        cutmix_scale: tuple[float, float] = (0.25, 0.5),
        cutmix_buffer: int = 2,
        **_: Any,
    ) -> None:
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_countries = list(train_countries)
        self.val_countries = list(val_countries)
        self.test_countries = list(test_countries)
        self.crop_size = crop_size
        self.load_boundaries = load_boundaries
        self.usable_only = usable_only
        self.boundary_dilate_px = int(boundary_dilate_px)
        # Curriculum schedule: {epoch_start: dilation_px}. If set, takes
        # precedence over ``boundary_dilate_px``. Dilation is applied on the
        # GPU side in on_after_batch_transfer so we can swap it per epoch
        # without re-spawning dataloader workers.
        if boundary_dilate_schedule is not None:
            self.boundary_dilate_schedule = {
                int(k): int(v) for k, v in boundary_dilate_schedule.items()
            }
        else:
            self.boundary_dilate_schedule = None
        self.return_sdf = bool(return_sdf)

        # 8 channels = window B (4) + window A (4); /10000 to reflectance.
        self.mean = torch.zeros(8)
        self.std = torch.full((8,), PLANET_SR_SCALE)
        self.preprocess_aug = bool(preprocess_aug)
        self.resize_aug = bool(resize_aug)
        self.swap_order = bool(swap_order)
        self.per_band_gamma_aug = bool(per_band_gamma_aug)
        self.per_band_gamma_range = (float(per_band_gamma_range[0]), float(per_band_gamma_range[1]))
        self.per_band_gamma_p = float(per_band_gamma_p)
        self.cutmix_aug = bool(cutmix_aug)
        self.cutmix_p = float(cutmix_p)
        self.cutmix_scale = (float(cutmix_scale[0]), float(cutmix_scale[1]))
        self.cutmix_buffer = int(cutmix_buffer)

        # Normalization branch: either fixed /PLANET_SR_SCALE or random divisor
        # in [0.5x, 1.5x] of PLANET_SR_SCALE (mirrors FTW PRUE preprocess_aug,
        # which uses [1500, 4500] around the /3000 fixed value).
        def random_divisor_normalize(x: Tensor) -> Tensor:
            if x.dim() != 4:
                return x
            divisors = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                PLANET_SR_SCALE * 0.5, PLANET_SR_SCALE * 1.5
            )
            return x / divisors

        # Per-band random gamma in [lo, hi] applied after normalization.
        # Each of the 8 channels gets an independent gamma per sample,
        # simulating cross-acquisition atmospheric-correction variation.
        _gamma_lo, _gamma_hi = self.per_band_gamma_range
        _gamma_p = self.per_band_gamma_p

        def per_band_gamma(x: Tensor) -> Tensor:
            if x.dim() != 4 or torch.rand(1).item() > _gamma_p:
                return x
            B, C = x.shape[0], x.shape[1]
            gammas = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                _gamma_lo, _gamma_hi
            )
            return x.clamp(min=1e-6) ** gammas

        train_augs: list[Any] = [
            kornia.contrib.Lambda(random_divisor_normalize)
            if self.preprocess_aug
            else K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
        ]
        if self.per_band_gamma_aug:
            train_augs.append(kornia.contrib.Lambda(per_band_gamma))
        if self.resize_aug:
            train_augs.append(
                K.RandomResizedCrop(
                    (self.crop_size, self.crop_size),
                    scale=(0.3, 0.9),
                    ratio=(0.75, 1.33),
                    p=0.5,
                )
            )
        self.train_aug = K.AugmentationSequential(*train_augs, data_keys=None)
        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None
        )

    def setup(self, stage: str) -> None:
        if stage == "fit":
            # If a schedule is set, defer dilation to on_after_batch_transfer
            # (GPU). The dataset always returns the original (un-dilated)
            # boundary, and the SDF target is also computed on the original.
            ds_dilate = 0 if self.boundary_dilate_schedule is not None else self.boundary_dilate_px
            self.train_dataset = FTWPlanet(
                root=self.root,
                countries=self.train_countries,
                split="train",
                transforms=_make_crop_transform(self.crop_size, train=True),
                load_boundaries=self.load_boundaries,
                usable_only=self.usable_only,
                boundary_dilate_px=ds_dilate,
                return_sdf=self.return_sdf,
                swap_order=self.swap_order,
            )
            print(f"[ftw-planet] train samples: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            # val/test always uses the original 1-px boundary for fair metrics.
            self.val_dataset = FTWPlanet(
                root=self.root,
                countries=self.val_countries,
                split="val",
                transforms=_make_crop_transform(self.crop_size, train=False),
                load_boundaries=self.load_boundaries,
                usable_only=self.usable_only,
            )
            print(f"[ftw-planet] val samples: {len(self.val_dataset)}")
        if stage == "test":
            self.test_dataset = FTWPlanet(
                root=self.root,
                countries=self.test_countries,
                split="test",
                transforms=_make_crop_transform(self.crop_size, train=False),
                load_boundaries=self.load_boundaries,
                usable_only=self.usable_only,
            )
            print(f"[ftw-planet] test samples: {len(self.test_dataset)}")

    def _loader(self, ds: FTWPlanet, shuffle: bool) -> DataLoader:
        kw: dict[str, Any] = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": True,
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            # 4 batches queued per worker so the GPU never waits on rasterio.
            kw["prefetch_factor"] = 4
        return DataLoader(ds, **kw)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)

    def _gpu_dilate_boundary(self, mask: Tensor, iters: int) -> Tensor:
        if iters <= 0:
            return mask
        b = (mask == 2).float().unsqueeze(1)  # (B,1,H,W)
        for _ in range(iters):
            b = torch.nn.functional.max_pool2d(b, kernel_size=3, stride=1, padding=1)
        b = b.squeeze(1) > 0.5
        out = mask.clone()
        # Don't overwrite ignore_index pixels (e.g. padded regions).
        valid = mask != PAD_IGNORE_INDEX
        out[b & valid] = 2
        return out

    def _current_curriculum_px(self) -> int:
        sched = self.boundary_dilate_schedule
        trainer = self.trainer
        if sched is None or trainer is None or not trainer.training:
            return 0
        ep = int(trainer.current_epoch)
        keys = [k for k in sched if k <= ep]
        return sched[max(keys)] if keys else 0

    def _cutmix_with_buffer(self, batch: dict) -> dict:
        """CutMix paste with ignore_index buffer along the cut rectangle border.

        Pastes a random rectangle from image[perm] into image[i] (same for
        mask and sdf if present), then paints a ``cutmix_buffer``-px ring of
        ``PAD_IGNORE_INDEX`` along the cut edges so the seg loss does not get
        contaminated by phantom (synthetic) class transitions.
        """
        image = batch["image"]
        mask = batch["mask"]
        if image.dim() != 4 or image.size(0) < 2:
            return batch
        if torch.rand(1).item() > self.cutmix_p:
            return batch
        B, _, H, W = image.shape
        perm = torch.randperm(B, device=image.device)
        lo, hi = self.cutmix_scale
        lam = torch.empty(1, device=image.device).uniform_(lo, hi).item()
        cut_h = max(8, int(H * lam))
        cut_w = max(8, int(W * lam))
        cy = int(torch.randint(0, max(1, H - cut_h), (1,), device=image.device).item())
        cx = int(torch.randint(0, max(1, W - cut_w), (1,), device=image.device).item())
        image[:, :, cy : cy + cut_h, cx : cx + cut_w] = image[perm][
            :, :, cy : cy + cut_h, cx : cx + cut_w
        ]

        # Mask may be (B,H,W) or (B,1,H,W); normalize to 3-D for indexing.
        squeezed = mask.dim() == 3
        if squeezed:
            mask = mask.unsqueeze(1)
        mask[:, :, cy : cy + cut_h, cx : cx + cut_w] = mask[perm][
            :, :, cy : cy + cut_h, cx : cx + cut_w
        ]
        b = self.cutmix_buffer
        if b > 0:
            y0, y1 = max(0, cy - b), min(H, cy + cut_h + b)
            x0, x1 = max(0, cx - b), min(W, cx + cut_w + b)
            mask[:, :, max(0, cy - b) : min(H, cy + b), x0:x1] = PAD_IGNORE_INDEX  # top
            mask[:, :, max(0, cy + cut_h - b) : min(H, cy + cut_h + b), x0:x1] = (
                PAD_IGNORE_INDEX  # bottom
            )
            mask[:, :, y0:y1, max(0, cx - b) : min(W, cx + b)] = PAD_IGNORE_INDEX  # left
            mask[:, :, y0:y1, max(0, cx + cut_w - b) : min(W, cx + cut_w + b)] = (
                PAD_IGNORE_INDEX  # right
            )
        batch["mask"] = mask.squeeze(1) if squeezed else mask

        if "sdf" in batch:
            sdf = batch["sdf"]
            squeezed_sdf = sdf.dim() == 3
            if squeezed_sdf:
                sdf = sdf.unsqueeze(1)
            sdf[:, :, cy : cy + cut_h, cx : cx + cut_w] = sdf[perm][
                :, :, cy : cy + cut_h, cx : cx + cut_w
            ]
            batch["sdf"] = sdf.squeeze(1) if squeezed_sdf else sdf
        return batch

    def on_after_batch_transfer(self, batch: dict, dataloader_idx: int = 0) -> dict:
        trainer = self.trainer
        training = trainer is not None and trainer.training
        if training:
            iters = self._current_curriculum_px()
            if iters > 0:
                batch["mask"] = self._gpu_dilate_boundary(batch["mask"], iters)
        # kornia AugmentationSequential treats non-tensor keys awkwardly; pop
        # string-valued metadata before augs and re-attach after.
        country = batch.pop("country", None)
        aug = self.train_aug if training else self.aug
        batch = aug(batch)
        if country is not None:
            batch["country"] = country
        if training and self.cutmix_aug:
            batch = self._cutmix_with_buffer(batch)
        return batch
