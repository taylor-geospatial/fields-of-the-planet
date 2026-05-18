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


def _ftw_s2_dataset(
    root: str,
    countries: list[str],
    split: str,
    transforms: Callable | None,
    load_boundaries: bool,
    swap_order: bool = False,
    **_: Any,
) -> Any:
    """Construct the FTW Sentinel-2 dataset from ``ftw_tools`` and inject
    ``country`` into each sample (needed for per-country val metrics).

    Returns a thin proxy whose ``__getitem__`` adds ``country`` derived from
    the file path. ``ftw_tools`` does not surface the country directly.
    """
    from ftw_tools.training.datasets import FTW as _FTW

    base = _FTW(
        root=root,
        countries=countries,
        split=split,
        transforms=transforms,
        load_boundaries=load_boundaries,
        temporal_options="stacked",
        swap_order=swap_order,
    )

    # Per-sample country lookup from the file path:
    # data/ftw/<country>/s2_images/window_b/<id>.tif
    country_for = [str(p["window_b"]).split("/ftw/")[-1].split("/")[0] for p in base.filenames]

    class _Wrapped:
        def __init__(self, base: Any, country_for: list[str]) -> None:
            self._base = base
            self._country_for = country_for
            self.filenames = base.filenames

        def __len__(self) -> int:
            return len(self._base)

        def __getitem__(self, i: int) -> dict:
            s = self._base[i]
            s["country"] = self._country_for[i]
            return s

    return _Wrapped(base, country_for)


PAD_IGNORE_INDEX = 3  # mask pad value; matches trainer.model.ignore_index=3


def _pad_to(x: Tensor, size: int, value: float = 0.0, mode: str = "constant") -> Tensor:
    """Pad bottom/right to reach ``size`` in H/W.

    ``mode="constant"`` uses ``value`` (used for masks: ``ignore_index=3``).
    ``mode="replicate"`` repeats edge pixels, keeping image values in
    distribution — preferred for image tensors so BatchNorm and the
    convolutional features don't see OOD zeros.
    """
    h, w = x.shape[-2], x.shape[-1]
    ph = max(size - h, 0)
    pw = max(size - w, 0)
    if ph == 0 and pw == 0:
        return x
    # F.pad order: (left, right, top, bottom)
    if mode == "replicate":
        # `replicate` requires 4-D input; expand if needed and squeeze after.
        squeeze = False
        if x.dim() == 3:
            x = x.unsqueeze(0)
            squeeze = True
        out = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        return out.squeeze(0) if squeeze else out
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


def _make_crop_transform(size: int, train: bool, pad_mode: str = "zero") -> Callable[[dict], dict]:
    """Crop image, mask, and (optionally) sdf with identical geometry.

    ``pad_mode``: ``"zero"`` (default) pads image with zeros; ``"replicate"``
    pads image with edge-replication so BN inputs stay in distribution.
    Mask always pads with ``ignore_index`` so loss/metrics skip the pad.
    """

    def _t(sample: dict) -> dict:
        img = sample["image"]
        msk = sample["mask"]
        # Pad all tensors to ``size``. Image: zeros or replicate; mask:
        # ignore_index; sdf: clip value (treated as "far from boundary").
        if pad_mode == "replicate":
            img = _pad_to(img, size, mode="replicate")
        else:
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
        # v3 augs
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
        pad_mode: str = "zero",
        dataset_backend: str = "planet",
        s2_data_scale: float = 3000.0,
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

        # 8 channels = window B (4) + window A (4); divisor depends on
        # source: /10000 for PlanetScope SR, /3000 for Sentinel-2 (FTW
        # convention).
        if dataset_backend not in ("planet", "s2"):
            raise ValueError(f"dataset_backend must be 'planet' or 's2', got {dataset_backend!r}")
        self.dataset_backend = dataset_backend
        self._data_scale = PLANET_SR_SCALE if dataset_backend == "planet" else float(s2_data_scale)
        self.mean = torch.zeros(8)
        self.std = torch.full((8,), self._data_scale)
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
        # v3 augs
        self.per_band_affine_aug = bool(per_band_affine_aug)
        self.per_band_affine_a_range = (
            float(per_band_affine_a_range[0]),
            float(per_band_affine_a_range[1]),
        )
        self.per_band_affine_b_range = (
            float(per_band_affine_b_range[0]),
            float(per_band_affine_b_range[1]),
        )
        self.per_band_affine_p = float(per_band_affine_p)
        self.gaussian_blur_aug = bool(gaussian_blur_aug)
        self.gaussian_blur_sigma = (float(gaussian_blur_sigma[0]), float(gaussian_blur_sigma[1]))
        self.gaussian_blur_p = float(gaussian_blur_p)
        self.gaussian_noise_aug = bool(gaussian_noise_aug)
        self.gaussian_noise_std = float(gaussian_noise_std)
        self.gaussian_noise_p = float(gaussian_noise_p)
        self.single_window_dropout_p = float(single_window_dropout_p)
        self.small_angle_rotation_aug = bool(small_angle_rotation_aug)
        self.small_angle_rotation_degrees = float(small_angle_rotation_degrees)
        self.small_angle_rotation_p = float(small_angle_rotation_p)
        self.shear_aug = bool(shear_aug)
        self.shear_degrees = float(shear_degrees)
        self.shear_p = float(shear_p)
        self.boundary_jitter_aug = bool(boundary_jitter_aug)
        self.boundary_jitter_max_px = int(boundary_jitter_max_px)
        self.boundary_jitter_p = float(boundary_jitter_p)
        if pad_mode not in ("zero", "replicate"):
            raise ValueError(f"pad_mode must be 'zero' or 'replicate', got {pad_mode!r}")
        self.pad_mode = pad_mode

        # Normalization branch: either fixed /``self._data_scale`` or random
        # divisor in [0.5x, 1.5x] of the fixed scale (mirrors FTW PRUE
        # preprocess_aug, which uses [1500, 4500] around the /3000 fixed
        # value for S2; we use [5000, 15000] around /10000 for PlanetScope).
        _scale = self._data_scale

        def random_divisor_normalize(x: Tensor) -> Tensor:
            if x.dim() != 4:
                return x
            divisors = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                _scale * 0.5, _scale * 1.5
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

        # Per-band random affine y = a*x + b. Strict generalization of gamma:
        # models inter-Dove-satellite radiometric offsets + atmospheric biases.
        _aff_a_lo, _aff_a_hi = self.per_band_affine_a_range
        _aff_b_lo, _aff_b_hi = self.per_band_affine_b_range
        _aff_p = self.per_band_affine_p

        def per_band_affine(x: Tensor) -> Tensor:
            if x.dim() != 4 or torch.rand(1).item() > _aff_p:
                return x
            B, C = x.shape[0], x.shape[1]
            a = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                _aff_a_lo, _aff_a_hi
            )
            b = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype).uniform_(
                _aff_b_lo, _aff_b_hi
            )
            return a * x + b

        train_augs: list[Any] = [
            kornia.contrib.Lambda(random_divisor_normalize)
            if self.preprocess_aug
            else K.Normalize(mean=self.mean, std=self.std),
            K.RandomRotation(p=0.5, degrees=90),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomSharpness(p=0.5),
        ]
        if self.small_angle_rotation_aug:
            train_augs.append(
                K.RandomAffine(
                    degrees=self.small_angle_rotation_degrees,
                    p=self.small_angle_rotation_p,
                )
            )
        if self.shear_aug:
            train_augs.append(
                K.RandomAffine(
                    degrees=0.0,
                    shear=self.shear_degrees,
                    p=self.shear_p,
                )
            )
        if self.per_band_gamma_aug:
            train_augs.append(kornia.contrib.Lambda(per_band_gamma))
        if self.per_band_affine_aug:
            train_augs.append(kornia.contrib.Lambda(per_band_affine))
        if self.gaussian_blur_aug:
            train_augs.append(
                K.RandomGaussianBlur(
                    kernel_size=(3, 3),
                    sigma=self.gaussian_blur_sigma,
                    p=self.gaussian_blur_p,
                )
            )
        if self.gaussian_noise_aug:
            train_augs.append(
                K.RandomGaussianNoise(
                    mean=0.0, std=self.gaussian_noise_std, p=self.gaussian_noise_p
                )
            )
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
        ds_ctor: Any
        extra: dict[str, Any]
        if self.dataset_backend == "s2":
            # S2 native patches are 256x256, no padding needed; use FTW
            # dataset via ftw_tools with country-injected wrapper.
            ds_ctor = _ftw_s2_dataset
            extra = {}  # S2 dataset doesn't take usable_only/boundary_dilate/return_sdf
        else:
            ds_ctor = FTWPlanet
            extra = {
                "usable_only": self.usable_only,
                "boundary_dilate_px": 0,  # set below in train branch
                "return_sdf": self.return_sdf,
                "swap_order": self.swap_order,
            }

        if stage == "fit":
            # If a schedule is set, defer dilation to on_after_batch_transfer
            # (GPU). The dataset always returns the original (un-dilated)
            # boundary, and the SDF target is also computed on the original.
            ds_dilate = 0 if self.boundary_dilate_schedule is not None else self.boundary_dilate_px
            train_extra = dict(extra)
            if self.dataset_backend == "planet":
                train_extra["boundary_dilate_px"] = ds_dilate
            else:
                # FTW S2 dataset supports swap_order natively
                train_extra["swap_order"] = self.swap_order
            self.train_dataset = ds_ctor(
                root=self.root,
                countries=self.train_countries,
                split="train",
                transforms=_make_crop_transform(self.crop_size, train=True, pad_mode=self.pad_mode),
                load_boundaries=self.load_boundaries,
                **train_extra,
            )
            print(f"[ftw-{self.dataset_backend}] train samples: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            # val/test always uses the original 1-px boundary for fair metrics.
            val_extra = dict(extra)
            if self.dataset_backend == "planet":
                val_extra["boundary_dilate_px"] = 0
                val_extra["return_sdf"] = False
                val_extra["swap_order"] = False
            else:
                val_extra["swap_order"] = False
            self.val_dataset = ds_ctor(
                root=self.root,
                countries=self.val_countries,
                split="val",
                transforms=_make_crop_transform(
                    self.crop_size, train=False, pad_mode=self.pad_mode
                ),
                load_boundaries=self.load_boundaries,
                **val_extra,
            )
            print(f"[ftw-{self.dataset_backend}] val samples: {len(self.val_dataset)}")
        if stage == "test":
            test_extra = dict(extra)
            if self.dataset_backend == "planet":
                test_extra["boundary_dilate_px"] = 0
                test_extra["return_sdf"] = False
                test_extra["swap_order"] = False
            else:
                test_extra["swap_order"] = False
            self.test_dataset = ds_ctor(
                root=self.root,
                countries=self.test_countries,
                split="test",
                transforms=_make_crop_transform(
                    self.crop_size, train=False, pad_mode=self.pad_mode
                ),
                load_boundaries=self.load_boundaries,
                **test_extra,
            )
            print(f"[ftw-{self.dataset_backend}] test samples: {len(self.test_dataset)}")

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
        # Accept both (B,H,W) and (B,1,H,W); normalize to (B,1,H,W) for max_pool.
        squeezed = mask.dim() == 3
        m = mask.unsqueeze(1) if squeezed else mask
        b = (m == 2).float()
        for _ in range(iters):
            b = torch.nn.functional.max_pool2d(b, kernel_size=3, stride=1, padding=1)
        b = b > 0.5
        out = m.clone()
        # Don't overwrite ignore_index pixels (e.g. padded regions).
        valid = m != PAD_IGNORE_INDEX
        out[b & valid] = 2
        return out.squeeze(1) if squeezed else out

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

    def _single_window_dropout(self, image: Tensor) -> Tensor:
        """Zero out either window B (channels 0:4) or window A (channels 4:8)
        for the whole batch with probability ``single_window_dropout_p``.
        """
        if self.single_window_dropout_p <= 0 or torch.rand(1).item() > self.single_window_dropout_p:
            return image
        image = image.clone()
        if torch.rand(1).item() < 0.5:
            image[:, 0:4] = 0.0
        else:
            image[:, 4:8] = 0.0
        return image

    def _random_boundary_jitter(self, mask: Tensor) -> Tensor:
        """Randomly dilate the boundary class by 0 .. boundary_jitter_max_px
        with probability ``boundary_jitter_p`` per batch. Models the
        thickness ambiguity of the GT under 3m all-touched rasterization.
        """
        if not self.boundary_jitter_aug or torch.rand(1).item() > self.boundary_jitter_p:
            return mask
        iters = int(torch.randint(0, self.boundary_jitter_max_px + 1, (1,)).item())
        if iters <= 0:
            return mask
        return self._gpu_dilate_boundary(mask, iters)

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
        if training:
            batch["image"] = self._single_window_dropout(batch["image"])
            batch["mask"] = self._random_boundary_jitter(batch["mask"])
            if self.cutmix_aug:
                batch = self._cutmix_with_buffer(batch)
        return batch
