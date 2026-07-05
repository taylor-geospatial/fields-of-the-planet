"""Shared inference infrastructure for evaluation: checkpoint loading, padding,
D4 test-time augmentation, and watershed instance separation.
"""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed


def load_task(ckpt: str | Path, device: torch.device) -> torch.nn.Module:
    """Load an ftw-planet checkpoint as an eval-mode task on ``device``."""
    from ftw_tools.training.trainers import CustomSemanticSegmentationTask

    from ftw_planet.trainers import FTWPlanetSegTask

    try:
        task = FTWPlanetSegTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    except (RuntimeError, KeyError, TypeError):
        task = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return task.eval().to(device)


def pad_to_min(
    image: torch.Tensor, mask: torch.Tensor, min_size: int = 0
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Zero-pad H,W to ``max(next-multiple-of-32, min_size)``.

    Setting ``min_size`` to the training crop size makes every inference patch
    at least as large as a training crop, removing train/eval size drift.
    Returns the padded tensors plus the original (H, W) for cropping back.
    """
    h, w = image.shape[-2], image.shape[-1]
    new_h = max(((h + 31) // 32) * 32, min_size)
    new_w = max(((w + 31) // 32) * 32, min_size)
    if (new_h, new_w) == (h, w):
        return image, mask, h, w
    image = F.pad(image, (0, new_w - w, 0, new_h - h), value=0.0)
    mask = F.pad(mask, (0, new_w - w, 0, new_h - h), value=3)
    return image, mask, h, w


def _d4_transforms() -> list[tuple[Callable, Callable]]:
    """The 8 D4 (flip + 90-degree rotation) forward/inverse transform pairs."""

    def r90(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, 1, dims=[-2, -1])

    def r270(x: torch.Tensor) -> torch.Tensor:
        return torch.rot90(x, 3, dims=[-2, -1])

    return [
        (lambda x: x, lambda x: x),
        (lambda x: torch.flip(x, dims=[-1]), lambda x: torch.flip(x, dims=[-1])),
        (lambda x: torch.flip(x, dims=[-2]), lambda x: torch.flip(x, dims=[-2])),
        (lambda x: torch.flip(x, dims=[-2, -1]), lambda x: torch.flip(x, dims=[-2, -1])),
        (r90, r270),
        (r270, r90),
        (lambda x: torch.flip(r90(x), dims=[-1]), lambda x: r270(torch.flip(x, dims=[-1]))),
        (lambda x: torch.flip(r90(x), dims=[-2]), lambda x: r270(torch.flip(x, dims=[-2]))),
    ]


@torch.inference_mode()
def predict(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities (1, C, H, W) for a single input patch."""
    return model(image).softmax(dim=1)


@torch.inference_mode()
def predict_tta(model: torch.nn.Module, image: torch.Tensor) -> torch.Tensor:
    """D4 test-time-augmented softmax probabilities: predict under each of the 8
    transforms, inverse-transform, and average."""
    probs = None
    transforms = _d4_transforms()
    for fwd, inv in transforms:
        p = inv(predict(model, fwd(image)))
        probs = p if probs is None else probs + p
    assert probs is not None
    return probs / len(transforms)


def watershed_instances(
    seg_pred: np.ndarray, distance: np.ndarray, h_min: float = 2.0, field_class: int = 1
) -> np.ndarray:
    """Split the predicted field mask into instances via marker-controlled
    watershed on ``distance`` (larger = deeper inside a field). Returns an
    instance-label image (0 = background, >=1 = field instance)."""
    field_mask = seg_pred == field_class
    if not field_mask.any():
        return np.zeros_like(seg_pred, dtype=np.int32)
    surface = (distance * field_mask).astype(np.float32)
    markers, _ = label(h_maxima(surface, h=h_min))
    if markers.max() == 0:
        markers, _ = label(field_mask)  # no strong peaks -> one per component
    return watershed(-surface, markers=markers, mask=field_mask).astype(np.int32)
