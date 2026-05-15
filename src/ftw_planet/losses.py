"""Soft-clDice auxiliary loss for the boundary class.

Reference: Shit et al., "clDice — a Novel Topology-Preserving Loss
Function for Tubular Structure Segmentation", CVPR 2021.

The soft skeleton is computed by iterated 3x3 max-min pooling, which
makes it differentiable end-to-end. We apply it to the predicted /
target boundary channel (class 2 in the FTW 3-class label scheme),
giving a loss that penalises *broken* boundary contours far more than a
plain Dice or CE term.
"""

import torch
import torch.nn.functional as F


def _soft_erode(x: torch.Tensor) -> torch.Tensor:
    # min-pool via -maxpool(-x), with kernel=3, stride=1, pad=1
    return -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)


def _soft_dilate(x: torch.Tensor) -> torch.Tensor:
    return F.max_pool2d(x, kernel_size=3, stride=1, padding=1)


def _soft_skeletonize(x: torch.Tensor, iters: int = 6) -> torch.Tensor:
    """Iterative soft erosion + opening — paper recipe."""
    img = x
    skel = F.relu(img - _soft_dilate(_soft_erode(img)))
    for _ in range(iters):
        img = _soft_erode(img)
        delta = F.relu(img - _soft_dilate(_soft_erode(img)))
        skel = skel + F.relu(delta - skel * delta)
    return skel


def soft_cldice_boundary(
    logits: torch.Tensor,
    target: torch.Tensor,
    boundary_class: int = 2,
    ignore_index: int = 3,
    iters: int = 6,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Soft-clDice on the boundary channel.

    Args:
        logits: (B, C, H, W) model output logits.
        target: (B, H, W) integer label map.
    """
    probs = logits.softmax(dim=1)
    p = probs[:, boundary_class : boundary_class + 1]  # (B,1,H,W)
    g = (target == boundary_class).unsqueeze(1).float()  # (B,1,H,W)

    if ignore_index is not None:
        valid = (target != ignore_index).unsqueeze(1).float()
        p = p * valid
        g = g * valid

    sp = _soft_skeletonize(p, iters=iters)
    sg = _soft_skeletonize(g, iters=iters)

    tprec = (sp * g).sum(dim=(1, 2, 3)) / sp.sum(dim=(1, 2, 3)).clamp_min(eps)
    tsens = (sg * p).sum(dim=(1, 2, 3)) / sg.sum(dim=(1, 2, 3)).clamp_min(eps)
    cldice = 2.0 * tprec * tsens / (tprec + tsens).clamp_min(eps)
    return (1.0 - cldice).mean()
