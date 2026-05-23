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

# ---------------------------------------------------------------------------
# VICReg alignment loss
# ---------------------------------------------------------------------------


def _off_diagonal(mat: torch.Tensor) -> torch.Tensor:
    """Return all off-diagonal elements of a square matrix as a 1-D tensor."""
    n = mat.size(0)
    return mat.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    lambda_: float = 25.0,
    mu: float = 25.0,
    nu: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg loss (Bardes et al., 2022) between two (B, D) embeddings.

    Three terms:
    * **Invariance** (lambda_): MSE between z1 and z2 — alignment.
    * **Variance** (mu): hinge loss pushing per-dim std above 1 — collapse prevention.
    * **Covariance** (nu): penalises off-diagonal covariance — decorrelation.

    Default weights match the original paper.
    """
    N, D = z1.shape
    # Invariance: direct embedding alignment
    inv = F.mse_loss(z1, z2)

    # Variance: each embedding dimension should have std > 1
    std1 = (z1.var(dim=0) + eps).sqrt()
    std2 = (z2.var(dim=0) + eps).sqrt()
    var = (F.relu(1.0 - std1).mean() + F.relu(1.0 - std2).mean()) / 2.0

    # Covariance: decorrelate embedding dimensions
    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.T @ z1c) / (N - 1)
    cov2 = (z2c.T @ z2c) / (N - 1)
    cov = (_off_diagonal(cov1).pow(2).sum() + _off_diagonal(cov2).pow(2).sum()) / D

    return lambda_ * inv + mu * var + nu * cov


# ---------------------------------------------------------------------------
# clDice loss
# ---------------------------------------------------------------------------


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
