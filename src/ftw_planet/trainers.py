"""ftw-planet trainer shim.

Subclasses the FTW ``CustomSemanticSegmentationTask`` to replace the
TensorBoard-only ``self.logger.experiment.add_scalar`` call in
``on_train_epoch_start`` with the logger-agnostic ``self.log``. Lets us
use ``WandbLogger`` (or any non-TB logger) without TB along for the ride.
"""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from ftw_tools.training.trainers import CustomSemanticSegmentationTask

from ftw_planet.losses import soft_cldice_boundary, vicreg_loss


class FTWPlanetSegTask(CustomSemanticSegmentationTask):
    def on_train_epoch_start(self) -> None:
        opts: Any = self.optimizers()
        opt = opts[0] if isinstance(opts, list) else opts
        lr = opt.optimizer.param_groups[0]["lr"]
        self.log("lr", lr, on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:  # type: ignore[override]
        super().on_validation_epoch_start()
        # country -> [tp, fp, fn] tallies for field class
        self._val_per_country: dict[str, list[int]] = {}

    def validation_step(  # type: ignore[override]
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        super().validation_step(batch, batch_idx, dataloader_idx)
        countries = batch.get("country", None)
        if not countries:
            return
        x = batch["image"]
        y = batch["mask"].squeeze(1) if batch["mask"].dim() == 4 else batch["mask"]
        with torch.inference_mode():
            y_hat = self(x)
        preds = y_hat.argmax(dim=1)
        ignore = self.hparams.get("ignore_index", 3)
        valid = y != ignore
        p_field = (preds == 1) & valid
        t_field = (y == 1) & valid
        tp = (p_field & t_field).flatten(1).sum(dim=1)
        fp = (p_field & ~t_field).flatten(1).sum(dim=1)
        fn = (~p_field & t_field).flatten(1).sum(dim=1)
        for i, c in enumerate(countries):
            d = self._val_per_country.setdefault(c, [0, 0, 0])
            d[0] += int(tp[i].item())
            d[1] += int(fp[i].item())
            d[2] += int(fn[i].item())

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        super().on_validation_epoch_end()
        pc = getattr(self, "_val_per_country", {})
        for c, (tp, fp, fn) in pc.items():
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            self.log(f"val/iou/field/{c}", iou, on_epoch=True, sync_dist=True)
            self.log(f"val/precision/field/{c}", prec, on_epoch=True, sync_dist=True)
            self.log(f"val/recall/field/{c}", rec, on_epoch=True, sync_dist=True)


class ClDiceSegTask(FTWPlanetSegTask):
    """Adds a soft-clDice term on the boundary channel to the existing loss.

    Knob: ``cldice_weight`` (defaults to 0.5). Total loss is
    ``criterion + cldice_weight * soft_cldice_boundary``.
    """

    def __init__(
        self,
        *args: Any,
        cldice_weight: float = 0.2,
        cldice_iters: int = 6,
        cldice_class: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.cldice_weight = float(cldice_weight)
        self.cldice_iters = int(cldice_iters)
        self.cldice_class = int(cldice_class)
        self.save_hyperparameters(ignore=[])

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"].squeeze(1)

        if self.hparams["model"] in ["fcsiamdiff", "fcsiamconc", "fcsiamavg"]:
            y_hat = self(rearrange(x, "b (t c) h w -> b t c h w", t=2))
        else:
            y_hat = self(x)

        base = self.criterion(y_hat, y)
        cld = soft_cldice_boundary(
            y_hat,
            y,
            boundary_class=self.cldice_class,
            ignore_index=self.hparams.get("ignore_index", 3),
            iters=self.cldice_iters,
        )
        loss = base + self.cldice_weight * cld

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_base", base, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_cldice", cld, on_step=False, on_epoch=True, sync_dist=True)
        self.train_metrics.update(y_hat, y)
        return loss


class SDFSegTask(FTWPlanetSegTask):
    """Multi-task: 3-class seg + parallel SDF regression head.

    The SDF (signed-distance-to-boundary in pixels, clipped) is predicted
    by a small conv on top of the smp.Unet decoder features. Loss is
    ``logcoshdice(seg) + sdf_weight * L1(sdf_pred, sdf_target)``.
    Validation / inference still call ``self.model(x)`` and only use the
    seg logits, so the existing val pipeline + metrics keep working.
    """

    def __init__(
        self,
        *args: Any,
        sdf_weight: float = 0.5,
        sdf_clip: float = 20.0,
        sdf_hidden: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.sdf_weight = float(sdf_weight)
        self.sdf_clip = float(sdf_clip)
        # smp.Unet decoder outputs (B, 16, H, W) at full res for the default
        # ``decoder_channels=(256,128,64,32,16)``. The SDF head is a small
        # conv stack that produces a single-channel (clipped, non-negative)
        # distance field.
        in_ch = 16
        self.sdf_head = nn.Sequential(
            nn.Conv2d(in_ch, sdf_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(sdf_hidden, 1, kernel_size=1),
            nn.Softplus(),  # non-negative output
        )
        self.save_hyperparameters(ignore=[])

    def _forward_dual(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unet = self.model
        feats = unet.encoder(x)
        dec = unet.decoder(feats)
        seg = unet.segmentation_head(dec)
        # SDF head outputs Softplus -> non-negative. We clamp to [0, 1] to
        # match the normalised target (raw target / sdf_clip).
        sdf = self.sdf_head(dec).squeeze(1)
        sdf = torch.clamp(sdf, max=1.0)
        return seg, sdf

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        sdf_target = batch["sdf"]  # (B,H,W)

        seg, sdf_pred = self._forward_dual(x)
        base = self.criterion(seg, y)

        # Normalise SDF target to [0, 1] so loss magnitudes match the seg
        # term without needing micro-tuned weights.
        sdf_target_n = sdf_target / self.sdf_clip
        # Mask out ignore_index pixels from SDF loss (padded regions).
        valid = (y != self.hparams.get("ignore_index", 3)).float()
        diff = (sdf_pred - sdf_target_n).abs() * valid
        sdf_loss = diff.sum() / valid.sum().clamp_min(1.0)

        loss = base + self.sdf_weight * sdf_loss

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_base", base, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_sdf", sdf_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.train_metrics.update(seg, y)
        return loss


class FrameFieldSegTask(SDFSegTask):
    """SDF + 4-PolyVector frame field aux head (Girard et al. CVPR 2021).

    Predicts a complex polynomial ``f(z) = z^4 + c2 z^2 + c0`` per pixel
    whose four roots are two pairs of opposite directions encoding the
    local boundary tangent(s). Supervised at boundary-adjacent pixels via
    ``|f(tau)|^2`` where ``tau`` is the unit tangent computed from the
    GT field-interior mask gradient (Sobel). At inference the frame field
    can be used to snap polygon vertices to corners and straighten edges
    along learned tangent directions.
    """

    def __init__(
        self,
        *args: Any,
        frame_weight: float = 0.1,
        frame_smooth_weight: float = 0.01,
        frame_hidden: int = 32,
        frame_grad_threshold: float = 0.05,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.frame_weight = float(frame_weight)
        self.frame_smooth_weight = float(frame_smooth_weight)
        self.frame_grad_threshold = float(frame_grad_threshold)
        self.frame_head = nn.Sequential(
            nn.Conv2d(16, frame_hidden, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(frame_hidden, 4, kernel_size=1),
        )
        sx: torch.Tensor = (
            torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
            / 8.0
        )
        sy: torch.Tensor = sx.transpose(2, 3).contiguous()
        self.register_buffer("sobel_x", sx)
        self.register_buffer("sobel_y", sy)
        self.save_hyperparameters(ignore=[])

    def _forward_triple(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        unet = self.model
        feats = unet.encoder(x)
        dec = unet.decoder(feats)
        seg = unet.segmentation_head(dec)
        sdf = torch.clamp(self.sdf_head(dec).squeeze(1), max=1.0)
        frame = self.frame_head(dec)  # (B, 4, H, W)
        return seg, sdf, frame

    def _compute_gt_tangent(
        self, y: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tangent direction at each pixel from gradient of field-interior mask.

        Returns (tx, ty, weight) each of shape (B, 1, H, W). Weight is 1 where
        the mask gradient magnitude exceeds ``frame_grad_threshold``.
        """
        field = (y == 1).float().unsqueeze(1)
        sx: torch.Tensor = self.get_buffer("sobel_x").to(field.dtype)
        sy: torch.Tensor = self.get_buffer("sobel_y").to(field.dtype)
        gx = F.conv2d(field, sx, padding=1)
        gy = F.conv2d(field, sy, padding=1)
        mag = (gx * gx + gy * gy).sqrt()
        m = mag.clamp(min=1e-3)
        nx, ny = gx / m, gy / m
        tx, ty = -ny, nx
        weight = (mag > self.frame_grad_threshold).float()
        return tx, ty, weight

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        x = batch["image"]
        y = batch["mask"].squeeze(1)
        sdf_target = batch["sdf"]

        seg, sdf_pred, frame = self._forward_triple(x)
        base = self.criterion(seg, y)

        sdf_target_n = sdf_target / self.sdf_clip
        ignore = self.hparams.get("ignore_index", 3)
        valid = (y != ignore).float()
        sdf_loss = ((sdf_pred - sdf_target_n).abs() * valid).sum() / valid.sum().clamp_min(1.0)

        # Frame field alignment loss: |f(tau)|^2 at boundary-adjacent pixels.
        tx, ty, fw = self._compute_gt_tangent(y)
        c0_re = frame[:, 0:1]
        c0_im = frame[:, 1:2]
        c2_re = frame[:, 2:3]
        c2_im = frame[:, 3:4]
        # tau^2 and tau^4 (tau = tx + i ty, complex squaring)
        t2_re = tx * tx - ty * ty
        t2_im = 2.0 * tx * ty
        t4_re = t2_re * t2_re - t2_im * t2_im
        t4_im = 2.0 * t2_re * t2_im
        # c2 * tau^2
        c2t2_re = c2_re * t2_re - c2_im * t2_im
        c2t2_im = c2_re * t2_im + c2_im * t2_re
        f_re = t4_re + c2t2_re + c0_re
        f_im = t4_im + c2t2_im + c0_im
        f_mag2 = f_re * f_re + f_im * f_im
        w = fw * valid.unsqueeze(1)
        frame_align = (f_mag2 * w).sum() / w.sum().clamp_min(1.0)

        # Spatial smoothness on (c0, c2): L2 of finite differences.
        c_dx = frame[:, :, :, 1:] - frame[:, :, :, :-1]
        c_dy = frame[:, :, 1:, :] - frame[:, :, :-1, :]
        frame_smooth = c_dx.pow(2).mean() + c_dy.pow(2).mean()

        loss = (
            base
            + self.sdf_weight * sdf_loss
            + self.frame_weight * frame_align
            + self.frame_smooth_weight * frame_smooth
        )

        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/loss_base", base, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/loss_sdf", sdf_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log(
            "train/loss_frame_align", frame_align, on_step=False, on_epoch=True, sync_dist=True
        )
        self.log(
            "train/loss_frame_smooth", frame_smooth, on_step=False, on_epoch=True, sync_dist=True
        )
        self.train_metrics.update(seg, y)
        return loss


# ---------------------------------------------------------------------------
# Paired Planet + S2 trainer with VICReg bottleneck alignment
# ---------------------------------------------------------------------------


class FTWPairedSegTask(FTWPlanetSegTask):
    """Joint Planet+S2 segmentation with VICReg cross-resolution alignment.

    Training batches contain paired samples from :class:`FTWPairedDataModule`:
    ``{planet_image, planet_mask, s2_image, s2_mask}``.  Both branches share
    the same UNet encoder/decoder.  The loss is:

        seg_loss(planet) + seg_loss(s2) + vicreg_weight * VICReg(z_pl, z_s2)

    where ``z_pl`` and ``z_s2`` are global-average-pooled bottleneck features
    (last encoder stage).  VICReg's invariance term pulls same-scene embeddings
    together; variance + covariance terms prevent representation collapse.

    Validation/test use Planet-only batches (``{image, mask}``), identical to
    :class:`FTWPlanetSegTask`.
    """

    def __init__(
        self,
        *args: Any,
        vicreg_weight: float = 1.0,
        vicreg_lambda: float = 25.0,
        vicreg_mu: float = 25.0,
        vicreg_nu: float = 1.0,
        grad_checkpointing: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vicreg_weight = float(vicreg_weight)
        self.vicreg_lambda = float(vicreg_lambda)
        self.vicreg_mu = float(vicreg_mu)
        self.vicreg_nu = float(vicreg_nu)
        self.grad_checkpointing = bool(grad_checkpointing)
        self.save_hyperparameters(ignore=[])

    def on_fit_start(self) -> None:
        """Enable gradient checkpointing on the timm backbone if requested.

        Both encoder forward passes share the same weights, so enabling
        checkpointing here halves the activation memory for encoder internals
        at the cost of ~33% extra compute (activations recomputed on backward).
        The UNet decoder skip connections still hold the encoder *outputs*
        (feature maps at each scale) — checkpointing only removes the
        intra-block intermediates.
        """
        super().on_fit_start()
        if not self.grad_checkpointing:
            return
        enc = self.model.encoder
        # smp wraps timm as enc.model; timm EfficientNet supports set_grad_checkpointing.
        timm_backbone = getattr(enc, "model", None)
        if timm_backbone is not None and hasattr(timm_backbone, "set_grad_checkpointing"):
            timm_backbone.set_grad_checkpointing(True)
            print("[FTWPairedSegTask] gradient checkpointing enabled on encoder backbone")

    def _encode_and_decode(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward through encoder+decoder; return (seg_logits, bottleneck_vec).

        ``bottleneck_vec`` is the global-average-pooled last encoder feature map
        (B, C) — suitable as input to VICReg.
        """
        unet = self.model
        feats = unet.encoder(x)          # list of feature maps, last = bottleneck
        dec = unet.decoder(feats)
        seg = unet.segmentation_head(dec)
        # Global avg pool over spatial dims → (B, C)
        z = feats[-1].mean(dim=[-2, -1])
        return seg, z

    def training_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        # Pop so the batch dict releases its references; tensors stay alive only
        # via our locals (and later the autograd graph).
        pl_x = batch.pop("planet_image")
        pl_y_raw = batch.pop("planet_mask")
        s2_x = batch.pop("s2_image")
        s2_y_raw = batch.pop("s2_mask")
        pl_y = pl_y_raw.squeeze(1) if pl_y_raw.dim() == 4 else pl_y_raw
        s2_y = s2_y_raw.squeeze(1) if s2_y_raw.dim() == 4 else s2_y_raw

        # Capture batch size before deleting input tensors; Lightning can't
        # infer it from the batch after we've popped everything.
        bs = pl_x.shape[0]

        pl_logits, z_pl = self._encode_and_decode(pl_x)
        del pl_x
        s2_logits, z_s2 = self._encode_and_decode(s2_x)
        del s2_x

        seg_pl = self.criterion(pl_logits, pl_y)
        seg_s2 = self.criterion(s2_logits, s2_y)
        seg = seg_pl + seg_s2

        vic = vicreg_loss(
            z_pl, z_s2,
            lambda_=self.vicreg_lambda,
            mu=self.vicreg_mu,
            nu=self.vicreg_nu,
        )
        loss = seg + self.vicreg_weight * vic

        log_kw: dict[str, Any] = {"batch_size": bs, "sync_dist": True}
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, **log_kw)
        self.log("train/loss_seg", seg, on_step=False, on_epoch=True, **log_kw)
        self.log("train/loss_seg_planet", seg_pl, on_step=False, on_epoch=True, **log_kw)
        self.log("train/loss_seg_s2", seg_s2, on_step=False, on_epoch=True, **log_kw)
        self.log("train/loss_vicreg", vic, on_step=False, on_epoch=True, **log_kw)
        # Track metrics on planet branch (primary eval modality)
        self.train_metrics.update(pl_logits, pl_y)
        return loss
