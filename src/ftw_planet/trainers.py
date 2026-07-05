"""ftw-planet trainer.

Subclasses ``CustomSemanticSegmentationTask`` to log the learning rate through
``self.log`` (logger-agnostic, unlike the base class's TensorBoard-only call)
and to track per-country field-class IoU/precision/recall during validation.
"""

from typing import Any

import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask


class FTWPlanetSegTask(CustomSemanticSegmentationTask):
    def on_train_epoch_start(self) -> None:
        opts: Any = self.optimizers()
        opt = opts[0] if isinstance(opts, list) else opts
        self.log("lr", opt.optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def on_validation_epoch_start(self) -> None:  # type: ignore[override]
        super().on_validation_epoch_start()
        self.val_per_country: dict[str, list[int]] = {}  # country -> [tp, fp, fn]

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
            preds = self(x).argmax(dim=1)
        ignore = self.hparams.get("ignore_index", 3)
        valid = y != ignore
        p_field = (preds == 1) & valid
        t_field = (y == 1) & valid
        tp = (p_field & t_field).flatten(1).sum(dim=1)
        fp = (p_field & ~t_field).flatten(1).sum(dim=1)
        fn = (~p_field & t_field).flatten(1).sum(dim=1)
        for i, c in enumerate(countries):
            d = self.val_per_country.setdefault(c, [0, 0, 0])
            d[0] += int(tp[i].item())
            d[1] += int(fp[i].item())
            d[2] += int(fn[i].item())

    def on_validation_epoch_end(self) -> None:  # type: ignore[override]
        super().on_validation_epoch_end()
        for c, (tp, fp, fn) in getattr(self, "val_per_country", {}).items():
            denom = tp + fp + fn
            iou = tp / denom if denom > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            self.log(f"val/iou/field/{c}", iou, on_epoch=True, sync_dist=True)
            self.log(f"val/precision/field/{c}", prec, on_epoch=True, sync_dist=True)
            self.log(f"val/recall/field/{c}", rec, on_epoch=True, sync_dist=True)
