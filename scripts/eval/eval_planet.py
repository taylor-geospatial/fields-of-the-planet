"""Per-country evaluation for the ftw-planet PRUE model.

Adapted from ``ftw_tools.training.eval.test``:

* swaps the S2 ``FTW`` dataset for :class:`ftw_planet.datasets.FTWPlanet`,
* uses /10000 (PlanetScope SR) normalization instead of /3000,
* pads variable-sized Planet patches to the nearest multiple of 32 so
  the UNet's stride-5 encoder accepts them (single sample per batch).

Reports pixel-level IoU/precision/recall and object-level
precision/recall/F1 per country, writes one CSV row per (checkpoint,
country) pair. Mirrors the schema of ftw-baselines' ``run_eval.py`` so
results can be concatenated directly.

Example:
    uv run scripts/eval_planet.py \\
        --ckpt logs/prue/ftw_planet-unet-efnet3-bf16/.../checkpoints/last.ckpt \\
        --out logs/eval_planet.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.metrics import get_object_level_metrics
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from torch.utils.data import DataLoader
from torchmetrics import JaccardIndex, MetricCollection, Precision, Recall
from tqdm import tqdm

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

COUNTRIES = [
    "austria",
    "belgium",
    "brazil",
    "cambodia",
    "corsica",
    "croatia",
    "denmark",
    "estonia",
    "finland",
    "france",
    "germany",
    "india",
    "kenya",
    "latvia",
    "lithuania",
    "luxembourg",
    "netherlands",
    "portugal",
    "rwanda",
    "slovakia",
    "slovenia",
    "south_africa",
    "spain",
    "sweden",
    "vietnam",
]


def _pad_mult32(image: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad H,W up to the next multiple of 32 so the UNet stride-5 stack accepts it."""
    h, w = image.shape[-2], image.shape[-1]
    new_h = ((h + 31) // 32) * 32
    new_w = ((w + 31) // 32) * 32
    ph, pw = new_h - h, new_w - w
    if ph or pw:
        image = F.pad(image, (0, pw, 0, ph), value=0.0)
        mask = F.pad(mask, (0, pw, 0, ph), value=3)  # ignore_index
    return image, mask


@torch.inference_mode()
def evaluate_country(
    model: torch.nn.Module,
    device: torch.device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    iou_threshold: float,
    model_predicts_3_classes: bool,
    presence_only: bool = False,
) -> dict[str, float]:
    ds = FTWPlanet(
        root=root, countries=[country], split=split, transforms=None, load_boundaries=True
    )
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    # 2-class evaluation (background vs field) — matches upstream test() default.
    metrics = MetricCollection(
        [
            JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=2, ignore_index=3),
        ]
    ).to(device)
    all_tps = all_fps = all_fns = 0

    for batch in tqdm(dl, desc=country, leave=False):
        image = batch["image"].to(device) / PLANET_SR_SCALE
        mask = batch["mask"].to(device)
        if presence_only:
            # Presence-only labels (e.g. kenya): background is untrusted, so
            # supervise metrics only on labeled polygon + boundary pixels.
            mask[mask == 0] = 3
        image, mask = _pad_mult32(image, mask)

        outputs = model(image).argmax(dim=1)
        if model_predicts_3_classes:
            # collapse: 0/2 -> 0 (bg + boundary), 1 -> 1 (field)
            outputs = (outputs == 1).long()
            mask_eval = mask.clone()
            mask_eval[mask_eval == 2] = 0  # treat boundary as bg for eval
            mask_eval[mask == 3] = 3  # preserve pad ignore
        else:
            mask_eval = mask

        metrics.update(outputs, mask_eval)
        out_np = outputs.squeeze(0).cpu().numpy().astype(np.uint8)
        msk_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)
        tps, fps, fns = get_object_level_metrics(msk_np, out_np, iou_threshold=iou_threshold)
        all_tps += tps
        all_fps += fps
        all_fns += fns

    res = metrics.compute()
    pix_iou = res["MulticlassJaccardIndex"][1].item()
    pix_prec = res["MulticlassPrecision"][1].item()
    pix_rec = res["MulticlassRecall"][1].item()
    obj_prec = all_tps / (all_tps + all_fps) if (all_tps + all_fps) else float("nan")
    obj_rec = all_tps / (all_tps + all_fns) if (all_tps + all_fns) else float("nan")
    obj_f1 = (
        2 * obj_prec * obj_rec / (obj_prec + obj_rec)
        if (obj_prec + obj_rec) and not (np.isnan(obj_prec) or np.isnan(obj_rec))
        else float("nan")
    )
    if presence_only:
        # FPs over unlabeled regions are unmeasurable -> precision/F1 invalid.
        obj_prec = float("nan")
        obj_f1 = float("nan")
    return {
        "pixel_level_iou": pix_iou,
        "pixel_level_precision": pix_prec,
        "pixel_level_recall": pix_rec,
        "object_level_precision": obj_prec,
        "object_level_recall": obj_rec,
        "object_level_f1": obj_f1,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--countries", nargs="*", default=None, help="Defaults to all 25.")
    p.add_argument(
        "--model-predicts-3-classes",
        action="store_true",
        help="Collapse 3-class predictions (0/1/2) to 2-class (bg/field) before eval.",
    )
    p.add_argument(
        "--presence-only-countries",
        nargs="*",
        default=[],
        help="Countries whose labels are presence-only (e.g. kenya): background "
        "pixels are masked to ignore_index=3 and object precision/F1 are NaN.",
    )
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"device={device} ckpt={args.ckpt}")

    tic = time.time()
    task = CustomSemanticSegmentationTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
    model = task.model.eval().to(device)
    print(f"loaded model in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(
                "train_checkpoint,countries,pixel_level_iou,pixel_level_precision,"
                "pixel_level_recall,object_level_precision,object_level_recall,"
                "object_level_f1\n"
            )

    countries = args.countries or COUNTRIES
    for country in countries:
        print(f"=== {country} ({args.split}) ===")
        try:
            m = evaluate_country(
                model,
                device,
                country,
                args.root,
                args.split,
                args.num_workers,
                args.iou_threshold,
                args.model_predicts_3_classes,
                presence_only=country in args.presence_only_countries,
            )
        except Exception as e:
            print(f"  skip {country}: {e}")
            continue
        line = (
            f"{args.ckpt},{country},{m['pixel_level_iou']:.6f},"
            f"{m['pixel_level_precision']:.6f},{m['pixel_level_recall']:.6f},"
            f"{m['object_level_precision']:.6f},{m['object_level_recall']:.6f},"
            f"{m['object_level_f1']:.6f}\n"
        )
        with args.out.open("a") as f:
            f.write(line)
        print(
            f"  iou={m['pixel_level_iou']:.4f} "
            f"prec={m['pixel_level_precision']:.4f} "
            f"rec={m['pixel_level_recall']:.4f} "
            f"objF1={m['object_level_f1']:.4f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
