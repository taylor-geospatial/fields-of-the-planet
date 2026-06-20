"""Per-patch polygon metrics: one CSV row per held-out patch.

Same inference + watershed + greedy-matching protocol as
``polygon_metrics_eval.py`` (TTA, min-pad 512, IoU>=0.5 matching), but instead
of aggregating per country it dumps a row per patch with the patch id parsed
from the dataset's file records. Aligning the patch id requires ``batch_size=1``
and ``shuffle=False`` so the DataLoader order matches the record order.

Columns: country, patch_id, n_gt, n_pred, obj_f1 (= RQ at IoU0.5), pq, sq,
pixel_iou.

Example:
    uv run scripts/eval/per_patch_metrics.py \\
        --ckpt logs/best_checkpoints/planet_efnet3_augmax_full_best.ckpt \\
        --out logs/per_patch/planet_b3.csv \\
        --dataset-backend planet --countries croatia \\
        --watershed --tta --min-pad-size 512
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "eval"))

from polygon_metrics_eval import (  # noqa: E402
    AP_IOU_THRESHOLDS,
    _extract_shapes,
    _match_shapes,
)
from postprocess_eval import _pad_min32, _predict, _predict_tta, watershed_instances  # noqa: E402

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet  # noqa: E402


def _patch_ids_for_dataset(ds, dataset_backend: str) -> list[str]:
    """Patch id per dataset record, in DataLoader (shuffle=False) order."""
    if dataset_backend == "s2":
        # ftw_tools FTW stores per-sample file records in ``filenames``.
        return [Path(rec["window_b"]).stem for rec in ds.filenames]
    # FTWPlanet stores explicit patch_id in ``records``.
    return [rec["patch_id"] for rec in ds.records]


def evaluate_country_per_patch(
    task,
    model,
    device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    use_tta: bool,
    use_watershed: bool,
    h_min: float,
    sdf_clip: float,
    min_pad_size: int,
    pad_mode: str,
    dataset_backend: str,
    s2_data_scale: float,
) -> list[dict]:
    if dataset_backend == "s2":
        from ftw_tools.training.datasets import FTW

        ds = FTW(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
            temporal_options="stacked",
        )
        scale = float(s2_data_scale)
    else:
        ds = FTWPlanet(
            root=root,
            countries=[country],
            split=split,
            transforms=None,
            load_boundaries=True,
        )
        scale = PLANET_SR_SCALE

    patch_ids = _patch_ids_for_dataset(ds, dataset_backend)
    if len(patch_ids) != len(ds):
        raise RuntimeError(
            f"patch_id count {len(patch_ids)} != dataset length {len(ds)} for {country}"
        )

    # batch_size=1, shuffle=False keeps the iteration order identical to the
    # record order so each batch lines up with patch_ids[i].
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    t05 = AP_IOU_THRESHOLDS[0]
    rows: list[dict] = []

    for i, batch in enumerate(tqdm(dl, desc=country, leave=False)):
        image = batch["image"].to(device) / scale
        mask = batch["mask"].to(device)
        image, mask, H, W = _pad_min32(image, mask, min_size=min_pad_size, pad_mode=pad_mode)

        if use_tta:
            probs, sdf = _predict_tta(task, model, image, sdf_clip)
        else:
            probs, sdf = _predict(task, model, image, sdf_clip)
        seg_pred = probs.argmax(dim=1)  # (1,H,W) 0/1/2

        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0
        mask_eval[mask == 3] = 3

        seg_np = seg_pred.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        if use_watershed:
            if sdf is not None:
                dist = sdf.squeeze(0).cpu().numpy().astype(np.float32)[:H, :W]
            else:
                boundary_np = (seg_np == 2).astype(np.uint8)
                dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
            inst_pred = watershed_instances(seg_np, dist, h_min=h_min, field_class=1)
            pred_bin = (inst_pred > 0).astype(np.uint8)
        else:
            pred_bin = (seg_np == 1).astype(np.uint8)

        gt_bin = (gt_np == 1).astype(np.uint8)

        gt_shapes = _extract_shapes(gt_bin)
        pred_shapes = _extract_shapes(pred_bin)
        m = _match_shapes(gt_shapes, pred_shapes, AP_IOU_THRESHOLDS)

        tps, fps, fns = m["per_t"][t05]
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        obj_f1 = (2 * p * r / (p + r)) if (p + r) else 0.0
        matched_ious = [iou for _, _, iou in m["matched_pairs_low"]]
        sq = float(np.mean(matched_ious)) if matched_ious else 0.0
        pq = sq * obj_f1

        inter = int((pred_bin & gt_bin).sum())
        union = int((pred_bin | gt_bin).sum())
        pixel_iou = inter / union if union else 0.0

        rows.append(
            {
                "country": country,
                "patch_id": patch_ids[i],
                "n_gt": len(gt_shapes),
                "n_pred": len(pred_shapes),
                "obj_f1": obj_f1,
                "pq": pq,
                "sq": sq,
                "pixel_iou": pixel_iou,
            }
        )
    return rows


def _load_task(ckpt: Path, device) -> torch.nn.Module:
    """Load a checkpoint using the exact task class named in its hparams.

    The checkpoint records its ``_class_path`` (e.g.
    ``ftw_planet.trainers.FTWPlanetSegTask``); we import and use that class so
    the loaded module exposes the right ``_forward_dual`` / ``sdf_head`` when
    present, and we fail loud if the class is missing rather than silently
    falling back to a base class with the wrong heads.
    """
    import importlib

    hp = torch.load(str(ckpt), map_location="cpu", weights_only=False).get("hyper_parameters", {})
    class_path = hp.get("_class_path")
    if class_path:
        module_name, cls_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_name), cls_name)
    else:
        cls = CustomSemanticSegmentationTask
    task = cls.load_from_checkpoint(str(ckpt), map_location="cpu")
    print(f"loaded as {type(task).__name__}")
    return task.eval().to(device)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--countries", nargs="*", default=None, required=True)
    p.add_argument("--tta", action="store_true")
    p.add_argument("--watershed", action="store_true")
    p.add_argument("--h-min", type=float, default=2.0)
    p.add_argument("--sdf-clip", type=float, default=20.0)
    p.add_argument("--min-pad-size", type=int, default=0)
    p.add_argument("--pad-mode", type=str, default="zero", choices=["zero", "replicate"])
    p.add_argument("--dataset-backend", type=str, default="planet", choices=["planet", "s2"])
    p.add_argument("--s2-data-scale", type=float, default=3000.0)
    args = p.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(
        f"device={device} ckpt={args.ckpt} backend={args.dataset_backend} "
        f"tta={args.tta} watershed={args.watershed}"
    )

    tic = time.time()
    task = _load_task(args.ckpt, device)
    model = task.model
    print(f"loaded model in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["country", "patch_id", "n_gt", "n_pred", "obj_f1", "pq", "sq", "pixel_iou"]
    if not args.out.exists():
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")

    for country in args.countries:
        print(f"=== {country} ({args.split}) ===")
        rows = evaluate_country_per_patch(
            task,
            model,
            device,
            country,
            args.root,
            args.split,
            args.num_workers,
            args.tta,
            args.watershed,
            args.h_min,
            args.sdf_clip,
            min_pad_size=args.min_pad_size,
            pad_mode=args.pad_mode,
            dataset_backend=args.dataset_backend,
            s2_data_scale=args.s2_data_scale,
        )
        with args.out.open("a") as f:
            for row in rows:
                f.write(
                    f"{row['country']},{row['patch_id']},{row['n_gt']},{row['n_pred']},"
                    f"{row['obj_f1']:.6f},{row['pq']:.6f},{row['sq']:.6f},{row['pixel_iou']:.6f}\n"
                )
        mean_f1 = float(np.mean([r["obj_f1"] for r in rows])) if rows else 0.0
        print(f"  wrote {len(rows)} rows, mean obj_f1={mean_f1:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
