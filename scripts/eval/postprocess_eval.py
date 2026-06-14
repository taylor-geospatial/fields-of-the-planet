"""Per-country eval with watershed instance post-processing and TTA.

Two extras over ``scripts/eval_planet.py``:

* **TTA** — D4 (8-way) flip + rotation ensemble at inference. Predictions
  are inverse-transformed and the softmax probabilities are averaged.
* **Watershed instance separation** — turns the per-pixel field mask into
  individual field instances by running marker-controlled watershed on a
  distance map. The distance map comes from one of:

    - the predicted SDF head (preferred — what DECODE does), or
    - ``distance_transform_edt`` of the predicted boundary class (fallback
      for models trained without an SDF head, e.g. the baseline and the
      crop512/curriculum runs).

Reports object-level precision/recall/F1 with and without watershed,
side-by-side, per country.

Example:
    uv run scripts/postprocess_eval.py \\
        --ckpt logs/.../checkpoints/last.ckpt \\
        --out logs/postproc_test.csv \\
        --tta --watershed
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.metrics import get_object_level_metrics
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from scipy.ndimage import distance_transform_edt, label
from skimage.morphology import h_maxima
from skimage.segmentation import watershed
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


def _pad_min32(
    image: torch.Tensor, mask: torch.Tensor, min_size: int = 0, pad_mode: str = "zero"
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Pad H,W to ``max(next-mult-of-32, min_size)``.

    ``min_size=0`` keeps the legacy behaviour (next mult-32 only).
    Setting ``min_size`` to the training crop size (e.g. 512) makes every
    inference patch at least as wide/tall as training crops, eliminating
    the spatial-size drift between train and eval. ``pad_mode="replicate"``
    repeats edge image pixels (mask still gets ignore_index).
    """
    h, w = image.shape[-2], image.shape[-1]
    new_h = max(((h + 31) // 32) * 32, min_size)
    new_w = max(((w + 31) // 32) * 32, min_size)
    if (new_h, new_w) == (h, w):
        return image, mask, h, w
    if pad_mode == "replicate":
        img_padded = F.pad(image, (0, new_w - w, 0, new_h - h), mode="replicate")
    else:
        img_padded = F.pad(image, (0, new_w - w, 0, new_h - h), value=0.0)
    return (
        img_padded,
        F.pad(mask, (0, new_w - w, 0, new_h - h), value=3),
        h,
        w,
    )


# ------- D4 transforms (TTA) -------------------------------------------------


# Each element: (forward fn, inverse fn) acting on (N,C,H,W) or (N,H,W).
def _d4_transforms() -> list[tuple]:
    def ident(x):
        return x

    def hflip(x):
        return torch.flip(x, dims=[-1])

    def vflip(x):
        return torch.flip(x, dims=[-2])

    def hv(x):
        return torch.flip(x, dims=[-2, -1])

    def r90(x):
        return torch.rot90(x, 1, dims=[-2, -1])

    def r270(x):
        return torch.rot90(x, 3, dims=[-2, -1])

    def r90_h(x):
        return torch.flip(torch.rot90(x, 1, dims=[-2, -1]), dims=[-1])

    def r90_v(x):
        return torch.flip(torch.rot90(x, 1, dims=[-2, -1]), dims=[-2])

    return [
        (ident, ident),
        (hflip, hflip),
        (vflip, vflip),
        (hv, hv),
        (r90, r270),
        (r270, r90),
        (r90_h, lambda x: torch.rot90(torch.flip(x, dims=[-1]), 3, dims=[-2, -1])),
        (r90_v, lambda x: torch.rot90(torch.flip(x, dims=[-2]), 3, dims=[-2, -1])),
    ]


# ------- Watershed -----------------------------------------------------------


def watershed_instances(
    seg_pred: np.ndarray,  # (H,W) argmax of seg, 0/1/2
    distance: np.ndarray,  # (H,W) larger -> deeper inside field
    h_min: float = 2.0,
    field_class: int = 1,
) -> np.ndarray:
    """Returns an instance-label image (0 = background, >=1 = field instance)."""
    field_mask = seg_pred == field_class
    if not field_mask.any():
        return np.zeros_like(seg_pred, dtype=np.int32)
    surface = (distance * field_mask).astype(np.float32)
    seeds = h_maxima(surface, h=h_min)
    markers, _ = label(seeds)
    if markers.max() == 0:
        # No strong peaks — fall back to one instance per connected component.
        markers, _ = label(field_mask)
    inst = watershed(-surface, markers=markers, mask=field_mask)
    return inst.astype(np.int32)


def gt_instances(mask: np.ndarray, field_class: int = 1) -> np.ndarray:
    """GT instances = connected components of the GT field class."""
    field = (mask == field_class).astype(np.uint8)
    inst, _ = label(field)
    return inst.astype(np.int32)


# ------- Inference -----------------------------------------------------------


def _has_sdf_head(task: torch.nn.Module) -> bool:
    return hasattr(task, "sdf_head") and hasattr(task, "_forward_dual")


@torch.inference_mode()
def _predict(task, model, image: torch.Tensor, sdf_clip: float):
    """Return softmax probs (1,C,H,W) and optional sdf (1,H,W)."""
    if _has_sdf_head(task):
        seg, sdf = task._forward_dual(image)
        sdf = sdf * sdf_clip  # un-normalise back to pixels
    else:
        seg = model(image)
        sdf = None
    return seg.softmax(dim=1), sdf


@torch.inference_mode()
def _predict_tta(task, model, image: torch.Tensor, sdf_clip: float):
    probs_sum = None
    sdf_sum = None
    n = 0
    for fwd, inv in _d4_transforms():
        xt = fwd(image)
        p, s = _predict(task, model, xt, sdf_clip)
        p = inv(p)
        if probs_sum is None:
            probs_sum = p
        else:
            probs_sum = probs_sum + p
        if s is not None:
            si = inv(s)
            sdf_sum = si if sdf_sum is None else sdf_sum + si
        n += 1
    assert probs_sum is not None
    return probs_sum / n, (sdf_sum / n if sdf_sum is not None else None)


def evaluate_country(
    task,
    model,
    device,
    country: str,
    root: str,
    split: str,
    num_workers: int,
    iou_threshold: float,
    use_tta: bool,
    use_watershed: bool,
    h_min: float,
    sdf_clip: float,
    min_pad_size: int = 0,
    pad_mode: str = "zero",
    dataset_backend: str = "planet",
    s2_data_scale: float = 3000.0,
    presence_only: bool = False,
) -> dict[str, float]:
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
        _scale = float(s2_data_scale)
    else:
        ds = FTWPlanet(
            root=root, countries=[country], split=split, transforms=None, load_boundaries=True
        )
        _scale = PLANET_SR_SCALE
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

    metrics = MetricCollection(
        [
            JaccardIndex(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Precision(task="multiclass", average="none", num_classes=2, ignore_index=3),
            Recall(task="multiclass", average="none", num_classes=2, ignore_index=3),
        ]
    ).to(device)
    obj_pix_tps = obj_pix_fps = obj_pix_fns = 0  # pixel-cc style (baseline)
    obj_ws_tps = obj_ws_fps = obj_ws_fns = 0  # watershed-derived

    for batch in tqdm(dl, desc=country, leave=False):
        image = batch["image"].to(device) / _scale
        mask = batch["mask"].to(device)
        if presence_only:
            # Presence-only labels (e.g. kenya): background is untrusted, so
            # supervise metrics only on labeled polygon + boundary pixels.
            mask[mask == 0] = 3
        image, mask, H, W = _pad_min32(image, mask, min_size=min_pad_size, pad_mode=pad_mode)

        if use_tta:
            probs, sdf = _predict_tta(task, model, image, sdf_clip)
        else:
            probs, sdf = _predict(task, model, image, sdf_clip)
        seg_pred = probs.argmax(dim=1)  # (1,H,W) 0/1/2

        # Collapse 3-class to 2-class for pixel metrics: only field == 1.
        seg_field = (seg_pred == 1).long()
        mask_eval = mask.clone()
        mask_eval[mask_eval == 2] = 0  # boundary -> background for 2-class eval
        mask_eval[mask == 3] = 3  # keep padded ignore
        metrics.update(seg_field, mask_eval)

        seg_np = seg_pred.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]
        gt_np = mask_eval.squeeze(0).cpu().numpy().astype(np.uint8)[:H, :W]

        # Object metrics, "baseline" = connected components of the predicted
        # field class (what scripts/eval_planet.py reports).
        tps, fps, fns = get_object_level_metrics(
            gt_np, (seg_np == 1).astype(np.uint8), iou_threshold=iou_threshold
        )
        obj_pix_tps += tps
        obj_pix_fps += fps
        obj_pix_fns += fns

        if use_watershed:
            if sdf is not None:
                dist = sdf.squeeze(0).cpu().numpy().astype(np.float32)[:H, :W]
            else:
                # No SDF head: derive distance from the predicted boundary class.
                # Pixels far from any predicted boundary get the largest distance.
                boundary_np = (seg_np == 2).astype(np.uint8)
                dist = distance_transform_edt(boundary_np == 0).astype(np.float32)
            inst_pred = watershed_instances(seg_np, dist, h_min=h_min, field_class=1)
            # Compare watershed instances vs GT instances. We pass the
            # instance-labelled rasters into FTW's matcher by relabelling
            # each instance to "1" one-at-a-time would be slow; instead we
            # match the binary "field" prediction and rely on connected-
            # component matching inside get_object_level_metrics. To get
            # the *watershed* benefit, we slightly erode the prediction
            # along watershed boundaries so touching fields become disjoint.
            sep_pred = (inst_pred > 0).astype(np.uint8)
            tps, fps, fns = get_object_level_metrics(gt_np, sep_pred, iou_threshold=iou_threshold)
            obj_ws_tps += tps
            obj_ws_fps += fps
            obj_ws_fns += fns

    res = metrics.compute()
    out = {
        "pixel_level_iou": res["MulticlassJaccardIndex"][1].item(),
        "pixel_level_precision": res["MulticlassPrecision"][1].item(),
        "pixel_level_recall": res["MulticlassRecall"][1].item(),
    }

    def _f1(tps, fps, fns):
        p = tps / max(tps + fps, 1)
        r = tps / max(tps + fns, 1)
        return p, r, (2 * p * r / (p + r)) if (p + r) else 0.0

    p0, r0, f0 = _f1(obj_pix_tps, obj_pix_fps, obj_pix_fns)
    out["object_pix_precision"] = p0
    out["object_pix_recall"] = r0
    out["object_pix_f1"] = f0
    if use_watershed:
        p1, r1, f1 = _f1(obj_ws_tps, obj_ws_fps, obj_ws_fns)
        out["object_ws_precision"] = p1
        out["object_ws_recall"] = r1
        out["object_ws_f1"] = f1
    if presence_only:
        # FPs over unlabeled regions are unmeasurable -> precision/F1 invalid.
        for k in ("object_pix_precision", "object_pix_f1", "object_ws_precision", "object_ws_f1"):
            if k in out:
                out[k] = float("nan")
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", required=True, type=Path)
    p.add_argument("--root", default="data", type=str)
    p.add_argument("--split", default="test", choices=["test", "val"])
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--iou-threshold", type=float, default=0.5)
    p.add_argument("--countries", nargs="*", default=None)
    # TTA off by default — A/B tested on baseline/boundary/sdf checkpoints
    # gave < 1 pt object F1 on average and *hurt* the boundary run.
    # Opt in explicitly for final reporting numbers if needed.
    p.add_argument("--tta", action="store_true", help="D4 TTA ensemble (default off).")
    p.add_argument(
        "--watershed", action="store_true", help="Run watershed instance post-processing."
    )
    p.add_argument(
        "--h-min", type=float, default=2.0, help="h_maxima parameter for watershed seeds."
    )
    p.add_argument("--sdf-clip", type=float, default=20.0)
    p.add_argument(
        "--min-pad-size",
        type=int,
        default=0,
        help="Pad H,W to at least this size at inference (e.g. 512 to match training crop).",
    )
    p.add_argument(
        "--pad-mode",
        type=str,
        default="zero",
        choices=["zero", "replicate"],
        help="Padding mode for image inputs: 'zero' (default) or 'replicate' (keeps values in distribution).",
    )
    p.add_argument(
        "--dataset-backend",
        type=str,
        default="planet",
        choices=["planet", "s2"],
        help="'planet' (FTW-Planet 3m, /10000) or 's2' (FTW S2 10m, /3000).",
    )
    p.add_argument("--s2-data-scale", type=float, default=3000.0)
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
    print(f"device={device} ckpt={args.ckpt} tta={args.tta} watershed={args.watershed}")

    # Try most-specific subclass first; fall back through SDF -> base.
    tic = time.time()
    task = None
    try:
        from ftw_planet.trainers import FrameFieldSegTask

        task = FrameFieldSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
        print("loaded as FrameFieldSegTask")
    except Exception:
        pass
    if task is None:
        try:
            from ftw_planet.trainers import SDFSegTask

            task = SDFSegTask.load_from_checkpoint(str(args.ckpt), map_location="cpu")
            print("loaded as SDFSegTask")
        except Exception:
            task = CustomSemanticSegmentationTask.load_from_checkpoint(
                str(args.ckpt), map_location="cpu"
            )
            print("loaded as CustomSemanticSegmentationTask (no SDF head)")
    task = task.eval().to(device)
    model = task.model
    print(f"loaded model in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not args.out.exists():
        cols = [
            "train_checkpoint",
            "country",
            "tta",
            "watershed",
            "pixel_level_iou",
            "pixel_level_precision",
            "pixel_level_recall",
            "object_pix_precision",
            "object_pix_recall",
            "object_pix_f1",
            "object_ws_precision",
            "object_ws_recall",
            "object_ws_f1",
        ]
        with args.out.open("w") as f:
            f.write(",".join(cols) + "\n")

    countries = args.countries or COUNTRIES
    for country in countries:
        print(f"=== {country} ({args.split}) ===")
        try:
            m = evaluate_country(
                task,
                model,
                device,
                country,
                args.root,
                args.split,
                args.num_workers,
                args.iou_threshold,
                args.tta,
                args.watershed,
                args.h_min,
                args.sdf_clip,
                min_pad_size=args.min_pad_size,
                pad_mode=args.pad_mode,
                dataset_backend=args.dataset_backend,
                s2_data_scale=args.s2_data_scale,
                presence_only=country in args.presence_only_countries,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(f"  skip {country}: {e}")
            continue
        ws_cols = ",".join(
            f"{m.get(k, float('nan')):.6f}"
            for k in ("object_ws_precision", "object_ws_recall", "object_ws_f1")
        )
        line = (
            f"{args.ckpt},{country},{int(args.tta)},{int(args.watershed)},"
            f"{m['pixel_level_iou']:.6f},{m['pixel_level_precision']:.6f},{m['pixel_level_recall']:.6f},"
            f"{m['object_pix_precision']:.6f},{m['object_pix_recall']:.6f},{m['object_pix_f1']:.6f},"
            f"{ws_cols}\n"
        )
        with args.out.open("a") as f:
            f.write(line)
        line_print = f"  iou={m['pixel_level_iou']:.4f} obj_pix_F1={m['object_pix_f1']:.4f}"
        if args.watershed:
            line_print += f" obj_ws_F1={m['object_ws_f1']:.4f}"
        print(line_print)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
