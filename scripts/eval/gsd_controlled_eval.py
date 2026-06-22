"""GSD-controlled pixel-IoU evaluation.

Question: does the Planet vs S2 pixel-IoU gap come from the model or from
the evaluation grid resolution?

For each patch we:
  1. Run the Planet model on the 3 m PlanetScope image -> pred_planet_3m
     (in the Planet label's UTM CRS, 3 m grid).
  2. Run the S2 model on the 10 m Sentinel-2 image -> pred_s2_10m
     (in the S2 label's EPSG:4326 grid, ~10 m).
  3. Score each prediction at *both* grids by warping (nearest neighbour)
     the prediction to the other grid and comparing against the GT raster
     defined on that grid.

So we get four conditions per country:
  planet@3m   pred_planet vs planet_gt_3m       (native, what paper reports)
  planet@10m  warp(pred_planet) vs s2_gt_10m    (Planet pred scored on S2 grid)
  s2@10m      pred_s2 vs s2_gt_10m              (native S2 numbers)
  s2@3m       warp(pred_s2) vs planet_gt_3m     (S2 pred scored on Planet grid)

The Planet GT raster and the S2 GT raster are independent rasterizations
of the *same* FTW polygons, so this is a clean controlled-grid comparison.

Outputs one CSV row per (model, eval_grid, country) with pixel IoU,
precision, recall, plus the underlying TP/FP/FN counts and a boundary-band
IoU (5 px ring around GT boundary in the eval grid).
"""

import argparse
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from rasterio.warp import Resampling, reproject
from scipy.ndimage import binary_dilation
from tqdm import tqdm

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

HELD_OUT_11 = [
    "belgium",
    "cambodia",
    "croatia",
    "germany",
    "kenya",
    "latvia",
    "lithuania",
    "portugal",
    "slovenia",
    "south_africa",
    "sweden",
]

IGNORE_INDEX = 3
S2_SCALE = 3000.0
FIELD_CLASS = 1


@dataclass
class Accum:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    btp: int = 0  # boundary-band TP
    bfp: int = 0
    bfn: int = 0
    btn: int = 0

    def add(
        self, pred: np.ndarray, gt: np.ndarray, ignore: np.ndarray, boundary: np.ndarray
    ) -> None:
        valid = ~ignore
        p = pred & valid
        g = gt & valid
        self.tp += int(np.logical_and(p, g).sum())
        self.fp += int(np.logical_and(p, ~g & valid).sum())
        self.fn += int(np.logical_and(~p & valid, g).sum())
        self.tn += int(np.logical_and(~p & valid, ~g & valid).sum())
        bvalid = valid & boundary
        bp = pred & bvalid
        bg = gt & bvalid
        self.btp += int(np.logical_and(bp, bg).sum())
        self.bfp += int(np.logical_and(bp, ~bg & bvalid).sum())
        self.bfn += int(np.logical_and(~bp & bvalid, bg).sum())
        self.btn += int(np.logical_and(~bp & bvalid, ~bg & bvalid).sum())

    def iou(self) -> float:
        denom = self.tp + self.fp + self.fn
        return self.tp / denom if denom else float("nan")

    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else float("nan")

    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else float("nan")

    def boundary_iou(self) -> float:
        denom = self.btp + self.bfp + self.bfn
        return self.btp / denom if denom else float("nan")


@dataclass
class CountryAccums:
    # (model, grid) -> Accum
    accs: dict[tuple[str, str], Accum] = field(default_factory=lambda: defaultdict(Accum))


# ---------- model loaders ----------------------------------------------------


def _load_task(ckpt: Path):
    """Try the FrameField / SDF / Planet base / S2 base task in order."""
    import argparse as _argparse

    from ftw_tools.training.trainers import CustomSemanticSegmentationTask

    from ftw_planet.trainers import FrameFieldSegTask, FTWPlanetSegTask, SDFSegTask

    last_err: Exception | None = None
    for cls in (FrameFieldSegTask, SDFSegTask, FTWPlanetSegTask, CustomSemanticSegmentationTask):
        try:
            return cls.load_from_checkpoint(str(ckpt), map_location="cpu")
        except (RuntimeError, KeyError, TypeError, _argparse.ArgumentError) as e:
            # Lightning + jsonargparse raises ArgumentError when the saved
            # class_path doesn't match the loader class; try the next one.
            last_err = e
            continue
    raise RuntimeError(f"could not load checkpoint {ckpt}: {last_err}")


# ---------- inference -------------------------------------------------------


def _pad_to_mult(image: torch.Tensor, mult: int = 32) -> tuple[torch.Tensor, int, int]:
    h, w = image.shape[-2], image.shape[-1]
    new_h = ((h + mult - 1) // mult) * mult
    new_w = ((w + mult - 1) // mult) * mult
    if (new_h, new_w) == (h, w):
        return image, h, w
    pad = (0, new_w - w, 0, new_h - h)
    return F.pad(image, pad, value=0.0), h, w


@torch.inference_mode()
def _argmax_pred(model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    image, h, w = _pad_to_mult(image, 32)
    out = model(image).argmax(dim=1)  # (1,H,W) 0/1/2
    return out.squeeze(0).cpu().numpy().astype(np.uint8)[:h, :w]


def _argmax_pred_rf(model: torch.nn.Module, image: torch.Tensor, resize_factor: int) -> np.ndarray:
    """resize_factor>1: bilinear-upsample the input, predict, then nearest-
    downsample the prediction back to the native grid. Mirrors ftw_tools'
    resize_factor inference so the S2 model sees the same finer pixel grid it is
    scored on in the headline tables; the returned map is at native resolution
    so the downstream warp/scoring is unchanged."""
    if resize_factor == 1:
        return _argmax_pred(model, image)
    h, w = image.shape[-2], image.shape[-1]
    up = torch.nn.functional.interpolate(
        image, size=(h * resize_factor, w * resize_factor), mode="bilinear", align_corners=False
    )
    upp, uh, uw = _pad_to_mult(up, 32)
    pred = model(upp).argmax(dim=1)[:, :uh, :uw].float()  # (1, h*rf, w*rf)
    pred = torch.nn.functional.interpolate(pred.unsqueeze(1), size=(h, w), mode="nearest")
    return pred.squeeze(1).squeeze(0).cpu().numpy().astype(np.uint8)


# ---------- raster IO + warp -----------------------------------------------


def _read_label(path: str) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1)
        meta = {
            "transform": src.transform,
            "crs": src.crs,
            "shape": (src.height, src.width),
        }
    return arr, meta


def _warp_nearest(pred: np.ndarray, src_meta: dict, dst_meta: dict) -> np.ndarray:
    dst = np.full(dst_meta["shape"], 255, dtype=np.uint8)
    reproject(
        source=pred,
        destination=dst,
        src_transform=src_meta["transform"],
        src_crs=src_meta["crs"],
        dst_transform=dst_meta["transform"],
        dst_crs=dst_meta["crs"],
        resampling=Resampling.nearest,
    )
    return dst


def _boundary_band(gt_field: np.ndarray, radius: int = 5) -> np.ndarray:
    """Pixels within `radius` of the GT field boundary (eroded XOR dilated)."""
    if not gt_field.any():
        return np.zeros_like(gt_field, dtype=bool)
    struct = np.ones((3, 3), dtype=bool)
    dil = gt_field.astype(bool)
    for _ in range(radius):
        dil = binary_dilation(dil, structure=struct)
    ero = gt_field.astype(bool)
    inv = ~ero
    for _ in range(radius):
        inv = binary_dilation(inv, structure=struct)
    ero = ~inv
    return dil & ~ero


# ---------- main per-patch routine ------------------------------------------


def _norm_image(arr: np.ndarray, scale: float) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / scale).unsqueeze(0)


def _read_pair(window_a: str, window_b: str) -> np.ndarray:
    """Stack [B, A] -> (8, H, W) uint16."""
    with rasterio.open(window_b) as src:
        b = src.read()
    with rasterio.open(window_a) as src:
        a = src.read()
    h = min(a.shape[1], b.shape[1])
    w = min(a.shape[2], b.shape[2])
    return np.concatenate([b[:, :h, :w], a[:, :h, :w]], axis=0)


def evaluate_country(
    country: str,
    planet_model: torch.nn.Module,
    s2_model: torch.nn.Module,
    device: torch.device,
    root: str,
    split: str,
    s2_resize_factor: int = 1,
) -> dict[tuple[str, str], Accum]:
    """Returns {(model, grid): Accum} for the country."""
    ds = FTWPlanet(
        root=root, countries=[country], split=split, transforms=None, load_boundaries=True
    )

    accs: dict[tuple[str, str], Accum] = defaultdict(Accum)

    s2_country_root = Path(root) / "ftw" / country
    for rec in tqdm(ds.records, desc=country, leave=False):
        pid = rec["patch_id"]
        s2_a = s2_country_root / "s2_images" / "window_a" / f"{pid}.tif"
        s2_b = s2_country_root / "s2_images" / "window_b" / f"{pid}.tif"
        s2_lbl_path = s2_country_root / "label_masks" / "semantic_2class" / f"{pid}.tif"
        if not (s2_a.exists() and s2_b.exists() and s2_lbl_path.exists()):
            # Missing S2 side — skip patch (must be paired for the controlled compare).
            continue

        # Planet image + GT
        planet_img = _read_pair(rec["window_a"], rec["window_b"])
        planet_gt_arr, planet_gt_meta = _read_label(rec["label"])

        # S2 image + GT
        s2_img = _read_pair(str(s2_a), str(s2_b))
        s2_gt_arr, s2_gt_meta = _read_label(str(s2_lbl_path))

        # Crop labels/images consistently
        ph = min(planet_img.shape[1], planet_gt_arr.shape[0])
        pw = min(planet_img.shape[2], planet_gt_arr.shape[1])
        planet_img = planet_img[:, :ph, :pw]
        planet_gt_arr = planet_gt_arr[:ph, :pw]
        planet_gt_meta["shape"] = (ph, pw)
        sh = min(s2_img.shape[1], s2_gt_arr.shape[0])
        sw = min(s2_img.shape[2], s2_gt_arr.shape[1])
        s2_img = s2_img[:, :sh, :sw]
        s2_gt_arr = s2_gt_arr[:sh, :sw]
        s2_gt_meta["shape"] = (sh, sw)

        # ---- inference ----
        p_in = _norm_image(planet_img, PLANET_SR_SCALE).to(device)
        pred_planet = _argmax_pred(planet_model, p_in)  # 3 m grid, 0/1/2

        s_in = _norm_image(s2_img, S2_SCALE).to(device)
        pred_s2 = _argmax_pred_rf(s2_model, s_in, s2_resize_factor)  # 10 m grid, 0/1/2

        # ---- prepare GT masks (2-class: field vs not-field; boundary->bg) ----
        # Planet GT: 0=bg, 1=field, 2=boundary (when load_boundaries semantics).
        planet_gt_field = planet_gt_arr == FIELD_CLASS
        planet_gt_ignore = planet_gt_arr == IGNORE_INDEX
        s2_gt_field = s2_gt_arr == FIELD_CLASS
        s2_gt_ignore = s2_gt_arr == IGNORE_INDEX

        planet_band = _boundary_band(planet_gt_field, radius=5)
        s2_band = _boundary_band(s2_gt_field, radius=2)  # ~5 Planet px ≈ 2 S2 px

        # ---- score predictions ----
        # Planet @ Planet grid (native)
        p_pred_field = pred_planet == FIELD_CLASS
        accs[("planet", "3m")].add(p_pred_field, planet_gt_field, planet_gt_ignore, planet_band)

        # S2 @ S2 grid (native)
        s_pred_field = pred_s2 == FIELD_CLASS
        accs[("s2", "10m")].add(s_pred_field, s2_gt_field, s2_gt_ignore, s2_band)

        # Planet @ S2 grid (warp Planet pred nearest into S2 EPSG:4326 grid)
        planet_meta_for_warp = {
            "transform": planet_gt_meta["transform"],
            "crs": planet_gt_meta["crs"],
            "shape": (ph, pw),
        }
        warped = _warp_nearest(p_pred_field.astype(np.uint8), planet_meta_for_warp, s2_gt_meta)
        warp_ignore = s2_gt_ignore | (warped == 255)
        accs[("planet", "10m")].add(
            warped.astype(bool) & (warped != 255), s2_gt_field, warp_ignore, s2_band
        )

        # S2 @ Planet grid (warp S2 pred nearest into Planet UTM 3 m grid)
        s2_meta_for_warp = {
            "transform": s2_gt_meta["transform"],
            "crs": s2_gt_meta["crs"],
            "shape": (sh, sw),
        }
        warped = _warp_nearest(s_pred_field.astype(np.uint8), s2_meta_for_warp, planet_gt_meta)
        warp_ignore = planet_gt_ignore | (warped == 255)
        accs[("s2", "3m")].add(
            warped.astype(bool) & (warped != 255), planet_gt_field, warp_ignore, planet_band
        )

    return accs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planet-ckpt", type=Path, required=True)
    ap.add_argument("--s2-ckpt", type=Path, required=True)
    ap.add_argument("--root", default="data", type=str)
    ap.add_argument("--split", default="test", choices=["test", "val"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--countries", nargs="*", default=None, help="Defaults to 11 held-out.")
    ap.add_argument(
        "--s2-resize-factor",
        type=int,
        default=1,
        help="Upsample the S2 input by this factor before inference (resize_factor; "
        "2 matches the upsample-512 headline eval). Prediction is downsampled back to native.",
    )
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
    )
    print(f"device={device}")

    tic = time.time()
    planet_task = _load_task(args.planet_ckpt).eval().to(device)
    s2_task = _load_task(args.s2_ckpt).eval().to(device)
    planet_model = planet_task.model
    s2_model = s2_task.model
    print(f"loaded both checkpoints in {time.time() - tic:.1f}s")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "country",
        "model",
        "eval_grid",
        "iou",
        "precision",
        "recall",
        "boundary_iou",
        "tp",
        "fp",
        "fn",
        "tn",
    ]
    with args.out.open("w") as f:
        f.write(",".join(cols) + "\n")

    countries = args.countries or HELD_OUT_11
    macro: dict[tuple[str, str], list[float]] = defaultdict(list)

    for country in countries:
        print(f"=== {country} ===")
        try:
            accs = evaluate_country(
                country,
                planet_model,
                s2_model,
                device,
                args.root,
                args.split,
                s2_resize_factor=args.s2_resize_factor,
            )
        except (RuntimeError, FileNotFoundError) as e:
            print(f"  skip {country}: {type(e).__name__}: {e}")
            continue
        for (model_name, grid), a in accs.items():
            iou = a.iou()
            prec = a.precision()
            rec = a.recall()
            biou = a.boundary_iou()
            macro[(model_name, grid)].append(iou)
            line = (
                f"{country},{model_name},{grid},"
                f"{iou:.6f},{prec:.6f},{rec:.6f},{biou:.6f},"
                f"{a.tp},{a.fp},{a.fn},{a.tn}\n"
            )
            with args.out.open("a") as f:
                f.write(line)
            print(f"  {model_name}@{grid}: IoU={iou:.4f} bIoU={biou:.4f}")

    print("\n=== macro IoU over countries ===")
    for k, vals in sorted(macro.items()):
        v = np.array(vals, dtype=np.float64)
        # average ignoring NaN (e.g. countries with no field pixels)
        m = float(np.nanmean(v))
        print(f"  {k[0]}@{k[1]:>4}: {m:.4f}  (n={len(vals)})")
        with args.out.open("a") as f:
            f.write(f"MACRO,{k[0]},{k[1]},{m:.6f},,,,,,,\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
