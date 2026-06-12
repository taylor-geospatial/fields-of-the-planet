"""Qualitative appendix figure: PRUE-FTP-B3 vs DelineateAnything (zero-shot)
on the same dense-smallholder rows used by ``make_qualitative_v8.py``.

4 columns x N rows:
  Planet RGB | GT instances | PRUE-FTP-B3 (TTA+WS) instances | DelineateAnything

DelineateAnything is the off-the-shelf YOLO11x-seg checkpoint from HF
(``torchgeo/delineate-anything``), run on window-B RGB uint8 with the
authors' canonical inference settings (conf=0.005, NMS IoU=0.5, FP16,
max_det=2000). Each color = one field instance (perturbed tab20).
"""

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from ftw_tools.training.trainers import CustomSemanticSegmentationTask
from huggingface_hub import hf_hub_download
from make_qualitative_v8 import (
    _instance_render,
    _predict_raw_and_tta,
    _resize_nn,
    _square_crop,
    _stretch,
    _watershed_instances,
)
from scipy.ndimage import label as cc_label
from ultralytics import (
    YOLO,  # ultralytics not in main CI deps; paper-scripts only
)

from ftw_planet.datasets import PLANET_SR_SCALE, FTWPlanet

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times"],
        "font.size": 8,
    }
)

SQUARE_SIZE = 256
RGB_REFL_CEILING = 3000.0


def _load_planet(country: str, pid: str):
    ds = FTWPlanet(root="data", countries=[country], split="test", load_boundaries=True)
    idx = next((i for i, r in enumerate(ds.records) if r["patch_id"] == str(pid)), None)
    if idx is None:
        raise RuntimeError(f"patch {country}:{pid} not found")
    s = ds[idx]
    x = s["image"]
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(np.asarray(x))
    y = s["mask"].cpu().numpy() if isinstance(s["mask"], torch.Tensor) else np.asarray(s["mask"])
    return x.float(), y.astype(np.uint8)


def _planet_rgb_for_window(country: str, pid: str, window: str):
    p = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(p) as src:
        rgb = src.read([3, 2, 1])
    return np.transpose(rgb, (1, 2, 0)).astype(np.float32) / PLANET_SR_SCALE


def _planet_rgb_uint8_for_yolo(country: str, pid: str, window: str):
    """Same RGB rescaling the eval script feeds DelineateAnything."""
    p = Path("data/planet") / country / f"window_{window}" / f"{pid}.tif"
    with rasterio.open(p) as src:
        rgb = src.read([3, 2, 1]).astype(np.float32)
    rgb = np.clip(rgb / RGB_REFL_CEILING, 0.0, 1.0)
    return np.transpose((rgb * 255).astype(np.uint8), (1, 2, 0))


def _yolo_instances(
    yolo: YOLO, rgb_u8: np.ndarray, imgsz: int, conf: float, iou: float, device: torch.device
) -> np.ndarray:
    H, W = rgb_u8.shape[:2]
    r = yolo.predict(
        source=rgb_u8,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=str(device) if device.type != "cpu" else "cpu",
        verbose=False,
        retina_masks=True,
        max_det=2000,
        half=device.type == "cuda",
    )[0]
    if r.masks is None or r.masks.data is None or len(r.masks.data) == 0:
        return np.zeros((H, W), dtype=np.int32)
    masks = r.masks.data.cpu().numpy().astype(np.uint8)  # ty: ignore[unresolved-attribute]  # YOLO Results.masks.data is always Tensor here; guarded by None-check above  # (N, h, w)
    inst = np.zeros((H, W), dtype=np.int32)
    for k in range(masks.shape[0]):
        m_k = masks[k]
        if m_k.shape != (H, W):
            m_k = m_k[:H, :W]
            if m_k.shape != (H, W):
                pad_h, pad_w = H - m_k.shape[0], W - m_k.shape[1]
                m_k = np.pad(m_k, ((0, max(0, pad_h)), (0, max(0, pad_w))))
        # Each detection becomes one connected instance; if YOLO returns a
        # multi-blob mask we keep them all under the same instance id so the
        # color count tracks detection count, not blob count.
        if m_k.sum() == 0:
            continue
        inst[(inst == 0) & (m_k > 0)] = k + 1
    return inst


def _gt_instances(mask: np.ndarray) -> np.ndarray:
    field = (mask == 1).astype(np.uint8)
    inst, _ = cc_label(field)
    return inst.astype(np.int32)


def _load_planet_task(ckpt: Path, device: torch.device):
    t = CustomSemanticSegmentationTask.load_from_checkpoint(str(ckpt), map_location="cpu")
    return t.model.eval().to(device)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--ckpt-planet",
        default=(
            "logs/prue/ftw_planet-unet-efnet3-crop512-v3-augmax-full/"
            "ftw-planet/mt6mdnl7/checkpoints/last.ckpt"
        ),
    )
    p.add_argument(
        "--rows",
        nargs="+",
        default=[
            "croatia:g10-3_00071_11:a",
            "slovenia:g13_00033_1:a",
            "austria:g83_00031_18:a",
            "lithuania:g11_00088_0:a",
        ],
    )
    p.add_argument("--out", default="paper/figs/qualitative_delineate.pdf")
    p.add_argument("--cell-size", type=int, default=SQUARE_SIZE)
    p.add_argument("--cell-h", type=float, default=1.05)
    p.add_argument("--cell-w", type=float, default=1.05)
    p.add_argument("--imgsz", type=int, default=1024)
    p.add_argument("--conf", type=float, default=0.005)
    p.add_argument("--iou", type=float, default=0.5)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    model_pl = _load_planet_task(args.ckpt_planet, device)
    ckpt = hf_hub_download(
        repo_id="torchgeo/delineate-anything",
        filename="delineate_anything_rgb_yolo11x-88ede029.pt",
    )
    yolo = YOLO(ckpt)

    rows = []
    for spec in args.rows:
        country, pid, window = (spec.split(":") + ["a"])[:3]
        print(f"  {country}:{pid} ({window})")
        x8, gt_full = _load_planet(country, pid)
        rgb_pl = _planet_rgb_for_window(country, pid, window)
        rgb_u8 = _planet_rgb_uint8_for_yolo(country, pid, window)

        _raw, tta_pl = _predict_raw_and_tta(model_pl, x8, device, scale=PLANET_SR_SCALE)
        inst_pl = _watershed_instances(tta_pl)
        inst_da = _yolo_instances(yolo, rgb_u8, args.imgsz, args.conf, args.iou, device)
        inst_gt = _gt_instances(gt_full)

        sq = args.cell_size
        rgb_pl_s = _resize_nn(_stretch(_square_crop(rgb_pl)), sq)
        inst_gt_s = _resize_nn(_square_crop(inst_gt), sq)
        inst_pl_s = _resize_nn(_square_crop(inst_pl), sq)
        inst_da_s = _resize_nn(_square_crop(inst_da), sq)
        print(
            f"    instances: gt={int(inst_gt.max())} "
            f"prue-hd={int(inst_pl.max())} delineate={int(inst_da.max())}"
        )
        rows.append((country, pid, window, rgb_pl_s, inst_gt_s, inst_pl_s, inst_da_s))

    n = len(rows)
    cols = 4
    _fig, axes = plt.subplots(
        n,
        cols,
        figsize=(cols * args.cell_w, n * args.cell_h),
        gridspec_kw={"wspace": 0.015, "hspace": 0.03},
    )
    if n == 1:
        axes = axes[None, :]
    col_titles = [
        "Input\nPlanet RGB",
        "GT instances",
        "PRUE-FTP-B3",
        "DelineateAnything",
    ]
    for i, (country, pid, window, rgb_pl, inst_gt, inst_pl, inst_da) in enumerate(rows):
        axes[i, 0].imshow(rgb_pl)
        axes[i, 1].imshow(_instance_render(inst_gt))
        axes[i, 2].imshow(_instance_render(inst_pl))
        axes[i, 3].imshow(_instance_render(inst_da))
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.35)
                s.set_color("#444444")
        axes[i, 0].set_ylabel(
            country.replace("_", " "),
            fontsize=7.5,
            fontweight="bold",
        )
        if i == 0:
            for j, t in enumerate(col_titles):
                axes[i, j].set_title(
                    t,
                    fontsize=7.5,
                    fontweight="bold",
                    pad=3,
                    linespacing=1.05,
                )

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.out, dpi=220, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
