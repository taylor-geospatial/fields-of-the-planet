"""Render image + GT + prediction for a handful of test patches."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from ftw_tools.training.trainers import CustomSemanticSegmentationTask

from ftw_planet.datasets import FTWPlanet, PLANET_SR_SCALE


def _pad32(x, value=0.0):
    h, w = x.shape[-2], x.shape[-1]
    nh, nw = ((h + 31) // 32) * 32, ((w + 31) // 32) * 32
    if (nh, nw) == (h, w):
        return x, h, w
    return F.pad(x, (0, nw - w, 0, nh - h), value=value), h, w


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--country", required=True)
    p.add_argument("--n", type=int, default=8)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = CustomSemanticSegmentationTask.load_from_checkpoint(args.ckpt, map_location="cpu")
    model = task.model.eval().to(device)

    ds = FTWPlanet(root="data", countries=[args.country], split="test", load_boundaries=True)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)

    fig, axes = plt.subplots(len(indices), 5, figsize=(20, 4 * len(indices)))
    if len(indices) == 1:
        axes = axes[None, :]
    for row, idx in enumerate(indices):
        s = ds[int(idx)]
        img8, gt = s["image"], s["mask"]  # (8,H,W), (H,W)
        img_in = (img8.unsqueeze(0).to(device) / PLANET_SR_SCALE)
        img_in, H, W = _pad32(img_in)
        with torch.inference_mode():
            pred3 = model(img_in).argmax(dim=1)[0, :H, :W].cpu().numpy()  # 0/1/2
        gt_np = gt.numpy()  # 0/1/2

        # Window A RGB (channels 4-7 in the stack are window A: B,G,R,NIR -> use R,G,B)
        a = img8[4:8].numpy()  # (4,H,W) -- order B,G,R,NIR
        rgb = np.stack([a[2], a[1], a[0]], axis=-1).astype(np.float32) / 3000.0
        rgb = np.clip(rgb, 0, 1)

        axes[row, 0].imshow(rgb); axes[row, 0].set_title("Window A RGB")
        axes[row, 1].imshow(gt_np, cmap="tab10", vmin=0, vmax=3); axes[row, 1].set_title("GT (0/1/2)")
        axes[row, 2].imshow(pred3, cmap="tab10", vmin=0, vmax=3); axes[row, 2].set_title("Pred 3-class")
        # 2-class diff: field gt vs field pred
        gt_field = (gt_np == 1).astype(np.uint8)
        pred_field = (pred3 == 1).astype(np.uint8)
        diff = np.zeros_like(gt_field, dtype=np.uint8)
        diff[(gt_field == 1) & (pred_field == 1)] = 1  # TP green
        diff[(gt_field == 0) & (pred_field == 1)] = 2  # FP red
        diff[(gt_field == 1) & (pred_field == 0)] = 3  # FN blue
        axes[row, 3].imshow(diff, cmap="tab10", vmin=0, vmax=3)
        axes[row, 3].set_title("TP/FP/FN")
        # Boundary class only
        axes[row, 4].imshow((pred3 == 2).astype(np.uint8), cmap="gray")
        axes[row, 4].set_title("Pred boundary mask")
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
