from __future__ import annotations

import os
import argparse
from pathlib import Path
from typing import Tuple

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from modules import ImprovedUNet2D, dice_per_channel
from dataset import get_data_loaders



# Visualization helpers
def make_label_colormap(num_classes: int) -> ListedColormap:
    palette = [
    (0.0, 0.0, 0.0),     # 0: background
    (1.0, 0.6, 0.0),     # 1: orange (muscle/tissue)
    (0.0, 0.7, 0.7),     # 2: teal (organ)
    (0.9, 0.0, 0.3),     # 3: magenta (tumor)
    (0.3, 0.8, 0.3),     # 4: green (prostate)
    (0.9, 0.9, 0.2),     # 5: yellow (bone)
    (0.5, 0.5, 1.0),     # 6: blue (boundary)
    ]

    return ListedColormap(palette[:num_classes])


def _norm01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    return (x - mn) / (mx - mn + eps)


def visualize_batch(
    images: torch.Tensor,
    gts_onehot: torch.Tensor,
    preds_probs: torch.Tensor,
    prostate_ch: int,
    out_dir: str,
    num_samples: int = 4,
) -> None:
    
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Move to numpy
    imgs = images.numpy()            # [B,1,H,W]
    gts  = gts_onehot.numpy()        # [B,C,H,W]
    prs  = preds_probs.numpy()       # [B,C,H,W]
    bsz, chs, H, W = gts.shape
    n = min(num_samples, bsz)

    # Figure 1: GT vs Pred
    cmap = make_label_colormap(chs)
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i in range(n):
        axes[i, 0].imshow(imgs[i, 0], cmap="gray")
        axes[i, 0].set_title(f"Sample {i+1}: MRI")
        axes[i, 0].axis("off")

        gt_arg = np.argmax(gts[i], axis=0)
        axes[i, 1].imshow(gt_arg, cmap=cmap, vmin=0, vmax=chs - 1)
        axes[i, 1].set_title("GT (argmax)")
        axes[i, 1].axis("off")

        pr_arg = np.argmax(prs[i], axis=0)
        axes[i, 2].imshow(pr_arg, cmap=cmap, vmin=0, vmax=chs - 1)
        axes[i, 2].set_title("Pred (argmax)")
        axes[i, 2].axis("off")

    fig.tight_layout()
    p1 = out_path / "predictions.png"
    fig.savefig(p1.as_posix(), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[predict] saved: {p1}")

    # Figure 2: Prostate overlay
    fig2, axes2 = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes2 = axes2.reshape(2, 1)

    for i in range(n):
        mri = imgs[i, 0]
        prob = prs[i, prostate_ch]
        mri01 = _norm01(mri)
        rgb = np.stack([mri01, mri01, mri01], axis=-1)

        binmask = (prob > 0.5).astype(np.float32)
        rgb[..., 0] = np.maximum(rgb[..., 0], binmask)  # red channel

        axes2[0, i].imshow(mri, cmap="gray")
        axes2[0, i].set_title(f"Sample {i+1}: MRI")
        axes2[0, i].axis("off")

        axes2[1, i].imshow(rgb)
        axes2[1, i].set_title("Prostate Overlay")
        axes2[1, i].axis("off")

    fig2.tight_layout()
    p2 = out_path / "overlays.png"
    fig2.savefig(p2.as_posix(), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"[predict] saved: {p2}")


# Inference pipeline
@torch.no_grad()
def predict_and_report(
    data_path: str,
    checkpoint_path: str,
    device: str = "cuda",
    num_samples: int = 4,
    out_dir: str = "./predictions",
) -> None:
    # Device
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[predict] device: {dev}")

    # Data
    print("[predict] loading data loaders ...")
    _, _, test_loader, num_classes, _ = get_data_loaders(
        base_path=data_path, batch_size=8, num_workers=2, out_size=(256, 256)
    )

    # Channel convention (update if your mapping changes)
    prostate_ch = 3 if num_classes > 3 else 0
    print(f"[predict] prostate channel = {prostate_ch}")

    # Checkpoint
    ckpt_path = Path(checkpoint_path)
    print(f"[predict] loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path.as_posix(), map_location=dev)
    saved_num_classes = int(ckpt.get("num_classes", num_classes))
    print(f"[predict] ckpt num_classes = {saved_num_classes}")

    # Model
    model = ImprovedUNet2D(in_ch=1, num_classes=saved_num_classes, base=32).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Batch
    xb, yb = next(iter(test_loader))  # xb: [B,1,H,W], yb: [B,C,H,W]
    xb = xb.to(dev)
    yb = yb.to(dev)

    # Forward
    logits = model(xb)                 # [B,C,H,W]
    probs = torch.sigmoid(logits)      # [B,C,H,W]

    # Metrics
    dice_vec = dice_per_channel(logits, yb)  # [C]
    print("\n[predict] Dice per channel (batch):")
    for c, v in enumerate(dice_vec.tolist()):
        print(f"  ch{c}: {v:.4f}")

    d_prostate = dice_vec[prostate_ch].item()
    print(f"\n[predict] prostate Dice (ch {prostate_ch}): {d_prostate:.4f}")
    print("[predict] spec OK (>=0.75)" if d_prostate >= 0.75 else "[predict] spec NOT met (<0.75)")

    # Visuals
    visualize_batch(
        images=xb.detach().cpu(),
        gts_onehot=yb.detach().cpu(),
        preds_probs=probs.detach().cpu(),
        prostate_ch=prostate_ch,
        out_dir=out_dir,
        num_samples=num_samples,
    )

    print(f"\n[predict] done. outputs -> {out_dir}")


# CLI
def main() -> None:
    parser = argparse.ArgumentParser(description="Predict & visualize with Improved U-Net")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root folder containing keras_slices_*")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_model.pth")
    parser.add_argument("--device", type=str, default="cuda",
                        help="cuda or cpu")
    parser.add_argument("--num_samples", type=int, default=4,
                        help="number of test samples to visualize")
    parser.add_argument("--out_dir", type=str, default="./predictions",
                        help="output directory for figures")

    args = parser.parse_args()

    predict_and_report(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        device=args.device,
        num_samples=args.num_samples,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
