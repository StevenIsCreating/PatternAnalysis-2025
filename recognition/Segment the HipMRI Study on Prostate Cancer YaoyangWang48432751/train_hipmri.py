# train_hipmri.py
# 2D UNet on HipMRI (Normal). Target: test Dice >= 0.75.
# - Two-dir dataset (images + masks)
# - Loss = 0.5*BCEWithLogits + 0.5*SoftDice
# - Light H/V flips
# - AMP (mixed precision)
# - Val threshold sweep; best threshold used for TEST

import os, random, numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from dataset_hipmri import HipMRI2DDataset
from modules import UNet2D

# ---- Reproducibility ----
def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True  # faster on fixed shapes
set_seed(42)

# ---- Device & AMP ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp_enabled = torch.cuda.is_available()
scaler = GradScaler(device if amp_enabled else "cpu", enabled=amp_enabled)

# ---- Absolute paths (edit if needed) ----
train_images = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_train"
train_masks  = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_seg_train"
val_images   = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_validate"
val_masks    = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_seg_validate"
test_images  = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_test"
test_masks   = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_seg_test"

for p in [train_images, train_masks, val_images, val_masks, test_images, test_masks]:
    assert os.path.exists(p), f"Path not found: {p}"

# ---- Datasets / Loaders ----
# HipMRI2DDataset yields: image [1,H,W] (z-norm), mask [1,H,W] in {0,1}
train_set = HipMRI2DDataset(images_dir=train_images, masks_dir=train_masks)
val_set   = HipMRI2DDataset(images_dir=val_images,   masks_dir=val_masks)
test_set  = HipMRI2DDataset(images_dir=test_images,  masks_dir=test_masks)

BATCH_SIZE = 8 if torch.cuda.is_available() else 4
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=amp_enabled)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=amp_enabled)
test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=amp_enabled)

# ---- Model / Loss / Optim ----
model = UNet2D(in_channels=1, out_channels=1).to(device)
bce_loss = nn.BCEWithLogitsLoss()

def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * targets).sum(dim=(1,2,3)) + eps
    den = probs.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3)) + eps
    return 1.0 - (num / den).mean()

def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return 0.5 * bce_loss(logits, targets) + 0.5 * soft_dice_loss(logits, targets)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0)

# ---- Metrics ----
@torch.no_grad()
def dice_at_threshold(logits: torch.Tensor, targets: torch.Tensor, thr: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    inter = (preds * targets).sum(dim=(1,2,3))
    denom = preds.sum(dim=(1,2,3)) + targets.sum(dim=(1,2,3))
    return ((2 * inter + eps) / (denom + eps)).mean()

@torch.no_grad()
def evaluate(loader, threshold: float = 0.5) -> float:
    model.eval(); scores = []
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        scores.append(dice_at_threshold(logits, y, thr=threshold).item())
    return float(np.mean(scores)) if scores else 0.0

# ---- Light augmentation ----
def maybe_augment(x: torch.Tensor, y: torch.Tensor, p_h: float = 0.5, p_v: float = 0.2):
    if random.random() < p_h:
        x = torch.flip(x, dims=[3]); y = torch.flip(y, dims=[3])
    if random.random() < p_v:
        x = torch.flip(x, dims=[2]); y = torch.flip(y, dims=[2])
    return x, y

# ---- Training ----
EPOCHS = 20
best_val_dice, best_thr = 0.0, 0.5
CKPT = "unet2d_hipmri_best.pth"

for epoch in range(1, EPOCHS + 1):
    model.train(); run = 0.0
    for images, masks in train_loader:
        images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
        images, masks = maybe_augment(images, masks)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=amp_enabled):
            logits = model(images)
            loss = combined_loss(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        run += loss.item()

    train_loss = run / max(1, len(train_loader))

    # threshold sweep on validation
    candidates = [round(t, 2) for t in np.arange(0.35, 0.56, 0.05)]
    val_scores = {t: evaluate(val_loader, threshold=t) for t in candidates}
    thr_epoch, dice_epoch = max(val_scores.items(), key=lambda kv: kv[1])

    print(f"[Epoch {epoch:02d}/{EPOCHS}] loss={train_loss:.4f} "
          f"val_dice@best_thr({thr_epoch:.2f})={dice_epoch:.4f} "
          f"candidates={ {t: round(v,4) for t,v in val_scores.items()} }")

    if dice_epoch > best_val_dice:
        best_val_dice, best_thr = dice_epoch, thr_epoch
        torch.save({"model": model.state_dict(),
                    "best_thr": best_thr,
                    "val_dice": best_val_dice,
                    "epoch": epoch}, CKPT)

print(f"[INFO] Best val Dice={best_val_dice:.4f} at thr={best_thr:.2f}; saved -> {CKPT}")

# ---- TEST evaluation ----
ckpt = torch.load(CKPT, map_location=device)
model.load_state_dict(ckpt["model"])
best_thr = float(ckpt.get("best_thr", 0.5))
model.eval()
test_dice = evaluate(test_loader, threshold=best_thr)
print(f"[TEST] Dice@thr={best_thr:.2f} = {test_dice:.4f}")
