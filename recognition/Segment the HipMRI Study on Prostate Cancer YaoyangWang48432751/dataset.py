from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import glob
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



# File discovery utilities
def build_pairs(img_dir: str, seg_dir: str) -> Tuple[List[str], List[str]]:
    # Keep glob for compatibility with various *.nii / *.nii.gz patterns
    img_paths_all = sorted(glob.glob(os.path.join(img_dir, "*.nii*")))
    img_list: List[str] = []
    seg_list: List[str] = []

    for img_path in img_paths_all:
        base = os.path.basename(img_path)
        # Swap prefix "case_" → "seg_"; if no prefix, just prepend seg_
        tail = base[len("case_"):] if base.startswith("case_") else base
        seg_name = f"seg_{tail}"
        candidate = os.path.join(seg_dir, seg_name)
        if os.path.exists(candidate):
            img_list.append(img_path)
            seg_list.append(candidate)

    return img_list, seg_list


def discover_labels(seg_files: Sequence[str], max_samples: int = 100) -> Tuple[List[int], Dict[int, int], int]:
    seen: set[int] = set()

    for seg_path in seg_files[:max_samples]:
        seg_np = nib.load(seg_path).get_fdata(caching="unchanged")
        # Expect (H, W) or (H, W, 1); if 3D, take the first slice axis=2
        if seg_np.ndim == 3:
            seg_np = seg_np[:, :, 0]
        seg_np = seg_np.astype(np.uint8, copy=False)

        # Update the global set with labels present in this mask
        # Using np.unique is fine for small label spaces
        for uid in np.unique(seg_np):
            seen.add(int(uid))

    label_ids = sorted(seen)
    label_to_ch = {lab: idx for idx, lab in enumerate(label_ids)}
    return label_ids, label_to_ch, len(label_ids)



# Dataset
class HipMRI2DSegDataset(Dataset):

    def __init__(
        self,
        img_files: Sequence[str],
        seg_files: Sequence[str],
        label_to_ch_map: Dict[int, int],
        num_classes: int,
        out_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
    ) -> None:
        if len(img_files) != len(seg_files):
            raise ValueError("Image/mask list length mismatch.")
        self.img_files = list(img_files)
        self.seg_files = list(seg_files)
        self.label_to_ch = dict(label_to_ch_map)
        self.num_classes = int(num_classes)
        self.out_size = tuple(out_size)
        self.normalize = bool(normalize)

    def __len__(self) -> int:
        return len(self.img_files)

    @staticmethod
    def _load_slice(path: str) -> np.ndarray:
        vol = nib.load(path).get_fdata(caching="unchanged")
        if vol.ndim == 3:
            vol = vol[:, :, 0]
        return vol

    @staticmethod
    def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = float(x.mean())
        sd = float(x.std())
        return (x - mu) / (sd + eps)

    def _to_one_hot(self, mask_hw: np.ndarray) -> torch.Tensor:
        h, w = mask_hw.shape
        onehot = np.zeros((self.num_classes, h, w), dtype=np.float32)
        # Fill channel-wise by exact match per label id
        for raw_label, ch_idx in self.label_to_ch.items():
            onehot[ch_idx] = (mask_hw == raw_label).astype(np.float32, copy=False)
        return torch.from_numpy(onehot)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.img_files[idx]
        seg_path = self.seg_files[idx]

        # --- image ---
        img_np = self._load_slice(img_path).astype(np.float32, copy=False)
        if self.normalize:
            img_np = self._zscore(img_np)
        img_t = torch.from_numpy(img_np).unsqueeze(0)  # [1, H, W]

        # --- mask ---
        seg_np = self._load_slice(seg_path).astype(np.uint8, copy=False)
        seg_t = self._to_one_hot(seg_np)  # [C, H, W]

        # --- resize to target shape ---
        # images: bilinear; masks: nearest; keep channels as leading dim
        img_t = F.interpolate(
            img_t.unsqueeze(0), size=self.out_size, mode="bilinear", align_corners=False
        ).squeeze(0)

        seg_t = F.interpolate(
            seg_t.unsqueeze(0), size=self.out_size, mode="nearest"
        ).squeeze(0)

        return img_t, seg_t


# Top-level loader factory
def get_data_loaders(
    base_path: str,
    batch_size: int = 8,
    num_workers: int = 2,
    out_size: Tuple[int, int] = (256, 256),
):

    # Resolve split folders
    img_train = os.path.join(base_path, "keras_slices_train")
    seg_train = os.path.join(base_path, "keras_slices_seg_train")
    img_val   = os.path.join(base_path, "keras_slices_validate")
    seg_val   = os.path.join(base_path, "keras_slices_seg_validate")
    img_test  = os.path.join(base_path, "keras_slices_test")
    seg_test  = os.path.join(base_path, "keras_slices_seg_test")

    # Match pairs
    train_imgs, train_segs = build_pairs(img_train, seg_train)
    val_imgs,   val_segs   = build_pairs(img_val,   seg_val)
    test_imgs,  test_segs  = build_pairs(img_test,  seg_test)

    print("Dataset splits:")
    print(f"  Train: {len(train_imgs)} files")
    print(f"  Val:   {len(val_imgs)} files")
    print(f"  Test:  {len(test_imgs)} files")

    # Determine label space from training masks (fixed channel order thereafter)
    label_ids, label_to_ch, num_classes = discover_labels(train_segs)
    print(f"\nLabel set ({num_classes} classes): {label_ids}")

    # Build datasets
    train_ds = HipMRI2DSegDataset(train_imgs, train_segs, label_to_ch, num_classes, out_size=out_size)
    val_ds   = HipMRI2DSegDataset(val_imgs,   val_segs,   label_to_ch, num_classes, out_size=out_size)
    test_ds  = HipMRI2DSegDataset(test_imgs,  test_segs,  label_to_ch, num_classes, out_size=out_size)

    # DataLoaders (shuffle only on train)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, num_classes, label_to_ch
