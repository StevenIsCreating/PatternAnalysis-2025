# dataset_hipmri.py
# Two-dir HipMRI dataset with flexible prefix/suffix normalization
# and optional resizing to a fixed target_size.

from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import nibabel as nib

def _list_with_exts(d: Path, exts=(".nii", ".nii.gz")):
    out = []
    for e in exts:
        out.extend(d.glob(f"*{e}"))
    return sorted(out)

def _strip_prefix(stem: str, prefixes) -> str:
    for p in prefixes:
        if stem.startswith(p):
            return stem[len(p):]
    return stem

def _strip_suffix(stem: str, suffixes) -> str:
    for s in suffixes:
        if stem.endswith(s):
            return stem[: -len(s)]
    return stem

def _normalize(stem: str, prefixes, suffixes) -> str:
    stem = _strip_prefix(stem, prefixes)
    stem = _strip_suffix(stem, suffixes)
    return stem

def _pair_from_two_dirs(img_dir: Path,
                        msk_dir: Path,
                        exts=(".nii", ".nii.gz"),
                        image_prefixes=("case_", "img_", "image_"),
                        mask_prefixes=("seg_", "mask_", "label_", "labels_", "gt_"),
                        mask_suffixes=("_mask", "_seg", "_label", "_labels", "_gt")):
    imgs = _list_with_exts(img_dir, exts)
    msks = _list_with_exts(msk_dir, exts)
    if not imgs:
        raise FileNotFoundError(f"No image files in {img_dir}")
    if not msks:
        raise FileNotFoundError(f"No mask files in {msk_dir}")

    img_map = {}
    for p in imgs:
        key = _normalize(p.stem, image_prefixes, ())
        img_map[key] = p

    msk_map = {}
    for p in msks:
        key = _normalize(p.stem, mask_prefixes, mask_suffixes)
        msk_map[key] = p

    common = sorted(set(img_map.keys()) & set(msk_map.keys()))
    if not common:
        sample_imgs = list(img_map.keys())[:6]
        sample_msks = list(msk_map.keys())[:6]
        raise RuntimeError(
            "No matched names between images and masks.\n"
            f"Sample image stems (norm): {sample_imgs}\n"
            f"Sample mask  stems (norm): {sample_msks}\n"
            "Extend prefixes/suffixes if needed."
        )

    img_list = [img_map[k] for k in common]
    msk_list = [msk_map[k] for k in common]
    return img_list, msk_list

class HipMRI2DDataset(Dataset):
    """
    Two-dir HipMRI dataset (NIfTI).
    Returns:
      image: [1,H,W] float32 (z-score normalized)
      mask : [1,H,W] float32 in {0,1}
    """

    def __init__(self,
                 images_dir: str = None,
                 masks_dir: str = None,
                 flat_dir: str = None,
                 image_prefixes=("case_", "img_", "image_"),
                 mask_prefixes=("seg_", "mask_", "label_", "labels_", "gt_"),
                 mask_suffixes=("_mask", "_seg", "_label", "_labels", "_gt"),
                 target_size: tuple | None = (256, 256),  # <- unify size here
                 dtype=np.float32):
        super().__init__()
        self.dtype = dtype
        self.target_size = target_size  # (H, W) or None

        if flat_dir is not None:
            raise NotImplementedError("Use two-dir mode for HipMRI.")

        if images_dir is None or masks_dir is None:
            raise ValueError("Provide images_dir and masks_dir for two-dir mode.")

        imgd, mskd = Path(images_dir), Path(masks_dir)
        if not imgd.exists(): raise FileNotFoundError(f"images_dir not found: {imgd}")
        if not mskd.exists(): raise FileNotFoundError(f"masks_dir not found: {mskd}")

        self.img_paths, self.msk_paths = _pair_from_two_dirs(
            imgd, mskd,
            exts=(".nii", ".nii.gz"),
            image_prefixes=image_prefixes,
            mask_prefixes=mask_prefixes,
            mask_suffixes=mask_suffixes
        )
        assert len(self.img_paths) == len(self.msk_paths), "Unpaired images and masks."

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = nib.load(str(self.img_paths[idx])).get_fdata()
        msk = nib.load(str(self.msk_paths[idx])).get_fdata()

        # binarize mask
        msk = (msk > 0.5).astype(self.dtype)

        # z-score normalize image
        img = img.astype(self.dtype)
        std = img.std() if img.std() > 1e-8 else 1.0
        img = (img - img.mean()) / std

        # to tensors [1,H,W]
        img = torch.from_numpy(np.expand_dims(img, 0)).float()
        msk = torch.from_numpy(np.expand_dims(msk, 0)).float()

        # optional resize to fixed size (image: bilinear, mask: nearest)
        if self.target_size is not None:
            H, W = self.target_size
            img = F.interpolate(img.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False).squeeze(0)
            msk = F.interpolate(msk.unsqueeze(0), size=(H, W), mode="nearest").squeeze(0)

        return img, msk
