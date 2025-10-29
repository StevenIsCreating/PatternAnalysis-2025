import os
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# === Local dataset root ===
OASIS_ROOT = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\OASIS"

# Define subdirectories
TRAIN_IMG_DIR = os.path.join(OASIS_ROOT, "keras_png_slices_train")
TRAIN_MSK_DIR = os.path.join(OASIS_ROOT, "keras_png_slices_seg_train")

VAL_IMG_DIR   = os.path.join(OASIS_ROOT, "keras_png_slices_validate")
VAL_MSK_DIR   = os.path.join(OASIS_ROOT, "keras_png_slices_seg_validate")

TEST_IMG_DIR  = os.path.join(OASIS_ROOT, "keras_png_slices_test")
TEST_MSK_DIR  = os.path.join(OASIS_ROOT, "keras_png_slices_seg_test")


class OasisSliceSegDataset(Dataset):
    """
    Dataset class for 2D OASIS brain MRI slice segmentation.
    Each sample contains one grayscale MRI slice and its binary segmentation mask.
    """
    def __init__(self, img_dir, msk_dir):
        super().__init__()
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.png")))
        self.msk_paths = sorted(glob.glob(os.path.join(msk_dir, "*.png")))
        assert len(self.img_paths) == len(self.msk_paths), \
            f"Image/Mask count mismatch: {len(self.img_paths)} vs {len(self.msk_paths)}"

        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # --- Load and normalize image ---
        img = Image.open(self.img_paths[idx]).convert("L")
        img_t = self.to_tensor(img)
        mean, std = img_t.mean(), img_t.std()
        img_t = (img_t - mean) / (std + 1e-6)

        # --- Load and binarize mask ---
        msk = Image.open(self.msk_paths[idx]).convert("L")
        msk_t = self.to_tensor(msk)
        msk_bin = (msk_t >= 0.5).float()

        return img_t, msk_bin


def get_dataloaders(batch_size=4, num_workers=2, shuffle_train=True):
    """Create train/val/test PyTorch DataLoaders."""
    train_ds = OasisSliceSegDataset(TRAIN_IMG_DIR, TRAIN_MSK_DIR)
    val_ds   = OasisSliceSegDataset(VAL_IMG_DIR, VAL_MSK_DIR)
    test_ds  = OasisSliceSegDataset(TEST_IMG_DIR, TEST_MSK_DIR)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_train,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader
