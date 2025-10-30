import torch
from modules import UNet2D
from dataset_hipmri import HipMRI2DDataset
import matplotlib.pyplot as plt

ckpt_path = "unet2d_hipmri_best.pth"

# ---- robust load (PyTorch 2.6) ----
try:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
except Exception:
    # file is trusted (created by yourself) -> allow pickle path
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

model = UNet2D(in_channels=1, out_channels=1)
model.load_state_dict(state, strict=True)
model.eval()

# ---- data ----
test_images = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_test"
test_masks  = r"D:\COMP3710A3\PatternAnalysis-2025\recognition\data\HipMRI_2D_data\keras_slices_data\keras_slices_seg_test"
ds = HipMRI2DDataset(images_dir=test_images, masks_dir=test_masks)

# ---- visualize one sample ----
x, y = ds[0]
with torch.no_grad():
    pred = torch.sigmoid(model(x.unsqueeze(0))).squeeze().numpy()

plt.subplot(1,3,1); plt.imshow(x[0], cmap="gray"); plt.title("Image")
plt.subplot(1,3,2); plt.imshow(y[0], cmap="gray"); plt.title("Ground Truth")
plt.subplot(1,3,3); plt.imshow(pred > 0.5, cmap="gray"); plt.title("Prediction")
plt.tight_layout(); plt.show()
