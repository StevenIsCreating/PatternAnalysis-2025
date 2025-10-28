import os
import torch
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from modules import UNet2D


@torch.no_grad()
def inference_and_visualize(
    weights_path="unet2d_oasis.pt",
    device=None,
    output_dir="predictions_viz",
    num_samples=3
):
    """
    Load a trained UNet2D model, run inference on the test set,
    and save side-by-side visualizations:
      - original image slice
      - ground truth mask
      - predicted mask (thresholded at 0.5)

    weights_path: path to the saved model weights (.pt) from training
    device: "cuda" or "cpu"; if None, auto-detect
    output_dir: directory to store the generated .png visualizations
    num_samples: how many test samples to visualize
    """
    # Select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device for inference: {device}")

    # Get loaders (we only need test_loader here)
    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=2, shuffle_train=False)

    # Build model and load weights
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    # Iterate over a few samples from the test set
    count = 0
    for imgs, masks in test_loader:
        imgs = imgs.to(device)      # shape (1,1,H,W)
        masks = masks.to(device)    # shape (1,1,H,W)

        # Forward pass
        logits = model(imgs)        # raw logits
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Move tensors to CPU for plotting
        img_np   = imgs[0, 0].detach().cpu().numpy()   # (H,W)
        mask_np  = masks[0, 0].detach().cpu().numpy()  # (H,W)
        pred_np  = preds[0, 0].detach().cpu().numpy()  # (H,W)

        # Plot: original, ground truth, prediction
        fig, axs = plt.subplots(1, 3, figsize=(10, 4))

        axs[0].imshow(img_np, cmap="gray")
        axs[0].set_title("Input slice")
        axs[0].axis("off")

        axs[1].imshow(mask_np, cmap="gray")
        axs[1].set_title("Ground truth mask")
        axs[1].axis("off")

        axs[2].imshow(pred_np, cmap="gray")
        axs[2].set_title("Predicted mask")
        axs[2].axis("off")

        fig.suptitle("Brain MRI segmentation (OASIS)")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"sample_{count}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)

        print(f"[Info] Saved visualization to {out_path}")

        count += 1
        if count >= num_samples:
            break


if __name__ == "__main__":
    """
    Typical usage on Rangpur after training:
    1. Run train.py to produce 'unet2d_oasis.pt'
    2. Run: python predict.py
    3. Check 'predictions_viz/' for PNG figures
    """
    inference_and_visualize(
        weights_path="unet2d_oasis.pt",
        device=None,                # auto-detect GPU
        output_dir="predictions_viz",
        num_samples=3               # save first 3 test examples
    )
