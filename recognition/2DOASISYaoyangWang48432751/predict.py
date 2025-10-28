import os
import torch
from dataset import get_dataloaders
from modules import UNet2D
from utils import visualize_prediction_triplet, dice_coefficient


@torch.no_grad()
def inference_and_visualize(
    weights_path="unet2d_oasis.pt",
    device=None,
    output_dir="predictions_viz",
    num_samples=3
):
    """
    Load a trained UNet2D model, run inference on the test set,
    save visualization (input / GT / prediction), and print Dice.
    """

    # Device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device for inference: {device}")

    # Data
    _, _, test_loader = get_dataloaders(batch_size=1, num_workers=2, shuffle_train=False)

    # Model
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    count = 0
    for imgs, masks in test_loader:
        imgs = imgs.to(device)      # (1,1,H,W)
        masks = masks.to(device)    # (1,1,H,W)

        logits = model(imgs)
        dice_val = dice_coefficient(logits, masks).item()

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Take first item in batch
        img_1   = imgs[0]          # (1,H,W)
        mask_1  = masks[0]         # (1,H,W)
        pred_1  = preds[0]         # (1,H,W)

        out_path = os.path.join(output_dir, f"sample_{count}_dice_{dice_val:.3f}.png")
        visualize_prediction_triplet(
            img_tensor=img_1,
            gt_mask_tensor=mask_1,
            pred_mask_tensor=pred_1,
            out_path=out_path,
            title=f"OASIS segmentation | Dice={dice_val:.3f}"
        )

        print(f"[Info] Saved {out_path} (Dice={dice_val:.3f})")

        count += 1
        if count >= num_samples:
            break


if __name__ == "__main__":
    inference_and_visualize(
        weights_path="unet2d_oasis.pt",
        device=None,
        output_dir="predictions_viz",
        num_samples=3
    )
