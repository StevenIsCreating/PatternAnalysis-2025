import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os
from dataset import get_dataloaders
from modules import UNet2D


def dice_coefficient(logits, targets, eps=1e-6):
    """
    Compute Dice coefficient for binary segmentation.

    logits: raw model output, shape (B,1,H,W)
    targets: ground truth mask in {0,1}, shape (B,1,H,W)
    eps: small constant to avoid division by zero
    """
    # Convert logits to binary mask via sigmoid + threshold
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def train_model(
    epochs=5,
    lr=1e-3,
    device=None,
    save_model_path="unet2d_oasis.pt",
    loss_curve_path="figs_loss_curve.png",
    dice_curve_path="figs_dice_curve.png"
):
    """
    Full training loop:
    - trains on train set
    - evaluates Dice on validation set each epoch
    - saves loss and Dice plots
    - evaluates final Dice on test set
    """

    # Select device (GPU if available, else CPU)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Create data loaders from the shared OASIS dataset on Rangpur
    train_loader, val_loader, test_loader = get_dataloaders()

    # Build model and optimizer
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()  # suitable for binary segmentation
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_dices = []

    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)         # (B,1,H,W)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # Validation Dice
        model.eval()
        with torch.no_grad():
            dice_values = []
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                logits = model(imgs)
                dice_val = dice_coefficient(logits, masks)
                dice_values.append(dice_val.item())

        avg_val_dice = sum(dice_values) / len(dice_values)
        val_dices.append(avg_val_dice)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"loss={avg_loss:.4f}  val_dice={avg_val_dice:.4f}")

    # Create plots folder path if needed
    # (We save figures in the current working directory by default.
    #  You can change this to a 'figs/' subfolder if you want.)
    # Plot training loss curve
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    # Plot validation Dice curve
    plt.figure()
    plt.plot(val_dices, label="val_dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_curve_path, dpi=150)
    plt.close()

    # Save model weights for later inference / predict.py
    torch.save(model.state_dict(), save_model_path)
    print(f"[Info] Saved model weights to {save_model_path}")

    # Final test Dice evaluation
    model.eval()
    with torch.no_grad():
        test_dice_values = []
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            dice_val = dice_coefficient(logits, masks)
            test_dice_values.append(dice_val.item())

    final_test_dice = sum(test_dice_values) / len(test_dice_values)
    print(f"[TEST] Dice = {final_test_dice:.4f}")

    return {
        "train_losses": train_losses,
        "val_dices": val_dices,
        "final_test_dice": final_test_dice,
    }


if __name__ == "__main__":
    # Important:
    # - On Rangpur, this script will run under sbatch via runner.sh.
    # - Locally, you can just run `python train.py`.
    #
    # You can tune epochs upward (e.g. 20+) on Rangpur to push Dice > 0.90.
    # For a quick smoke test, keep epochs small (e.g. 3-5).
    results = train_model(
        epochs=5,
        lr=1e-3,
        device=None,  # auto-detect GPU
        save_model_path="unet2d_oasis.pt",
        loss_curve_path="figs_loss_curve.png",
        dice_curve_path="figs_dice_curve.png"
    )

    print("[Summary]", results)
