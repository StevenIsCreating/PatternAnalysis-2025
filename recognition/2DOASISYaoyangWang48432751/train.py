import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from modules import UNet2D  # ensure your modules.py defines UNet2D
from utils import dice_coefficient, save_training_curves


def train_model(
    epochs=10,
    lr=1e-4,
    device=None,
    save_model_path="unet2d_oasis_local.pth",
    loss_curve_path="loss_curve_local.png",
    dice_curve_path="dice_curve_local.png"
):
    """
    Full training loop for 2D OASIS brain MRI segmentation.
    Automatically detects GPU (RTX 4070) and trains locally.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # --- Load data ---
    train_loader, val_loader, test_loader = get_dataloaders()

    # --- Model / loss / optimizer ---
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_dices = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # --- Validation ---
        model.eval()
        dices = []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                d = dice_coefficient(logits, masks)
                dices.append(d.item())

        avg_val_dice = sum(dices) / len(dices)
        val_dices.append(avg_val_dice)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"loss={avg_loss:.4f}  val_dice={avg_val_dice:.4f}")

    # --- Save outputs ---
    save_training_curves(train_losses, val_dices,
                         loss_curve_path=loss_curve_path,
                         dice_curve_path=dice_curve_path)
    torch.save(model.state_dict(), save_model_path)

    print(f"[INFO] Model saved: {save_model_path}")
    print(f"[INFO] Curves saved: {loss_curve_path}, {dice_curve_path}")

    # --- Final test evaluation ---
    model.eval()
    test_dices = []
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            d = dice_coefficient(logits, masks)
            test_dices.append(d.item())

    final_test_dice = sum(test_dices) / len(test_dices)
    print(f"[TEST] Dice = {final_test_dice:.4f}")

    return {
        "train_losses": train_losses,
        "val_dices": val_dices,
        "final_test_dice": final_test_dice
    }


if __name__ == "__main__":
    results = train_model()
    print("[Summary]", results)
