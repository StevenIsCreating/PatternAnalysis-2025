import torch
import torch.nn as nn
import torch.optim as optim
from dataset import get_dataloaders
from modules import UNet2D
from utils import dice_coefficient, save_training_curves


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
    - saves loss and Dice curves
    - evaluates final Dice on test set
    """

    # Select device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    # Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Model / optimizer / loss
    model = UNet2D(in_channels=1, out_channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    val_dices = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)

        # validation
        model.eval()
        with torch.no_grad():
            dices = []
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)
                logits = model(imgs)
                d = dice_coefficient(logits, masks)
                dices.append(d.item())

        avg_val_dice = sum(dices) / len(dices)
        val_dices.append(avg_val_dice)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"loss={avg_loss:.4f}  val_dice={avg_val_dice:.4f}")

    # save training curves as .png
    save_training_curves(
        train_losses,
        val_dices,
        loss_curve_path=loss_curve_path,
        dice_curve_path=dice_curve_path
    )

    # save model weights for inference
    torch.save(model.state_dict(), save_model_path)
    print(f"[Info] Saved model weights to {save_model_path}")

    # final test dice
    model.eval()
    with torch.no_grad():
        test_dices = []
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            logits = model(imgs)
            d = dice_coefficient(logits, masks)
            test_dices.append(d.item())

    final_test_dice = sum(test_dices) / len(test_dices)
    print(f"[TEST] Dice = {final_test_dice:.4f}")

    return {
        "train_losses": train_losses,
        "val_dices": val_dices,
        "final_test_dice": final_test_dice,
    }


if __name__ == "__main__":
    results = train_model(
        epochs=5,
        lr=1e-3,
        device=None,  # auto
        save_model_path="unet2d_oasis.pt",
        loss_curve_path="figs_loss_curve.png",
        dice_curve_path="figs_dice_curve.png"
    )
    print("[Summary]", results)
