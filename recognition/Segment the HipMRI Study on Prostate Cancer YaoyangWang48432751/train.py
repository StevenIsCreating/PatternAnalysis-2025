from __future__ import annotations
from tqdm import tqdm
from modules import ImprovedUNet2D, dice_loss, dice_per_channel
from dataset import get_data_loaders
import os
import argparse
import json
import torch
import torch.optim as optim
import matplotlib.pyplot as plt


# Core loops
def run_one_epoch(model, loader, optimizer=None, device="cuda"):
    train = optimizer is not None
    model.train(train)

    tot = 0.0
    steps = 0

    desc = "train" if train else "val"
    for xb, yb in tqdm(loader, desc=desc):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        logits = model(xb)
        loss = dice_loss(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        tot += float(loss.item())
        steps += 1

    return tot / max(steps, 1)


@torch.no_grad()
def evaluate_dice(model, loader, device="cuda"):
    model.eval()
    acc = None
    n = 0

    for xb, yb in tqdm(loader, desc="test"):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)

        d = dice_per_channel(model(xb), yb)  # [C]
        acc = d.clone() if acc is None else acc + d
        n += 1

    return (acc / max(n, 1)).cpu()

# Visualization
def plot_training_curves(train_losses, val_losses, save_path="training_curves.png"):
    """
    Save train/val loss curves.
    """
    plt.figure(figsize=(9, 5))
    xs = range(1, len(train_losses) + 1)

    plt.plot(xs, train_losses, marker='o', linewidth=2, label='train')
    plt.plot(xs, val_losses, marker='s', linewidth=2, label='val')

    plt.xlabel('epoch')
    plt.ylabel('dice loss')
    plt.title('training curves')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training curves saved at {save_path}")
    plt.close()

# CLI / main
def main():
    parser = argparse.ArgumentParser(description="Train Improved U-Net on HipMRI slices")
    parser.add_argument('--data_path', type=str, required=True, help='HipMRI_2D root')
    parser.add_argument('--epochs', type=int, default=20, help='epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--base_channels', type=int, default=32, help='U-Net base channels')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='checkpoint dir')
    parser.add_argument('--device', type=str, default='cuda', help='cuda/cpu')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # device choice
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"Using device: {device}")

    # data
    print("Loading dataset, please wait...")
    train_loader, val_loader, test_loader, num_classes, label_to_ch = get_data_loaders(
        args.data_path, batch_size=args.batch_size, num_workers=2
    )

    # model
    print(f"Building Improved U-Net model with {num_classes} output classes...")
    model = ImprovedUNet2D(in_ch=1, num_classes=num_classes, base=args.base_channels).to(device)

    # params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: total={total:,}, trainable={trainable:,}")

    # optim
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    best_val = float("inf")
    train_losses, val_losses = [], []
    print(f"Training will run for {args.epochs} epochs.")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")

        tr = run_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        va = run_one_epoch(model, val_loader, optimizer=None, device=device)
        train_losses.append(tr)
        val_losses.append(va)

        print(f"Epoch {epoch} completed | Train loss: {tr:.4f} | Val loss: {va:.4f}")

        # save best
        if va < best_val:
            best_val = va
            ckpt_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": va,
                    "num_classes": num_classes,
                    "label_to_ch": label_to_ch,
                },
                ckpt_path,
            )
            print(f"New best model found at epoch {epoch}, saved to {ckpt_path}")

    # curves
    plot_training_curves(
        train_losses, val_losses,
        save_path=os.path.join(args.save_dir, "training_curves.png"),
    )

    # test (load best first)
    print("\nLoading the best saved model for final evaluation...")
    ckpt = torch.load(os.path.join(args.save_dir, "best_model.pth"), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    test_dice = evaluate_dice(model, test_loader, device=device)  # [C]

    print("\nDice score per channel on test set:")
    print("--------------------------------")
    for c, v in enumerate(test_dice.tolist()):
        print(f"Channel {c}: {v:.4f}")

    mean_dice = float(test_dice.mean().item())
    prostate_ch = 3 if len(test_dice) > 3 else 0
    prostate_dice = float(test_dice[prostate_ch].item())

    print(f"\nAverage Dice: {mean_dice:.4f}")
    print(f"Prostate Dice (channel {prostate_ch}): {prostate_dice:.4f}")

    # results json
    results = {
        "test_dice_per_channel": test_dice.tolist(),
        "mean_dice": mean_dice,
        "prostate_dice": prostate_dice,
        "num_classes": int(num_classes),
        "label_to_ch": label_to_ch,
    }
    with open(os.path.join(args.save_dir, "test_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Test results saved as JSON at {os.path.join(args.save_dir, 'test_results.json')}")

    # requirement check
    if prostate_dice >= 0.75:
        print("\nThe requirement has been met (prostate Dice ≥ 0.75).")
    else:
        print("\nThe requirement was not met (prostate Dice < 0.75).")


if __name__ == "__main__":
    main()
