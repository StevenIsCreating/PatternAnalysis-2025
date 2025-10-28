import torch
import matplotlib.pyplot as plt
import os


def dice_coefficient(logits, targets, eps=1e-6):
    """
    Compute Dice coefficient for binary segmentation.

    logits: raw model output from the network, shape (B,1,H,W)
    targets: ground truth binary mask in {0,1}, shape (B,1,H,W)
    eps: numerical stability term
    """
    probs = torch.sigmoid(logits)          # convert logits -> probabilities
    preds = (probs > 0.5).float()          # threshold to binary {0,1}

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))

    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def save_training_curves(train_losses, val_dices,
                         loss_curve_path="figs_loss_curve.png",
                         dice_curve_path="figs_dice_curve.png"):
    """
    Save training curves (loss over epochs, Dice over epochs) as .png files.

    train_losses: list[float] of average training loss per epoch
    val_dices: list[float] of average validation Dice per epoch
    loss_curve_path: output path for the loss curve image
    dice_curve_path: output path for the dice curve image
    """
    # loss curve
    plt.figure()
    plt.plot(train_losses, label="train_loss")
    plt.xlabel("epoch")
    plt.ylabel("BCEWithLogitsLoss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_curve_path, dpi=150)
    plt.close()

    # dice curve
    plt.figure()
    plt.plot(val_dices, label="val_dice")
    plt.xlabel("epoch")
    plt.ylabel("Dice")
    plt.title("Validation Dice")
    plt.legend()
    plt.tight_layout()
    plt.savefig(dice_curve_path, dpi=150)
    plt.close()


def visualize_prediction_triplet(img_tensor,
                                 gt_mask_tensor,
                                 pred_mask_tensor,
                                 out_path,
                                 title="Brain MRI segmentation (OASIS)"):
    """
    Save a side-by-side visualization of:
      - input image
      - ground truth mask
      - predicted mask
    to a PNG file at out_path.

    img_tensor: (1,H,W) or (H,W) torch.Tensor
    gt_mask_tensor: same spatial size as img_tensor
    pred_mask_tensor: same spatial size as img_tensor
    out_path: string path to save .png
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Ensure tensors are on CPU and in numpy format
    img_np = img_tensor.detach().cpu()
    gt_np = gt_mask_tensor.detach().cpu()
    pred_np = pred_mask_tensor.detach().cpu()

    if img_np.ndim == 3:
        # (1,H,W) -> (H,W)
        img_np = img_np[0]
    if gt_np.ndim == 3:
        gt_np = gt_np[0]
    if pred_np.ndim == 3:
        pred_np = pred_np[0]

    img_np = img_np.numpy()
    gt_np = gt_np.numpy()
    pred_np = pred_np.numpy()

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(img_np, cmap="gray")
    plt.title("Input slice")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_np, cmap="gray")
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_np, cmap="gray")
    plt.title("Predicted mask")
    plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
