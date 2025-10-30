from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Utility helpers
def center_crop(tensor: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
    _, _, h, w = tensor.shape
    dh = max((h - target_h) // 2, 0)
    dw = max((w - target_w) // 2, 0)
    th = min(target_h, h)
    tw = min(target_w, w)
    return tensor[:, :, dh:dh + th, dw:dw + tw]


def conv_block(c_in: int, c_out: int, dilation: int = 1) -> nn.Sequential:
    pad2 = dilation
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=pad2, dilation=dilation, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


# Building blocks
class Down(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = conv_block(c_in, c_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, c_in: int, c_out: int) -> None:

        super().__init__()
        self.up = nn.ConvTranspose2d(c_in, c_in // 2, kernel_size=2, stride=2)
        # After upsampling we concatenate with the skip (same channel count as up output),
        # therefore the conv block sees c_in channels.
        self.conv = conv_block(c_in, c_out)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        _, _, h, w = x.shape
        skip = center_crop(skip, h, w)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)



# Model
class ImprovedUNet2D(nn.Module):
    def __init__(self, in_ch: int = 1, num_classes: int = 6, base: int = 32) -> None:
        super().__init__()

        # Encoder
        self.enc1 = conv_block(in_ch, base)            # H×W,    C=base
        self.enc2 = Down(base, base * 2)               # H/2×W/2
        self.enc3 = Down(base * 2, base * 4)           # H/4×W/4
        self.enc4 = Down(base * 4, base * 8)           # H/8×W/8

        # Bottleneck (down once more). You can optionally increase dilation here.
        self.bottleneck = Down(base * 8, base * 16)    # H/16×W/16

        # Decoder
        self.up4 = Up(base * 16, base * 8)
        self.up3 = Up(base * 8,  base * 4)
        self.up2 = Up(base * 4,  base * 2)
        self.up1 = Up(base * 2,  base)

        # Classifier head
        self.outc = nn.Conv2d(base, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        s1 = self.enc1(x)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        s4 = self.enc4(s3)

        b = self.bottleneck(s4)

        x = self.up4(b, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        return self.outc(x)



# Metrics & Loss
def dice_per_channel(pred_logits: torch.Tensor,
                     target_onehot: torch.Tensor,
                     eps: float = 1e-6) -> torch.Tensor:

    probs = torch.sigmoid(pred_logits)
    pred = (probs > 0.5).to(target_onehot.dtype)

    dims = (0, 2, 3)
    inter = (pred * target_onehot).sum(dim=dims)
    denom = pred.sum(dim=dims) + target_onehot.sum(dim=dims)
    return (2.0 * inter + eps) / (denom + eps)


def dice_loss(pred_logits: torch.Tensor,
              target_onehot: torch.Tensor,
              eps: float = 1e-6) -> torch.Tensor:

    probs = torch.sigmoid(pred_logits)
    dims = (0, 2, 3)
    inter = (probs * target_onehot).sum(dim=dims)
    denom = probs.sum(dim=dims) + target_onehot.sum(dim=dims)
    dice = (2.0 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()
