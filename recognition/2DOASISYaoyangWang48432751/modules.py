import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive conv -> batchnorm -> ReLU blocks.
    This is the basic building unit in U-Net.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Down(nn.Module):
    """
    Down-sampling block:
    maxpool to shrink spatial size by 2,
    then a DoubleConv to increase channels.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Up-sampling block:
    1) use ConvTranspose2d to upsample (spatial x2, channel halves),
    2) concatenate skip connection from encoder,
    3) run DoubleConv to fuse.
    
    The tricky part: after concat, channels = (from upsample) + (from skip).
    We must set DoubleConv input channels accordingly.
    """
    def __init__(self, in_ch, out_ch):
        """
        in_ch: number of channels BEFORE DoubleConv, i.e.
               channels_from_upsample + channels_from_skip
        out_ch: desired output channels AFTER this Up block
        """
        super().__init__()

        # we assume upsample halves the channels.
        # Example: if bottom had 1024 ch,
        #   upconv turns it into 512 ch, spatial x2.
        #
        # So usually:
        #   upconv in_ch = out_ch * 2
        # but we cannot hardcode, so we take half dynamically here.
        #
        # BUT PyTorch ConvTranspose2d needs explicit numbers, so we
        # will pass them from the caller via specific channel layout.
        #
        # To make this robust and explicit, we will NOT try to infer.
        # Instead we expect caller to give "up_in_ch" and "skip_ch".
        # Easiest way: rewrite constructor signature to take both.
        raise NotImplementedError("Use UpBlock below instead of Up directly.")


class UpBlock(nn.Module):
    """
    Safer explicit version of Up:
    - up_in_ch:  channels coming from the lower layer BEFORE upsample
                 (e.g. 1024)
    - skip_ch:   channels from the encoder skip connection
                 (e.g. 512)
    - out_ch:    channels we want to have AFTER fusion
                 (e.g. 512 -> then maybe 256 -> etc.)
    """
    def __init__(self, up_in_ch, skip_ch, out_ch):
        super().__init__()

        # Step 1: upsample. This turns (up_in_ch) -> (skip_ch)
        # For example: 1024 -> 512, spatial x2.
        self.upconv = nn.ConvTranspose2d(
            up_in_ch,          # in_channels
            skip_ch,           # out_channels after upsample
            kernel_size=2,
            stride=2
        )

        # Step 2: after concat with skip, channels = skip_ch + skip_ch
        # Example: 512 (upsampled) + 512 (skip) = 1024.
        # DoubleConv will reduce that to out_ch.
        self.fuse = DoubleConv(skip_ch + skip_ch, out_ch)

    def forward(self, x_low, x_skip):
        """
        x_low:  tensor from deeper layer (smaller spatial size)
        x_skip: tensor from encoder (same spatial size after upsample)
        """
        x_low_up = self.upconv(x_low)

        # In case of odd size mismatch due to pooling/rounding,
        # we do a center crop on x_skip to match x_low_up.
        if x_skip.size()[2:] != x_low_up.size()[2:]:
            diff_y = x_skip.size(2) - x_low_up.size(2)
            diff_x = x_skip.size(3) - x_low_up.size(3)
            x_skip = x_skip[
                :,
                :,
                diff_y // 2 : x_skip.size(2) - diff_y // 2,
                diff_x // 2 : x_skip.size(3) - diff_x // 2
            ]

        x = torch.cat([x_skip, x_low_up], dim=1)
        x = self.fuse(x)
        return x


class UNet2D(nn.Module):
    """
    A standard 5-level U-Net:
    Encoder channels: 1 -> 64 -> 128 -> 256 -> 512 -> 1024
    Decoder mirrors that back to 1 output channel.
    """
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        # ----- Encoder -----
        self.enc1 = DoubleConv(in_channels, 64)    # out: 64 ch
        self.enc2 = Down(64,   128)                # out: 128 ch
        self.enc3 = Down(128,  256)                # out: 256 ch
        self.enc4 = Down(256,  512)                # out: 512 ch
        self.enc5 = Down(512,  1024)               # bottom: 1024 ch

        # ----- Decoder -----
        # Up from 1024 -> (upsample to 512), concat with 512 -> fuse -> 512
        self.up1 = UpBlock(up_in_ch=1024, skip_ch=512, out_ch=512)

        # Up from 512 -> (upsample to 256), concat with 256 -> fuse -> 256
        self.up2 = UpBlock(up_in_ch=512, skip_ch=256, out_ch=256)

        # Up from 256 -> (upsample to 128), concat with 128 -> fuse -> 128
        self.up3 = UpBlock(up_in_ch=256, skip_ch=128, out_ch=128)

        # Up from 128 -> (upsample to 64), concat with 64 -> fuse -> 64
        self.up4 = UpBlock(up_in_ch=128, skip_ch=64, out_ch=64)

        # Final 1x1 conv to get per-pixel logits for binary segmentation
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # ----- Encoder path -----
        x1 = self.enc1(x)      # [B,  64, H,   W  ]
        x2 = self.enc2(x1)     # [B, 128, H/2, W/2]
        x3 = self.enc3(x2)     # [B, 256, H/4, W/4]
        x4 = self.enc4(x3)     # [B, 512, H/8, W/8]
        x5 = self.enc5(x4)     # [B,1024, H/16,W/16]

        # ----- Decoder path with skip connections -----
        d4 = self.up1(x5, x4)  # expect 1024 -> 512 up, cat 512 -> fuse -> 512
        d3 = self.up2(d4, x3)  # expect 512  -> 256 up, cat 256 -> fuse -> 256
        d2 = self.up3(d3, x2)  # expect 256  -> 128 up, cat 128 -> fuse -> 128
        d1 = self.up4(d2, x1)  # expect 128  -> 64  up, cat 64  -> fuse -> 64

        out = self.out_conv(d1)  # [B, out_channels, H, W]
        return out
