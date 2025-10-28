import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Two consecutive conv -> batch norm -> ReLU blocks.
    This is the basic building block of U-Net.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """
    Downscaling block: maxpool followed by DoubleConv.
    Used in the encoder path of U-Net.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x


class Up(nn.Module):
    """
    Upscaling block:
    - upsample using transposed convolution
    - concatenate with the corresponding encoder feature map (skip connection)
    - apply DoubleConv to fuse features
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # transposed conv for learned upsampling
        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x_decoder, x_encoder):
        """
        x_decoder: feature map from the decoder (lower resolution)
        x_encoder: skip connection from the encoder (higher resolution)
        """
        x_decoder = self.up(x_decoder)

        # Spatial alignment:
        # Due to possible odd/even size differences after pooling,
        # we crop or pad so that both tensors match spatially.
        diff_y = x_encoder.size(2) - x_decoder.size(2)
        diff_x = x_encoder.size(3) - x_decoder.size(3)

        if diff_y != 0 or diff_x != 0:
            # pad (left, right, top, bottom)
            x_decoder = F.pad(
                x_decoder,
                [diff_x // 2, diff_x - diff_x // 2,
                 diff_y // 2, diff_y - diff_y // 2]
            )

        # concatenate along channel dimension
        x = torch.cat([x_encoder, x_decoder], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    """
    Final 1x1 convolution to map to the desired number of output channels.
    For binary segmentation we use out_ch = 1.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet2D(nn.Module):
    """
    U-Net for 2D medical image segmentation.

    - Encoder path (contracting): DoubleConv + pooling
    - Decoder path (expanding): transposed conv upsample + skip connections
    - Output: raw logits (NOT passed through sigmoid here)
      so that BCEWithLogitsLoss can be applied directly in training.
    """
    def __init__(self, in_channels=1, out_channels=1, base_ch=64):
        """
        in_channels: number of channels in the input image (1 for grayscale MRI slices)
        out_channels: number of classes / masks (1 for binary brain mask)
        base_ch: number of feature channels in the first layer
        """
        super().__init__()

        # Encoder
        self.inc   = DoubleConv(in_channels, base_ch)          # level 1
        self.down1 = Down(base_ch, base_ch * 2)                 # level 2
        self.down2 = Down(base_ch * 2, base_ch * 4)             # level 3
        self.down3 = Down(base_ch * 4, base_ch * 8)             # level 4
        self.down4 = Down(base_ch * 8, base_ch * 16)            # bottleneck

        # Decoder
        self.up1 = Up(base_ch * 16, base_ch * 8)
        self.up2 = Up(base_ch * 8,  base_ch * 4)
        self.up3 = Up(base_ch * 4,  base_ch * 2)
        self.up4 = Up(base_ch * 2,  base_ch)

        self.outc = OutConv(base_ch, out_channels)

    def forward(self, x):
        # Encoder / contracting path
        x1 = self.inc(x)        # shape: base_ch
        x2 = self.down1(x1)     # shape: base_ch*2
        x3 = self.down2(x2)     # shape: base_ch*4
        x4 = self.down3(x3)     # shape: base_ch*8
        x5 = self.down4(x4)     # shape: base_ch*16

        # Decoder / expanding path with skip connections
        x = self.up1(x5, x4)    # combine bottleneck with x4
        x = self.up2(x,  x3)    # combine with x3
        x = self.up3(x,  x2)    # combine with x2
        x = self.up4(x,  x1)    # combine with x1
        logits = self.outc(x)   # final 1x1 conv -> logits

        return logits
