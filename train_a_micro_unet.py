"""Model A: Micro U-Net -- skip-addition encoder-decoder.

Architecture:
  Encoder:     stride-2 convs     3 -> 8 -> 16 -> 32   (3 levels)
  Bottleneck:  32 -> 32           (single 3x3 conv)
  Decoder:     bilinear up + conv + skip-add   32 -> 16 -> 8
  Head:        1x1 conv -> sigmoid

~21K params  |  ~34M MACs  |  ~68M FLOPs  |  est. 5-10ms on i5-11400
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils import run_training


class MicroUNet(nn.Module):
    """Micro U-Net with additive skip connections."""

    def __init__(self, half_res=False):
        super().__init__()
        self.half_res = half_res
        # Encoder (stride-2 downsampling)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        # Head
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                           # 96x96x8
        e2 = self.enc2(e1)                           # 48x48x16
        b = self.bottleneck(self.enc3(e2))           # 24x24x32

        d3 = self.dec3(F.interpolate(
            b, size=e2.shape[2:], mode="bilinear",
            align_corners=False)) + e2               # 48x48x16
        d2 = self.dec2(F.interpolate(
            d3, size=e1.shape[2:], mode="bilinear",
            align_corners=False)) + e1               # 96x96x8

        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(self.head(d2))


def main():
    p = argparse.ArgumentParser(description="Model A: Micro U-Net")
    p.add_argument("--data-dir", default="synthetic_data")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--save-onnx", default="model_a_micro_unet.onnx")
    p.add_argument("--half-res", action="store_true",
                    help="Output 128x128 heatmap (skip final upsample)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(MicroUNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
