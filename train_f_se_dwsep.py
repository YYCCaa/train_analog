"""Model F: SE-Enhanced DWSep (Model B+).

Architecture:
  Encoder:     Conv 3->32, DWSepSE 32->64->96  (3 levels, down to 32x32)
  Bottleneck:  2x residual DWSepSE(96)
  Decoder:     bilinear up + 1x1 project + add skip + DWSepSE
  Head:        1x1 conv -> sigmoid

Builds on Model B's proven DWSep pattern with two improvements:
  - Squeeze-and-Excitation channel attention after each DWSep block
    (suppresses noise/artifact features, amplifies pointer-relevant ones)
  - Residual connections in bottleneck for gradient stability
  - Channel progression 32->64->96 (SE compensates for fewer channels
    vs B's 128 by making each channel more effective)

~65K params  |  est. 8ms P95 on i5-11400 (ONNX, 1 thread)
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils_v2 import run_training


class ChannelSE(nn.Module):
    """Squeeze-and-Excitation channel attention (conv-based for clean ONNX).

    GAP -> Conv1x1(ch -> ch/r) -> ReLU -> Conv1x1(ch/r -> ch) -> Sigmoid -> scale.
    """

    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(self.pool(x))


class DWSepSE(nn.Module):
    """Depthwise separable + Squeeze-and-Excitation.

    DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU -> SE(out_ch).
    """

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.se = ChannelSE(out_ch, reduction=4)

    def forward(self, x):
        return self.se(self.block(x))


class DWSepResSE(nn.Module):
    """Residual DWSep + SE for bottleneck: output = input + SE(DWSep(input))."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
        )
        self.se = ChannelSE(ch, reduction=4)

    def forward(self, x):
        return x + self.se(self.block(x))


class SEDWSepNet(nn.Module):
    """Model F: SE-enhanced DWSep encoder-decoder (Model B+)."""

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))          # 128x128x32
        self.enc2 = DWSepSE(32, 64, stride=2)                    # 64x64x64
        self.enc3 = DWSepSE(64, 96, stride=2)                    # 32x32x96

        # Bottleneck: 2x residual DWSep + SE
        self.bn1 = DWSepResSE(96)
        self.bn2 = DWSepResSE(96)

        # Decoder with additive skips
        self.up3_proj = nn.Sequential(
            nn.Conv2d(96, 64, 1, bias=False),
            nn.BatchNorm2d(64))
        self.dec3 = DWSepSE(64, 64)                              # 64x64x64

        self.up2_proj = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False),
            nn.BatchNorm2d(32))
        self.dec2 = DWSepSE(32, 32)                              # 128x128x32

        # Head
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                        # 128x128x32
        e2 = self.enc2(e1)                                        # 64x64x64
        e3 = self.enc3(e2)                                        # 32x32x96

        b = self.bn2(self.bn1(e3))                                # 32x32x96

        # Decode level 3
        up3 = self.up3_proj(F.interpolate(
            b, size=e2.shape[2:], mode="bilinear",
            align_corners=False))
        d3 = self.dec3(up3 + e2)                                  # 64x64x64

        # Decode level 2
        up2 = self.up2_proj(F.interpolate(
            d3, size=e1.shape[2:], mode="bilinear",
            align_corners=False))
        d2 = self.dec2(up2 + e1)                                  # 128x128x32

        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(self.head(d2))


def main():
    p = argparse.ArgumentParser(
        description="Model F: SE-Enhanced DWSep (Model B+)")
    p.add_argument("--data-dir", default="sv2")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--save-onnx", default="model_f_se_dwsep.onnx")
    p.add_argument("--half-res", action="store_true", default=False,
                    help="Output 128x128 heatmap (default: on)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(SEDWSepNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
