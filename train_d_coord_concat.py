"""Model D: Wide Residual DWSep with CoordConv + Concatenation Skips.

Architecture:
  Encoder:     CoordConv 3+2 -> 32, DWSep 32->64->96  (3 levels, down to 32x32)
  Bottleneck:  2x pre-activation residual DWSep(96)
  Decoder:     bilinear up + concat skip + 1x1 reduce + DWSep
  Head:        1x1 conv -> sigmoid

Key innovations over Model A/C:
  - CoordConv at input for spatial awareness in heatmap regression
  - Concatenation skips (richer than additive) with 1x1 channel reduction
  - Pre-activation residual connections in bottleneck for gradient stability
  - 96 bottleneck channels (3x Model A) via DWSep for param efficiency

~52K params  |  est. 5ms P95 on i5-11400 (ONNX, 1 thread)
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils_v2 import run_training


class CoordConv(nn.Module):
    """Adds 2 normalised coordinate channels (x, y in [-1, 1]) before conv."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1,
                 bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch + 2, out_ch, kernel_size, stride=stride,
                              padding=padding, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        yy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
        yy = yy.view(1, 1, H, 1).expand(B, 1, H, W)
        xx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
        xx = xx.view(1, 1, 1, W).expand(B, 1, H, W)
        return self.conv(torch.cat([x, xx, yy], dim=1))


class DWSepDown(nn.Module):
    """Depthwise separable with stride-2 downsampling."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWSepRes(nn.Module):
    """Pre-activation residual DWSep block: BN-ReLU-DW-BN-ReLU-PW + skip."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
        )

    def forward(self, x):
        return x + self.block(x)


class DWSep(nn.Module):
    """Standard DWSep: DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class CoordConcatNet(nn.Module):
    """Model D: CoordConv + concat-skip DWSep encoder-decoder."""

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res

        # Encoder
        self.enc1 = nn.Sequential(
            CoordConv(3, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))          # 128x128x32
        self.enc2 = DWSepDown(32, 64)                             # 64x64x64
        self.enc3 = DWSepDown(64, 96)                             # 32x32x96

        # Bottleneck: 2x pre-activation residual DWSep
        self.bn1 = DWSepRes(96)
        self.bn2 = DWSepRes(96)

        # Decoder level 3: concat(up_96, skip_64) = 160 -> reduce -> DWSep
        self.dec3_reduce = nn.Sequential(
            nn.Conv2d(160, 64, 1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.dec3 = DWSep(64, 64)

        # Decoder level 2: concat(up_64, skip_32) = 96 -> reduce -> DWSep
        self.dec2_reduce = nn.Sequential(
            nn.Conv2d(96, 32, 1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.dec2 = DWSep(32, 32)

        # Head
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                                        # 128x128x32
        e2 = self.enc2(e1)                                        # 64x64x64
        e3 = self.enc3(e2)                                        # 32x32x96

        b = self.bn2(self.bn1(e3))                                # 32x32x96

        # Decode level 3: upsample + concat + reduce + refine
        up3 = F.interpolate(b, size=e2.shape[2:], mode="bilinear",
                            align_corners=False)
        d3 = self.dec3(self.dec3_reduce(
            torch.cat([up3, e2], dim=1)))                         # 64x64x64

        # Decode level 2: upsample + concat + reduce + refine
        up2 = F.interpolate(d3, size=e1.shape[2:], mode="bilinear",
                            align_corners=False)
        d2 = self.dec2(self.dec2_reduce(
            torch.cat([up2, e1], dim=1)))                         # 128x128x32

        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(self.head(d2))


def main():
    p = argparse.ArgumentParser(
        description="Model D: CoordConv + Concat Skips")
    p.add_argument("--data-dir", default="sv2")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--save-onnx", default="model_d_coord_concat.onnx")
    p.add_argument("--half-res", action="store_true", default=False,
                    help="Output 128x128 heatmap (default: on)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(CoordConcatNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
