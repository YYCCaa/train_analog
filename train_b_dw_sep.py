"""Model B: Depthwise Separable encoder-decoder -- skip-addition.

Architecture:
  Encoder:     regular 3->32, DW-sep stride-2  32 -> 64 -> 128
  Bottleneck:  DW-sep 128 -> 128  (x2)
  Decoder:     bilinear up + DW-sep + skip-add  128 -> 64 -> 32
  Head:        1x1 conv -> sigmoid

~17K params  |  ~26M MACs  |  ~52M FLOPs  |  est. 3-8ms on i5-11400
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils_v2 import run_training


class DWSep(nn.Module):
    """Depthwise separable: DW 3x3 -> BN -> ReLU -> PW 1x1 -> BN -> ReLU."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DWSepNet(nn.Module):
    """Depthwise separable encoder-decoder with additive skip connections."""

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res
        # First layer is regular conv (DW-sep on 3ch input is inefficient)
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.enc2 = DWSep(32, 64, stride=2)
        self.enc3 = DWSep(64, 128, stride=2)
        # Bottleneck (two DW-sep layers for capacity)
        self.bn1 = DWSep(128, 128)
        self.bn2 = DWSep(128, 128)
        # Decoder
        self.dec3 = DWSep(128, 64)
        self.dec2 = DWSep(64, 32)
        # Head
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)                            # 128x128x32
        e2 = self.enc2(e1)                            # 64x64x64
        b = self.bn2(self.bn1(self.enc3(e2)))         # 32x32x128

        d3 = self.dec3(F.interpolate(
            b, size=e2.shape[2:], mode="bilinear",
            align_corners=False)) + e2                # 64x64x64
        d2 = self.dec2(F.interpolate(
            d3, size=e1.shape[2:], mode="bilinear",
            align_corners=False)) + e1                # 128x128x32

        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(self.head(d2))


def main():
    p = argparse.ArgumentParser(description="Model B: Depthwise Separable")
    p.add_argument("--data-dir", default="sv2")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3)
    p.add_argument("--save-onnx", default="model_b_dw_sep.onnx")
    p.add_argument("--half-res", action="store_true",
                    help="Output 128x128 heatmap (skip final upsample)", default=False)
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(DWSepNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
