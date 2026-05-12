"""Model E: Inverted Residual Hourglass.

Architecture:
  Stem:        Conv 3->24  (stride-2)
  Encoder:     InvRes(24->48, s=2), InvRes(48->64, s=2)
  Bottleneck:  2x InvRes(64->64) with residual
  Decoder:     bilinear up + 1x1 project + add skip + InvRes
  Head:        1x1 conv -> sigmoid

MobileNetV2-style inverted residual blocks:
  - Expand channels (1x1) -> depthwise conv (3x3) -> project (1x1, linear)
  - Residual connection when in_ch == out_ch and stride == 1
  - Linear bottleneck: no activation after final projection
  - ReLU6 bounds activations (helps under heavy augmentation)

~71K params  |  est. 6ms P95 on i5-11400 (ONNX, 1 thread)
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils_v2 import run_training


class InvRes(nn.Module):
    """MobileNetV2 inverted residual block.

    thin -> expand (1x1) -> DW (3x3) -> project (1x1, linear)
    Residual shortcut when stride=1 and in_ch==out_ch.
    """

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)

        layers = []
        # Expand (skip if expand_ratio == 1)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid),
                nn.ReLU6(inplace=True),
            ]
        # Depthwise
        layers += [
            nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                      groups=mid, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True),
        ]
        # Project (linear -- no activation)
        layers += [
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class InvResNet(nn.Module):
    """Model E: Inverted Residual Hourglass encoder-decoder."""

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res

        # Stem: regular conv (DW on 3ch is wasteful)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True))                              # 128x128x24

        # Encoder
        self.enc2 = InvRes(24, 48, stride=2, expand_ratio=2)     # 64x64x48
        self.enc3 = InvRes(48, 64, stride=2, expand_ratio=2)     # 32x32x64

        # Bottleneck: 2x inverted residuals (with residual connections)
        self.bn1 = InvRes(64, 64, stride=1, expand_ratio=2)
        self.bn2 = InvRes(64, 64, stride=1, expand_ratio=2)

        # Decoder: 1x1 project to match skip channels, then add
        self.up3_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False),
            nn.BatchNorm2d(48))
        self.dec3 = InvRes(48, 48, stride=1, expand_ratio=2)     # 64x64x48

        self.up2_proj = nn.Sequential(
            nn.Conv2d(48, 24, 1, bias=False),
            nn.BatchNorm2d(24))
        self.dec2 = InvRes(24, 24, stride=1, expand_ratio=2)     # 128x128x24

        # Head
        self.head = nn.Conv2d(24, 1, 1)

    def forward(self, x):
        e1 = self.stem(x)                                        # 128x128x24
        e2 = self.enc2(e1)                                        # 64x64x48
        e3 = self.enc3(e2)                                        # 32x32x64

        b = self.bn2(self.bn1(e3))                                # 32x32x64

        # Decode level 3
        up3 = self.up3_proj(F.interpolate(
            b, size=e2.shape[2:], mode="bilinear",
            align_corners=False))
        d3 = self.dec3(up3 + e2)                                  # 64x64x48

        # Decode level 2
        up2 = self.up2_proj(F.interpolate(
            d3, size=e1.shape[2:], mode="bilinear",
            align_corners=False))
        d2 = self.dec2(up2 + e1)                                  # 128x128x24

        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return torch.sigmoid(self.head(d2))


def main():
    p = argparse.ArgumentParser(
        description="Model E: Inverted Residual Hourglass")
    p.add_argument("--data-dir", default="sv2")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--save-onnx", default="model_e_inv_res.onnx")
    p.add_argument("--half-res", action="store_true", default=False,
                    help="Output 128x128 heatmap (default: on)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(InvResNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
