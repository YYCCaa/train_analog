"""Model C: Direct encoder-decoder -- no skip connections.

Architecture:
  Encoder:     stride-2 convs  3 -> 8 -> 16 -> 32 -> 32  (4 levels, 12x12)
  Bottleneck:  32 -> 32  (single 3x3 conv)
  Decoder:     bilinear upsample + conv  32 -> 32 -> 16 -> 8
  Head:        1x1 conv -> sigmoid

~40K params  |  ~37M MACs  |  ~74M FLOPs  |  est. 5-10ms on i5-11400
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_utils import run_training


class DirectNet(nn.Module):
    """Direct encoder-decoder without skip connections.

    Deeper encoding (4 levels down to 12x12) captures more global context,
    compensating for the lack of skip connections.
    """

    def __init__(self, half_res=False):
        super().__init__()
        self.half_res = half_res
        # Encoder (4 levels of stride-2 downsampling)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),      # -> 96
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1, bias=False),     # -> 48
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1, bias=False),    # -> 24
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),    # -> 12
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        # Decoder (4 levels of 2x upsampling)
        self.dec4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.dec3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(
            nn.Conv2d(16, 8, 3, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True))
        # Head
        self.head = nn.Conv2d(8, 1, 1)

    def forward(self, x):
        x = self.bottleneck(self.encoder(x))         # 12x12x32

        x = self.dec4(F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False))                    # 24x24x32
        x = self.dec3(F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False))                    # 48x48x16
        x = self.dec2(F.interpolate(
            x, scale_factor=2, mode="bilinear",
            align_corners=False))                    # 96x96x8

        if not self.half_res:
            x = F.interpolate(x, scale_factor=2, mode="bilinear",
                              align_corners=False)
        return torch.sigmoid(self.head(x))


def main():
    p = argparse.ArgumentParser(description="Model C: Direct (no skips)")
    p.add_argument("--data-dir", default="synthetic_data")
    p.add_argument("--epochs", type=int, default=130)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--save-onnx", default="model_c_direct.onnx")
    p.add_argument("--half-res", action="store_true",
                    help="Output 128x128 heatmap (skip final upsample)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    run_training(DirectNet(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
