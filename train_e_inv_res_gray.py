"""Model E (grayscale): Inverted Residual Hourglass on 1-channel input.

Identical to ``train_e_inv_res.InvResNet`` except the stem convolution takes
1 input channel instead of 3:

    InvResNet      stem Conv2d(3 -> 24, 3x3, s2)   3 * 24 * 9 = 648 weights
    InvResNetGray  stem Conv2d(1 -> 24, 3x3, s2)   1 * 24 * 9 = 216 weights
                                                   -> 432 fewer parameters

Everything after the stem (encoder, bottleneck, decoder, head) is shared
code, so the forward pass is inherited unchanged.

    (B,1,256,256) --stem--> (B,24,128,128) --enc/bn/dec--> (B,24,128,128)
                  --[up x2 unless --half-res]--> head --> (B,1,256|128,256|128)

Trained with ``train_utils_v2_gray.run_training`` on grayscale PNGs (e.g.
the ``sv7`` output of generate_synthetic_v7_mp.py).
"""

import argparse

import torch.nn as nn

from train_e_inv_res import InvResNet
from train_utils_v2_gray import IN_CHANNELS, run_training


class InvResNetGray(InvResNet):
    """Model E with a 1-channel stem.

    Input : (B, 1, 256, 256) float32 in [0, 1]
    Output: (B, 1, 256, 256) float32 in [0, 1]  (128x128 if half_res)
    """

    def __init__(self, half_res=True):
        super().__init__(half_res=half_res)
        # Stem: (B, 1, 256, 256) -> (B, 24, 128, 128) float32
        self.stem = nn.Sequential(
            nn.Conv2d(IN_CHANNELS, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU6(inplace=True))                              # 128x128x24


def main():
    p = argparse.ArgumentParser(
        description="Model E (grayscale): Inverted Residual Hourglass")
    p.add_argument("--data-dir", default="sv7")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--sigma", type=float, default=3.0)
    p.add_argument("--model-name", default="model_e_inv_res_gray",
                    help="base name for all output artifacts: "
                         "<name>.pth/.onnx/.tflite, <name>_final.*, "
                         "<name>_p99.pth, and the diag/<name>/ plot "
                         "directory (default: model_e_inv_res_gray)")
    p.add_argument("--save-onnx", default=None,
                    help="explicit .onnx output path; overrides "
                         "--model-name if given")
    p.add_argument("--half-res", action="store_true", default=False,
                    help="Output 128x128 heatmap (default: off, 256x256)")
    p.add_argument("--num-workers", type=int, default=4,
                    help="DataLoader worker processes (0 = load in main process)")
    p.add_argument("--benchmark-only", action="store_true")
    args = p.parse_args()
    args.save_onnx = args.save_onnx or f"{args.model_name}.onnx"
    run_training(InvResNetGray(half_res=args.half_res), args)


if __name__ == "__main__":
    main()
