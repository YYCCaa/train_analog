"""Train the autocal Model E variant (Inverted Residual encoder + regression head)."""

import argparse

from autocal_utils import AutocalModelE, run_autocal_training


def main():
    p = argparse.ArgumentParser(description="Autocal Model E trainer")
    p.add_argument("--data-dir", default="sv3")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--w-param", type=float, default=0.1)
    p.add_argument("--w-mark", type=float, default=1.0)
    p.add_argument("--save-onnx", default="autocal_model_e.onnx")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    run_autocal_training(AutocalModelE(), args)


if __name__ == "__main__":
    main()
