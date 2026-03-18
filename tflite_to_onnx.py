"""Convert any TFLite model to ONNX format using tf2onnx.

Usage:
    python tflite_to_onnx.py model.tflite
    python tflite_to_onnx.py model.tflite -o model.onnx
    python tflite_to_onnx.py model.tflite --opset 17
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Convert TFLite to ONNX")
    parser.add_argument("tflite", help="Path to input .tflite file")
    parser.add_argument("-o", "--output", default=None,
                        help="Output .onnx path (default: same name, .onnx extension)")
    parser.add_argument("--opset", type=int, default=13,
                        help="ONNX opset version (default: 13)")
    args = parser.parse_args()

    if not os.path.isfile(args.tflite):
        print(f"ERROR: {args.tflite} not found")
        sys.exit(1)

    output = args.output or os.path.splitext(args.tflite)[0] + ".onnx"

    cmd = [
        sys.executable, "-m", "tf2onnx.convert",
        "--tflite", args.tflite,
        "--output", output,
        "--opset", str(args.opset),
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print("Conversion failed.")
        sys.exit(result.returncode)

    print(f"\nSaved: {output}  ({os.path.getsize(output) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
