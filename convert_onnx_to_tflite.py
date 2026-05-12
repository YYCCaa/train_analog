"""Convert an ONNX model to TFLite via onnx2tf.

Usage:
    python convert_onnx_to_tflite.py model_b_dw_sep.onnx
    python convert_onnx_to_tflite.py model_b_dw_sep.onnx -o my_model.tflite
"""

import argparse
import os
import shutil
import tempfile

import onnx2tf


def convert(onnx_path, tflite_path):
    """Convert ONNX model to float32 TFLite."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    with tempfile.TemporaryDirectory() as tmp:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tmp,
            non_verbose=True,
            output_signaturedefs=True
        )
        shutil.copy2(os.path.join(tmp, f"{stem}_float32.tflite"), tflite_path)
    src_kb = os.path.getsize(onnx_path) / 1024
    dst_kb = os.path.getsize(tflite_path) / 1024
    print(f"ONNX:   {onnx_path}  ({src_kb:.1f} KB)")
    print(f"TFLite: {tflite_path}  ({dst_kb:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description="Convert ONNX model to TFLite")
    p.add_argument("onnx", help="Path to input ONNX model")
    p.add_argument("-o", "--output", default=None,
                   help="Output TFLite path (default: same name, .tflite)")
    args = p.parse_args()

    out = args.output or args.onnx.replace(".onnx", ".tflite")
    convert(args.onnx, out)


if __name__ == "__main__":
    main()
