"""
Convert tip_heatmap_net.pth  →  tip_heatmap_net.tflite

Path: PyTorch → ONNX → onnx2tf → TFLite (float32).

The original .pth file is kept unchanged.
"""

import argparse
import os
import shutil

import numpy as np
import onnx
import torch

from train import TipHeatmapNet

ONNX_PATH = "tip_heatmap_net.onnx"
TFLITE_DIR = "tflite_output"


def export_onnx(pth_path, onnx_path):
    """Step 1: PyTorch → ONNX."""
    model = TipHeatmapNet()
    model.load_state_dict(torch.load(pth_path, map_location="cpu",
                                     weights_only=True))
    model.eval()

    dummy = torch.randn(1, 3, 192, 192)

    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # static shape for TFLite
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX exported and verified: {onnx_path}")


def convert_to_tflite(onnx_path, output_dir):
    """Step 2: ONNX → TFLite via onnx2tf."""
    import onnx2tf

    onnx2tf.convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=output_dir,
        non_verbose=True,
    )
    print(f"onnx2tf conversion complete → {output_dir}/")


def verify(pth_path, tflite_path):
    """Step 3: Compare PyTorch vs TFLite outputs on the same input."""
    # PyTorch inference
    model = TipHeatmapNet()
    model.load_state_dict(torch.load(pth_path, map_location="cpu",
                                     weights_only=True))
    model.eval()

    np.random.seed(0)
    inp_np = np.random.rand(1, 3, 192, 192).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.from_numpy(inp_np)).numpy()

    # TFLite inference
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter

    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    # TFLite expects NHWC
    inp_nhwc = inp_np.transpose(0, 2, 3, 1)
    interp.set_tensor(inp_detail["index"], inp_nhwc)
    interp.invoke()
    tfl_out_raw = interp.get_tensor(out_detail["index"])

    # onnx2tf output may be NHWC [1,192,192,1]; reshape to match PyTorch [1,1,192,192]
    if tfl_out_raw.shape == (1, 192, 192, 1):
        tfl_out = tfl_out_raw.transpose(0, 3, 1, 2)
    else:
        tfl_out = tfl_out_raw

    max_diff = np.abs(pt_out - tfl_out).max()
    mean_diff = np.abs(pt_out - tfl_out).mean()

    # Compare argmax (predicted tip pixel)
    pt_idx = pt_out[0, 0].flatten().argmax()
    tfl_idx = tfl_out[0, 0].flatten().argmax()
    pt_y, pt_x = divmod(int(pt_idx), 192)
    tfl_y, tfl_x = divmod(int(tfl_idx), 192)

    print(f"\nVerification (PyTorch vs TFLite):")
    print(f"  Max pixel diff : {max_diff:.6f}")
    print(f"  Mean pixel diff: {mean_diff:.6f}")
    print(f"  PyTorch  tip   : ({pt_x}, {pt_y})")
    print(f"  TFLite   tip   : ({tfl_x}, {tfl_y})")
    print(f"  Tip match      : {'YES' if (pt_x == tfl_x and pt_y == tfl_y) else 'NO (see diffs above)'}")


def main():
    parser = argparse.ArgumentParser(description="Convert .pth to .tflite")
    parser.add_argument("--pth", default="tip_heatmap_net.pth",
                        help="Input PyTorch weights")
    parser.add_argument("--tflite", default="tip_heatmap_net.tflite",
                        help="Output TFLite file")
    parser.add_argument("--skip-verify", action="store_true")
    args = parser.parse_args()

    print(f"=== Step 1: PyTorch → ONNX ===")
    export_onnx(args.pth, ONNX_PATH)

    print(f"\n=== Step 2: ONNX → TFLite ===")
    convert_to_tflite(ONNX_PATH, TFLITE_DIR)

    # Find the generated .tflite file and move to final location
    candidates = [f for f in os.listdir(TFLITE_DIR) if f.endswith(".tflite")]
    if not candidates:
        print(f"ERROR: no .tflite found in {TFLITE_DIR}/")
        return
    src = os.path.join(TFLITE_DIR, candidates[0])
    shutil.copy2(src, args.tflite)
    print(f"\nTFLite model: {args.tflite}  ({os.path.getsize(args.tflite) / 1e6:.1f} MB)")

    if not args.skip_verify:
        print(f"\n=== Step 3: Verify ===")
        verify(args.pth, args.tflite)

    # Clean up intermediates
    os.remove(ONNX_PATH)
    shutil.rmtree(TFLITE_DIR, ignore_errors=True)
    print("\nCleaned up intermediate files.")
    print(f"Original .pth kept: {args.pth}")


if __name__ == "__main__":
    main()
