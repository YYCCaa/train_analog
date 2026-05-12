"""Convert a .pth checkpoint to .tflite via ONNX as intermediate.

Usage:
    python convert_pth_to_tflite.py --model f --pth model_f_se_dwsep.pth
    python convert_pth_to_tflite.py --model b --pth model_b_dw_sep_p99.pth --half-res
    python convert_pth_to_tflite.py --model a --pth model_a.pth --out my_model.tflite
"""

import argparse
import os
import shutil
import tempfile

import onnx
import torch

IMG_SIZE = 256

# ── model registry ───────────────────────────────────────────────────────

MODELS = {}


def _register(key, import_path, class_name):
    MODELS[key] = (import_path, class_name)


_register("a", "train_a_micro_unet", "MicroUNet")
_register("b", "train_b_dw_sep", "DWSepNet")
_register("c", "train_c_direct", "DirectNet")
_register("d", "train_d_coord_concat", "CoordConcatNet")
_register("e", "train_e_inv_res", "InvResNet")
_register("f", "train_f_se_dwsep", "SEDWSepNet")


def load_model(key, half_res=False):
    if key not in MODELS:
        raise ValueError(f"Unknown model key '{key}'. Choose from: {list(MODELS.keys())}")
    module_name, class_name = MODELS[key]
    import importlib
    mod = importlib.import_module(module_name)
    cls = getattr(mod, class_name)
    return cls(half_res=half_res)


# ── conversion ───────────────────────────────────────────────────────────

def pth_to_onnx(model, pth_path, onnx_path, opset=17):
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, onnx_path, opset_version=opset,
                      input_names=["input"], output_names=["heatmap"])
    onnx.checker.check_model(onnx.load(onnx_path))
    print(f"ONNX: {onnx_path}  ({os.path.getsize(onnx_path) / 1024:.1f} KB)")


def onnx_to_tflite(onnx_path, tflite_path):
    import onnx2tf
    with tempfile.TemporaryDirectory() as tmp:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tmp,
            non_verbose=True,
        )
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        src = os.path.join(tmp, f"{stem}_float32.tflite")
        shutil.copy2(src, tflite_path)
    print(f"TFLite: {tflite_path}  ({os.path.getsize(tflite_path) / 1024:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description="Convert .pth to .tflite")
    p.add_argument("--model", required=True, choices=list(MODELS.keys()),
                   help="Model architecture key (a/b/c/d/e/f)")
    p.add_argument("--pth", required=True, help="Path to .pth checkpoint")
    p.add_argument("--out", default=None,
                   help="Output .tflite path (default: same stem as pth)")
    p.add_argument("--half-res", action="store_true",
                   help="Use half-resolution (128x128) output")
    p.add_argument("--keep-onnx", action="store_true",
                   help="Keep intermediate ONNX file")
    args = p.parse_args()

    stem = os.path.splitext(args.pth)[0]
    onnx_path = stem + ".onnx"
    tflite_path = args.out or (stem + ".tflite")

    model = load_model(args.model, half_res=args.half_res)
    pth_to_onnx(model, args.pth, onnx_path)
    onnx_to_tflite(onnx_path, tflite_path)

    if not args.keep_onnx:
        os.remove(onnx_path)
        print(f"Removed intermediate: {onnx_path}")

    print(f"\nDone: {args.pth} -> {tflite_path}")


if __name__ == "__main__":
    main()
