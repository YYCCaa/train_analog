"""
Validate that the PyTorch (.pth) and TFLite (.tflite) models produce
the same predictions on freshly generated synthetic dial images.

Generates a small batch of synthetic images, runs both models, and
compares the predicted tip coordinates and angles.
"""

import math
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

from generate_synthetic import generate_image, IMG_SIZE
from train import TipHeatmapNet

# Paths
PTH_PATH = "tip_heatmap_net.pth"
TFLITE_PATH = "tip_heatmap_net.tflite"
NUM_SAMPLES = 5000
CENTER = IMG_SIZE // 2


def load_pytorch_model(pth_path):
    model = TipHeatmapNet()
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def load_tflite_model(tflite_path):
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return interp


def preprocess(img_path):
    """Load image and return float32 array in [0,1], shape (H, W, 3)."""
    img = Image.open(img_path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def predict_pytorch(model, img_hwc):
    """Run PyTorch model. img_hwc: (H, W, 3) float32 [0,1]."""
    tensor = torch.from_numpy(img_hwc.transpose(2, 0, 1)).unsqueeze(0)  # 1x3xHxW
    with torch.no_grad():
        heatmap = model(tensor)[0, 0].numpy()  # HxW
    idx = heatmap.flatten().argmax()
    y, x = divmod(int(idx), heatmap.shape[1])
    return x, y, heatmap


def predict_tflite(interp, img_hwc):
    """Run TFLite model. img_hwc: (H, W, 3) float32 [0,1]."""
    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    inp = img_hwc[np.newaxis, ...]  # 1xHxWx3 (NHWC)
    interp.set_tensor(inp_detail["index"], inp)
    interp.invoke()
    raw = interp.get_tensor(out_detail["index"])

    # Reshape to HxW regardless of output layout
    if raw.ndim == 4 and raw.shape[-1] == 1:
        heatmap = raw[0, :, :, 0]  # NHWC -> HxW
    elif raw.ndim == 4 and raw.shape[1] == 1:
        heatmap = raw[0, 0, :, :]  # NCHW -> HxW
    else:
        heatmap = raw.reshape(IMG_SIZE, IMG_SIZE)

    idx = heatmap.flatten().argmax()
    y, x = divmod(int(idx), heatmap.shape[1])
    return x, y, heatmap


def tip_to_angle(tip_x, tip_y):
    dx = tip_x - CENTER
    dy = -(tip_y - CENTER)
    return math.degrees(math.atan2(dx, dy)) % 360


def main():
    if not os.path.exists(PTH_PATH):
        print(f"ERROR: {PTH_PATH} not found"); sys.exit(1)
    if not os.path.exists(TFLITE_PATH):
        print(f"ERROR: {TFLITE_PATH} not found"); sys.exit(1)

    print(f"Loading PyTorch model: {PTH_PATH}")
    pt_model = load_pytorch_model(PTH_PATH)

    print(f"Loading TFLite model:  {TFLITE_PATH}")
    tfl_interp = load_tflite_model(TFLITE_PATH)

    print(f"\nGenerating {NUM_SAMPLES} synthetic images and comparing predictions...\n")

    tip_dists = []
    angle_diffs = []
    heatmap_max_diffs = []
    heatmap_mean_diffs = []
    mismatches = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(NUM_SAMPLES):
            angle_deg = int(np.random.uniform(0, 360))
            img_path = os.path.join(tmpdir, f"dial_{i:04d}.png")
            gt_tip_x, gt_tip_y = generate_image(angle_deg, img_path)

            img_hwc = preprocess(img_path)

            pt_x, pt_y, pt_hm = predict_pytorch(pt_model, img_hwc)
            tfl_x, tfl_y, tfl_hm = predict_tflite(tfl_interp, img_hwc)

            # Tip distance between the two models
            dist = math.sqrt((pt_x - tfl_x) ** 2 + (pt_y - tfl_y) ** 2)
            tip_dists.append(dist)

            # Angle difference
            pt_angle = tip_to_angle(pt_x, pt_y)
            tfl_angle = tip_to_angle(tfl_x, tfl_y)
            adiff = abs(pt_angle - tfl_angle)
            if adiff > 180:
                adiff = 360 - adiff
            angle_diffs.append(adiff)

            # Heatmap diffs
            max_diff = np.abs(pt_hm - tfl_hm).max()
            mean_diff = np.abs(pt_hm - tfl_hm).mean()
            heatmap_max_diffs.append(max_diff)
            heatmap_mean_diffs.append(mean_diff)

            match = "OK" if dist == 0 else f"DIFF {dist:.1f}px"
            if dist > 0:
                mismatches += 1

            print(f"  [{i+1:3d}/{NUM_SAMPLES}] angle={angle_deg:3d}deg  "
                  f"PT=({pt_x:3d},{pt_y:3d})  TFL=({tfl_x:3d},{tfl_y:3d})  "
                  f"tip_dist={dist:.1f}px  hm_max={max_diff:.6f}  {match}")

    # Summary
    print("\n" + "=" * 65)
    print("SUMMARY")
    print("=" * 65)
    print(f"  Samples tested       : {NUM_SAMPLES}")
    print(f"  Exact tip matches    : {NUM_SAMPLES - mismatches}/{NUM_SAMPLES}")
    print(f"  Tip distance (mean)  : {np.mean(tip_dists):.2f} px")
    print(f"  Tip distance (max)   : {np.max(tip_dists):.2f} px")
    print(f"  Angle diff   (mean)  : {np.mean(angle_diffs):.2f} deg")
    print(f"  Angle diff   (max)   : {np.max(angle_diffs):.2f} deg")
    print(f"  Heatmap max-diff     : {np.max(heatmap_max_diffs):.6f}")
    print(f"  Heatmap mean-diff    : {np.mean(heatmap_mean_diffs):.6f}")

    # Pass/fail
    max_tip = np.max(tip_dists)
    if max_tip <= 2.0:
        print(f"\n  PASS  (max tip distance {max_tip:.2f}px <= 2px tolerance)")
    else:
        print(f"\n  FAIL  (max tip distance {max_tip:.2f}px > 2px tolerance)")
        sys.exit(1)


if __name__ == "__main__":
    main()
