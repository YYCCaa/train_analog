"""
TFLite inference script for tip-heatmap U-Net.

Drop-in replacement for infer.py that uses the TFLite model instead of
PyTorch.  No PyTorch or CUDA dependency — only NumPy and a TFLite runtime.

Handles arbitrary input sizes:
  - Axes larger than TARGET_SIZE are centre-cropped (even on both sides).
  - Axes smaller than TARGET_SIZE are padded with edge mean colour
    (even on both sides).
  - The centre of the raw input always maps to the centre of the
    preprocessed 192x192 canvas.

Centre convention (matches generate_synthetic.py):
    cx = cy = TARGET_SIZE // 2  (= 96 for 192x192)

Output: clockwise angle in degrees from the positive-y axis (up)
    through the preprocessed-image centre, i.e.
        angle = atan2(tip_x - cx, -(tip_y - cy))   [then % 360]
    0 deg = up, 90 deg = right, 180 deg = down, 270 deg = left.
"""

import argparse
import math

import numpy as np
from PIL import Image

TARGET_SIZE = 192
CENTER = TARGET_SIZE // 2  # 96  — must match generate_synthetic.CENTER


def _load_interpreter(tflite_path):
    """Load TFLite interpreter from ai_edge_litert or tensorflow."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    interp = Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    return interp


def preprocess(img_pil):
    """Crop / pad to TARGET_SIZE x TARGET_SIZE keeping input centre aligned.

    Returns:
        arr    : TARGET_SIZE x TARGET_SIZE x 3  float32 [0,1]  (NHWC-ready)
        offset : (ox, oy) — shift applied to original coords
                 orig = preprocessed + offset
    """
    w, h = img_pil.size  # PIL (width, height)
    arr = np.array(img_pil)  # H x W x 3

    # --- x axis (width, axis=1) ---
    if w > TARGET_SIZE:
        crop_l = (w - TARGET_SIZE) // 2
        arr = arr[:, crop_l: crop_l + TARGET_SIZE, :]
        ox = crop_l
    elif w < TARGET_SIZE:
        pad_total = TARGET_SIZE - w
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        edge_mean = arr[:, [0, -1], :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((arr.shape[0], TARGET_SIZE, 3), dtype=np.uint8)
        for c in range(3):
            new[:, :pad_l, c] = edge_mean[c]
            new[:, pad_l:pad_l + w, c] = arr[:, :, c]
            if pad_r > 0:
                new[:, -pad_r:, c] = edge_mean[c]
        arr = new
        ox = -pad_l
    else:
        ox = 0

    # --- y axis (height, axis=0) ---
    cur_h = arr.shape[0]
    if cur_h > TARGET_SIZE:
        crop_t = (cur_h - TARGET_SIZE) // 2
        arr = arr[crop_t: crop_t + TARGET_SIZE, :, :]
        oy = crop_t
    elif cur_h < TARGET_SIZE:
        pad_total = TARGET_SIZE - cur_h
        pad_t = pad_total // 2
        pad_b = pad_total - pad_t
        edge_mean = arr[[0, -1], :, :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((TARGET_SIZE, arr.shape[1], 3), dtype=np.uint8)
        for c in range(3):
            new[:pad_t, :, c] = edge_mean[c]
            new[pad_t:pad_t + cur_h, :, c] = arr[:, :, c]
            if pad_b > 0:
                new[-pad_b:, :, c] = edge_mean[c]
        arr = new
        oy = -pad_t
    else:
        oy = 0

    arr = arr.astype(np.float32) / 255.0
    return arr, (ox, oy)


def tip_to_angle(tip_x, tip_y, cx=CENTER, cy=CENTER):
    """Convert tip pixel coords to CW angle from positive-y axis (up).

    Same convention as generate_synthetic.py:
        0 deg = up, 90 deg = right, 180 deg = down, 270 deg = left.
    """
    dx = tip_x - cx
    dy = -(tip_y - cy)  # flip y so up is positive
    angle = math.degrees(math.atan2(dx, dy)) % 360
    return angle


def predict(interp, img_pil):
    """Return (angle_deg, tip_x_orig, tip_y_orig, heatmap).

    Parameters:
        interp  : TFLite Interpreter (already allocated)
        img_pil : PIL.Image in RGB mode
    """
    arr, (ox, oy) = preprocess(img_pil)

    inp_detail = interp.get_input_details()[0]
    out_detail = interp.get_output_details()[0]

    inp = arr[np.newaxis, ...]  # 1 x H x W x 3  (NHWC)
    interp.set_tensor(inp_detail["index"], inp)
    interp.invoke()
    raw = interp.get_tensor(out_detail["index"])

    # Reshape to H x W regardless of output layout
    if raw.ndim == 4 and raw.shape[-1] == 1:
        heatmap = raw[0, :, :, 0]   # NHWC → H x W
    elif raw.ndim == 4 and raw.shape[1] == 1:
        heatmap = raw[0, 0, :, :]   # NCHW → H x W
    else:
        heatmap = raw.reshape(TARGET_SIZE, TARGET_SIZE)

    idx = heatmap.flatten().argmax()
    pred_y, pred_x = divmod(int(idx), heatmap.shape[1])

    # Angle is computed in preprocessed space (centre = CENTER, CENTER)
    angle = tip_to_angle(pred_x, pred_y)

    # Map tip back to original image space
    orig_x = pred_x + ox
    orig_y = pred_y + oy

    return angle, orig_x, orig_y, heatmap


def main():
    parser = argparse.ArgumentParser(
        description="Infer pointer tip location (TFLite)")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="tip_heatmap_net.tflite",
                        help="Path to TFLite model")
    parser.add_argument("--save-heatmap", default=None,
                        help="Save predicted heatmap as greyscale image")
    args = parser.parse_args()

    interp = _load_interpreter(args.model)

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    print(f"Input: {args.image}  ({w}x{h})")
    print(f"Centre convention: ({CENTER}, {CENTER}) in "
          f"{TARGET_SIZE}x{TARGET_SIZE} canvas")

    angle, tip_x, tip_y, heatmap = predict(interp, img)
    print(f"Predicted tip (original coords): ({tip_x}, {tip_y})")
    print(f"Angle: {angle:.1f} deg  "
          f"(CW from up, 0=up 90=right 180=down 270=left)")

    if args.save_heatmap:
        hm_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(hm_uint8, "L").save(args.save_heatmap)
        print(f"Heatmap saved to {args.save_heatmap}")


if __name__ == "__main__":
    main()
