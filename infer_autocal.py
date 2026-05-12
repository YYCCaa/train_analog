"""Run autocal inference on a single image and draw the predicted geometry.

Note on intentional duplication
--------------------------------
``unpack_8`` duplicates ``autocal_utils.unpack_params_from_8``.  The
autocal_utils version is torch-based; this script is intentionally
torch-free at inference (ONNX Runtime + numpy only).

``draw_geometry`` duplicates the polyline math from
``generate_synthetic_v3.draw_dial_boundary`` and from
``autocal_utils.canonical_to_image_np``.  Same rationale — PIL-only at
inference time.
"""

import argparse
import math

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from autocal_utils import IMG_SIZE, derive_marks_np


def preprocess(img_path, img_size=IMG_SIZE):
    """Load and resize/pad an image to (img_size, img_size).

    Resizes the longer side to fit, then edge-mean pads the rest.
    Returns (chw_float_array, original_image_resized).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = img_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.array(img, dtype=np.uint8)
    pad_left = (img_size - new_w) // 2
    pad_top = (img_size - new_h) // 2
    pad_right = img_size - new_w - pad_left
    pad_bottom = img_size - new_h - pad_top

    if pad_left or pad_right:
        edge = arr[:, [0, -1], :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((new_h, img_size, 3), dtype=np.uint8)
        new[:, :pad_left] = edge
        new[:, pad_left:pad_left + new_w] = arr
        new[:, pad_left + new_w:] = edge
        arr = new
    if pad_top or pad_bottom:
        edge = arr[[0, -1], :, :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((img_size, img_size, 3), dtype=np.uint8)
        new[:pad_top, :] = edge
        new[pad_top:pad_top + new_h, :] = arr
        new[pad_top + new_h:, :] = edge
        arr = new

    chw = arr.astype(np.float32) / 255.0
    chw = chw.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return chw, Image.fromarray(arr, "RGB")


def unpack_8(p):
    """Unpack the 8-vector network output back to 6 raw params (numpy).

    Duplicates ``autocal_utils.unpack_params_from_8`` intentionally so
    that this script has no PyTorch dependency at inference time.
    """
    cx_n, cy_n, a_n, b_n, sin2, cos2, sin_p, cos_p = p
    cx = cx_n * IMG_SIZE
    cy = cy_n * IMG_SIZE
    a = a_n * (IMG_SIZE / 2)
    b = b_n * (IMG_SIZE / 2)
    theta = 0.5 * math.atan2(sin2, cos2)
    phi_0 = math.atan2(sin_p, cos_p)
    return cx, cy, a, b, theta, phi_0


def draw_geometry(img, cx, cy, a, b, theta, phi_0):
    """Overlay predicted ellipse and 10 marks on the image."""
    d = ImageDraw.Draw(img)
    # Ellipse as polyline
    pts = []
    for k in range(181):
        t = 2 * math.pi * k / 180
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = cx + (a * cos_t * math.cos(theta) - b * sin_t * math.sin(theta))
        y = cy + (a * cos_t * math.sin(theta) + b * sin_t * math.cos(theta))
        pts.append((x, y))
    d.line(pts, fill=(0, 255, 0), width=2)
    # 10 marks
    marks = derive_marks_np(cx, cy, a, b, theta, phi_0)
    for i in range(10):
        mx, my = marks[i]
        d.ellipse([mx - 5, my - 5, mx + 5, my + 5],
                  outline=(255, 0, 0), width=2)
        d.text((mx + 6, my + 6), str(i), fill=(255, 0, 0))
    # Center
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 0))
    return img


def main():
    p = argparse.ArgumentParser(description="Autocal inference")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", default="autocal_overlay.png")
    args = p.parse_args()

    sess = ort.InferenceSession(args.model,
                                providers=["CPUExecutionProvider"])
    inp_arr, vis_img = preprocess(args.image)
    out = sess.run(None, {"input": inp_arr})[0][0]  # (8,)
    cx, cy, a, b, theta, phi_0 = unpack_8(out)
    print(f"cx={cx:.2f} cy={cy:.2f} a={a:.2f} b={b:.2f} "
          f"theta={math.degrees(theta):.2f}° phi_0={math.degrees(phi_0):.2f}°")
    vis_img = draw_geometry(vis_img, cx, cy, a, b, theta, phi_0)
    vis_img.save(args.output)
    print(f"Overlay saved: {args.output}")


if __name__ == "__main__":
    main()
