"""
Inference script for tip-heatmap U-Net.

Handles arbitrary input sizes:
  - Axes larger than TARGET_SIZE are centre-cropped (even on both sides).
  - Axes smaller than TARGET_SIZE are padded with edge mean colour
    (even on both sides).
  - The centre of the raw input always maps to the centre of the
    preprocessed 192×192 canvas.

Centre convention (matches generate_synthetic.py):
    cx = cy = TARGET_SIZE // 2  (= 96 for 192×192)

Output: clockwise angle in degrees from the positive-y axis (up)
    through the preprocessed-image centre, i.e.
        angle = atan2(tip_x - cx, -(tip_y - cy))   [then % 360]
    0° = up, 90° = right, 180° = down, 270° = left.
"""

import argparse
import math

import numpy as np
import torch
from PIL import Image

from train import TipHeatmapNet

TARGET_SIZE = 192
CENTER = TARGET_SIZE // 2  # 96  — must match generate_synthetic.CENTER


def preprocess(img_pil):
    """Crop / pad to TARGET_SIZE×TARGET_SIZE keeping input centre aligned.

    Returns:
        tensor : 1×3×TARGET_SIZE×TARGET_SIZE  float32 [0,1]
        offset : (ox, oy) — shift applied to original coords
                 orig = preprocessed + offset
    """
    w, h = img_pil.size  # PIL (width, height)
    arr = np.array(img_pil)  # H×W×3

    # --- x axis (width, axis=1) ---
    if w > TARGET_SIZE:
        crop_l = (w - TARGET_SIZE) // 2
        crop_r = w - TARGET_SIZE - crop_l
        arr = arr[:, crop_l: crop_l + TARGET_SIZE, :]
        ox = crop_l  # preprocessed_x + ox = original_x
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
        crop_b = cur_h - TARGET_SIZE - crop_t
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

    tensor = torch.from_numpy(arr.astype(np.float32) / 255.0)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # 1×3×H×W
    return tensor, (ox, oy)


def tip_to_angle(tip_x, tip_y, cx=CENTER, cy=CENTER):
    """Convert tip pixel coords to CW angle from positive-y axis (up).

    Same convention as generate_synthetic.py:
        0° = up, 90° = right, 180° = down, 270° = left.
    """
    dx = tip_x - cx
    dy = -(tip_y - cy)  # flip y so up is positive
    angle = math.degrees(math.atan2(dx, dy)) % 360
    return angle


@torch.no_grad()
def predict(model, img_pil, device):
    """Return (angle_deg, tip_x_orig, tip_y_orig, heatmap)."""
    tensor, (ox, oy) = preprocess(img_pil)
    tensor = tensor.to(device)

    heatmap = model(tensor)[0, 0]  # H×W
    idx = heatmap.view(-1).argmax().item()
    pred_y, pred_x = divmod(idx, heatmap.size(1))

    # Angle is computed in preprocessed space (centre = CENTER, CENTER)
    angle = tip_to_angle(pred_x, pred_y)

    # Map tip back to original image space
    orig_x = pred_x + ox
    orig_y = pred_y + oy

    return angle, orig_x, orig_y, heatmap.cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Infer pointer tip location")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--model", default="tip_heatmap_net.pth",
                        help="Path to trained weights")
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-heatmap", default=None,
                        help="Save predicted heatmap as greyscale image")
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    model = TipHeatmapNet()
    model.load_state_dict(torch.load(args.model, map_location=device,
                                     weights_only=True))
    model.to(device).eval()

    img = Image.open(args.image).convert("RGB")
    w, h = img.size
    print(f"Input: {args.image}  ({w}x{h})")
    print(f"Centre convention: ({CENTER}, {CENTER}) in {TARGET_SIZE}x{TARGET_SIZE} canvas")

    angle, tip_x, tip_y, heatmap = predict(model, img, device)
    print(f"Predicted tip (original coords): ({tip_x}, {tip_y})")
    print(f"Angle: {angle:.1f} deg  (CW from up, 0=up 90=right 180=down 270=left)")

    if args.save_heatmap:
        hm_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(hm_uint8, "L").save(args.save_heatmap)
        print(f"Heatmap saved to {args.save_heatmap}")


if __name__ == "__main__":
    main()
