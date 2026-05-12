"""
TFLite inference script for tip-heatmap models.

Drop-in replacement for infer_onnx.py that uses a TFLite runtime instead of
ONNX Runtime.  No PyTorch or CUDA dependency -- only NumPy and a TFLite runtime.

Handles arbitrary input sizes:
  - Axes larger than TARGET_SIZE are centre-cropped (even on both sides).
  - Axes smaller than TARGET_SIZE are padded with edge mean colour
    (even on both sides).
  - The centre of the raw input always maps to the centre of the
    preprocessed 256x256 canvas.

Handles half-res models (128x128 output) automatically by upscaling the
heatmap to 256x256 before finding the peak.

Centre convention (matches generate_synthetic.py):
    cx = cy = TARGET_SIZE // 2  (= 128 for 256x256)

Output: clockwise angle in degrees from the positive-y axis (up)
    through the preprocessed-image centre, i.e.
        angle = atan2(tip_x - cx, -(tip_y - cy))   [then % 360]
    0 deg = up, 90 deg = right, 180 deg = down, 270 deg = left.
"""

import argparse
import math
import time
import tensorflow as tf
import numpy as np
from PIL import Image

TARGET_SIZE = 256
CENTER = TARGET_SIZE // 2  # 128  -- must match generate_synthetic.CENTER


def preprocess(img_pil):
    """Crop / pad to TARGET_SIZE x TARGET_SIZE keeping input centre aligned.

    Returns:
        arr    : TARGET_SIZE x TARGET_SIZE x 3  float32 [0,1]  (HWC)
        offset : (ox, oy) -- shift applied to original coords
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


def load_interpreter(tflite_path, threads=1):
    """Load TFLite interpreter with XNNPACK delegate."""

    Interpreter = tf.lite.Interpreter
    interp = Interpreter(
        model_path=tflite_path,
        num_threads=threads,
        experimental_delegates=[
            Interpreter.load_delegate("XNNPACK", {"num_threads": threads}),
        ] if hasattr(Interpreter, "load_delegate") else None,
    )
    interp.allocate_tensors()
    return interp


def predict(interp, img_pil, timings=None):
    """Return (angle_deg, tip_x_orig, tip_y_orig, heatmap).

    Parameters:
        interp  : TFLite Interpreter (already allocated)
        img_pil : PIL.Image in RGB mode
        timings : if a dict is passed, per-stage times (ms) are stored in it
    """
    t0 = time.perf_counter()
    arr, (ox, oy) = preprocess(img_pil)

    inp_detail = interp.get_input_details()[0]
    inp_shape = inp_detail["shape"]  # e.g. [1, 256, 256, 3] or [1, 3, 256, 256]

    # Prepare input in the layout the model expects
    if inp_shape[-1] == 3:
        inp = arr[np.newaxis, ...]              # NHWC: 1 x H x W x 3
    else:
        inp = arr.transpose(2, 0, 1)[np.newaxis, ...]  # NCHW: 1 x 3 x H x W

    t1 = time.perf_counter()

    interp.set_tensor(inp_detail["index"], inp)
    interp.invoke()
    raw = interp.get_tensor(interp.get_output_details()[0]["index"])
    t2 = time.perf_counter()

    # Reshape to H x W regardless of output layout
    if raw.ndim == 4 and raw.shape[-1] == 1:
        heatmap = raw[0, :, :, 0]   # NHWC -> H x W
    elif raw.ndim == 4 and raw.shape[1] == 1:
        heatmap = raw[0, 0, :, :]   # NCHW -> H x W
    else:
        heatmap = raw.squeeze()

    # Upscale to TARGET_SIZE if model outputs lower resolution (e.g. half-res 128x128)
    if heatmap.shape[0] != TARGET_SIZE or heatmap.shape[1] != TARGET_SIZE:
        heatmap = np.array(Image.fromarray(heatmap, "F").resize(
            (TARGET_SIZE, TARGET_SIZE), Image.BILINEAR))

    idx = heatmap.flatten().argmax()
    pred_y, pred_x = divmod(int(idx), heatmap.shape[1])

    # Angle is computed in preprocessed space (centre = CENTER, CENTER)
    angle = tip_to_angle(pred_x, pred_y)

    # Map tip back to original image space
    orig_x = pred_x + ox
    orig_y = pred_y + oy
    t3 = time.perf_counter()

    if timings is not None:
        timings["preprocess"] = (t1 - t0) * 1000
        timings["inference"]  = (t2 - t1) * 1000
        timings["postprocess"] = (t3 - t2) * 1000
        timings["total"]      = (t3 - t0) * 1000

    return angle, orig_x, orig_y, heatmap


def benchmark(interp, img_pil, n_warmup=20, n_runs=100):
    """Benchmark inference with per-stage timing breakdown."""
    for _ in range(n_warmup):
        predict(interp, img_pil)

    stages = {"preprocess": [], "inference": [], "postprocess": [], "total": []}
    for _ in range(n_runs):
        t = {}
        predict(interp, img_pil, timings=t)
        for k in stages:
            stages[k].append(t[k])

    print(f"\nBenchmark ({n_runs} runs):")
    print(f"  {'Stage':<14s} {'Mean':>8s} {'Median':>8s} {'P95':>8s}"
          f" {'Min':>8s} {'Max':>8s}")
    print(f"  {'-'*54}")
    for name in ("preprocess", "inference", "postprocess", "total"):
        s = np.array(stages[name])
        print(f"  {name:<14s} {s.mean():7.2f}ms {np.median(s):7.2f}ms"
              f" {np.percentile(s, 95):7.2f}ms {s.min():7.2f}ms"
              f" {s.max():7.2f}ms")

    total = np.array(stages["total"])
    ok = np.percentile(total, 95) < 30
    print(f"\n  {'PASS' if ok else 'WARN'}: target P95 total < 30ms on i5-11400")
    return stages


def run_batch(model_path, image_pattern, indices, threads=8, save_heatmaps=True):
    """Run one TFLite model on a batch of images.

    Parameters:
        model_path    : path to the TFLite model (single model for all images)
        image_pattern : format string with {n}, e.g. "model_analog_main_{n}_ana{n}.png"
        indices       : list of n values, e.g. [0, 1, 2, 3]
        threads       : interpreter threads
        save_heatmaps : if True, save heatmaps as hm_{n}.png
    """
    interp = load_interpreter(model_path, threads=threads)
    print(f"Model: {model_path}")
    for n in indices:
        image_path = image_pattern.format(n=n)
        print(f"\n{'='*60}")
        print(f"[n={n}]  image={image_path}")

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        print(f"  Input size: {w}x{h}")

        angle, tip_x, tip_y, heatmap = predict(interp, img)
        print(f"  Tip (orig coords): ({tip_x}, {tip_y})")
        print(f"  Angle: {angle:.1f} deg")

        if save_heatmaps:
            hm_path = f"hm_{n}.png"
            hm_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(hm_uint8, "L").save(hm_path)
            print(f"  Heatmap saved: {hm_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Infer pointer tip location (TFLite)")
    parser.add_argument("image", nargs="?", default=None,
                        help="Path to input image (single-image mode)")
    parser.add_argument("--model", default="model_a_micro_unet.tflite",
                        help="Path to TFLite model (single-image mode)")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of interpreter threads")
    parser.add_argument("--save-heatmap", default=None,
                        help="Save predicted heatmap as greyscale image")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run latency benchmark on the given image")
    parser.add_argument("--benchmark-runs", type=int, default=100,
                        help="Number of timed runs for benchmark")
    parser.add_argument("--batch", action="store_true",
                        help="Batch mode: run n=0..3 with model_analog_main_{n}_ana{n}")
    parser.add_argument("--batch-indices", default="0,1,2,3",
                        help="Comma-separated indices for batch mode (default: 0,1,2,3)")
    args = parser.parse_args()

    if args.batch:
        indices = [int(x) for x in args.batch_indices.split(",")]
        run_batch(
            model_path=args.model,
            image_pattern="model_analog_main_{n}_ana{n}.png",
            indices=indices,
            threads=args.threads,
        )
        return

    if args.image is None:
        parser.error("image is required in single-image mode (or use --batch)")

    interp = load_interpreter(args.model, threads=args.threads)

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

    if args.benchmark:
        benchmark(interp, img, n_runs=args.benchmark_runs)


if __name__ == "__main__":
    main()
