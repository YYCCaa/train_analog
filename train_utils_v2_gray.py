"""Shared training pipeline for tiny heatmap models -- 1-channel (grayscale).

This is the grayscale counterpart of ``train_utils_v2.py``.  The loss, the
metrics, the diagnostics and the ONNX->TFLite / ORT optimisation helpers do
not care how many input channels the model has, so they are imported from
``train_utils_v2`` unchanged.  Only the pieces that hard-code the 3-channel
input shape are redefined here with ``IN_CHANNELS = 1``.

Reused from train_utils_v2 (channel-agnostic):
    IMG_SIZE, make_heatmap, weighted_mse_loss, peak_sharpness_loss,
    _dist_stats, train_one_epoch, evaluate, save_radial_error_plot,
    print_worst_n, export_tflite, optimise_onnx

Redefined here (channel-dependent):
    DialDatasetGray   -- PNG -> convert("L") -> (1, 256, 256) tensor
    export_onnx       -- dummy input (1, 1, 256, 256)
    benchmark         -- ORT input  (1, 1, 256, 256)
    verify            -- PT vs ORT on (1, 1, 256, 256)
    run_training      -- v2's run_training calls the module-level
                         DialDataset / export_onnx / verify / benchmark by
                         name, so it is copied here to bind the gray versions
                         (and to expose args.num_workers).

Data shape flow:

    dial_XXXX.png                 (mode "L", or RGB -> converted to "L")
    256 x 256  uint8 [0,255]
          |  np.float32 / 255, add channel axis
          v
    tensor (1, 256, 256) float32 [0,1]      -- one sample
          |  DataLoader batches
          v
    tensor (B, 1, 256, 256) float32         -- model input
          |  InvResNetGray (stem Conv2d 1->24, ...)
          v
    heatmap (B, 1, H, W) float32 [0,1]      H = W = 256, or 128 with --half-res
          |  argmax  -> (x, y) tip pixel, * (256 / W)
          v
    tip distance error in 256-px image coordinates
"""

import csv
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

from train_utils_v2 import (  # noqa: F401  (re-exported for callers)
    IMG_SIZE,
    _dist_stats,
    evaluate,
    export_tflite,
    make_heatmap,
    optimise_onnx,
    peak_sharpness_loss,
    print_worst_n,
    save_radial_error_plot,
    train_one_epoch,
    weighted_mse_loss,
)

IN_CHANNELS = 1  # int: number of input image channels (grayscale)


# ── dataset ──────────────────────────────────────────────────────────────

class DialDatasetGray(Dataset):
    """Synthetic dial images (grayscale) with Gaussian heatmap targets.

    Each item is a tuple:
        img      -- torch.float32, (1, img_size, img_size), values in [0, 1]
        heatmap  -- torch.float32, (1, output_size, output_size), peak 1.0
        filename -- str, e.g. "dial_0000.png"

    ``convert("L")`` means RGB datasets (e.g. sv22) load fine too; they are
    collapsed to luma with PIL's ITU-R 601-2 weights.
    """

    def __init__(self, data_dir, img_size=IMG_SIZE, output_size=None, sigma=3.0):
        self.data_dir = data_dir                      # str: folder with PNGs + labels.csv
        self.img_size = img_size                      # int: input side length in px (256)
        self.output_size = output_size or img_size    # int: heatmap side length (256 or 128)
        self.sigma = sigma                            # float: Gaussian std in heatmap px
        self.samples = []                             # list[dict{filename:str, tip_x:float, tip_y:float}]
        with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                self.samples.append({
                    "filename": row["filename"],
                    "tip_x": float(row["tip_x"]),
                    "tip_y": float(row["tip_y"]),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, s["filename"])).convert("L")
        img = np.array(img, dtype=np.float32) / 255.0  # (H, W) float32 in [0,1]
        img = img[None, :, :]                           # (1, H, W) float32
        out_scale = self.output_size / self.img_size   # float: label px -> heatmap px
        hm = make_heatmap(s["tip_x"] * out_scale, s["tip_y"] * out_scale,
                          self.output_size, self.sigma)  # (out, out) float32
        return torch.from_numpy(img), torch.from_numpy(hm).unsqueeze(0), s["filename"]


# ── ONNX export ──────────────────────────────────────────────────────────

def export_onnx(model, path, opset=17):
    """Export ``model`` to ONNX with a (1, 1, 256, 256) float32 input."""
    model.eval().cpu()
    dummy = torch.randn(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE)  # (1, 1, 256, 256) float32
    torch.onnx.export(model, dummy, path, opset_version=opset,
                      input_names=["input"], output_names=["heatmap"])
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX exported: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


# ── benchmark ────────────────────────────────────────────────────────────

def benchmark(path, n_warmup=20, n_runs=100):
    """Single-image CPU inference latency (1 thread, worst-case)."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    # inp: (1, 1, 256, 256) float32 in [0,1) -- random gray image
    inp = np.random.rand(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {"input": inp})

    times = []  # list[float]: per-run latency in ms
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": inp})
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)  # (n_runs,) float64 ms
    print(f"\nLatency (1 thread, {n_runs} runs):")
    print(f"  Mean:   {t.mean():.1f} ms")
    print(f"  Median: {np.median(t):.1f} ms")
    print(f"  P95:    {np.percentile(t, 95):.1f} ms")
    print(f"  Min:    {t.min():.1f} ms")
    print(f"  Max:    {t.max():.1f} ms")
    ok = np.percentile(t, 95) < 30
    print(f"  {'PASS' if ok else 'WARN'}: target P95 < 30ms on i5-11400")
    return t


def verify(model, path):
    """Compare PyTorch vs ONNX outputs on a random (1, 1, 256, 256) input."""
    model.eval().cpu()
    np.random.seed(0)
    # inp: (1, 1, 256, 256) float32 in [0,1)
    inp = np.random.rand(1, IN_CHANNELS, IMG_SIZE, IMG_SIZE).astype(np.float32)
    with torch.no_grad():
        pt = model(torch.from_numpy(inp)).numpy()  # (1, 1, H, W) float32
    ort_out = ort.InferenceSession(
        path, providers=["CPUExecutionProvider"]
    ).run(None, {"input": inp})[0]                 # (1, 1, H, W) float32
    diff = np.abs(pt - ort_out).max()
    out_w = pt.shape[-1]
    py, px = divmod(int(pt[0, 0].flatten().argmax()), out_w)
    oy, ox = divmod(int(ort_out[0, 0].flatten().argmax()), out_w)
    print(f"Verify: max_diff={diff:.6f}  PT=({px},{py})  ONNX=({ox},{oy})  "
          f"match={'YES' if px == ox and py == oy else 'NO'}")


# ── full pipeline ────────────────────────────────────────────────────────

def run_training(model, args):
    """Train -> export ONNX -> optimise -> verify -> benchmark (grayscale).

    Same behaviour as ``train_utils_v2.run_training`` but with
    ``DialDatasetGray`` and the 1-channel export/verify/benchmark above.
    ``args.num_workers`` (int, default 4) sets DataLoader workers.
    """
    if args.benchmark_only:
        benchmark(args.save_onnx)
        return

    half_res = getattr(args, 'half_res', False)            # bool: 128x128 heatmap
    output_size = IMG_SIZE // 2 if half_res else IMG_SIZE   # int: heatmap side px
    sigma = args.sigma / 2 if half_res else args.sigma      # float: sigma in heatmap px
    num_workers = getattr(args, "num_workers", 4)           # int: DataLoader workers

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Sigma: {sigma}  |  Output: {output_size}x{output_size}"
          f"  |  In-channels: {IN_CHANNELS}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    ds = DialDatasetGray(args.data_dir, output_size=output_size, sigma=sigma)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, num_workers=num_workers, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=num_workers, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    peak_weight = getattr(args, 'peak_weight', 0.00003)

    model.to(device)
    crit = weighted_mse_loss
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    print(f"Peak weight: {peak_weight}")

    pth = args.save_onnx.replace(".onnx", ".pth")
    best = float("inf")
    best_p99 = float("inf")

    # Diagnostics output directory (named after model)
    model_stem = os.path.splitext(os.path.basename(args.save_onnx))[0]
    diag_dir = os.path.join("diag", model_stem)
    os.makedirs(diag_dir, exist_ok=True)

    for ep in range(1, args.epochs + 1):
        tl, ts = train_one_epoch(model, train_ldr, crit, opt, device,
                                 peak_weight=peak_weight)
        vl, vs, val_records = evaluate(model, val_ldr, crit, device)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:3d}/{args.epochs}  train={tl:.6f}  val={vl:.6f}  lr={lr:.2e}")
        print(f"  val  mean={vs['mean']:.2f}  med={vs['median']:.2f}  "
              f"p95={vs['p95']:.1f}  p99={vs['p99']:.1f}  "
              f"worst={vs['worst']:.1f} [{vs['worst_file']}]")
        print(f"  trn  mean={ts['mean']:.2f}  med={ts['median']:.2f}  "
              f"p95={ts['p95']:.1f}  p99={ts['p99']:.1f}  "
              f"worst={ts['worst']:.1f} [{ts['worst_file']}]")
        if vs['mean'] < best:
            best = vs['mean']
            torch.save(model.state_dict(), pth)
            print(f"  -> saved best mean (dist={best:.2f}px)")
            save_radial_error_plot(val_records, diag_dir, ep)
            print_worst_n(val_records, n=20)
        if vs['p99'] < best_p99:
            best_p99 = vs['p99']
            pth_p99 = pth.replace(".pth", "_p99.pth")
            torch.save(model.state_dict(), pth_p99)
            print(f"  -> saved best p99 (p99={best_p99:.2f}px)")
            save_radial_error_plot(val_records, diag_dir, ep)
            print_worst_n(val_records, n=20)

    # Save final epoch model
    pth_final = pth.replace(".pth", "_final.pth")
    torch.save(model.state_dict(), pth_final)
    print(f"\nFinal epoch model saved: {pth_final}")

    onnx_final = args.save_onnx.replace(".onnx", "_final.onnx")
    export_onnx(model, onnx_final)

    tflite_final = onnx_final.replace(".onnx", ".tflite")
    export_tflite(onnx_final, tflite_final)

    # Load best model for primary export
    print(f"\nBest val distance: {best:.2f}px")
    model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))

    export_onnx(model, args.save_onnx)
    verify(model, args.save_onnx)

    tflite_path = args.save_onnx.replace(".onnx", ".tflite")
    export_tflite(args.save_onnx, tflite_path)

    print("\n=== Benchmark ===")
    benchmark(args.save_onnx)
