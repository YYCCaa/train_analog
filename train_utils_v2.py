"""Shared training pipeline for tiny heatmap models (v2: anti-diffuse loss).

Same as train_utils.py but adds a peak-sharpness penalty to MSE so the
model cannot satisfy the loss with broad, flat blobs.

Loss = MSE + peak_weight * (1 - max(pred_heatmap))

The penalty drives the predicted peak toward 1.0, matching the target
Gaussian's peak height.  When the prediction is already sharp the penalty
is near zero and MSE dominates.
"""

import csv
import math
import os
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split

IMG_SIZE = 256


# ── dataset ──────────────────────────────────────────────────────────────

def make_heatmap(tip_x, tip_y, size, sigma):
    """2-D Gaussian heatmap centred on (tip_x, tip_y)."""
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    return np.exp(-((xx - tip_x) ** 2 + (yy - tip_y) ** 2) / (2 * sigma ** 2))


class DialDataset(Dataset):
    """Synthetic dial images with Gaussian heatmap targets."""

    def __init__(self, data_dir, img_size=IMG_SIZE, output_size=None, sigma=3.0):
        self.data_dir = data_dir
        self.img_size = img_size
        self.output_size = output_size or img_size
        self.sigma = sigma
        self.samples = []
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
        img = Image.open(os.path.join(self.data_dir, s["filename"])).convert("RGB")
        img = np.array(img, dtype=np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # HWC -> CHW
        out_scale = self.output_size / self.img_size
        hm = make_heatmap(s["tip_x"] * out_scale, s["tip_y"] * out_scale,
                          self.output_size, self.sigma)
        return torch.from_numpy(img), torch.from_numpy(hm).unsqueeze(0), s["filename"]


# ── training ─────────────────────────────────────────────────────────────

def weighted_mse_loss(pred, target, w_pos=10.0):
    """MSE with higher weight near the ground-truth peak (target > 0.1)."""
    weight = torch.ones_like(target)
    weight[target > 0.1] = w_pos
    return (weight * (pred - target) ** 2).mean()


def peak_sharpness_loss(preds):
    """Penalise diffuse predictions: mean(1 - max_per_sample)."""
    B = preds.size(0)
    peaks = preds.view(B, -1).max(dim=1).values  # max activation per sample
    return (1.0 - peaks).mean()


def _dist_stats(dists, files):
    """Compute mean, median, p95, p99, worst dist and worst file from lists."""
    if not dists:
        return {"mean": 0, "median": 0, "p95": 0, "p99": 0,
                "worst": 0, "worst_file": ""}
    arr = np.array(dists)
    idx = int(arr.argmax())
    return {
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "worst": float(arr.max()),
        "worst_file": files[idx],
    }


def train_one_epoch(model, loader, criterion, optimizer, device,
                    peak_weight=0):
    model.train()
    total = 0.0
    all_dists = []
    all_files = []
    for imgs, targets, fnames in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets) + peak_weight * peak_sharpness_loss(preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * imgs.size(0)
        scale = IMG_SIZE / preds.shape[-1]
        with torch.no_grad():
            for i in range(preds.size(0)):
                pm, gm = preds[i, 0], targets[i, 0]
                pi = pm.view(-1).argmax().item()
                gi = gm.view(-1).argmax().item()
                py, px = divmod(pi, pm.size(1))
                gy, gx = divmod(gi, gm.size(1))
                d = math.sqrt((px - gx) ** 2 + (py - gy) ** 2) * scale
                all_dists.append(d)
                all_files.append(fnames[i])
    stats = _dist_stats(all_dists, all_files)
    return total / len(loader.dataset), stats


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (loss, stats_dict, records) where records is a list of
    (gt_angle_deg, dist_px, filename) per validation sample."""
    model.eval()
    total_loss = 0.0
    all_dists = []
    all_files = []
    records = []  # (angle, dist, filename)
    cx = cy = IMG_SIZE / 2.0
    for imgs, targets, fnames in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        total_loss += criterion(preds, targets).item() * imgs.size(0)
        scale = IMG_SIZE / preds.shape[-1]
        for i in range(preds.size(0)):
            pm, gm = preds[i, 0], targets[i, 0]
            pi = pm.view(-1).argmax().item()
            gi = gm.view(-1).argmax().item()
            py, px = divmod(pi, pm.size(1))
            gy, gx = divmod(gi, gm.size(1))
            d = math.sqrt((px - gx) ** 2 + (py - gy) ** 2) * scale
            all_dists.append(d)
            all_files.append(fnames[i])
            # ground-truth angle (CW from up, 0=up 90=right)
            gt_x = gx * scale
            gt_y = gy * scale
            angle = math.degrees(math.atan2(gt_x - cx, -(gt_y - cy))) % 360
            records.append((angle, d, fnames[i]))
    stats = _dist_stats(all_dists, all_files)
    return total_loss / len(loader.dataset), stats, records


# ── diagnostics (radial plot + worst list) ────────────────────────────────

def save_radial_error_plot(records, out_dir, epoch, threshold=50.0):
    """Polar plot: count of samples with dist > threshold per angle bin."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_bins = 72
    bin_width = 360.0 / n_bins
    bin_counts = [0] * n_bins
    for angle, dist, _ in records:
        if dist > threshold:
            b = int(angle / bin_width) % n_bins
            bin_counts[b] += 1

    thetas = [math.radians(i * bin_width + bin_width / 2) for i in range(n_bins)]
    # Close the loop
    thetas.append(thetas[0])
    bin_counts.append(bin_counts[0])

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))
    ax.plot(thetas, bin_counts, "o-", markersize=3)
    ax.fill(thetas, bin_counts, alpha=0.15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)  # clockwise
    total_bad = sum(bin_counts[:-1])  # exclude duplicated closing point
    ax.set_title(f"Val dist>{threshold:.0f}px by angle (ep {epoch}, n={total_bad})", pad=15)
    ax.set_ylabel("# images")

    path = os.path.join(out_dir, f"radial_ep{epoch:03d}.png")
    fig.savefig(path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> radial plot: {path}")


def print_worst_n(records, n=20):
    """Print the n samples with highest distance error."""
    sorted_recs = sorted(records, key=lambda r: r[1], reverse=True)
    print(f"  -> worst {n} validation samples:")
    for i, (angle, dist, fname) in enumerate(sorted_recs[:n]):
        print(f"     {i+1:2d}. {dist:7.1f}px  angle={angle:5.1f}deg  {fname}")


# ── ONNX export & optimisation ──────────────────────────────────────────

def export_onnx(model, path, opset=17):
    model.eval().cpu()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(model, dummy, path, opset_version=opset,
                      input_names=["input"], output_names=["heatmap"])
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX exported: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")


def export_tflite(onnx_path, tflite_path):
    """Convert ONNX model to TFLite via onnx2tf."""
    try:
        import onnx2tf
    except ImportError:
        print("WARNING: onnx2tf not installed, skipping TFLite export.  "
              "Install with: pip install onnx2tf")
        return
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=tmp,
            non_verbose=True,
        )
        stem = os.path.splitext(os.path.basename(onnx_path))[0]
        shutil.copy2(os.path.join(tmp, f"{stem}_float32.tflite"), tflite_path)
    print(f"TFLite exported: {tflite_path}  ({os.path.getsize(tflite_path) / 1024:.1f} KB)")


def optimise_onnx(src, dst):
    """ORT graph optimisation (BN fusion, constant folding, etc.)."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    opts.optimized_model_filepath = dst
    ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
    print(f"Optimised: {dst}  ({os.path.getsize(dst) / 1024:.1f} KB)")


# ── benchmark ────────────────────────────────────────────────────────────

def benchmark(path, n_warmup=20, n_runs=100):
    """Single-image CPU inference latency (1 thread, worst-case)."""
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
    inp = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)

    for _ in range(n_warmup):
        sess.run(None, {"input": inp})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"input": inp})
        times.append((time.perf_counter() - t0) * 1000)

    t = np.array(times)
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
    """Compare PyTorch vs ONNX outputs on random input."""
    model.eval().cpu()
    np.random.seed(0)
    inp = np.random.rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    with torch.no_grad():
        pt = model(torch.from_numpy(inp)).numpy()
    ort_out = ort.InferenceSession(
        path, providers=["CPUExecutionProvider"]
    ).run(None, {"input": inp})[0]
    diff = np.abs(pt - ort_out).max()
    out_w = pt.shape[-1]
    py, px = divmod(int(pt[0, 0].flatten().argmax()), out_w)
    oy, ox = divmod(int(ort_out[0, 0].flatten().argmax()), out_w)
    print(f"Verify: max_diff={diff:.6f}  PT=({px},{py})  ONNX=({ox},{oy})  "
          f"match={'YES' if px == ox and py == oy else 'NO'}")


# ── full pipeline ────────────────────────────────────────────────────────

def run_training(model, args):
    """Train -> export ONNX -> optimise -> verify -> benchmark."""
    if args.benchmark_only:
        benchmark(args.save_onnx)
        return

    half_res = getattr(args, 'half_res', False)
    output_size = IMG_SIZE // 2 if half_res else IMG_SIZE
    sigma = args.sigma / 2 if half_res else args.sigma

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  Sigma: {sigma}  |  Output: {output_size}x{output_size}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    ds = DialDataset(args.data_dir, output_size=output_size, sigma=sigma)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size,
                           shuffle=True, num_workers=4, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)
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
