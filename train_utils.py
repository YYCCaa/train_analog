"""Shared training pipeline for tiny heatmap models.

Provides dataset loading, training loop, ONNX export, optimisation,
benchmarking, and verification.  Used by all model variants (A/B/C).

Note: This module defines its own DialDataset with an explicit sigma
parameter, avoiding the default-capture bug in train.py's make_heatmap.
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

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    total_dist = worst_dist = 0.0
    worst_file = ""
    count = 0
    for imgs, targets, fnames in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        preds = model(imgs)
        loss = criterion(preds, targets)
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
                total_dist += d
                if d > worst_dist:
                    worst_dist = d
                    worst_file = fnames[i]
                count += 1
    mean_dist = total_dist / max(count, 1)
    return total / len(loader.dataset), mean_dist, worst_dist, worst_file


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = total_dist = worst_dist = 0.0
    worst_file = ""
    count = 0
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
            total_dist += d
            if d > worst_dist:
                worst_dist = d
                worst_file = fnames[i]
            count += 1
    return total_loss / len(loader.dataset), total_dist / max(count, 1), worst_dist, worst_file


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
                           shuffle=True, num_workers=0, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size,
                         shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    model.to(device)
    crit = nn.MSELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr,
                     weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    pth = args.save_onnx.replace(".onnx", ".pth")
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        tl, td, tw, t_wf = train_one_epoch(model, train_ldr, crit, opt, device)
        vl, vd, vw, v_wf = evaluate(model, val_ldr, crit, device)
        sched.step()
        lr = opt.param_groups[0]["lr"]
        print(f"Epoch {ep:3d}/{args.epochs}  train={tl:.6f}  "
              f"val={vl:.6f}  dist={vd:.2f}px  worst={vw:.1f}px [{v_wf}]  "
              f"t_dist={td:.2f}px  t_worst={tw:.1f}px [{t_wf}]  lr={lr:.2e}")
        if vd < best:
            best = vd
            torch.save(model.state_dict(), pth)
            print(f"  -> saved (dist={best:.2f}px)")

    print(f"\nBest val distance: {best:.2f}px")
    model.load_state_dict(torch.load(pth, map_location="cpu", weights_only=True))

    export_onnx(model, args.save_onnx)
    verify(model, args.save_onnx)

    tflite_path = args.save_onnx.replace(".onnx", ".tflite")
    export_tflite(args.save_onnx, tflite_path)

    print("\n=== Benchmark ===")
    benchmark(args.save_onnx)
