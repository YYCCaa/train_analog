"""Autocalibration shared utilities.

Contents:
  - Geometry math: canonical_to_image_np/torch, derive_marks_np/torch.
  - Parameter packing: pack_params_to_8, unpack_params_from_8.
  - Loss: autocal_loss (SmoothL1 on params + derived marks).
  - Dataset: AutoCalibDataset (PyTorch Dataset over generated images + labels.csv).
  - Backbones: AutocalModelE, AutocalModelF (regression head replaces heatmap decoder).
  - Training: train_one_epoch_autocal, evaluate_autocal, run_autocal_training, _export_onnx_autocal.
"""

import csv
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


IMG_SIZE = 256  # shared with generate_synthetic_v3 and train_autocal_*
N_MARKS = 10
DEG_PER_MARK = 360.0 / N_MARKS
RAD_PER_MARK = math.radians(DEG_PER_MARK)


# ── Geometry: canonical-circle → ellipse map ────────────────────────────

def canonical_to_image_np(t, r_c, cx, cy, a, b, theta):
    """Map a canonical-unit-circle point to ellipse image coords (numpy/float).

    For a canonical point at angle `t` and radius `r_c ∈ [0, 1]`, the image
    position is:
        x = cx + r_c · (a·cos(t)·cosθ - b·sin(t)·sinθ)
        y = cy + r_c · (a·cos(t)·sinθ + b·sin(t)·cosθ)

    Major axis is along (cosθ, sinθ); minor axis is perpendicular.
    """
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return x, y


def canonical_to_image_torch(t, r_c, cx, cy, a, b, theta):
    """Differentiable batched version of canonical_to_image.

    All inputs are 1-D tensors of shape (B,). Returns (B, 2).
    """
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    cos_o = torch.cos(theta)
    sin_o = torch.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return torch.stack([x, y], dim=-1)


def derive_marks_np(cx, cy, a, b, theta, phi_0):
    """Compute the 10 scale-mark positions in image coords.

    Mark i lies on the ellipse at canonical angle t_i = phi_0 + i · 36°.
    Returns array of shape (10, 2).
    """
    marks = np.empty((N_MARKS, 2), dtype=np.float64)
    for i in range(N_MARKS):
        t_i = phi_0 + i * RAD_PER_MARK
        x, y = canonical_to_image_np(t=t_i, r_c=1.0,
                                     cx=cx, cy=cy, a=a, b=b, theta=theta)
        marks[i, 0] = x
        marks[i, 1] = y
    return marks


def derive_marks_torch(cx, cy, a, b, theta, phi_0):
    """Differentiable batched mark derivation.

    All inputs are 1-D tensors of shape (B,). Returns (B, 10, 2).
    """
    device = cx.device
    dtype = cx.dtype
    i = torch.arange(N_MARKS, device=device, dtype=dtype)
    # Broadcast: phi_0 (B,1) + i*(10,) -> (B, 10).
    # `_bc` suffix marks tensors unsqueezed from (B,) to (B, 1) for broadcasting
    # over the 10 marks dimension.
    t_i = phi_0.unsqueeze(-1) + i * RAD_PER_MARK
    cos_t = torch.cos(t_i)
    sin_t = torch.sin(t_i)
    cos_o = torch.cos(theta).unsqueeze(-1)
    sin_o = torch.sin(theta).unsqueeze(-1)
    a_bc = a.unsqueeze(-1)
    b_bc = b.unsqueeze(-1)
    cx_bc = cx.unsqueeze(-1)
    cy_bc = cy.unsqueeze(-1)
    x = cx_bc + a_bc * cos_t * cos_o - b_bc * sin_t * sin_o
    y = cy_bc + a_bc * cos_t * sin_o + b_bc * sin_t * cos_o
    return torch.stack([x, y], dim=-1)


# ── Parameter packing ────────────────────────────────────────────────────

def pack_params_to_8(cx, cy, a, b, theta, phi_0, img_size=IMG_SIZE):
    """Pack 6 raw params into 8 normalized scalars suitable for regression.

    Layout: [cx_n, cy_n, a_n, b_n, sin(2θ), cos(2θ), sin(φ₀), cos(φ₀)]
      cx_n, cy_n: divided by IMG_SIZE
      a_n, b_n: divided by IMG_SIZE/2
      θ uses 2θ because, with a ≥ b enforced, θ has period π (an axis line).
    Returns tensor of shape (B, 8).
    """
    cx_n = cx / img_size
    cy_n = cy / img_size
    a_n = a / (img_size / 2)
    b_n = b / (img_size / 2)
    sin_2theta = torch.sin(2 * theta)
    cos_2theta = torch.cos(2 * theta)
    sin_p = torch.sin(phi_0)
    cos_p = torch.cos(phi_0)
    return torch.stack(
        [cx_n, cy_n, a_n, b_n, sin_2theta, cos_2theta, sin_p, cos_p], dim=-1)


def unpack_params_from_8(packed, img_size=IMG_SIZE):
    """Inverse of pack_params_to_8.

    Returns (cx, cy, a, b, theta, phi_0), each shape (B,).

    A small epsilon is added to the cosine argument of atan2 so that the
    gradient is finite even when the network outputs an all-zero vector
    (typical random-init scenario).  The bias is 1e-4, small enough that
    the round-trip error for real (sin, cos) pairs stays below 5e-5.
    """
    _EPS = 1e-4
    cx_n, cy_n, a_n, b_n, sin_2theta, cos_2theta, sin_p, cos_p = packed.unbind(-1)
    cx = cx_n * img_size
    cy = cy_n * img_size
    a = a_n * (img_size / 2)
    b = b_n * (img_size / 2)
    theta = 0.5 * torch.atan2(sin_2theta, cos_2theta + _EPS)
    phi_0 = torch.atan2(sin_p, cos_p + _EPS)
    return cx, cy, a, b, theta, phi_0


# ── Loss ──────────────────────────────────────────────────────────────────

def autocal_loss(pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                 gt_marks, img_size=IMG_SIZE, w_param=0.1, w_mark=1.0):
    """Combined SmoothL1 loss on packed params and derived marks.

    Args:
        pred_8: (B, 8) — network output (the packed prediction).
        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0: ground truth, each (B,).
        gt_marks: (B, 10, 2) — ground truth mark positions in pixel coords.
        img_size: used to normalize the mark loss to unit scale.
        w_param, w_mark: weighting; mark loss carries the main signal.

    Returns: (total_loss, param_loss, mark_loss)
    """
    gt_8 = pack_params_to_8(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                            img_size=img_size)

    l_param = F.smooth_l1_loss(pred_8, gt_8)

    cx_p, cy_p, a_p, b_p, theta_p, phi_0_p = unpack_params_from_8(
        pred_8, img_size=img_size)
    pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_0_p)

    pred_marks_n = pred_marks / img_size
    gt_marks_n = gt_marks / img_size
    l_mark = F.smooth_l1_loss(pred_marks_n, gt_marks_n)

    total = w_param * l_param + w_mark * l_mark
    return total, l_param, l_mark


# ── Dataset ──────────────────────────────────────────────────────────────

class AutoCalibDataset(Dataset):
    """Synthetic autocal images with regression targets.

    __getitem__ returns:
        img:          (3, IMG_SIZE, IMG_SIZE) float32 in [0, 1]
        params_raw:   (6,) float32 — [cx, cy, a, b, theta, phi_0]
        marks_raw:    (10, 2) float32 — pixel coords of 10 marks
        filename:     str
    """

    def __init__(self, data_dir, img_size=IMG_SIZE):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                marks_arr = np.empty((N_MARKS, 2), dtype=np.float32)
                for i in range(N_MARKS):
                    marks_arr[i, 0] = float(row[f"mark_{i}_x"])
                    marks_arr[i, 1] = float(row[f"mark_{i}_y"])
                self.samples.append({
                    "filename": row["filename"],
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "a": float(row["a"]),
                    "b": float(row["b"]),
                    "theta": float(row["theta"]),
                    "phi_0": float(row["phi_0"]),
                    "marks": marks_arr,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, s["filename"])).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW

        params_raw = torch.tensor([s["cx"], s["cy"], s["a"], s["b"],
                                   s["theta"], s["phi_0"]], dtype=torch.float32)
        marks_raw = torch.from_numpy(s["marks"])
        return torch.from_numpy(arr), params_raw, marks_raw, s["filename"]


# ── Backbone architectures (regression heads) ────────────────────────────

# Building blocks copied from train_e_inv_res.py and train_f_se_dwsep.py.
# If the autocal versions need to diverge from the train_*.py originals
# (different SE reduction, dropout, etc.), that is intentional — these serve
# a different task. Otherwise keep them in sync.

class _InvRes(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                             groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
                   nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class _ChannelSE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.pool(x))


class _DWSepSE(nn.Module):
    """Depthwise separable conv + Squeeze-and-Excitation channel attention."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.se = _ChannelSE(out_ch, reduction=4)

    def forward(self, x):
        return self.se(self.block(x))


class _DWSepResSE(nn.Module):
    """Residual DWSep + SE for the bottleneck (output = input + SE(DWSep(input)))."""

    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.se = _ChannelSE(ch, reduction=4)

    def forward(self, x):
        return x + self.se(self.block(x))


class _RegressionHead(nn.Module):
    """Global avg pool → 1x1 conv ReLU → 1x1 conv to 8 outputs.

    Conv-only design exports cleanly to ONNX.
    """

    def __init__(self, in_ch, hidden=64, out=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out, 1, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.flatten(1)  # (B, 8)


class AutocalModelE(nn.Module):
    """Autocal variant of Model E.

    Stem and encoder/bottleneck taken from InvResNet (train_e_inv_res.py).
    The decoder and sigmoid heatmap head are replaced by _RegressionHead,
    producing 8 packed parameters per image instead of a 128x128 heatmap.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6(inplace=True))         # 128x128x24
        self.enc2 = _InvRes(24, 48, stride=2, expand_ratio=2)    # 64x64x48
        self.enc3 = _InvRes(48, 64, stride=2, expand_ratio=2)    # 32x32x64
        self.bn1 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.bn2 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.head = _RegressionHead(64)

    def forward(self, x):
        x = self.stem(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bn2(self.bn1(x))
        return self.head(x)


class AutocalModelF(nn.Module):
    """Autocal variant of Model F.

    Encoder/bottleneck taken from SEDWSepNet (train_f_se_dwsep.py).
    The decoder and sigmoid heatmap head are replaced by _RegressionHead,
    producing 8 packed parameters per image.
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))           # 128x128x32
        self.enc2 = _DWSepSE(32, 64, stride=2)                    # 64x64x64
        self.enc3 = _DWSepSE(64, 96, stride=2)                    # 32x32x96
        self.bn1 = _DWSepResSE(96)
        self.bn2 = _DWSepResSE(96)
        self.head = _RegressionHead(96)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bn2(self.bn1(x))
        return self.head(x)


# ── Training utilities ───────────────────────────────────────────────────

def train_one_epoch_autocal(model, loader, optimizer, device,
                            img_size=IMG_SIZE, w_param=0.1, w_mark=1.0):
    """One training epoch. Returns (mean_loss, mean_mark_error_px)."""
    model.train()
    total_loss = 0.0
    total_mark_err = 0.0
    n = 0
    for imgs, params_raw, marks_raw, _fnames in loader:
        imgs = imgs.to(device)
        params_raw = params_raw.to(device)
        marks_raw = marks_raw.to(device)

        pred_8 = model(imgs)

        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0 = params_raw.unbind(-1)
        loss, _lp, _lm = autocal_loss(
            pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
            marks_raw, img_size=img_size, w_param=w_param, w_mark=w_mark)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cx_p, cy_p, a_p, b_p, theta_p, phi_p = unpack_params_from_8(
                pred_8, img_size=img_size)
            pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_p)
            mark_err = torch.norm(pred_marks - marks_raw, dim=-1).mean()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mark_err += mark_err.item() * bs
        n += bs

    return total_loss / n, total_mark_err / n


@torch.no_grad()
def evaluate_autocal(model, loader, device, img_size=IMG_SIZE,
                     w_param=0.1, w_mark=1.0):
    """Evaluate on a validation loader. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_mark_err = []
    all_center_err = []
    n = 0
    for imgs, params_raw, marks_raw, _fnames in loader:
        imgs = imgs.to(device)
        params_raw = params_raw.to(device)
        marks_raw = marks_raw.to(device)

        pred_8 = model(imgs)

        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0 = params_raw.unbind(-1)
        loss, _lp, _lm = autocal_loss(
            pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
            marks_raw, img_size=img_size, w_param=w_param, w_mark=w_mark)

        cx_p, cy_p, a_p, b_p, theta_p, phi_p = unpack_params_from_8(
            pred_8, img_size=img_size)
        pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_p)
        per_sample_mark_err = torch.norm(pred_marks - marks_raw, dim=-1).mean(dim=-1)
        per_sample_center_err = torch.sqrt(
            (cx_p - gt_cx) ** 2 + (cy_p - gt_cy) ** 2)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        all_mark_err.extend(per_sample_mark_err.cpu().tolist())
        all_center_err.extend(per_sample_center_err.cpu().tolist())
        n += bs

    mark_arr = np.array(all_mark_err)
    center_arr = np.array(all_center_err)
    return {
        "loss": total_loss / n,
        "mark_err_mean": float(mark_arr.mean()),
        "mark_err_p95": float(np.percentile(mark_arr, 95)),
        "mark_err_p99": float(np.percentile(mark_arr, 99)),
        "center_err_mean": float(center_arr.mean()),
    }


def run_autocal_training(model, args):
    """Full training pipeline: train → save best → export ONNX → benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    ds = AutoCalibDataset(args.data_dir)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pth_path = args.save_onnx.replace(".onnx", ".pth")
    best_mark_err = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mark_err = train_one_epoch_autocal(
            model, train_ldr, optimizer, device,
            w_param=args.w_param, w_mark=args.w_mark)
        val = evaluate_autocal(
            model, val_ldr, device,
            w_param=args.w_param, w_mark=args.w_mark)
        sched.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_mark={train_mark_err:.2f}px  "
              f"val_loss={val['loss']:.4f} mark_mean={val['mark_err_mean']:.2f}px "
              f"p95={val['mark_err_p95']:.2f} p99={val['mark_err_p99']:.2f} "
              f"center_mean={val['center_err_mean']:.2f}px  "
              f"lr={lr:.2e}  ({dt:.1f}s)")

        if val["mark_err_mean"] < best_mark_err:
            best_mark_err = val["mark_err_mean"]
            torch.save(model.state_dict(), pth_path)
            print(f"  -> saved best mark_err_mean={best_mark_err:.2f}px")

    # Export best.
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    _export_onnx_autocal(model, args.save_onnx)


def _export_onnx_autocal(model, path, img_size=IMG_SIZE):
    import onnx
    model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"], output_names=["params_8"])
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX exported: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
