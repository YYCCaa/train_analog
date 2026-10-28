"""Fast pointer-tip heatmap trainer for the fixed v9 dataset (tiny model, < 500 KB).

Why a new trainer: ``train_utils_v2_gray`` decodes PNGs in DataLoader workers,
loops ``.item()`` per sample and scores against the target-heatmap argmax.  This
one decodes every PNG once into a uint8 cache, keeps the whole dataset in RAM,
augments on the GPU and scores the decoded float prediction against the float
label.  The old files are left untouched.

Data flow (shapes per batch, B = --batch-size, S = 256, H = W = 128):

    <data-dir>/dial_*.png + labels.csv        (fixed dataset, never written)
          |  first run, or labels.csv newer than the .npy: mp.Pool PNG decode   build_cache()
          v
    <data-dir>_cache_images.npy  uint8 (N, S, S)      memmap-able .npy
    <data-dir>_cache_meta.npz    filenames (N,) str, labels (N, 2) float32
          |  load_cache(): one RAM copy, torch uint8 (N, S, S) on the CPU
          v
    split_indices(): the exact random_split(seed 42) permutation
          |  per epoch: random permutation of the train indices
          v
    BatchFeeder: index_select on the CPU -> pinned staging (B, S, S) uint8
          |  .to(cuda, non_blocking)          no DataLoader, no workers
          v
    GPU: dihedral_images / dihedral_points  (8 flip/rot90 transforms, exact labels)
         [optional photometric_aug]  -> float /255 -> (B, 1, S, S) float32
          |  bf16 autocast, channels_last
          v
    TipNet  (B, 1, 256, 256) -> logits (B, 1, 128, 128)
      encoder  stem s2 -> InvRes stages s4, s8, s16, s32  (8x8 deep map)
      context  global-avg-pool branch broadcast-added + CoordConv x/y channels
      decoder  FPN top-down: project, x2 bilinear, + lateral skip, InvRes refine
               1/32 -> 1/16 -> 1/8 -> 1/4 -> 1/2, 1x1 head -> 1 logit map
          |
          +--> loss: spatial softmax over 128*128 logits
          |          CE vs normalised Gaussian at u=(x-0.5)/2, v=(y-0.5)/2 (sigma 1 hm px)
          |          + coord_weight * L1(soft-argmax expectation (DSNT), (u, v))
          |
          +--> decode: argmax -> 5x5 softmax-weighted centroid -> x = 2u+0.5, y = 2v+0.5
                       error = |pred - label| in 256-px image coordinates

    Per epoch: full-val evaluation with the EMA weights -> metrics.csv,
    best_p99.pth (state_dict), last.pth, worst_val.csv, train.log.
    End: reload best_p99.pth, re-evaluate (fp32), export model.onnx (opset 17),
    check ONNX Runtime against torch.

Usage:
    python train_v9_tip.py --run-name v9_tipnet_a
    python train_v9_tip.py --epochs 1 --limit 20000 --run-name bench
"""

import argparse
import copy
import csv
import datetime
import math
import multiprocessing as mp
import os
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from train_e_inv_res import InvRes

# ── constants ────────────────────────────────────────────────────────────

IMG_SIZE = 256        # int: input image side (px)
HM_SIZE = 128         # int: heatmap side (px); heatmap px j covers image px 2j, 2j+1
PTH_LIMIT_BYTES = 500 * 1024   # int: hard limit for the saved state_dict .pth (bytes)
PARAM_BUDGET = 115_000         # int: trainable-parameter budget at the default width
OVER_PX = 7.0                  # float: error threshold counted as n_over_7px (image px)
DEFAULT_DATA_DIR = "C:/Users/Administrator/PycharmProjects/train_dial_data/sv9_150k"  # str
DEFAULT_RUNS_DIR = "C:/Users/Administrator/PycharmProjects/train_dial_data/runs"      # str
METRIC_FIELDS = ["epoch", "lr", "train_loss", "val_loss", "mean", "median", "p95",
                 "p99", "max", "n_over_7px", "epoch_seconds", "imgs_per_sec"]  # list[str]


# ── logging ──────────────────────────────────────────────────────────────

class Tee:
    """File-like object that writes every message to several streams.

    streams: list of text streams (e.g. [sys.__stdout__, open log file]).
    """

    def __init__(self, *streams):
        self.streams = streams  # tuple[TextIO]: destinations for every write

    def write(self, msg):
        for s in self.streams:
            s.write(msg)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def fmt_secs(s):
    """Seconds (float) -> 'HhMMmSSs' / 'MmSSs' / 'S.Ss' string."""
    s = float(s)
    if s < 60:
        return f"{s:.1f}s"
    m, sec = divmod(int(round(s)), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else f"{m}m{sec:02d}s"


# ── data cache ───────────────────────────────────────────────────────────

def read_labels_csv(data_dir):
    """labels.csv -> (filenames list[str] (N,), labels float32 ndarray (N, 2) = tip_x, tip_y)."""
    names = []   # list[str]: PNG file names in labels.csv row order
    xy = []      # list[tuple(float, float)]: tip (x, y) in image px
    with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
        for row in csv.DictReader(f):
            names.append(row["filename"])
            xy.append((float(row["tip_x"]), float(row["tip_y"])))
    return names, np.asarray(xy, dtype=np.float32).reshape(-1, 2)


def cache_paths(cache_prefix):
    """Cache prefix (str) -> (images .npy path, meta .npz path)."""
    return cache_prefix + "_images.npy", cache_prefix + "_meta.npz"


def _load_png(data_dir, name):
    """One PNG -> uint8 ndarray (IMG_SIZE, IMG_SIZE); raises on a wrong size."""
    with Image.open(os.path.join(data_dir, name)) as im:
        if im.mode != "L":
            im = im.convert("L")
        arr = np.asarray(im, dtype=np.uint8)  # (H, W) uint8
    if arr.shape != (IMG_SIZE, IMG_SIZE):
        raise ValueError(f"{name}: shape {arr.shape}, expected {(IMG_SIZE, IMG_SIZE)}")
    return arr


def _decode_rows(job):
    """Pool worker: decode PNGs into rows [start, start+len(names)) of the images .npy.

    job: tuple(images_path str, start int, names list[str], data_dir str).
    Returns the number of rows written (int).
    """
    images_path, start, names, data_dir = job
    mm = np.load(images_path, mmap_mode="r+")  # (N, S, S) uint8 memmap
    for i, name in enumerate(names):
        mm[start + i] = _load_png(data_dir, name)
    mm.flush()
    del mm
    return len(names)


def build_cache(data_dir, cache_prefix, workers=None, chunk=512):
    """Decode every PNG of data_dir into the cache files at cache_prefix.

    Writes <prefix>_images.npy (uint8 (N, S, S)) via a .partial file renamed at the
    end, then <prefix>_meta.npz (filenames (N,) str, labels (N, 2) float32).
    """
    names, labels = read_labels_csv(data_dir)
    n = len(names)                       # int: number of images
    img_path, meta_path = cache_paths(cache_prefix)
    os.makedirs(os.path.dirname(os.path.abspath(img_path)), exist_ok=True)
    tmp_path = img_path[:-4] + ".partial.npy"   # str: written first, renamed when complete
    print(f"[cache] building {img_path}  ({n} images, {n * IMG_SIZE * IMG_SIZE / 2**30:.2f} GiB)")
    t0 = time.time()
    mm = np.lib.format.open_memmap(tmp_path, mode="w+", dtype=np.uint8,
                                   shape=(n, IMG_SIZE, IMG_SIZE))  # (N, S, S) uint8
    del mm  # header + zero file written; workers re-open with mode r+
    jobs = [(tmp_path, s, names[s:s + chunk], data_dir) for s in range(0, n, chunk)]
    n_workers = max(1, min(workers or os.cpu_count() or 1, 16, len(jobs)))
    done = 0          # int: rows decoded so far
    next_report = 0   # int: row count at which the next progress line prints
    if n_workers == 1 or n < 2000:
        for job in jobs:
            done += _decode_rows(job)
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_workers) as pool:
            for k in pool.imap_unordered(_decode_rows, jobs):
                done += k
                if done >= next_report:
                    el = time.time() - t0
                    print(f"[cache] {done}/{n}  {el:.0f}s  "
                          f"{done / max(el, 1e-6):.0f} img/s", flush=True)
                    next_report += max(n // 10, 1)
    os.replace(tmp_path, img_path)
    np.savez(meta_path, filenames=np.asarray(names), labels=labels,
             data_dir=np.asarray(os.path.abspath(data_dir)))
    print(f"[cache] done in {fmt_secs(time.time() - t0)}")


def cache_is_stale(data_dir, cache_prefix):
    """True if the images .npy exists but labels.csv is newer than it.

    The generator rewrites labels.csv on every run and the cache is built afterwards, so a newer labels.csv
    means the dataset was regenerated (possibly with the same seed: identical labels, new pixels).
    """
    img_path, _ = cache_paths(cache_prefix)    # str: images .npy
    if not os.path.exists(img_path):
        return False
    # int ns: mtimes of labels.csv and of the cache images
    return os.stat(os.path.join(data_dir, "labels.csv")).st_mtime_ns > os.stat(img_path).st_mtime_ns


def cache_is_valid(data_dir, cache_prefix):
    """True if the cache files exist, match labels.csv (N, first/last filename, labels) and are not stale."""
    img_path, meta_path = cache_paths(cache_prefix)
    if cache_is_stale(data_dir, cache_prefix):
        return False
    if not (os.path.exists(img_path) and os.path.exists(meta_path)):
        return False
    names, labels = read_labels_csv(data_dir)
    try:
        meta = np.load(meta_path)
        c_names = meta["filenames"]      # (N,) str
        c_labels = meta["labels"]        # (N, 2) float32
        shape = np.load(img_path, mmap_mode="r").shape
    except (OSError, ValueError, KeyError):
        return False
    n = len(names)
    return (len(c_names) == n and shape == (n, IMG_SIZE, IMG_SIZE) and n > 0
            and str(c_names[0]) == names[0] and str(c_names[-1]) == names[-1]
            and np.array_equal(c_labels, labels))


def ensure_cache(data_dir, cache_prefix, workers=None):
    """Build the cache unless it is valid (missing, mismatched or older than labels.csv); then re-check."""
    if cache_is_stale(data_dir, cache_prefix):
        img_path, _ = cache_paths(cache_prefix)    # str: images .npy
        print(f"[cache] labels.csv is newer than {img_path}: rebuilding", flush=True)
    if not cache_is_valid(data_dir, cache_prefix):
        build_cache(data_dir, cache_prefix, workers=workers)
        if not cache_is_valid(data_dir, cache_prefix):
            raise RuntimeError("cache build finished but does not match labels.csv")


def load_cache(data_dir, cache_prefix, limit=None, workers=None):
    """Build the cache if missing/stale, then load it into RAM.

    Returns (images uint8 torch (n, S, S) CPU, labels float32 torch (n, 2),
    filenames list[str] (n,)), n = min(N, limit).  One RAM copy of the images.
    """
    ensure_cache(data_dir, cache_prefix, workers=workers)
    img_path, meta_path = cache_paths(cache_prefix)
    meta = np.load(meta_path)
    mm = np.load(img_path, mmap_mode="r")             # (N, S, S) uint8 memmap
    n = len(mm) if not limit else min(int(limit), len(mm))
    need = n * IMG_SIZE * IMG_SIZE                    # int: bytes for the RAM copy
    try:
        import psutil
        avail = psutil.virtual_memory().available     # int: free RAM bytes
        if need > 0.9 * avail:
            print(f"[cache] WARNING: images need {need / 2**30:.1f} GiB, "
                  f"only {avail / 2**30:.1f} GiB available")
    except ImportError:
        pass
    t0 = time.time()
    images = torch.empty((n, IMG_SIZE, IMG_SIZE), dtype=torch.uint8)  # (n, S, S) uint8
    dst = images.numpy()                              # numpy view on the same memory
    step = 8192                                       # int: rows copied per chunk
    for s in range(0, n, step):
        dst[s:s + step] = mm[s:min(s + step, n)]
    del mm
    labels = torch.from_numpy(np.array(meta["labels"][:n], dtype=np.float32))  # (n, 2)
    names = [str(s) for s in meta["filenames"][:n]]   # list[str] (n,)
    print(f"[cache] loaded {n} images ({need / 2**30:.2f} GiB) in {fmt_secs(time.time() - t0)}")
    return images, labels, names


# ── split ────────────────────────────────────────────────────────────────

def split_sizes(n, val_split):
    """(n_train, n_val) for n samples; n_val = int(n * val_split), at least 1."""
    n_val = max(1, int(n * val_split))
    return n - n_val, n_val


def split_indices(n, val_split, seed=42):
    """Exact indices of random_split(range(n), [n_train, n_val], Generator().manual_seed(seed)).

    Returns (train_idx, val_idx): int64 torch tensors (n_train,), (n_val,).
    """
    n_train, n_val = split_sizes(n, val_split)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))  # (n,) int64
    return perm[:n_train].clone(), perm[n_train:n_train + n_val].clone()


# ── GPU augmentation ─────────────────────────────────────────────────────

def dihedral_images(x, k):
    """Apply one of the 8 flip/rot90 transforms per sample.

    x: (B, ..., S, S) tensor (any dtype); k: (B,) int64 in [0, 8).
    k >= 4 -> horizontal flip (x' = S-1-x) first; then rot90 (k % 4) times
    (torch.rot90 over the last two dims, i.e. counter-clockwise on screen).
    No host sync: every candidate is computed and selected with torch.where.
    """
    view = (-1,) + (1,) * (x.dim() - 1)            # tuple: broadcast k over image dims
    x = torch.where((k >= 4).view(view), x.flip(-1), x)
    r = (k % 4).view(view)                         # (B, 1, ...) int64: rot90 count
    out = x
    for n_rot in (1, 2, 3):
        out = torch.where(r == n_rot, torch.rot90(x, n_rot, dims=(-2, -1)), out)
    return out


def dihedral_points(xy, k, size=IMG_SIZE):
    """Transform points exactly like dihedral_images transforms pixels.

    xy: (B, 2) float (x, y) with pixel i centred at coordinate i; k: (B,) int64.
    Flip: (x, y) -> (m-x, y); one rot90: (x, y) -> (y, m-x), with m = size-1.
    Returns (B, 2) float.
    """
    m = float(size - 1)
    x, y = xy[:, 0], xy[:, 1]
    x = torch.where(k >= 4, m - x, x)
    r = k % 4                                       # (B,) int64
    nx = torch.where(r == 1, y, torch.where(r == 2, m - x, torch.where(r == 3, m - y, x)))
    ny = torch.where(r == 1, m - x, torch.where(r == 2, m - y, torch.where(r == 3, x, y)))
    return torch.stack([nx, ny], dim=1)


def photometric_aug(x, gen=None):
    """Mild per-sample photometric jitter on (B, 1, S, S) float in [0, 1].

    contrast 0.8-1.2 about the image mean, brightness +-0.08, gamma e^+-0.2,
    Gaussian noise sigma 0-0.02; result clamped to [0, 1].
    """
    b = x.shape[0]
    shp = (b, 1, 1, 1)
    u = lambda lo, hi: torch.empty(shp, device=x.device).uniform_(lo, hi, generator=gen)  # noqa: E731
    mean = x.mean(dim=(1, 2, 3), keepdim=True)       # (B, 1, 1, 1)
    x = (x - mean) * u(0.8, 1.2) + mean + u(-0.08, 0.08)
    x = x.clamp(0, 1).pow(torch.exp(u(-0.2, 0.2)))
    x = x + torch.randn(x.shape, device=x.device, generator=gen) * u(0.0, 0.02)
    return x.clamp_(0, 1)


# ── batching (no DataLoader) ─────────────────────────────────────────────

class BatchFeeder:
    """Gathers uint8 image rows on the CPU and ships them to the device.

    images: uint8 CPU tensor (N, S, S); labels: float32 tensor (N, 2) (moved to device).
    On CUDA two pinned staging buffers (B, S, S) uint8 alternate, so the CPU gather of
    batch k+1 overlaps the GPU work of batch k; a CUDA event guards each buffer.
    """

    def __init__(self, images, labels, device, batch_size):
        self.images = images                              # (N, S, S) uint8 CPU
        self.labels = labels.to(device)                   # (N, 2) float32 on device
        self.device = device                              # torch.device
        self.batch_size = batch_size                      # int: max rows per batch
        self.cuda = device.type == "cuda"                 # bool
        self.n_stage = 2 if self.cuda else 1              # int: staging buffers
        self.stage = [torch.empty((batch_size, IMG_SIZE, IMG_SIZE), dtype=torch.uint8,
                                  pin_memory=self.cuda) for _ in range(self.n_stage)]
        self.events = [None] * self.n_stage               # list[torch.cuda.Event | None]
        self.turn = 0                                     # int: next staging buffer

    def batches(self, indices, drop_last=False):
        """Yield (x_u8 (B, S, S) uint8 on device, xy (B, 2) float32 on device, idx (B,) int64 CPU)."""
        n = len(indices)
        stop = n - n % self.batch_size if drop_last else n
        indices_dev = indices.to(self.device)             # (M,) int64: one copy per pass
        for s in range(0, stop, self.batch_size):
            idx = indices[s:s + self.batch_size]          # (B,) int64 CPU
            b = len(idx)
            buf = self.stage[self.turn]
            if self.events[self.turn] is not None:
                self.events[self.turn].synchronize()      # previous copy from buf finished
            torch.index_select(self.images, 0, idx, out=buf[:b])
            if self.cuda:
                x = buf[:b].to(self.device, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record()
                self.events[self.turn] = ev
            else:
                x = buf[:b].clone()
            self.turn = (self.turn + 1) % self.n_stage
            yield x, self.labels[indices_dev[s:s + b]], idx


def to_input(x_u8):
    """(B, S, S) uint8 on device -> (B, 1, S, S) float32 in [0, 1], channels_last."""
    return x_u8.unsqueeze(1).float().div_(255.0).contiguous(memory_format=torch.channels_last)


# ── models ───────────────────────────────────────────────────────────────

def _ch(c, width):
    """Scale a base channel count c (int) by width (float); multiple of 4, at least 8."""
    return max(8, int(round(c * width / 4.0)) * 4)


def conv_bn(cin, cout, k=1, stride=1, act=False):
    """Conv(k x k, no bias) + BN [+ ReLU6] as nn.Sequential: (B, cin, H, W) -> (B, cout, H/s, W/s)."""
    layers = [nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=False),
              nn.BatchNorm2d(cout)]
    if act:
        layers.append(nn.ReLU6(inplace=True))
    return nn.Sequential(*layers)


class TopDown(nn.Module):
    """One FPN decoder step: project + x2 bilinear upsample the coarse map, add the
    lateral skip (1x1 projected if its width differs), refine with an InvRes block.

    coarse (B, c_coarse, h, w) + skip (B, c_skip, 2h, 2w) -> (B, c_out, 2h, 2w).
    """

    def __init__(self, c_coarse, c_skip, c_out, expand=2):
        super().__init__()
        self.proj = conv_bn(c_coarse, c_out)                     # 1x1 before upsampling (cheaper)
        self.lateral = (nn.Identity() if c_skip == c_out
                        else conv_bn(c_skip, c_out))             # 1x1 skip projection
        self.refine = InvRes(c_out, c_out, stride=1, expand_ratio=expand)

    def forward(self, coarse, skip):
        up = F.interpolate(self.proj(coarse), scale_factor=2, mode="bilinear",
                           align_corners=False)                  # (B, c_out, 2h, 2w)
        return self.refine(up + self.lateral(skip))


class TipNet(nn.Module):
    """MobileNetV2-style encoder-decoder emitting one tip logit map.

    Input : (B, 1, 256, 256) float32 in [0, 1]   (256 x 256 only: fixed CoordConv grid)
    Output: (B, 1, 128, 128) float32 logits (spatial softmax gives the tip distribution)

    Base widths (width = 1.0):
      enc  stem 16 @1/2 (+InvRes 16) | 24 @1/4 | 32 @1/8 | 48 @1/16 | 64 @1/32 (8x8)
      ctx  global-avg-pool -> 1x1 squeeze/expand (64->16->64), broadcast-added to the 8x8 map;
           2 CoordConv channels (x, y in [-1, 1]) concatenated, 1x1 fuse -> 48
      dec  48 @1/32 -> 40 @1/16 -> 32 @1/8 -> 24 @1/4 -> 16 @1/2 -> 1x1 head -> 1
    The receptive field of the 8x8 map spans the whole 256-px image, and the pooled
    branch gives every position the global dial context (hub, decoys).
    """

    def __init__(self, width=1.0, coordconv=True):
        super().__init__()
        c1, c2, c3, c4, c5 = (_ch(c, width) for c in (16, 24, 32, 48, 64))  # encoder widths
        d5, d4, d3, d2, d1 = (_ch(c, width) for c in (48, 40, 32, 24, 16))  # decoder widths
        self.coordconv = coordconv                                         # bool
        # encoder: (B,1,256,256) -> e1 (B,c1,128,128) e2 (B,c2,64,64) e3 (B,c3,32,32)
        #          e4 (B,c4,16,16) e5 (B,c5,8,8)
        self.stem = conv_bn(1, c1, k=3, stride=2, act=True)
        self.enc1 = InvRes(c1, c1, stride=1, expand_ratio=2)
        self.enc2 = nn.Sequential(InvRes(c1, c2, 2, 3), InvRes(c2, c2, 1, 3))
        self.enc3 = nn.Sequential(InvRes(c2, c3, 2, 3), InvRes(c3, c3, 1, 3))
        self.enc4 = nn.Sequential(InvRes(c3, c4, 2, 3), InvRes(c4, c4, 1, 2))
        self.enc5 = nn.Sequential(InvRes(c4, c5, 2, 2), InvRes(c5, c5, 1, 2))
        # global context: (B,c5,8,8) -> mean -> (B,c5,1,1) -> 1x1 squeeze (c5/4) -> ReLU6
        #                 -> 1x1 expand (c5) -> broadcast add onto every 8x8 position
        self.ctx = nn.Sequential(nn.Conv2d(c5, c5 // 4, 1), nn.ReLU6(inplace=True),
                                 nn.Conv2d(c5 // 4, c5, 1))
        self.fuse = conv_bn(c5 + (2 if coordconv else 0), d5, act=True)    # -> (B,d5,8,8)
        # decoder (top-down)
        self.up4 = TopDown(d5, c4, d4)   # -> (B,d4,16,16)
        self.up3 = TopDown(d4, c3, d3)   # -> (B,d3,32,32)
        self.up2 = TopDown(d3, c2, d2)   # -> (B,d2,64,64)
        self.up1 = TopDown(d2, c1, d1)   # -> (B,d1,128,128)
        self.head = nn.Conv2d(d1, 1, 1)  # -> (B,1,128,128) logits
        # CoordConv grid for the 8x8 deep map: (1, 2, 8, 8) float32, not saved in the .pth
        g = torch.linspace(-1.0, 1.0, IMG_SIZE // 32)
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy])[None], persistent=False)

    def forward(self, x):
        e1 = self.enc1(self.stem(x))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e5 = e5 + self.ctx(e5.mean(dim=(2, 3), keepdim=True))
        if self.coordconv:   # the (1, 2, 8, 8) grid assumes a 256x256 input
            coords = self.coords.to(e5.dtype).expand(e5.shape[0], -1, -1, -1)
            e5 = torch.cat([e5, coords], dim=1)
        p = self.fuse(e5)
        p = self.up4(p, e4)
        p = self.up3(p, e3)
        p = self.up2(p, e2)
        p = self.up1(p, e1)
        return self.head(p)


class InvResNetGrayLogits(nn.Module):
    """Adapter for arch 'e': the existing InvResNetGray (half-res) without its sigmoid.

    Input (B, 1, 256, 256) float32 -> logits (B, 1, 128, 128) float32.
    The wrapped module's parameters are reused unchanged (state_dict keys 'net.*').
    """

    def __init__(self):
        super().__init__()
        from train_e_inv_res_gray import InvResNetGray
        self.net = InvResNetGray(half_res=True)

    def forward(self, x):
        n = self.net
        e1 = n.stem(x)
        e2 = n.enc2(e1)
        e3 = n.enc3(e2)
        b = n.bn2(n.bn1(e3))
        d3 = n.dec3(n.up3_proj(F.interpolate(b, size=e2.shape[2:], mode="bilinear",
                                             align_corners=False)) + e2)
        d2 = n.dec2(n.up2_proj(F.interpolate(d3, size=e1.shape[2:], mode="bilinear",
                                             align_corners=False)) + e1)
        return n.head(d2)


ARCHS = {
    "tipnet": lambda width: TipNet(width=width),
    "e": lambda width: InvResNetGrayLogits(),
}  # dict[str, callable(width float) -> nn.Module]: --arch registry


def build_model(arch="tipnet", width=1.0):
    """--arch registry lookup -> nn.Module mapping (B,1,256,256) to logits (B,1,128,128)."""
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[arch](width)


def count_params(model):
    """Number of trainable parameters (int)."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def clean_state_dict(model):
    """state_dict with contiguous fp32/int CPU copies (what every .pth here stores)."""
    return {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}


def state_dict_bytes(model, name="best_p99.pth"):
    """Size in bytes (int) of torch.save(clean_state_dict(model)) saved as `name`.

    The file name matters: torch's zip format stores it in every record name, so the
    check saves under the same name as the real checkpoint (in a temp folder).
    """
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, name)
        torch.save(clean_state_dict(model), path)
        return os.path.getsize(path)


# ── loss ─────────────────────────────────────────────────────────────────

def image_to_heatmap(xy):
    """Image px (B, 2) -> heatmap px (B, 2): u = (x - 0.5) / 2, v = (y - 0.5) / 2."""
    return (xy - 0.5) * 0.5


def heatmap_to_image(uv):
    """Heatmap px (B, 2) -> image px (B, 2): x = 2u + 0.5, y = 2v + 0.5."""
    return uv * 2.0 + 0.5


def gaussian_target(uv, h, w, sigma, normalise=True):
    """Gaussian maps centred at uv (B, 2) heatmap px -> (B, h, w) float32.

    normalise=True: each map sums to 1 (softmax-CE target); False: peak value 1 (MSE target).
    """
    xs = torch.arange(w, device=uv.device, dtype=torch.float32)   # (w,)
    ys = torch.arange(h, device=uv.device, dtype=torch.float32)   # (h,)
    gx = torch.exp(-(xs[None] - uv[:, :1]) ** 2 / (2 * sigma ** 2))  # (B, w)
    gy = torch.exp(-(ys[None] - uv[:, 1:]) ** 2 / (2 * sigma ** 2))  # (B, h)
    g = gy[:, :, None] * gx[:, None, :]                            # (B, h, w)
    if normalise:
        g = g / g.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    return g


def soft_argmax(logp):
    """DSNT expectation of a log-probability map (B, h, w) -> (B, 2) heatmap px (u, v)."""
    p = logp.exp()
    h, w = p.shape[-2:]
    xs = torch.arange(w, device=p.device, dtype=p.dtype)
    ys = torch.arange(h, device=p.device, dtype=p.dtype)
    return torch.stack([(p.sum(1) * xs).sum(1), (p.sum(2) * ys).sum(1)], dim=1)


def tip_loss(logits, xy, kind="ce", sigma=1.0, coord_weight=0.1):
    """Training loss for logits (B, 1, h, w) and float labels xy (B, 2) image px -> scalar.

    kind 'ce' : spatial-softmax CE vs a normalised Gaussian (sigma heatmap px)
                + coord_weight * L1(DSNT expectation, label) in heatmap px.
    kind 'mse': old weighted MSE of sigmoid(logits) vs a peak-1 Gaussian (w_pos 10).
    """
    logits = logits.float()[:, 0]                       # (B, h, w) float32
    b, h, w = logits.shape
    uv = image_to_heatmap(xy)                           # (B, 2) heatmap px
    if kind == "mse":
        target = gaussian_target(uv, h, w, sigma, normalise=False)
        weight = torch.where(target > 0.1, 10.0, 1.0)
        return (weight * (torch.sigmoid(logits) - target) ** 2).mean()
    logp = F.log_softmax(logits.reshape(b, -1), dim=1)  # (B, h*w)
    target = gaussian_target(uv, h, w, sigma).reshape(b, -1)
    ce = -(target * logp).sum(1).mean()
    if coord_weight <= 0:
        return ce
    l1 = (soft_argmax(logp.reshape(b, h, w)) - uv).abs().sum(1).mean()
    return ce + coord_weight * l1


# ── decode ───────────────────────────────────────────────────────────────

def decode_logits(logits, win=5):
    """Logits (B, 1, h, w) -> tip (B, 2) float32 in image px.

    argmax -> softmax-weighted centroid of the win x win logits around it (cells
    outside the map are ignored) -> x = 2u + 0.5, y = 2v + 0.5.  Fully vectorised.
    """
    lg = logits.float()[:, 0]                           # (B, h, w)
    b, h, w = lg.shape
    r = win // 2
    flat = lg.reshape(b, -1).argmax(dim=1)              # (B,) int64
    iy, ix = flat // w, flat % w                        # (B,) int64 each
    pad = F.pad(lg, (r, r, r, r), value=float("-inf"))  # (B, h+2r, w+2r)
    d = torch.arange(-r, r + 1, device=lg.device)       # (win,) int64 offsets
    yy = (iy[:, None, None] + d[None, :, None] + r).expand(b, win, win)  # padded rows
    xx = (ix[:, None, None] + d[None, None, :] + r).expand(b, win, win)  # padded cols
    bb = torch.arange(b, device=lg.device)[:, None, None].expand(b, win, win)
    patch = pad[bb, yy, xx].reshape(b, -1)              # (B, win*win)
    wts = torch.softmax(patch, dim=1).reshape(b, win, win)
    v = iy.float() + (wts.sum(2) * d.float()[None]).sum(1)   # (B,) heatmap rows
    u = ix.float() + (wts.sum(1) * d.float()[None]).sum(1)   # (B,) heatmap cols
    return heatmap_to_image(torch.stack([u, v], dim=1))


# ── EMA ──────────────────────────────────────────────────────────────────

class ModelEma:
    """Exponential moving average of a model's parameters and float buffers.

    module: deep copy of the model (eval mode, no grad) holding the averaged weights.
    decay: float target decay; early steps use min(decay, (1+t)/(10+t)).
    """

    def __init__(self, model, decay):
        self.module = copy.deepcopy(model).eval()       # nn.Module: EMA weights
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay                              # float
        self.updates = 0                                # int: update() calls so far
        ema_sd, src_sd = self.module.state_dict(), model.state_dict()
        self.pairs_f = [(ema_sd[k], src_sd[k]) for k in ema_sd if ema_sd[k].is_floating_point()]
        self.pairs_i = [(ema_sd[k], src_sd[k]) for k in ema_sd if not ema_sd[k].is_floating_point()]

    @torch.no_grad()
    def update(self):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        torch._foreach_lerp_([e for e, _ in self.pairs_f], [s for _, s in self.pairs_f], 1.0 - d)
        for e, s in self.pairs_i:
            e.copy_(s)


# ── schedule ─────────────────────────────────────────────────────────────

def lr_at(step, total_steps, warmup_steps, base_lr):
    """Learning rate (float) for 1-based optimizer step: linear warmup, then cosine to 0."""
    if warmup_steps > 0 and step <= warmup_steps:
        return base_lr * step / warmup_steps
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t)))


def param_groups(model, weight_decay):
    """AdamW groups: decay on conv/linear weights (ndim > 1), none on BN params and biases."""
    decay, no_decay = [], []   # list[nn.Parameter] each
    for p in model.parameters():
        if p.requires_grad:
            (decay if p.ndim > 1 else no_decay).append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


# ── evaluation ───────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, feeder, indices, args, amp):
    """Full pass over `indices` (int64 CPU (M,)).

    Returns dict: loss float, err float32 ndarray (M,) image px, pred float32 (M, 2),
    label float32 (M, 2), idx int64 (M,), all in `indices` order.
    """
    model.eval()
    dev = feeder.device
    errs, preds, labs, idxs = [], [], [], []      # lists of per-batch tensors
    loss_sum = torch.zeros((), device=dev)        # scalar float32 on device
    for x_u8, xy, idx in feeder.batches(indices):
        x = to_input(x_u8)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
            logits = model(x)
        loss_sum += tip_loss(logits, xy, args.loss, args.sigma, args.coord_weight) * len(idx)
        pred = decode_logits(logits)               # (B, 2) image px
        errs.append(torch.linalg.vector_norm(pred - xy, dim=1))
        preds.append(pred)
        labs.append(xy)
        idxs.append(idx)
    return {"loss": float(loss_sum) / max(1, len(indices)),
            "err": torch.cat(errs).cpu().numpy(),
            "pred": torch.cat(preds).cpu().numpy(),
            "label": torch.cat(labs).cpu().numpy(),
            "idx": torch.cat(idxs).numpy()}


def err_stats(err):
    """Error array (M,) image px -> dict of mean, median, p95, p99, max, n_over_7px."""
    return {"mean": float(err.mean()), "median": float(np.median(err)),
            "p95": float(np.percentile(err, 95)), "p99": float(np.percentile(err, 99)),
            "max": float(err.max()), "n_over_7px": int((err > OVER_PX).sum())}


def write_worst(path, res, names, n=200):
    """Write the n largest-error val samples (filename, label xy, pred xy, err) to CSV."""
    order = np.argsort(-res["err"])[:n]            # (n,) int64 positions into res arrays
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "label_x", "label_y", "pred_x", "pred_y", "err"])
        for j in order:
            w.writerow([names[int(res["idx"][j])],
                        f"{res['label'][j, 0]:.2f}", f"{res['label'][j, 1]:.2f}",
                        f"{res['pred'][j, 0]:.2f}", f"{res['pred'][j, 1]:.2f}",
                        f"{res['err'][j]:.3f}"])


# ── ONNX export ──────────────────────────────────────────────────────────

def export_and_check_onnx(model, path, samples):
    """Export model (CPU, fp32, eval) to ONNX opset 17 and compare ORT with torch.

    samples: float32 tensor (K, 1, 256, 256) checked one by one.
    Returns (ok bool, max_abs_diff float, size_kb float).
    """
    import onnx
    import onnxruntime as ort
    model = model.cpu().float().eval()
    dummy = torch.zeros(1, 1, IMG_SIZE, IMG_SIZE)          # (1, 1, 256, 256) float32
    torch.onnx.export(model, dummy, path, opset_version=17, input_names=["input"],
                      output_names=["logits"], do_constant_folding=True)
    onnx.checker.check_model(onnx.load(path))
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    max_diff, same_argmax = 0.0, True
    with torch.no_grad():
        for i in range(samples.shape[0]):
            inp = samples[i:i + 1].float().contiguous()     # (1, 1, 256, 256)
            pt = model(inp).numpy()                         # (1, 1, 128, 128)
            on = sess.run(["logits"], {"input": inp.numpy()})[0]
            max_diff = max(max_diff, float(np.abs(pt - on).max()))
            same_argmax &= int(pt.argmax()) == int(on.argmax())
    ok = same_argmax and max_diff < 1e-3
    return ok, max_diff, os.path.getsize(path) / 1024


# ── training ─────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Fast tip-heatmap trainer for the v9 dataset")
    p.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="dataset folder with dial_*.png + labels.csv (never written)")
    p.add_argument("--cache", default=None,
                   help="cache prefix (default <data-dir>_cache -> _images.npy + _meta.npz)")
    p.add_argument("--cache-workers", type=int, default=None,
                   help="PNG decode processes for the first cache build (default min(cpu, 16))")
    p.add_argument("--limit", type=int, default=None, help="use only the first N images")
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--arch", default="tipnet", choices=sorted(ARCHS))
    p.add_argument("--width", type=float, default=1.0, help="tipnet width multiplier")
    p.add_argument("--loss", default="ce", choices=["ce", "mse"])
    p.add_argument("--sigma", type=float, default=1.0, help="target Gaussian sigma (heatmap px)")
    p.add_argument("--coord-weight", type=float, default=0.1,
                   help="weight of the DSNT L1 term (heatmap px)")
    p.add_argument("--aug", default="dihedral", choices=["dihedral", "none"])
    p.add_argument("--photo-aug", action="store_true", help="mild photometric jitter (GPU)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--ema", type=float, default=0.999, help="EMA decay (0 = off)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", default=None, help="default: <arch>_<timestamp>")
    p.add_argument("--out", default=None, help=f"default {DEFAULT_RUNS_DIR}/<run-name>")
    p.add_argument("--resume", default=None, help="<out>/last.pth of an earlier run")
    p.add_argument("--export-tflite", action="store_true",
                   help="also convert model.onnx to TFLite (slow; needs onnx2tf)")
    p.add_argument("--no-amp", action="store_true", help="disable bf16 autocast")
    p.add_argument("--cpu", action="store_true", help="force CPU")
    return p.parse_args(argv)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)   # seeds CUDA too


def main(argv=None):
    args = parse_args(argv)
    run_name = args.run_name or f"{args.arch}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out = (args.out                                     # str: run output folder
           or (os.path.dirname(os.path.abspath(args.resume)) if args.resume else None)
           or os.path.join(DEFAULT_RUNS_DIR, run_name))
    os.makedirs(out, exist_ok=True)
    log_f = open(os.path.join(out, "train.log"), "a", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)
    print(f"\n=== train_v9_tip  {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  run={run_name}")
    print("args: " + " ".join(f"{k}={v}" for k, v in vars(args).items()))

    seed_everything(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    amp = dev.type == "cuda" and not args.no_amp     # bool: bf16 autocast on
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # ---- model + budget check (before touching the data) ----
    model = build_model(args.arch, args.width)
    n_params = count_params(model)                  # int
    pth_bytes = state_dict_bytes(model)             # int
    print(f"model: arch={args.arch} width={args.width}  params={n_params:,} "
          f"(budget {PARAM_BUDGET:,})  pth={pth_bytes / 1024:.1f} KB "
          f"(limit {PTH_LIMIT_BYTES / 1024:.0f} KB)")
    if pth_bytes >= PTH_LIMIT_BYTES:
        print(f"REFUSED: state_dict .pth would be {pth_bytes / 1024:.1f} KB >= 500 KB")
        sys.exit(2)
    if n_params > PARAM_BUDGET:
        print(f"WARNING: {n_params:,} params exceeds the {PARAM_BUDGET:,} budget")

    # ---- data ----
    cache = args.cache or (os.path.abspath(args.data_dir).rstrip("\\/") + "_cache")
    images, labels, names = load_cache(args.data_dir, cache, args.limit, args.cache_workers)
    n = len(names)
    train_idx, val_idx = split_indices(n, args.val_split)
    print(f"data: {n} images  train={len(train_idx)}  val={len(val_idx)}  device={dev} "
          f"amp={'bf16' if amp else 'off'}  aug={args.aug}{'+photo' if args.photo_aug else ''}")
    feeder = BatchFeeder(images, labels, dev, max(args.batch_size, args.eval_batch_size))

    model = model.to(dev).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(param_groups(model, args.weight_decay), lr=args.lr)
    ema = ModelEma(model, args.ema) if args.ema > 0 else None
    eval_model = (lambda: ema.module) if ema else (lambda: model)  # noqa: E731

    steps_per_epoch = max(1, len(train_idx) // args.batch_size)   # int: drop_last batches
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(round(args.warmup_epochs * steps_per_epoch))
    start_epoch, step = 1, 0          # int, int: first epoch to run, optimizer steps done
    best_p99, best_epoch = float("inf"), 0
    metrics_path = os.path.join(out, "metrics.csv")
    state_path = os.path.join(out, "train_state.pt")   # full state for --resume

    if args.resume:
        rs_path = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "train_state.pt")
        st = torch.load(rs_path, map_location=dev, weights_only=True)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        if ema and st.get("ema") is not None:
            ema.module.load_state_dict(st["ema"])
            ema.updates = st["ema_updates"]
        start_epoch, step = st["epoch"] + 1, st["step"]
        best_p99, best_epoch = st["best_p99"], st["best_epoch"]
        print(f"resumed from {rs_path}: next epoch {start_epoch}, best p99 {best_p99:.3f}")
    if not args.resume or not os.path.exists(metrics_path):
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(METRIC_FIELDS)

    run_t0 = time.time()
    epoch_times = []                     # list[float]: seconds per finished epoch
    gen = torch.Generator()              # CPU generator for the per-epoch permutation
    for epoch in range(start_epoch, args.epochs + 1):
        ep_t0 = time.time()
        model.train()
        gen.manual_seed(args.seed * 100_003 + epoch)
        order = train_idx[torch.randperm(len(train_idx), generator=gen)]  # (n_train,) int64
        loss_sum = torch.zeros((), device=dev)
        seen = 0
        lr = args.lr
        for x_u8, xy, _ in feeder_batches(feeder, order, args.batch_size):
            b = x_u8.shape[0]
            if args.aug == "dihedral":
                k = torch.randint(0, 8, (b,), device=dev)
                x_u8 = dihedral_images(x_u8, k)
                xy = dihedral_points(xy, k)
            x = to_input(x_u8)
            if args.photo_aug:
                x = photometric_aug(x).contiguous(memory_format=torch.channels_last)
            step += 1
            lr = lr_at(step, total_steps, warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp):
                logits = model(x)
            loss = tip_loss(logits, xy, args.loss, args.sigma, args.coord_weight)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if ema:
                ema.update()
            loss_sum += loss.detach() * b
            seen += b
        if dev.type == "cuda":
            torch.cuda.synchronize()
        train_secs = time.time() - ep_t0
        train_loss = float(loss_sum) / max(1, seen)

        res = evaluate(eval_model(), feeder, val_idx, args, amp)
        st = err_stats(res["err"])
        ep_secs = time.time() - ep_t0
        ips = seen / max(train_secs, 1e-9)
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{lr:.6g}", f"{train_loss:.6f}", f"{res['loss']:.6f}",
                                    f"{st['mean']:.4f}", f"{st['median']:.4f}", f"{st['p95']:.4f}",
                                    f"{st['p99']:.4f}", f"{st['max']:.4f}", st["n_over_7px"],
                                    f"{ep_secs:.2f}", f"{ips:.1f}"])
        sd = clean_state_dict(eval_model())
        torch.save(sd, os.path.join(out, "last.pth"))
        tag = ""
        if np.isfinite(st["p99"]) and st["p99"] < best_p99:
            best_p99, best_epoch = st["p99"], epoch
            torch.save(sd, os.path.join(out, "best_p99.pth"))
            write_worst(os.path.join(out, "worst_val.csv"), res, names)
            tag = "  * best p99 -> best_p99.pth, worst_val.csv"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "ema": ema.module.state_dict() if ema else None,
                    "ema_updates": ema.updates if ema else 0, "epoch": epoch, "step": step,
                    "best_p99": best_p99, "best_epoch": best_epoch}, state_path)
        epoch_times.append(ep_secs)
        elapsed = time.time() - run_t0
        eta = (sum(epoch_times) / len(epoch_times)) * (args.epochs - epoch)
        print(f"[ep {epoch:3d}/{args.epochs}] lr={lr:.2e} train={train_loss:.4f} "
              f"val={res['loss']:.4f} | err mean={st['mean']:.3f} med={st['median']:.3f} "
              f"p95={st['p95']:.3f} p99={st['p99']:.3f} max={st['max']:.1f} "
              f"n>7px={st['n_over_7px']} | {fmt_secs(ep_secs)} ({ips:.0f} img/s) "
              f"| elapsed {fmt_secs(elapsed)} ETA {fmt_secs(eta)}{tag}")

    # ---- end of training: reload best, re-evaluate in fp32, export ONNX ----
    best_path = os.path.join(out, "best_p99.pth")
    if not os.path.exists(best_path):   # every epoch had a non-finite p99 (diverged)
        print("WARNING: no finite val p99 in any epoch; using last.pth")
        best_path = os.path.join(out, "last.pth")
    final = build_model(args.arch, args.width)
    final.load_state_dict(torch.load(best_path, map_location="cpu", weights_only=True))
    final = final.to(dev).to(memory_format=torch.channels_last)
    res = evaluate(final, feeder, val_idx, args, amp=False)
    st = err_stats(res["err"])
    pth_kb = os.path.getsize(best_path) / 1024
    print(f"best_p99.pth re-evaluated (fp32): mean={st['mean']:.3f} med={st['median']:.3f} "
          f"p95={st['p95']:.3f} p99={st['p99']:.3f} max={st['max']:.1f} n>7px={st['n_over_7px']}")
    print(f"FINAL best_p99={st['p99']:.3f}px at epoch {best_epoch}, "
          f"pth_kb={pth_kb:.1f}, params={n_params}")

    onnx_path = os.path.join(out, "model.onnx")
    samples = to_input(images[val_idx[:8]]).float().cpu().contiguous()  # (<=8, 1, 256, 256)
    ok, diff, onnx_kb = export_and_check_onnx(final, onnx_path, samples)
    print(f"ONNX: {onnx_path}  {onnx_kb:.1f} KB  ORT vs torch max|diff|={diff:.2e} "
          f"argmax {'match' if ok else 'MISMATCH'} -> {'OK' if ok else 'FAILED'}")
    if args.export_tflite:
        from train_utils_v2 import export_tflite
        export_tflite(onnx_path, os.path.join(out, "model.tflite"))
    print(f"total time {fmt_secs(time.time() - run_t0)}")
    sys.stdout.flush()
    if not ok:
        sys.exit(1)


def feeder_batches(feeder, order, batch_size):
    """Training batches of exactly batch_size from `order` (drop_last), via the feeder."""
    saved = feeder.batch_size
    feeder.batch_size = batch_size
    try:
        yield from feeder.batches(order, drop_last=True)
    finally:
        feeder.batch_size = saved


if __name__ == "__main__":
    main()
