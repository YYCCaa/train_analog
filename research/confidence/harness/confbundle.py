"""Shared, read-only data + helpers for the TipNet v9 confidence research.  NEVER write into this folder.

Import it from your own worktree root (so `train_v9_tip` and `tip_webapp` resolve to your copy):

    import sys; sys.path.insert(0, r"<this folder>")
    import confbundle as cb
    val = cb.load_split("val")          # dict, see load_split
    model = cb.load_model()             # TipNet, CUDA, full float32 (TF32 off)
    lg = cb.logits(model, val["images"][:512])   # (512, 128, 128) float32

Coordinates: image px, pixel i centred at i. Heatmap cell u covers image px 2u, 2u+1; x = 2u + 0.5.
"""

import sys
from pathlib import Path

import numpy as np

# BUNDLE: Path, this folder
BUNDLE = Path(__file__).resolve().parent
# IMG_SIZE / HM_SIZE: int, model input side (px) and heatmap side (cells)
IMG_SIZE, HM_SIZE = 256, 128
# NOISE: hier_noise blocks NB, block_variance BV, pixel_variance PV, blur radius BLUR (v9_config.yaml steps 11 + 14)
NB, BV, PV, BLUR = 5, 15, 5, 2


def load_split(split):
    """split "val" (22500 images, report here) or "train" (20000-image sample, tune / fit here) -> dict:

    images (n, 256, 256) uint8 memmap (read-only; slice it, don't load all of it twice)
    rows   (n,) int64   row in the full sv9_150k dataset
    label  (n, 2) float32  ground-truth tip (x, y), image px
    tip    (n, 2) float64  clean decoded tip (argmax -> 5x5 softmax centroid), image px
    err    (n,) float64  |tip - label|, image px
    conf   (n,) float64  current confidence = global-softmax mass of the 5x5 cells around the argmax cell
    m9     (n,) float64  same mass over 9x9 cells
    peak   (n,) float64  probability of the argmax cell alone
    runner (n,) float64  5x5 mass of the second candidate peak (webapp find_candidates; 0 = none)
    """
    meta = dict(np.load(BUNDLE / f"{split}_meta.npz"))
    meta["images"] = np.load(BUNDLE / f"{split}_images.npy", mmap_mode="r")
    return meta


def load_stability():
    """The noise-stability probe on the val best / worst 1000 -> dict.

    pos (2000,) int64 positions into the VAL split; group (2000,) 0 = best 1000, 1 = worst 1000 (by clean err);
    shift (2000, 10) float64 |noisy tip - clean tip| px under 10 training-style noise masks per image;
    jumper (2000,) bool = shift.max(1) > 7 px (12 images); plus tip / conf / err of the noisy trials.
    """
    d = dict(np.load(BUNDLE / "stab_stats.npz"))
    d["jumper"] = d["shift"].max(axis=1) > 7.0
    return d


def load_model(device="cuda"):
    """TipNet with the shipped weights on `device`, eval mode, channels_last, full float32.

    Needs `train_v9_tip` importable: run from your worktree root (it is put on sys.path here).
    TF32 is switched off: with it the logits drift ~0.09 from the float32 TFLite the webapp runs;
    without it they match within ~1e-4.
    """
    import torch
    sys.path.insert(0, str(Path.cwd()))
    import train_v9_tip as tv
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = tv.build_model("tipnet").to(device).eval()
    model.load_state_dict(torch.load(BUNDLE / "tip.pth", map_location=device, weights_only=True))
    return model.to(memory_format=torch.channels_last)


def logits(model, images_u8, batch=512):
    """(n, 256, 256) uint8 (numpy / memmap slice) -> (n, 128, 128) float32 numpy logits."""
    import torch
    dev = next(model.parameters()).device
    out = np.empty((len(images_u8), HM_SIZE, HM_SIZE), np.float32)
    with torch.no_grad():
        for s in range(0, len(images_u8), batch):
            x = torch.from_numpy(np.ascontiguousarray(images_u8[s:s + batch])).to(dev)
            x = x.unsqueeze(1).float().div_(255.0).contiguous(memory_format=torch.channels_last)
            out[s:s + batch] = model(x).float()[:, 0].cpu().numpy()
    return out


def noise_mask(rng, size=IMG_SIZE):
    """One training-style noise mask (generator steps 11 + 14) -> (size, size) int16, about -20..20 gray levels."""
    from PIL import Image, ImageFilter
    block_u8 = np.clip(rng.integers(-BV, BV + 1, (NB, NB), dtype=np.int16) + 128, 0, 255).astype(np.uint8)
    block_up = np.array(Image.fromarray(block_u8).resize((size, size), Image.BICUBIC), dtype=np.int16) - 128
    combined = block_up + rng.integers(-PV, PV + 1, (size, size), dtype=np.int16)
    as_u8 = Image.fromarray(np.clip(combined + 128, 0, 255).astype(np.uint8))
    return np.array(as_u8.filter(ImageFilter.GaussianBlur(radius=BLUR)), dtype=np.int16) - 128


def inside_dial(img):
    """(256, 256) uint8 -> (256, 256) bool, convex hull of the non-black dial region (where training adds noise)."""
    from scipy.spatial import ConvexHull, Delaunay
    ys, xs = np.nonzero(img > 0)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    gy, gx = np.mgrid[0:img.shape[0], 0:img.shape[1]]
    inside = hull.find_simplex(np.column_stack([gx.ravel(), gy.ravel()])) >= 0
    return inside.reshape(img.shape) | (img > 0)


def apply_mask(img, mask, inside):
    """img (256, 256) uint8 + mask (256, 256) int16 inside the dial -> (256, 256) uint8."""
    return np.clip(img.astype(np.int16) + np.where(inside, mask, 0), 0, 255).astype(np.uint8)
