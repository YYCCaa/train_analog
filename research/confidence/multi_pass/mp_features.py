"""Multi-pass consistency features for the TipNet v9 confidence research (family 2: multi-pass).

For every image of a split this runs the clean pass plus 7 extra passes of each of 4 input-transform
families, maps every pass back into the clean frame and stores SMALL per-pass features (no maps):

    gray (256,256) uint8 ──T_j──▶ TipNet ──▶ logits_j (128,128) ──decode──▶ tip_j (variant frame)
                                                 │                              │ T_j^-1
                                                 │ softmax                      ▼
                                                 ▼                         tip_j (clean frame)
                                  probs_j ── 5x5 / 9x9 mass at own argmax       (m5own, m9own)
                                          ── 5x5 / 9x9 mass at T_j(clean argmax cell)  (m5c, m9c)
                                          ── T_j^-1 (exact families only) ──▶ running mean map
                                             ──▶ at K = 2, 4, 8 passes: decoded tip + own 5x5 / 9x9 mass

Families (7 extra passes each; K passes = the clean pass + the first K-1 extras):
    dih    dihedral k in DIH_ORDER (train_v9_tip.dihedral_images: k>=4 flip first, then rot90 k%4 times)
    noise  training-style masks (confbundle.noise_mask inside confbundle.inside_dial), 7 per image
    shift  translation by SHIFTS px with black fill (even px, so the heatmap shifts by whole cells)
    scale  resample about the centre 127.5 by SCALES (bilinear, re-quantised to the uint8 grid)

Coordinates: pixel i is centred at i; heatmap cell u covers px 2u, 2u+1, x = 2u + 0.5.
Run from the repo root:   python research/confidence/multi_pass/mp_features.py --check
                          python research/confidence/multi_pass/mp_features.py --split train
Output: research/confidence/multi_pass/_cache/feat_<split>.npz (not committed; regenerate with this script).
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# BUNDLE: Path, the read-only research bundle (data, weights, harness)
BUNDLE = Path(r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-PycharmProjects-train-dial"
              r"\f292a1d6-f81e-4d2c-b353-1ee00d6e9308\scratchpad\confbundle")
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(Path.cwd()))
import confbundle as cb  # noqa: E402
import train_v9_tip as tv  # noqa: E402

# HERE: Path, this folder; CACHE: Path, where the feature files go (gitignored by not being added)
HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
# DEV: str, torch device
DEV = "cuda"
# S / H: int, image side (px) and heatmap side (cells)
S, H = cb.IMG_SIZE, cb.HM_SIZE
# CENTRE: float, image centre in the pixel-centred convention ((S-1)/2 = 127.5)
CENTRE = (S - 1) / 2.0
# DIH_ORDER: list[int] 7 dihedral codes, prefix order (K=2: rot180; K=4: + hflip, vflip; K=8: all 8)
DIH_ORDER = [2, 4, 6, 1, 3, 5, 7]
# SHIFTS: list[(dx, dy)] 7 translations in image px (even, so heatmaps move by whole cells)
SHIFTS = [(6, 0), (-6, 0), (0, 6), (0, -6), (6, 6), (-6, -6), (6, -6)]
# SCALES: list[float] 7 zoom factors about CENTRE (>1 = dial drawn bigger)
SCALES = [0.95, 1.05, 0.97, 1.03, 0.96, 1.04, 0.98]
# KS: tuple[int] pass budgets reported (clean pass included)
KS = (2, 4, 8)
# N_BANK: int, size of the fixed noise-mask bank; MASK_SEED: int seed of the bank; DRAW_SEED: per-split draws
N_BANK, MASK_SEED, DRAW_SEED = 512, 1234, {"train": 11, "val": 22}
# FAMILIES: tuple[str] family names in storage order
FAMILIES = ("dih", "noise", "shift", "scale")


# ── transforms: each has img(x, ctx, j) -> x', fwd(xy) / inv(xy) point maps, map_inv(m) or None ──────

def _full(k, n):
    """(n,) int64 CUDA tensor filled with k (dihedral code per sample)."""
    return torch.full((n,), k, dtype=torch.int64, device=DEV)


def shift2d(t, dx, dy, fill=0.0):
    """Translate the last two dims of t (..., h, w) by (dx, dy) whole cells: out[y, x] = t[y-dy, x-dx]; gaps = fill."""
    out = torch.full_like(t, fill)
    h, w = t.shape[-2:]
    out[..., max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = \
        t[..., max(-dy, 0):h + min(-dy, 0), max(-dx, 0):w + min(-dx, 0)]
    return out


class Identity:
    """The clean pass: no transform."""
    exact = True

    def img(self, x, ctx):
        return x

    def fwd(self, xy):
        return xy

    def inv(self, xy):
        return xy

    def map_inv(self, m):
        return m


class Dihedral:
    """One of the 8 flips / rot90s exactly as training applied them (k: int in [0, 8))."""
    exact = True

    def __init__(self, k):
        self.k = k                                        # int: forward code
        self.kinv = (4 - k) % 4 if k < 4 else k           # int: inverse code (reflections are involutions)

    def img(self, x, ctx):
        return tv.dihedral_images(x, _full(self.k, len(x)))

    def fwd(self, xy):
        return tv.dihedral_points(xy, _full(self.k, len(xy)))

    def inv(self, xy):
        return tv.dihedral_points(xy, _full(self.kinv, len(xy)))

    def map_inv(self, m):
        """(B, 128, 128) map in the variant frame -> clean frame (a 2x-subsampled flip / rot is the same op)."""
        return tv.dihedral_images(m, _full(self.kinv, len(m)))


class Shift:
    """Translate by (dx, dy) image px, black fill (dx, dy even ints)."""
    exact = True

    def __init__(self, dx, dy):
        self.dx, self.dy = dx, dy                         # int, int: image px

    def img(self, x, ctx):
        return shift2d(x, self.dx, self.dy, 0.0)

    def fwd(self, xy):
        return xy + torch.tensor([self.dx, self.dy], dtype=xy.dtype, device=xy.device)

    def inv(self, xy):
        return xy - torch.tensor([self.dx, self.dy], dtype=xy.dtype, device=xy.device)

    def map_inv(self, m):
        return shift2d(m, -self.dx // 2, -self.dy // 2, 0.0)


class Scale:
    """Resample about CENTRE by factor s (bilinear, zeros outside), re-quantised to /255 steps."""
    exact = False

    def __init__(self, s):
        self.s = s                                        # float: zoom factor

    def img(self, x, ctx):
        theta = torch.tensor([[1 / self.s, 0, 0], [0, 1 / self.s, 0]], dtype=x.dtype, device=x.device)
        grid = F.affine_grid(theta.expand(len(x), 2, 3), list(x.shape), align_corners=False)
        y = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
        return torch.round(y * 255.0) / 255.0

    def fwd(self, xy):
        return CENTRE + self.s * (xy - CENTRE)

    def inv(self, xy):
        return CENTRE + (xy - CENTRE) / self.s

    map_inv = None


class Noise:
    """Training-style noise mask j of the image's 7 drawn masks (ctx['masks'] (B,7,S,S) int16, ctx['inside'] bool)."""
    exact = True

    def __init__(self, j):
        self.j = j                                        # int: which of the 7 drawn masks

    def img(self, x, ctx):
        u8 = ctx["u8"].to(torch.int16)                    # (B, S, S) int16
        m = torch.where(ctx["inside"], ctx["masks"][:, self.j], torch.zeros_like(u8))
        return (u8 + m).clamp_(0, 255).float().div_(255.0)[:, None]

    def fwd(self, xy):
        return xy

    def inv(self, xy):
        return xy

    def map_inv(self, m):
        return m


def family_transforms():
    """-> dict fam -> list of 7 transform objects in prefix order."""
    return {"dih": [Dihedral(k) for k in DIH_ORDER], "noise": [Noise(j) for j in range(7)],
            "shift": [Shift(dx, dy) for dx, dy in SHIFTS], "scale": [Scale(s) for s in SCALES]}


# ── fast inside_dial (same convex hull, half-plane test per row instead of Delaunay over 65536 points) ──

def fast_inside(img):
    """(S, S) uint8 -> (S, S) bool: convex hull of the non-black region (matches confbundle.inside_dial)."""
    from scipy.spatial import ConvexHull
    nz = img > 0                                           # (S, S) bool
    rows = np.flatnonzero(nz.any(1))                       # (r,) int64 rows with any non-black pixel
    if len(rows) < 3:
        return nz
    first = nz[rows].argmax(1)                             # (r,) leftmost non-black column per row
    last = S - 1 - nz[rows][:, ::-1].argmax(1)             # (r,) rightmost
    pts = np.concatenate([np.column_stack([first, rows]), np.column_stack([last, rows])]).astype(np.float64)
    hull = ConvexHull(pts)
    v = pts[hull.vertices]                                 # (E, 2) float64 CCW (x, y)
    v1 = np.roll(v, -1, axis=0)
    a = -(v1[:, 1] - v[:, 1])                              # (E,) coefficient of x in the edge function
    b = v1[:, 0] - v[:, 0]                                 # (E,) coefficient of y
    c = -(a * v[:, 0] + b * v[:, 1])                       # (E,) constant: f(p) = a x + b y + c >= 0 inside
    ys = np.arange(S, dtype=np.float64)[:, None]           # (S, 1)
    rhs = -(b[None] * ys + c[None]) - 1e-9                 # (S, E): need a x >= rhs
    with np.errstate(divide="ignore", invalid="ignore"):
        bound = rhs / a[None]                              # (S, E)
    lo = np.where(a[None] > 0, bound, -np.inf).max(1)      # (S,) float64 smallest x inside per row
    hi = np.where(a[None] < 0, bound, np.inf).min(1)       # (S,) largest x
    ok = np.where(a[None] == 0, rhs <= 0, True).all(1)     # (S,) rows not cut off by a horizontal edge
    xs = np.arange(S, dtype=np.float64)[None]              # (1, S)
    inside = (xs >= lo[:, None]) & (xs <= hi[:, None]) & ok[:, None]
    return inside | nz


# ── per-pass statistics on the GPU ──────────────────────────────────────────────────────────────────

def window_mass(probs, cy, cx):
    """probs (B, H, H) + cell (cy, cx) (B,) int64 -> (m5, m9) (B,) each: 5x5 / 9x9 mass, clipped to the map."""
    b = len(probs)
    pad = F.pad(probs, (4, 4, 4, 4))                      # (B, H+8, H+8) zeros outside
    d = torch.arange(9, device=probs.device)               # (9,) offsets into the padded map
    yy = (cy[:, None, None] + d[None, :, None]).expand(b, 9, 9)
    xx = (cx[:, None, None] + d[None, None, :]).expand(b, 9, 9)
    bb = torch.arange(b, device=probs.device)[:, None, None].expand(b, 9, 9)
    patch = pad[bb, yy, xx]                                # (B, 9, 9)
    return patch[:, 2:7, 2:7].sum((1, 2)), patch.sum((1, 2))


def argmax_cell(m):
    """(B, H, H) map -> (cy, cx) (B,) int64 each (first max in row-major order, like the decoder)."""
    flat = m.reshape(len(m), -1).argmax(1)
    return flat // H, flat % H


def to_cell(xy):
    """Image px (B, 2) -> nearest heatmap cell (cy, cx) (B,) int64, clamped to the map."""
    c = torch.round((xy - 0.5) / 2.0).long().clamp_(0, H - 1)
    return c[:, 1], c[:, 0]


def run_pass(model, x):
    """(B, 1, S, S) float input -> (logits (B, H, H) float32, probs (B, H, H) float32)."""
    lg = model(x.contiguous(memory_format=torch.channels_last)).float()[:, 0]
    probs = torch.softmax(lg.reshape(len(lg), -1), 1).reshape(lg.shape)
    return lg, probs


def decode(lg):
    """(B, H, H) logits -> (B, 2) float32 tip, image px (train_v9_tip.decode_logits)."""
    return tv.decode_logits(lg[:, None])


# ── unit check: transform, predict, map back ─────────────────────────────────────────────────────────

def unit_check(model, n=128):
    """Clean confident val images through every geometric transform must map back within ~0.5 px.

    Prints per transform: bias (mean dx, dy), median / p95 distance to the clean tip, and the
    max |decode(map_inv(logits)) - inv(decode(logits))| (exact families). Raises on failure.
    Also shows that a deliberately wrong flip mapping (x -> 256 - x) fails it (the check has teeth).
    """
    val = cb.load_split("val")
    pos = np.flatnonzero((val["conf"] > 0.95) & (val["err"] < 0.5))[:n]   # (n,) int64 val positions
    u8 = torch.from_numpy(np.ascontiguousarray(val["images"][pos])).to(DEV)
    x0 = u8.float().div(255.0)[:, None]                                    # (n, 1, S, S) float32
    lab = torch.from_numpy(val["label"][pos]).to(DEV)                      # (n, 2) float32
    with torch.no_grad():
        lg0, _ = run_pass(model, x0)
        tip0 = decode(lg0)                                                 # (n, 2) clean tip
        meta_gap = (tip0.double().cpu().numpy() - val["tip"][pos]).__abs__().max()
        print(f"[check] {len(pos)} images; clean tip vs bundle meta max |diff| {meta_gap:.2e} px")
        assert meta_gap < 1e-2, "clean decode does not match the bundle meta"
        fams = family_transforms()
        ok = True
        for fam in ("dih", "shift", "scale"):
            for tr in fams[fam]:
                lg, _ = run_pass(model, tr.img(x0, None))
                tip_v = decode(lg)                                         # variant frame
                back = tr.inv(tip_v)                                       # clean frame
                d = (back - tip0).norm(dim=1)
                bias = (back - tip0).mean(0).cpu().numpy()
                lab_gap = (tip_v - tr.fwd(lab)).norm(dim=1).median().item()
                exact_gap = float("nan")
                if tr.exact:
                    exact_gap = (decode(tr.map_inv(lg)) - back).abs().max().item()
                med, p95 = d.median().item(), d.quantile(0.95).item()
                good = med < 0.5 and np.hypot(*bias) < 0.25 and (not tr.exact or exact_gap < 1e-3)
                ok &= good
                name = f"{fam}:{getattr(tr, 'k', getattr(tr, 's', None)) if fam != 'shift' else (tr.dx, tr.dy)}"
                print(f"[check] {name:<16} bias ({bias[0]:+.3f},{bias[1]:+.3f})  median {med:.3f}  p95 {p95:.3f}"
                      f"  |pred - fwd(label)| median {lab_gap:.3f}  map_inv gap {exact_gap:.1e}  "
                      f"{'ok' if good else 'FAIL'}")
        # wrong mapping on purpose: horizontal flip mapped back with x -> 256 - x instead of 255 - x
        lg, _ = run_pass(model, x0.flip(-1))
        tip_v = decode(lg)
        wrong = torch.stack([256.0 - tip_v[:, 0], tip_v[:, 1]], 1)
        right = torch.stack([255.0 - tip_v[:, 0], tip_v[:, 1]], 1)
        print(f"[check] hflip, wrong map 256-x: median {(wrong - tip0).norm(dim=1).median().item():.3f} px; "
              f"right map 255-x: {(right - tip0).norm(dim=1).median().item():.3f} px")
    # fast_inside vs the bundle's inside_dial
    gaps = [int((fast_inside(val["images"][i]) != cb.inside_dial(val["images"][i])).sum()) for i in pos[:32]]
    print(f"[check] fast_inside vs inside_dial: mismatched px per image max {max(gaps)} (of {S * S})")
    ok &= max(gaps) <= 64
    if not ok:
        raise SystemExit("[check] FAILED")
    print("[check] all transforms map back correctly")


# ── feature extraction over a split ──────────────────────────────────────────────────────────────────

def extract(model, split, batch=256, limit=None):
    """Run 1 + 4 x 7 passes on every image of `split` and save the per-pass features (see module doc)."""
    data = cb.load_split(split)
    imgs = data["images"]                                  # (n, S, S) uint8 memmap
    n = len(imgs) if limit is None else min(limit, len(imgs))
    bank = torch.from_numpy(np.stack([cb.noise_mask(np.random.default_rng([MASK_SEED, i]))
                                      for i in range(N_BANK)])).to(DEV)    # (N_BANK, S, S) int16
    draw_rng = np.random.default_rng(DRAW_SEED[split])
    draws = np.stack([draw_rng.choice(N_BANK, 7, replace=False) for _ in range(n)])  # (n, 7) int64
    fams = family_transforms()
    # out: dict name -> numpy array, see the module doc for the meaning of each key
    out = {"clean_tip": np.zeros((n, 2), np.float32), "clean_conf": np.zeros(n, np.float32),
           "clean_m9": np.zeros(n, np.float32), "clean_cell": np.zeros((n, 2), np.int16),
           "noise_draws": draws.astype(np.int16)}
    for fam in FAMILIES:
        out[f"{fam}_tip"] = np.zeros((n, 7, 2), np.float32)        # tips mapped back to the clean frame
        for key in ("m5own", "m9own", "m5c", "m9c"):
            out[f"{fam}_{key}"] = np.zeros((n, 7), np.float32)
        if fams[fam][0].exact:
            out[f"{fam}_avgtip"] = np.zeros((n, len(KS), 2), np.float32)  # decode of the K-pass mean map
            out[f"{fam}_avgm5"] = np.zeros((n, len(KS)), np.float32)      # its 5x5 mass at its own argmax
            out[f"{fam}_avgm9"] = np.zeros((n, len(KS)), np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(n, s + batch)
            u8_np = np.ascontiguousarray(imgs[s:e])                         # (b, S, S) uint8
            inside = np.stack([fast_inside(im) for im in u8_np])            # (b, S, S) bool
            ctx = {"u8": torch.from_numpy(u8_np).to(DEV), "inside": torch.from_numpy(inside).to(DEV),
                   "masks": bank[torch.from_numpy(draws[s:e]).to(DEV)]}     # masks (b, 7, S, S) int16
            x0 = ctx["u8"].float().div(255.0)[:, None]
            lg0, p0 = run_pass(model, x0)
            tip0 = decode(lg0)
            cy, cx = argmax_cell(lg0)
            m5, m9 = window_mass(p0, cy, cx)
            out["clean_tip"][s:e] = tip0.cpu().numpy()
            out["clean_conf"][s:e] = m5.cpu().numpy()
            out["clean_m9"][s:e] = m9.cpu().numpy()
            out["clean_cell"][s:e] = torch.stack([cy, cx], 1).cpu().numpy()
            centre0 = torch.stack([2.0 * cx + 0.5, 2.0 * cy + 0.5], 1).float()   # (b, 2) clean argmax cell centre
            for fam in FAMILIES:
                acc = p0.clone() if fams[fam][0].exact else None             # (b, H, H) running sum of maps
                for j, tr in enumerate(fams[fam]):
                    lg, p = run_pass(model, tr.img(x0, ctx))
                    out[f"{fam}_tip"][s:e, j] = tr.inv(decode(lg)).cpu().numpy()
                    oy, ox = argmax_cell(lg)
                    a5, a9 = window_mass(p, oy, ox)
                    vy, vx = to_cell(tr.fwd(centre0))
                    c5, c9 = window_mass(p, vy, vx)
                    for key, val_ in (("m5own", a5), ("m9own", a9), ("m5c", c5), ("m9c", c9)):
                        out[f"{fam}_{key}"][s:e, j] = val_.cpu().numpy()
                    if acc is not None:
                        acc += tr.map_inv(p)
                        if j + 2 in KS:                                      # j+2 passes so far
                            ki = KS.index(j + 2)
                            avg = acc / (j + 2)
                            out[f"{fam}_avgtip"][s:e, ki] = decode(torch.log(avg.clamp_min(1e-30))).cpu().numpy()
                            ay, ax = argmax_cell(avg)
                            v5, v9 = window_mass(avg, ay, ax)
                            out[f"{fam}_avgm5"][s:e, ki] = v5.cpu().numpy()
                            out[f"{fam}_avgm9"][s:e, ki] = v9.cpu().numpy()
            if (s // batch) % 10 == 0 or e == n:
                el = time.time() - t0
                print(f"[{split}] {e}/{n}  {el:.0f} s  ({e * 29 / el:.0f} passes/s)", flush=True)
    meta = data
    gap = np.abs(out["clean_tip"] - meta["tip"][:n]).max()
    cgap = np.abs(out["clean_conf"] - meta["conf"][:n]).max()
    print(f"[{split}] clean tip vs meta max {gap:.2e} px, conf vs meta max {cgap:.2e}")
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"feat_{split}{'' if limit is None else f'_{n}'}.npz"
    np.savez(path, **out, dih_order=np.array(DIH_ORDER), shifts=np.array(SHIFTS), scales=np.array(SCALES),
             ks=np.array(KS))
    print(f"[{split}] saved {path}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="run the coordinate-mapping unit check and exit")
    p.add_argument("--split", choices=["train", "val"])
    p.add_argument("--limit", type=int, default=None, help="first n images only (smoke run)")
    p.add_argument("--batch", type=int, default=256, help="images per GPU batch (<= 512)")
    a = p.parse_args(argv)
    model = cb.load_model()
    if a.check:
        unit_check(model)
        return
    extract(model, a.split, batch=min(a.batch, 512), limit=a.limit)


if __name__ == "__main__":
    main()
