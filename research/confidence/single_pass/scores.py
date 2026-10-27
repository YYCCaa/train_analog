"""Final single-pass confidence scores for TipNet v9: one numpy function per method, one logit map in, one float out.

Every score reads only the logit map the normal prediction already produces (passes = 1):

    logits (128, 128) float32  ──prep()──▶  Map: p (128,128) float64 global softmax, lmax/lse float,
                                                 argmax cell (iy, ix) int, decoded tip (u, v) float cells,
                                                 ideal target marginals gy/gx (128,) float64 (sigma 1 cell)
                               ──METHODS[name](logits)──▶  float, HIGHER = MORE TRUSTWORTHY

    heatmap cell grid around the decoded tip (u, v); d = Euclidean distance in cells
    ┌─────────────────────────────────────────┐
    │  far mass (decoys, rim marks)  d >> 4   │   mom2 weighs it by d^2: a 0.01 decoy 30 cells away
    │      ┌───────────────────┐              │   costs 9 cells^2, as much as a 3x wider peak
    │      │ ring  2 < d <= 4  │ blur, harmless│
    │      │   ┌──────────┐    │              │   disc4/ideal: mass inside d <= 4 over what a perfect
    │      │   │ core d<=2│ ◀── tip (u, v)    │   sigma-1 target centred at the same sub-cell point holds
    │      │   └──────────┘    │              │
    │      └───────────────────┘              │   bc: overlap sum sqrt(p * g) with that perfect target
    └─────────────────────────────────────────┘

Usage (from the worktree root):
    python research/confidence/single_pass/scores.py --bundle <bundle dir> --cache <cache dir>
        scores every method of FINAL on both splits with eval_conf.evaluate (-> results.jsonl), times the
        numpy post-processing per map, writes every method's scores to <cache>/scores_<slug>.npz
        (the best 3 were copied here as scores_bc_T1.5.npz, scores_conf_sub_m9k16.npz, scores_kl_T2.npz).
    python research/confidence/single_pass/scores.py --bundle <bundle dir> --cache <cache dir> --time-only
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

# H: int, heatmap side (cells)
H = 128
# GRID: (128,) float64 cell indices 0..127
GRID = np.arange(H, dtype=np.float64)
# OFF5: (5,) float64 offsets -2..2 of the shipped 5x5 decode window
OFF5 = np.arange(-2, 3, dtype=np.float64)
# HERE: Path, this folder (results.jsonl, scores_*.npz are written here)
HERE = Path(__file__).resolve().parent


@dataclass
class Map:
    """Shared one-map quantities (what the shipped decode already computes, plus the ideal target marginals)."""
    L: np.ndarray        # (128, 128) float64 logits
    p: np.ndarray        # (128, 128) float64 global softmax, sums to 1
    lmax: float          # max logit
    lse: float           # logsumexp of the logits
    iy: int              # argmax row (cell)
    ix: int              # argmax col (cell)
    u: float             # decoded tip column, heatmap cells (x_img = 2u + 0.5)
    v: float             # decoded tip row, heatmap cells
    gy: np.ndarray       # (128,) float64 ideal sigma-1 target row marginal at v, sums to 1
    gx: np.ndarray       # (128,) float64 ideal target column marginal at u, sums to 1


def prep(logits):
    """(128, 128) float logits -> Map (shipped softmax + argmax + 5x5 logit-softmax centroid decode)."""
    L = np.asarray(logits, dtype=np.float64)
    flat = int(np.argmax(L))
    iy, ix = divmod(flat, H)
    lmax = float(L[iy, ix])
    e = np.exp(L - lmax)
    z = float(e.sum())
    p = e / z
    y0, y1, x0, x1 = max(0, iy - 2), min(H, iy + 3), max(0, ix - 2), min(H, ix + 3)
    w = e[y0:y1, x0:x1] / e[y0:y1, x0:x1].sum()           # (<=5, <=5) 5x5 logit softmax (off-map cells dropped)
    v = float((w.sum(1) * GRID[y0:y1]).sum())
    u = float((w.sum(0) * GRID[x0:x1]).sum())
    gy = np.exp(-(GRID - v) ** 2 / 2)
    gx = np.exp(-(GRID - u) ** 2 / 2)
    return Map(L, p, lmax, lmax + np.log(z), iy, ix, u, v, gy / gy.sum(), gx / gx.sum())


def _win(m, r):
    """Row / col slices of the (2r+1)^2 square around the argmax, clipped to the map."""
    return slice(max(0, m.iy - r), m.iy + r + 1), slice(max(0, m.ix - r), m.ix + r + 1)


# ── methods: (128, 128) logits -> float, higher = more trustworthy ──────────────────────────────────

def conf(logits):
    """Shipped: global-softmax mass of the 5x5 cells around the argmax."""
    m = prep(logits)
    ys, xs = _win(m, 2)
    return float(m.p[ys, xs].sum())


def m9(logits):
    """Baseline: the same mass over 9x9 cells."""
    m = prep(logits)
    ys, xs = _win(m, 4)
    return float(m.p[ys, xs].sum())


def conf_sub(logits):
    """conf / its sub-cell ceiling: the 5x5 mass a perfect sigma-1 target at (u, v) would have (0.962..0.982)."""
    m = prep(logits)
    ys, xs = _win(m, 2)
    return float(m.p[ys, xs].sum() / (m.gy[ys].sum() * m.gx[xs].sum()))


def disc_sub(logits, r=4.0, T=1.0):
    """Mass of softmax(L / T) within distance r cells of the decoded tip / the same mass of the perfect target."""
    m = prep(logits)
    pt = m.p if T == 1.0 else np.exp((m.L - m.lmax) / T)
    pt = pt / pt.sum()
    k = int(np.ceil(r)) + 2                                # the disc fits in this square around the argmax
    ys, xs = _win(m, k)
    d2 = (GRID[ys, None] - m.v) ** 2 + (GRID[None, xs] - m.u) ** 2   # (rows, cols) cells^2
    inside = d2 <= r * r
    ideal = (np.outer(m.gy[ys], m.gx[xs]) * inside).sum()
    return float((pt[ys, xs] * inside).sum() / ideal)


def neg_moment(logits, a=2.0, T=1.0):
    """-sum p_T * d^a over the whole map: minus the predicted |error|^a (cells) if the map were the posterior."""
    m = prep(logits)
    pt = m.p if T == 1.0 else np.exp((m.L - m.lmax) / T)
    pt = pt / pt.sum()
    d2 = (GRID[:, None] - m.v) ** 2 + (GRID[None, :] - m.u) ** 2
    return float(-(pt * (d2 if a == 2.0 else d2 ** (a / 2))).sum())


def bhatt(logits, T=1.0):
    """Bhattacharyya overlap sum sqrt(p_T * g) with the perfect sigma-1 target g at (u, v); 1 = identical maps."""
    m = prep(logits)
    pt = m.p if T == 1.0 else np.exp((m.L - m.lmax) / T)
    pt = pt / pt.sum()
    return float((np.sqrt(pt) * np.outer(np.sqrt(m.gy), np.sqrt(m.gx))).sum())


def neg_kl(logits, T=1.0):
    """-KL(g || p_T): target conformity of the tempered map to the perfect sigma-1 target g at (u, v) (nats)."""
    m = prep(logits)
    z = (m.L - m.lmax) / T
    logp = z - np.log(np.exp(z).sum())                     # (128, 128) log p_T
    logg = -((GRID[:, None] - m.v) ** 2 + (GRID[None, :] - m.u) ** 2) / 2
    logg = logg - np.log(np.exp(-(GRID - m.v) ** 2 / 2).sum()) - np.log(np.exp(-(GRID - m.u) ** 2 / 2).sum())
    g = np.exp(logg)
    return float(-(g * (logg - logp)).sum())


def conf_sub_far(logits, k=16.0):
    """conf_sub * m9^k: the sub-cell-normalised 5x5 mass with the far mass (outside 9x9) counted (1 + k) times.

    log score ~ -(ring deficit) - (1 + k) * far mass; k = 0 is conf_sub, k -> inf ranks like m9.
    """
    m = prep(logits)
    ys, xs = _win(m, 2)
    y9, x9 = _win(m, 4)
    return float(np.log(m.p[ys, xs].sum() / (m.gy[ys].sum() * m.gx[xs].sum())) + k * np.log(m.p[y9, x9].sum()))


def tempered_conf(logits, T=2.0):
    """5x5 mass around the argmax of softmax(L / T): T > 1 lifts faint far peaks and the background."""
    m = prep(logits)
    pt = np.exp((m.L - m.lmax) / T)
    ys, xs = _win(m, 2)
    return float(pt[ys, xs].sum() / pt.sum())


def margin_out(logits, R=4):
    """Logit margin: max logit minus the best logit at Chebyshev distance > R cells from the argmax."""
    m = prep(logits)
    ys, xs = _win(m, R)
    rest = m.L.copy()
    rest[ys, xs] = -np.inf
    return float(m.lmax - rest.max())


def neg_floor(logits):
    """Minus the log-probability of the median 9x9 local maximum (plateaus merged): the map's background floor."""
    m = prep(logits)
    mf = ndimage.maximum_filter(m.L, size=9, mode="constant", cval=-np.inf)
    lab, n = ndimage.label(m.L >= mf, structure=np.ones((3, 3), bool))  # (128,128) int, one id per plateau
    vals = ndimage.maximum(m.L, lab, np.arange(1, n + 1))                # (n,) one logit per plateau
    return float(-(np.median(vals) - m.lse))


def neg_entropy(logits):
    """Minus the entropy (nats) of the global softmax."""
    m = prep(logits)
    return float((m.p * np.log(np.maximum(m.p, 1e-300))).sum())


def floor_margin(logits):
    """max logit minus the median 9x9 local-max logit (plateaus merged) = -log(floor / main peak)."""
    m = prep(logits)
    mf = ndimage.maximum_filter(m.L, size=9, mode="constant", cval=-np.inf)
    lab, n = ndimage.label(m.L >= mf, structure=np.ones((3, 3), bool))
    return float(m.lmax - np.median(ndimage.maximum(m.L, lab, np.arange(1, n + 1))))


# FINAL: list of (method name, function, kwargs, hyperparameter note); kwargs are the TRAIN-split argmax of
# AUROC>3px over the grids in explore.py / tune_grid.py (README rule 2; val never used for tuning).
FINAL = [
    ("conf (shipped, ref)", conf, {}, "reference: shipped 5x5 mass"),
    ("m9 (ref)", m9, {}, "reference: 9x9 mass"),
    ("sp_conf_sub", conf_sub, {}, "conf / sub-cell ceiling; 0 hp"),
    ("sp_disc_sub r3 T0.85", disc_sub, {"r": 3.0, "T": 0.85}, "disc mass / ideal; r, T tuned on train"),
    ("sp_mom a1 T1.25", neg_moment, {"a": 1.0, "T": 1.25}, "-E_pT[d^a]; a, T tuned on train"),
    ("sp_mom2 (a2 T1)", neg_moment, {"a": 2.0, "T": 1.0}, "-E_p[d^2], untuned reference"),
    ("sp_bc T1.5", bhatt, {"T": 1.5}, "Bhattacharyya overlap with ideal; T tuned on train"),
    ("sp_kl T2", neg_kl, {"T": 2.0}, "-KL(g||p_T); T tuned on train"),
    ("sp_conf T2", tempered_conf, {"T": 2.0}, "5x5 mass of softmax(L/T); T tuned on train"),
    ("sp_conf_sub*m9^16", conf_sub_far, {"k": 16.0}, "log conf_sub + k log m9; k tuned on train"),
    ("sp_neg_entropy", neg_entropy, {}, "-entropy; 0 hp"),
    ("sp_margin out4", margin_out, {"R": 4}, "lmax - best logit outside Chebyshev 4; R tuned on train"),
    ("sp_floor -log(med lm prob)", neg_floor, {}, "coordinator probe: -log median local-max prob; 0 hp"),
    ("sp_floor lmax-med lm", floor_margin, {}, "coordinator probe: -log(floor / main peak); 0 hp"),
]


def score_split(lg, fn, kw):
    """lg (n, 128, 128) float32 memmap -> (n,) float64 scores of fn(map, **kw)."""
    return np.array([fn(lg[i], **kw) for i in range(len(lg))], dtype=np.float64)


def time_method(maps, fn, kw, repeats=3):
    """maps: list of (128, 128) float32 -> best-of-repeats mean CPU ms per map of fn (single thread numpy)."""
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        for mp in maps:
            fn(mp, **kw)
        best = min(best, (time.perf_counter() - t0) / len(maps) * 1e3)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cache", required=True, help="folder with logits_<split>.npy (dump_logits.py)")
    ap.add_argument("--only", nargs="*", default=None, help="method names to run (default: all of FINAL)")
    ap.add_argument("--out", default=str(HERE / "results.jsonl"))
    ap.add_argument("--time-only", action="store_true", help="only time each method (best of 7 x 1000 val maps)")
    args = ap.parse_args()
    sys.path.insert(0, args.bundle)
    import eval_conf as ev
    cache = Path(args.cache)
    lgs = {s: np.load(cache / f"logits_{s}.npy", mmap_mode="r") for s in ("train", "val")}   # (n,128,128) f32
    timing_maps = [np.array(lgs["val"][i]) for i in range(1000)]
    if args.time_only:
        t_prep = time_method(timing_maps, lambda mp: prep(mp), {}, repeats=7)
        print(f"prep (shared softmax + argmax + decode): {t_prep:.3f} ms/map")
        for name, fn, kw, _ in FINAL:
            ms = time_method(timing_maps, fn, kw, repeats=7)
            print(f"{name:<30} {ms:.3f} ms/map  (+{ms - t_prep:.3f} over prep)", flush=True)
        return
    ev.print_header()
    for name, fn, kw, note in FINAL:
        if args.only and name not in args.only:
            continue
        ms = time_method(timing_maps, fn, kw)
        sc = {}
        for split in ("train", "val"):
            sc[split] = score_split(lgs[split], fn, kw)
            ev.evaluate(split, sc[split], name, passes=1, notes=f"{note}; cpu {ms:.3f} ms/map", out=args.out)
        print(f"    cpu {ms:.3f} ms/map", flush=True)
        # every method's scores go to the cache; the best <= 3 are copied to scores_<method>.npz in this folder
        slug = re.sub(r"[^A-Za-z0-9.]+", "_", name).strip("_")
        np.savez(cache / f"scores_{slug}.npz", val=sc["val"].astype(np.float32), train=sc["train"].astype(np.float32))


if __name__ == "__main__":
    main()
