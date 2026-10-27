"""Hyperparameter grid for the best single-pass families, scored on the TRAIN split only (README rule 2).

Families (p_T = softmax(L / T) over all cells; d = distance in cells from the shipped decoded tip (u, v);
g = ideal sigma = 1 cell target centred at (u, v)):

    mom(a, T)   = -sum p_T * d^a              predicted |error|^a under the map (higher = trustworthy)
    bc(T)       =  sum sqrt(p_T * g)          Bhattacharyya overlap with the ideal target
    kl(T)       = -KL(g || p_T)               target conformity
    disc(r, T)  =  sum_{d <= r} p_T / sum_{d <= r} g      disc mass over its sub-cell ceiling

    logits_train.npy (20000, 128, 128) float32 --chunks--> stats {(family, params): (20000,) float64}
        --> table: family params | AUROC>3px | AP>3px | rho      (eval_conf metric functions)

Usage (from the worktree root):
    python research/confidence/single_pass/tune_grid.py --bundle <bundle dir> --cache <cache dir> [--split train]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from features import GRID, decode

# TEMPS: tuple[float], softmax temperatures tried
TEMPS = (0.5, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)
# ALPHAS: tuple[float], distance exponents of the moment family
ALPHAS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
# DISC_R: tuple[float], disc radii (cells) of the normalised disc mass
DISC_R = (2.0, 3.0, 4.0, 5.0, 6.0)


def chunk_stats(L):
    """L (n, 128, 128) float64 logits -> {(family, a, T): (n,) float64}, higher = trustworthy."""
    _, _, u, v = decode(L)
    dy2 = (GRID[None, :] - v[:, None]) ** 2                # (n, 128) cells^2
    dx2 = (GRID[None, :] - u[:, None]) ** 2
    d2 = dy2[:, :, None] + dx2[:, None, :]                 # (n, 128, 128)
    d = np.sqrt(d2)
    gy, gx = np.exp(-dy2 / 2), np.exp(-dx2 / 2)
    g = (gy / gy.sum(1, keepdims=True))[:, :, None] * (gx / gx.sum(1, keepdims=True))[:, None, :]
    logg = np.log(np.maximum(g, 1e-300))
    pw = {a: d ** a for a in ALPHAS}                       # {a: (n, 128, 128)} distance powers
    disc = {r: d2 <= r * r for r in DISC_R}                # {r: (n, 128, 128) bool}
    gdisc = {r: (g * disc[r]).sum((1, 2)) for r in DISC_R}
    out = {}
    for T in TEMPS:
        z = L / T
        z = z - z.max((1, 2), keepdims=True)
        lse = np.log(np.exp(z).sum((1, 2)))
        logp = z - lse[:, None, None]
        p = np.exp(logp)
        for a in ALPHAS:
            out[("mom", a, T)] = -(p * pw[a]).sum((1, 2))
        out[("bc", 0, T)] = np.sqrt(p * g).sum((1, 2))
        out[("kl", 0, T)] = -(g * (logg - logp)).sum((1, 2))
        for r in DISC_R:
            out[("disc", r, T)] = (p * disc[r]).sum((1, 2)) / gdisc[r]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--split", default="train", choices=["train"], help="tuning split (train only)")
    ap.add_argument("--chunk", type=int, default=200)
    args = ap.parse_args()
    sys.path.insert(0, args.bundle)
    import eval_conf as ev
    lg = np.load(Path(args.cache) / f"logits_{args.split}.npy", mmap_mode="r")   # (N, 128, 128) float32
    err = np.load(Path(args.bundle) / f"{args.split}_meta.npz")["err"]          # (N,) float64 image px
    t0, parts = time.time(), []
    for s in range(0, len(lg), args.chunk):
        parts.append(chunk_stats(np.asarray(lg[s:s + args.chunk], dtype=np.float64)))
    stats = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez(Path(args.cache) / f"grid_{args.split}.npz", **{f"{k[0]}|{k[1]:g}|{k[2]:g}": v for k, v in stats.items()})
    print(f"{len(stats)} grid scores in {time.time() - t0:.1f} s")
    bad3 = err > ev.BAD_PX
    rows = [(k, ev.auroc(v, bad3), ev.average_precision(v, bad3), spearmanr(v, -err).statistic)
            for k, v in stats.items()]
    rows.sort(key=lambda r: -r[1])
    print(f"{'family':<6}{'a/r':>5}{'T':>6}{'AUROC>3':>9}{'AP>3':>8}{'rho':>8}")
    for k, a, p, r in rows:
        print(f"{k[0]:<6}{k[1]:>5g}{k[2]:>6g}{a:>9.4f}{p:>8.3f}{r:>8.3f}")


if __name__ == "__main__":
    main()
