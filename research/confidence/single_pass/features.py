"""Single-pass map statistics of TipNet v9 logit maps: the candidate-feature table every score is built from.

One logit map per image is all a feature may use (passes = 1). Data flow (n = maps in a chunk):

    logits_<split>.npy  float32 (N, 128, 128)       (dump_logits.py)
        |  chunk of n maps -> float64
        v
    L (n, 128, 128)  --global softmax-->  p (n, 128, 128), each map sums to 1
        |  argmax (iy, ix) -> 5x5 logit-softmax centroid (u, v)     = the shipped decode
        |  ideal target g (n, 128, 128): sigma = 1 cell Gaussian centred at (u, v), sums to 1
        v
    feature dict  {name: (N,) float64}   -> <cache>/features_<split>.npz

Feature families (all float64, one value per map; names used by explore.py / scores.py):

    window masses   m{w}       square w x w window around the argmax (w odd), clipped to the map
                    c{r}       disc of radius r cells around the decoded tip (u, v)
                    ic{r}, im{w}  the same masses of the ideal target g (the sub-cell ceiling)
    second peak     lm_n       number of 9x9 local maxima (plateaus merged)
                    lm2        logit of the best local max other than the top one (-1e9 when none)
                    lm2_m5     5x5 mass of that peak (0 when none)
                    lm_med     median logit over all local maxima (the "background floor")
                    out{R}     max logit over cells with Chebyshev distance > R from the argmax
    shape           ent        entropy of p (nats)
                    mom2       sum p * d^2 over the map (d: cells from (u, v))
                    mom2_{R}   the same restricted to d <= R, normalised by the mass inside
                    lam1, lam2 eigenvalues of the p-covariance inside d <= 3 (cells^2), lam1 >= lam2
                    kl_gp      KL(g || p);  tv = 0.5 sum |p - g|;  bc = sum sqrt(p g)
    logit space     lmax, lse  max logit, logsumexp;  q10, q50, q90 logit quantiles over the map
    temperature     m5_T{T}, m9_T{T}  window masses of softmax(L / T)
    decode          u, v       decoded tip, heatmap cells (x = 2u + 0.5)

Usage (from the worktree root):
    python research/confidence/single_pass/features.py --cache <dir with logits_*.npy>
"""

import argparse
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

# H: int, heatmap side (cells)
H = 128
# WINDOWS: tuple[int], odd square-window sides around the argmax cell
WINDOWS = (3, 5, 7, 9, 11, 13, 17, 21, 31)
# RADII: tuple[float], disc radii (cells) around the decoded tip
RADII = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0)
# OUT_R: tuple[int], Chebyshev exclusion radii (cells) for the "best logit outside R" margins
OUT_R = (2, 3, 4, 6, 8, 12)
# TEMPS: tuple[float], softmax temperatures for the tempered masses
TEMPS = (0.5, 0.7, 1.5, 2.0, 3.0)
# PEAK_NB: int, local-maximum neighbourhood (cells), as tip_webapp/app/inference.py local_maxima
PEAK_NB = 9
# GRID: (128,) float64 cell index 0..127
GRID = np.arange(H, dtype=np.float64)


def patch(a, iy, ix, r, fill):
    """(n, H, H) array, centres iy / ix (n,) int -> (n, 2r+1, 2r+1) windows around them, `fill` off the map."""
    n = len(a)
    ap = np.pad(a, ((0, 0), (r, r), (r, r)), constant_values=fill)
    # off: (2r+1,) int window offsets; rows / cols: (n, 2r+1) indices into the padded array
    off = np.arange(2 * r + 1)
    rows, cols = iy[:, None] + off, ix[:, None] + off
    return ap[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]


def box_mass(S, iy, ix, r):
    """Integral image S (n, H+1, H+1) -> (n,) sums over rows iy-r..iy+r, cols ix-r..ix+r clipped to the map."""
    n = len(S)
    y0, y1 = np.clip(iy - r, 0, H), np.clip(iy + r + 1, 0, H)
    x0, x1 = np.clip(ix - r, 0, H), np.clip(ix + r + 1, 0, H)
    k = np.arange(n)
    return S[k, y1, x1] - S[k, y0, x1] - S[k, y1, x0] + S[k, y0, x0]


def integral(p):
    """(n, H, H) -> (n, H+1, H+1) zero-padded 2-D cumulative sum (integral image)."""
    S = np.zeros((len(p), H + 1, H + 1))
    S[:, 1:, 1:] = p.cumsum(1).cumsum(2)
    return S


def decode(L):
    """Shipped decode. L (n, H, H) float64 -> iy, ix (n,) int argmax cell; u, v (n,) float64 5x5 centroid (cells)."""
    n = len(L)
    flat = L.reshape(n, -1).argmax(1)
    iy, ix = flat // H, flat % H
    P = patch(L, iy, ix, 2, -np.inf)                     # (n, 5, 5) logits, -inf off the map
    w = np.exp(P - P.max((1, 2), keepdims=True))
    w /= w.sum((1, 2), keepdims=True)
    off = np.arange(-2, 3, dtype=np.float64)             # (5,) cell offsets
    v = iy + (w.sum(2) * off).sum(1)
    u = ix + (w.sum(1) * off).sum(1)
    return iy, ix, u, v


def local_max_stats(L, iy, ix):
    """9x9 local maxima of each map, plateaus merged (8-connected equal maxima count once).

    L (n, H, H) float64; iy, ix (n,) argmax cell -> dict of (n,) arrays: lm_n count, lm2 best other peak logit,
    lm2_y / lm2_x its cell (-1 when none), lm_med median peak logit (top peak included).
    """
    n = len(L)
    mf = ndimage.maximum_filter(L, size=(1, PEAK_NB, PEAK_NB), mode="constant", cval=-np.inf)
    is_max = L >= mf                                       # (n, H, H) bool
    struct = np.zeros((3, 3, 3), bool)
    struct[1] = True                                       # 8-connectivity inside a map, never across maps
    lab, n_lab = ndimage.label(is_max, structure=struct)  # lab (n, H, H) int32, 0 = background
    idx = np.arange(1, n_lab + 1)
    # one representative cell per component: its first cell in C order
    pos = ndimage.minimum_position(np.arange(lab.size).reshape(lab.shape), lab, idx)
    pos = np.array(pos, dtype=np.int64).reshape(-1, 3)    # (n_lab, 3) (map, y, x)
    m_id, py, px = pos[:, 0], pos[:, 1], pos[:, 2]
    val = L[m_id, py, px]                                  # (n_lab,) peak logit
    top_lab = lab[np.arange(n), iy, ix]                    # (n,) component holding the argmax
    is_top = lab[m_id, py, px] == top_lab[m_id]
    out = {"lm_n": np.bincount(m_id, minlength=n).astype(np.float64),
           "lm2": np.full(n, -1e9), "lm2_y": np.full(n, -1, np.int64), "lm2_x": np.full(n, -1, np.int64),
           "lm_med": np.zeros(n)}
    # best non-top component per map: sort by (map, -value), take the first non-top row of each map
    order = np.lexsort((-val, m_id))
    o_m, o_top = m_id[order], is_top[order]
    cand = order[~o_top]
    first = np.unique(m_id[cand], return_index=True)
    sel = cand[first[1]]
    out["lm2"][first[0]] = val[sel]
    out["lm2_y"][first[0]], out["lm2_x"][first[0]] = py[sel], px[sel]
    # median peak logit per map
    order = np.lexsort((val, m_id))
    starts = np.searchsorted(m_id[order], np.arange(n))
    counts = np.bincount(m_id, minlength=n)
    sv = val[order]
    lo = starts + (counts - 1) // 2
    hi = starts + counts // 2
    out["lm_med"] = 0.5 * (sv[lo] + sv[hi])
    del o_m
    return out


def chunk_features(L):
    """L (n, H, H) float64 logits -> dict {name: (n,) float64} (see the module docstring)."""
    n = len(L)
    f = {}
    iy, ix, u, v = decode(L)
    f["u"], f["v"] = u, v
    lmax = L.max((1, 2))
    lse = lmax + np.log(np.exp(L - lmax[:, None, None]).sum((1, 2)))
    f["lmax"], f["lse"] = lmax, lse
    logp = L - lse[:, None, None]                          # (n, H, H) log-probabilities
    p = np.exp(logp)
    S = integral(p)
    for w in WINDOWS:
        f[f"m{w}"] = box_mass(S, iy, ix, w // 2)
    # d2: (n, H, H) squared distance (cells^2) from the decoded tip
    dy2 = (GRID[None, :] - v[:, None]) ** 2                # (n, H)
    dx2 = (GRID[None, :] - u[:, None]) ** 2                # (n, H)
    d2 = dy2[:, :, None] + dx2[:, None, :]
    # ideal target g, sigma = 1 cell, normalised over the map (separable)
    gy, gx = np.exp(-dy2 / 2), np.exp(-dx2 / 2)
    zy, zx = gy.sum(1), gx.sum(1)
    g = (gy / zy[:, None])[:, :, None] * (gx / zx[:, None])[:, None, :]
    logg = -d2 / 2 - np.log(zy)[:, None, None] - np.log(zx)[:, None, None]
    Sg = integral(g)
    for w in WINDOWS:
        f[f"im{w}"] = box_mass(Sg, iy, ix, w // 2)
    for r in RADII:
        inside = d2 <= r * r
        f[f"c{r:g}"] = (p * inside).sum((1, 2))
        f[f"ic{r:g}"] = (g * inside).sum((1, 2))
    # shape / entropy / conformity
    f["ent"] = -(p * logp).sum((1, 2))
    f["mom2"] = (p * d2).sum((1, 2))
    for R in (2.0, 3.0, 5.0):
        inside = d2 <= R * R
        f[f"mom2_{R:g}"] = (p * d2 * inside).sum((1, 2)) / (p * inside).sum((1, 2))
        f[f"imom2_{R:g}"] = (g * d2 * inside).sum((1, 2)) / (g * inside).sum((1, 2))
    inside = d2 <= 9.0
    pw = p * inside
    mw = pw.sum((1, 2))
    ey = (pw.sum(2) * GRID).sum(1) / mw
    ex = (pw.sum(1) * GRID).sum(1) / mw
    cyy = (pw.sum(2) * (GRID[None] - ey[:, None]) ** 2).sum(1) / mw
    cxx = (pw.sum(1) * (GRID[None] - ex[:, None]) ** 2).sum(1) / mw
    cxy = (pw * (GRID[None, :, None] - ey[:, None, None]) * (GRID[None, None, :] - ex[:, None, None])).sum((1, 2)) / mw
    tr, det = cyy + cxx, cyy * cxx - cxy ** 2
    disc = np.sqrt(np.maximum(tr ** 2 / 4 - det, 0))
    f["lam1"], f["lam2"] = tr / 2 + disc, tr / 2 - disc
    f["kl_gp"] = (g * (logg - logp)).sum((1, 2))
    f["tv"] = 0.5 * np.abs(p - g).sum((1, 2))
    f["bc"] = np.sqrt(p * g).sum((1, 2))
    # logit quantiles
    q = np.quantile(L.reshape(n, -1), [0.1, 0.5, 0.9], axis=1)
    f["q10"], f["q50"], f["q90"] = q
    # local maxima / second peak
    lm = local_max_stats(L, iy, ix)
    f["lm_n"], f["lm2"], f["lm_med"] = lm["lm_n"], lm["lm2"], lm["lm_med"]
    has2 = lm["lm2_y"] >= 0
    m5_2 = np.zeros(n)
    m5_2[has2] = box_mass(S[has2], lm["lm2_y"][has2], lm["lm2_x"][has2], 2)
    f["lm2_m5"] = m5_2
    # best logit outside a Chebyshev radius R of the argmax
    cy = np.abs(GRID[None, :] - iy[:, None])               # (n, H)
    cx = np.abs(GRID[None, :] - ix[:, None])
    cheb = np.maximum(cy[:, :, None], cx[:, None, :])      # (n, H, H)
    for R in OUT_R:
        f[f"out{R}"] = np.where(cheb > R, L, -np.inf).max((1, 2))
    # tempered window masses
    for T in TEMPS:
        z = L / T
        pt = np.exp(z - z.max((1, 2), keepdims=True))
        pt /= pt.sum((1, 2), keepdims=True)
        St = integral(pt)
        f[f"m5_T{T:g}"] = box_mass(St, iy, ix, 2)
        f[f"m9_T{T:g}"] = box_mass(St, iy, ix, 4)
    return f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True, help="folder with logits_<split>.npy; features_<split>.npz go here")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--chunk", type=int, default=250)
    args = ap.parse_args()
    cache = Path(args.cache)
    for split in args.splits:
        lg = np.load(cache / f"logits_{split}.npy", mmap_mode="r")   # (N, H, H) float32 memmap
        t0, parts = time.time(), []
        for s in range(0, len(lg), args.chunk):
            parts.append(chunk_features(np.asarray(lg[s:s + args.chunk], dtype=np.float64)))
        feats = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        np.savez(cache / f"features_{split}.npz", **feats)
        print(f"{split}: {len(lg)} maps, {len(feats)} features in {time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
