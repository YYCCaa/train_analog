"""Cheap per-image features of a TipNet v9 logit map, for learned confidence models (family 3).

    logits (B, 128, 128) float32 ──map_features()──▶ feats (B, F) float32   (F = len(FEATURE_NAMES))
                                 └─────────────────▶ aux dict: tip (B, 2) image px, iy / ix (B,) argmax cell

Everything is one-pass map statistics of the normal prediction (passes = 1). It runs batched in torch on
CPU or CUDA; the CPU cost per image is measured with B = 1 in fit_feats.py.

Feature groups (all heights / masses are of the GLOBAL softmax probs p = softmax over all 16384 cells):

    window masses   m3 .. m25 around the argmax cell, as logit(m) = log m - log(1 - m)
    ceiling         logit(m5) - logit(ceil5): ceil5 = 5x5 mass a perfect sigma-1 Gaussian would have at the
                    decoded sub-cell offset (the shipped conf cannot reach 1, its ceiling is 0.962-0.982)
    peaks           9x9 local maxima of p (the webapp's candidates): top-3 runner 5x5 masses (log), runner
                    distance, logit margin max - runner, number of candidates over 1e-3 / 1e-2
    floor           median height of all local maxima (log) and floor / main peak: the map's background level
    spread          entropy, E_p[distance to argmax], far-weighted distance, log-prob quantiles
    shape           peak curvature in x / y, decode-window entropy, sub-cell offset (du, dv), |offset|
    position        tip position relative to the image centre, raw max logit and logsumexp
"""

import math

import torch
import torch.nn.functional as F

# HM: int, heatmap side (cells)
HM = 128
# WINDOWS: tuple[int], odd window sides (cells) of the argmax-centred masses
WINDOWS = (3, 5, 7, 9, 13, 17, 25)
# PEAK_NB: int, a local maximum is the largest p in its PEAK_NB x PEAK_NB window (webapp PEAK_NEIGHBOURHOOD)
PEAK_NB = 9
# N_RUNNERS: int, how many runner-up peaks are described
N_RUNNERS = 3
# EPS: float, floor inside logs of masses / heights
EPS = 1e-12

# FEATURE_NAMES: list[str], column names of the feature matrix, in order
FEATURE_NAMES = (
    [f"lgt_m{w}" for w in WINDOWS]
    + ["lgt_m5_minus_ceil", "log_far9", "log_far17", "log_peak", "log_m5", "log_m9"]
    + [f"log_run{k + 1}" for k in range(N_RUNNERS)]
    + ["run1_dist", "run1_margin", "n_cand_1e3", "n_cand_1e2", "log_runsum"]
    + ["log_lmax_median", "log_floor_over_peak"]
    + ["entropy", "mean_dist", "far_dist", "logp_q50", "logp_q90", "logp_q99", "log_n_supp"]
    + ["curv_x", "curv_y", "win_entropy", "du", "dv", "abs_off"]
    + ["pos_x", "pos_y", "pos_r", "lmax_raw", "lse_raw"]
)


def _box_sums(S, iy, ix, r):
    """Sums of p in the (2r+1)^2 window around (iy, ix), clipped to the map.

    S (B, HM+1, HM+1) float64 integral image of p; iy, ix (B,) int64 -> (B,) float64.
    """
    b = torch.arange(S.shape[0], device=S.device)
    y0, y1 = (iy - r).clamp(0, HM), (iy + r + 1).clamp(0, HM)
    x0, x1 = (ix - r).clamp(0, HM), (ix + r + 1).clamp(0, HM)
    return S[b, y1, x1] - S[b, y0, x1] - S[b, y1, x0] + S[b, y0, x0]


def _ceil5(off):
    """5x5 mass fraction of a sigma-1 Gaussian centred `off` cells from the window centre, per axis.

    off (B,) float64 sub-cell offset in [-2, 2] -> (B,) float64 in (0, 1].
    """
    k = torch.arange(-64, 65, device=off.device, dtype=torch.float64)       # (129,) cell offsets
    g = torch.exp(-(k[None] - off[:, None]) ** 2 / 2.0)                      # (B, 129)
    inside = (k.abs() <= 2).to(torch.float64)                                # (129,) 1 inside the 5x5 window
    return (g * inside).sum(1) / g.sum(1)


def _logit(m):
    """logit of a mass in [0, 1] (B,) float64 -> (B,) float64, clipped away from 0 and 1."""
    m = m.clamp(EPS, 1 - 1e-15)
    return torch.log(m) - torch.log1p(-m)


@torch.no_grad()
def map_features(logits):
    """Feature matrix of a batch of logit maps.

    logits: (B, 128, 128) float32 tensor (any device).
    Returns feats (B, F) float32 tensor and aux dict {tip (B, 2) float64 image px, iy (B,), ix (B,) int64,
    logp (B, 128, 128) float32 global log-probs}.
    """
    L = logits.float()
    B = L.shape[0]
    dev = L.device
    flat = L.reshape(B, -1)                                                  # (B, 16384) float32
    # float64 log-softmax: 1 - m9 is ~1e-5, and float32 probs (sum 1 +- 1e-6) would move logit(m9) by ~0.1
    lse = torch.logsumexp(flat.double(), dim=1)                              # (B,) float64
    logp64 = L.double() - lse[:, None, None]                                 # (B, 128, 128) float64
    logp = logp64.float()                                                    # (B, 128, 128) float32 (returned)
    p = logp64.exp()                                                         # (B, 128, 128) float64
    am = flat.argmax(dim=1)                                                  # (B,) int64 row-major argmax
    iy, ix = am // HM, am % HM                                               # (B,) int64 cell
    bidx = torch.arange(B, device=dev)                                       # (B,) int64

    # integral image S (B, 129, 129) float64 of p, for clipped box sums
    S = F.pad(p.cumsum(1).cumsum(2), (1, 0, 1, 0))
    masses = {w: _box_sums(S, iy, ix, w // 2) for w in WINDOWS}              # w -> (B,) float64

    # decode window (5x5 logits softmax, off-map cells weigh 0) -> sub-cell offset du, dv and entropy
    pad = F.pad(L, (2, 2, 2, 2), value=float("-inf"))                        # (B, 132, 132)
    d = torch.arange(-2, 3, device=dev)                                      # (5,) int64
    yy = (iy[:, None, None] + d[None, :, None] + 2).expand(B, 5, 5)
    xx = (ix[:, None, None] + d[None, None, :] + 2).expand(B, 5, 5)
    patch = pad[bidx[:, None, None].expand(B, 5, 5), yy, xx].reshape(B, -1).double()  # (B, 25)
    wts = torch.softmax(patch, dim=1)                                        # (B, 25) float64
    w55 = wts.reshape(B, 5, 5)
    dv = (w55.sum(2) * d.double()[None]).sum(1)                              # (B,) row offset, cells
    du = (w55.sum(1) * d.double()[None]).sum(1)                              # (B,) col offset, cells
    win_ent = -(wts * torch.log(wts.clamp_min(1e-300))).sum(1)               # (B,) nats
    tip = torch.stack([2 * (ix.double() + du) + 0.5, 2 * (iy.double() + dv) + 0.5], 1)  # (B, 2) image px
    ceil5 = _ceil5(du) * _ceil5(dv)                                          # (B,) float64

    # local maxima of p over a 9x9 window (webapp candidates) and their 5x5 masses
    pf = p.float()[:, None]                                                  # (B, 1, 128, 128) float32
    pooled = F.max_pool2d(pf, PEAK_NB, stride=1, padding=PEAK_NB // 2)[:, 0]
    is_max = p.float() >= pooled                                             # (B, 128, 128) bool
    m5map = F.avg_pool2d(pf, 5, stride=1, padding=2, count_include_pad=True)[:, 0].double() * 25  # (B,128,128)
    gy, gx = torch.meshgrid(torch.arange(HM, device=dev), torch.arange(HM, device=dev), indexing="ij")
    cheb = torch.maximum((gy[None] - iy[:, None, None]).abs(), (gx[None] - ix[:, None, None]).abs())  # (B,128,128)
    runner_ok = is_max & (cheb > PEAK_NB // 2)                               # other peaks, outside the main 9x9
    run_m = torch.where(runner_ok, m5map, torch.zeros_like(m5map)).reshape(B, -1)  # (B, 16384) float64
    top_m, top_i = run_m.topk(N_RUNNERS, dim=1)                              # (B, 3) masses, flat indices
    r1y, r1x = top_i[:, 0] // HM, top_i[:, 0] % HM
    run1_dist = torch.sqrt(((r1y - iy) ** 2 + (r1x - ix) ** 2).double())     # (B,) cells
    run1_margin = (flat.gather(1, am[:, None]) - flat.gather(1, top_i[:, :1]))[:, 0].double()  # (B,) logits
    has_run = top_m[:, 0] > 0
    run1_dist = torch.where(has_run, run1_dist, torch.full_like(run1_dist, 0.0))
    run1_margin = torch.where(has_run, run1_margin, torch.full_like(run1_margin, 40.0))
    n_c3 = ((run_m > 1e-3).sum(1)).double()
    n_c2 = ((run_m > 1e-2).sum(1)).double()

    # background floor: median height of all local maxima (plateaus counted per cell), relative to the peak
    heights = torch.where(is_max, p, torch.full_like(p, float("nan"))).reshape(B, -1)  # (B, 16384) float64
    lmax_med = torch.nanmedian(heights, dim=1).values                        # (B,) float64
    peak = p.reshape(B, -1).gather(1, am[:, None])[:, 0]                     # (B,) float64

    # spread / entropy / quantiles of log-probs
    ent = -(p * logp64).sum((1, 2))                                          # (B,) nats
    dist = torch.sqrt(((gy[None] - iy[:, None, None]) ** 2 + (gx[None] - ix[:, None, None]) ** 2).double())
    mean_dist = (p * dist).sum((1, 2))                                       # (B,) cells
    far_dist = (p * dist * (cheb > 4)).sum((1, 2))                           # (B,) cells, outside the 9x9
    q = torch.quantile(logp.reshape(B, -1).float(), torch.tensor([0.5, 0.9, 0.99], device=dev), dim=1).T.double()
    n_supp = (p > 1e-4).sum((1, 2)).double()

    # peak curvature (logits), cells off the map replaced by the centre value
    def at(dy, dx):
        """Logit at (iy+dy, ix+dx) clamped to the map -> (B,) float64."""
        return L[bidx, (iy + dy).clamp(0, HM - 1), (ix + dx).clamp(0, HM - 1)].double()
    c = at(0, 0)
    curv_x = 2 * c - at(0, -1) - at(0, 1)
    curv_y = 2 * c - at(-1, 0) - at(1, 0)

    cols = ([_logit(masses[w]) for w in WINDOWS]
            + [_logit(masses[5]) - _logit(ceil5), torch.log((1 - masses[9]).clamp_min(EPS)), torch.log((1 - masses[17]).clamp_min(EPS)),
               torch.log(peak + EPS), torch.log(masses[5] + EPS), torch.log(masses[9] + EPS)]
            + [torch.log(top_m[:, k] + EPS) for k in range(N_RUNNERS)]
            + [run1_dist, run1_margin, n_c3, n_c2, torch.log(top_m.sum(1) + EPS)]
            + [torch.log(lmax_med + EPS), torch.log(lmax_med + EPS) - torch.log(peak + EPS)]
            + [ent, mean_dist, far_dist, q[:, 0], q[:, 1], q[:, 2], torch.log1p(n_supp)]
            + [curv_x, curv_y, win_ent, du, dv, du.abs() + dv.abs()]
            + [tip[:, 0] / 128 - 1, tip[:, 1] / 128 - 1, torch.sqrt((tip[:, 0] - 128) ** 2 + (tip[:, 1] - 128) ** 2) / 128,
               c, lse.double()])
    feats = torch.stack(cols, 1).float()                                     # (B, F) float32
    assert feats.shape[1] == len(FEATURE_NAMES)
    return feats, {"tip": tip, "iy": iy, "ix": ix, "logp": logp}


def window_mass_check():
    """Self-check of the box sums on a random map (no data needed) -> max abs error (float)."""
    g = torch.Generator().manual_seed(0)
    L = torch.randn(3, HM, HM, generator=g) * 3
    f, aux = map_features(L)
    p = torch.softmax(L.reshape(3, -1), 1).reshape(3, HM, HM).double()
    errs = []
    for b in range(3):
        iy, ix = int(aux["iy"][b]), int(aux["ix"][b])
        m5 = p[b, max(0, iy - 2):iy + 3, max(0, ix - 2):ix + 3].sum()
        errs.append(abs(math.log(float(m5)) - float(f[b, FEATURE_NAMES.index("log_m5")])))
    return max(errs)


if __name__ == "__main__":
    print("features:", len(FEATURE_NAMES), "box-sum self-check err:", window_mass_check())
