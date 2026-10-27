"""Independent re-check of conf-single: two one-pass scores beat the shipped confidence, and the "floor" is lse.

Written from scratch in torch (no import of the agent's code). Per image, from the clean logits L (128,128):

    argmax (iy, ix) ─▶ 5x5 logit-softmax centroid (u, v) = decoded tip, cells
    g = ideal σ=1 target at (u, v): outer(gy, gx), each marginal normalised over the 128 cells
    bc T1.5      = Σ sqrt(softmax(L / 1.5)) · sqrt(g)                         (1 = the map IS the ideal target)
    conf_sub·m9^16 = log( m5 / g-mass in the same 5x5 window ) + 16 · log m9   (m5 / m9: global-softmax masses)

Floor claim: median local-max prob = exp(floor_logit - lse). If floor_logit barely varies, the "floor" signal
is just -lse (≈ the peak logit). Tested on the val best / worst 1000 of the stability probe (cb.load_stability).
Run from the repo root; prints only (log via redirect).
"""

import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage
from scipy.stats import mannwhitneyu, spearmanr

# BUNDLE: str, the read-only research bundle (data + meta + the harness in ../harness; not in the repo)
BUNDLE = (r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-PycharmProjects-train-dial"
          r"\f292a1d6-f81e-4d2c-b353-1ee00d6e9308\scratchpad\confbundle")
# HERE: Path, this folder;  AGENT: Path, conf-single's folder (its committed score files)
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent / "single_pass"
sys.path.insert(0, BUNDLE)
sys.path.insert(0, str(Path.cwd()))
import confbundle as cb  # noqa: E402
import eval_conf as ev  # noqa: E402

# G: (128,) float64 cell indices on the GPU
DEV = torch.device("cuda")
G = torch.arange(128, dtype=torch.float64, device=DEV)


def win_sum(m, iy, ix, r):
    """m (B,128,128), iy/ix (B,) -> (B,) sum over the (2r+1)^2 window around (iy, ix), clipped at the border."""
    pad = torch.nn.functional.pad(m, (r, r, r, r))
    d = torch.arange(2 * r + 1, device=m.device)
    b = torch.arange(len(m), device=m.device)[:, None, None]
    return pad[b, iy[:, None, None] + d[None, :, None], ix[:, None, None] + d[None, None, :]].sum(dim=(1, 2))


def vec_sum(v, i, r):
    """v (B,128), i (B,) -> (B,) sum of v over indices i-r..i+r, clipped (the 1-D marginal of a clipped window)."""
    pad = torch.nn.functional.pad(v, (r, r))
    d = torch.arange(2 * r + 1, device=v.device)
    return pad[torch.arange(len(v), device=v.device)[:, None], i[:, None] + d[None, :]].sum(1)


def scores(L):
    """L (B,128,128) float64 logits on DEV -> dict of (B,) numpy float64: conf, bc15, csm16, lse, lmax."""
    B = len(L)
    flat = L.reshape(B, -1)
    a = flat.argmax(1)
    iy, ix = a // 128, a % 128
    lse = torch.logsumexp(flat, 1)
    p = torch.exp(L - lse[:, None, None])                               # (B,128,128) global softmax
    # decoded tip: softmax of the 5x5 logits around the argmax (off-map cells = -inf), centroid in cells
    pad = torch.nn.functional.pad(L, (2, 2, 2, 2), value=float("-inf"))
    d = torch.arange(5, device=DEV)
    b = torch.arange(B, device=DEV)[:, None, None]
    w = torch.softmax(pad[b, iy[:, None, None] + d[None, :, None], ix[:, None, None] + d[None, None, :]]
                      .reshape(B, -1), 1).reshape(B, 5, 5)
    off = (d - 2).double()
    v = iy.double() + (w.sum(2) * off).sum(1)
    u = ix.double() + (w.sum(1) * off).sum(1)
    gy = torch.exp(-(G[None] - v[:, None]) ** 2 / 2); gy = gy / gy.sum(1, keepdim=True)   # (B,128)
    gx = torch.exp(-(G[None] - u[:, None]) ** 2 / 2); gx = gx / gx.sum(1, keepdim=True)   # (B,128)
    pT = torch.softmax(flat / 1.5, 1).reshape(B, 128, 128)
    bc = (pT.sqrt() * gy.sqrt()[:, :, None] * gx.sqrt()[:, None, :]).sum(dim=(1, 2))
    m5, m9 = win_sum(p, iy, ix, 2), win_sum(p, iy, ix, 4)
    ideal5 = vec_sum(gy, iy, 2) * vec_sum(gx, ix, 2)
    csm = torch.log(m5 / ideal5) + 16.0 * torch.log(m9)
    return {k: t.cpu().numpy() for k, t in (("conf", m5), ("bc15", bc), ("csm16", csm), ("lse", lse),
                                              ("lmax", flat.max(1).values))}


def run_split(model, split):
    """Score every image of a bundle split -> (meta dict, dict of (n,) float64 arrays)."""
    d = cb.load_split(split)
    acc = {}
    for s in range(0, len(d["err"]), 512):
        lg = torch.from_numpy(cb.logits(model, d["images"][s:s + 512])).to(DEV).double()
        for k, v in scores(lg).items():
            acc.setdefault(k, []).append(v)
    return d, {k: np.concatenate(v) for k, v in acc.items()}


def main():
    model = cb.load_model()
    out = {}
    ev.print_header()
    for split in ("train", "val"):
        d, s = run_split(model, split)
        out[split] = s
        print(f"[sanity {split}] max |my conf - bundle conf| = {np.abs(s['conf'] - d['conf']).max():.2e}")
        for mine, fname in (("bc15", "scores_bc_T1.5.npz"), ("csm16", "scores_conf_sub_m9k16.npz")):
            theirs = np.load(AGENT / fname)[split]
            print(f"[agree {split}] {mine}: max |mine - agent| = {np.abs(s[mine] - theirs).max():.2e}, "
                  f"spearman {spearmanr(s[mine], theirs).statistic:.6f}")
        ev.evaluate(split, d["conf"], "conf (shipped)")
        ev.evaluate(split, s["bc15"], "MINE bc T1.5")
        ev.evaluate(split, s["csm16"], "MINE conf_sub*m9^16")
        ev.evaluate(split, -s["lse"], "MINE -lse (probe)")
        ev.evaluate(split, s["lmax"], "MINE lmax (probe)")

    # paired bootstrap of the gain over conf on val
    d = cb.load_split("val")
    bad, n, rng = d["err"] > 3.0, len(d["err"]), np.random.default_rng(1)
    for name in ("bc15", "csm16"):
        da, dp = [], []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            da.append(ev.auroc(out["val"][name][i], bad[i]) - ev.auroc(d["conf"][i], bad[i]))
            dp.append(ev.average_precision(out["val"][name][i], bad[i]) - ev.average_precision(d["conf"][i], bad[i]))
        print(f"[boot] {name} - conf: dAUROC3 {np.mean(da):+.4f} [{np.percentile(da, 2.5):+.4f}, "
              f"{np.percentile(da, 97.5):+.4f}]   dAP3 {np.mean(dp):+.4f} [{np.percentile(dp, 2.5):+.4f}, "
              f"{np.percentile(dp, 97.5):+.4f}]")

    # floor claim, on the best / worst 1000: log(median local-max prob) = floor_logit - lse
    stab = cb.load_stability()
    pos, g = stab["pos"], stab["group"]                                 # (2000,) val positions, 0 best / 1 worst
    lg = cb.logits(model, d["images"][np.sort(pos)])
    lg = lg[np.argsort(np.argsort(pos))]                                # back to stability-probe order
    floor = np.empty(len(pos))                                          # (2000,) median 9x9 local-max logit
    for k, L in enumerate(lg.astype(np.float64)):
        mf = ndimage.maximum_filter(L, size=9, mode="constant", cval=-np.inf)
        lab, m = ndimage.label(L >= mf, structure=np.ones((3, 3), bool))  # plateaus merged into one peak
        floor[k] = np.median(ndimage.maximum(L, lab, np.arange(1, m + 1)))
    lse = out["val"]["lse"][pos]
    hmed = np.exp(floor - lse)                                          # (2000,) median-peak probability
    print(f"[floor] sd over 2000: floor logit {floor.std():.2f}, lse {lse.std():.2f}; "
          f"spearman(log hmed, -lse) {spearmanr(np.log(hmed), -lse).statistic:.3f}")
    for nm, x in (("median-peak prob (hmed)", hmed), ("floor logit alone", floor), ("-lse alone", -lse)):
        auc = mannwhitneyu(x[g == 1], x[g == 0]).statistic / ((g == 1).sum() * (g == 0).sum())
        print(f"[floor] P(worst > best) for {nm:<24} {auc:.3f}")
    np.savez(HERE / "verify_single.npz", **{f"{sp}_{k}": v for sp in out for k, v in out[sp].items()}, floor=floor)


if __name__ == "__main__":
    main()
