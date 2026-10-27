"""Independent re-check of conf-multi's claim: dihedral mean-map mass beats the shipped confidence.

Written from scratch (no import of the agent's code). For every val image and dihedral code k:

    img ──T_k──▶ TipNet ──▶ logits_k ──softmax over all cells──▶ P_k (128,128), Σ=1
        P_k ──T_k^-1──▶ back in the clean frame          (heatmap flips/rotations are exact: cell u ↔ 127-u)
    M_K = mean of the K back-mapped maps                  (K = 2: codes {0,7}; 4: {0,4,5,7}; 8: all)
    m5_K / m9_K = Σ M_K over the 5x5 / 9x9 cells around the CLEAN argmax (clipped at the border)
    TTA tip = decode_logits(log M_8)                       (argmax -> 5x5 softmax centroid of log M)

Then: max |mine - agent npz|, the frozen harness on mine, a paired bootstrap of ΔAUROC>3px / ΔAP>3px vs conf,
and the TTA tip's p99 / >7px count. Run from the repo root; prints only (log via redirect).
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

# BUNDLE: str, the read-only research bundle (data + meta + the harness in ../harness; not in the repo)
BUNDLE = (r"C:\Users\ADMINI~1\AppData\Local\Temp\claude\C--Users-Administrator-PycharmProjects-train-dial"
          r"\f292a1d6-f81e-4d2c-b353-1ee00d6e9308\scratchpad\confbundle")
# HERE: Path, this folder;  AGENT: Path, conf-multi's folder (its committed score files)
HERE = Path(__file__).resolve().parent
AGENT = HERE.parent / "multi_pass"
sys.path.insert(0, BUNDLE)
sys.path.insert(0, str(Path.cwd()))
import confbundle as cb  # noqa: E402
import eval_conf as ev  # noqa: E402
import train_v9_tip as tv  # noqa: E402

# SUBSETS: dict K -> list[int] dihedral codes (code 0 = identity first), the agent's train-chosen subsets
SUBSETS = {2: [0, 7], 4: [0, 4, 5, 7], 8: list(range(8))}
# BATCH: int, images per forward pass (bundle rule: <= 512)
BATCH = 256


def inv_code(k):
    """Dihedral code k (int) -> code of its inverse: rotations invert to (4-k)%4, flip+rot are involutions."""
    return (4 - k) % 4 if k < 4 else k


def window_sum(maps, iy, ix, r):
    """maps (B,128,128) float64, iy/ix (B,) int64 -> (B,) float64 sum over the (2r+1)^2 cells, clipped."""
    pad = torch.nn.functional.pad(maps, (r, r, r, r))      # zeros outside the map
    d = torch.arange(2 * r + 1, device=maps.device)
    yy = (iy[:, None, None] + d[None, :, None])             # (B, w, 1) padded rows
    xx = (ix[:, None, None] + d[None, None, :])             # (B, 1, w) padded cols
    b = torch.arange(len(maps), device=maps.device)[:, None, None]
    return pad[b, yy, xx].sum(dim=(1, 2))


def main():
    dev = torch.device("cuda")
    model = cb.load_model()
    val = cb.load_split("val")
    n = len(val["err"])
    # self-test of the inverse codes on a random map: T_k^-1(T_k(a)) == a for all 8 codes
    a = torch.rand(8, 128, 128, device=dev)
    ks = torch.arange(8, device=dev)
    back = tv.dihedral_images(tv.dihedral_images(a, ks), torch.tensor([inv_code(k) for k in range(8)], device=dev))
    assert torch.equal(back, a), "inverse dihedral codes are wrong"

    out = {f"m{w}_k{K}": np.empty(n) for w in (5, 9) for K in SUBSETS}   # (n,) float64 each
    conf_mine = np.empty(n)                                              # (n,) clean 5x5 mass, vs bundle conf
    tta_tip = np.empty((n, 2))                                           # (n, 2) image px, decode of log M_8
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, BATCH):
            x = torch.from_numpy(np.ascontiguousarray(val["images"][s:s + BATCH])).to(dev)   # (B,256,256) u8
            B = len(x)
            back_maps = []                                               # 8 x (B,128,128) float64, clean frame
            for k in range(8):
                kk = torch.full((B,), k, dtype=torch.int64, device=dev)
                lg = model(tv.to_input(tv.dihedral_images(x, kk))).float()[:, 0]          # (B,128,128)
                p = torch.softmax(lg.double().reshape(B, -1), dim=1).reshape(B, 128, 128)
                back_maps.append(tv.dihedral_images(p, torch.full((B,), inv_code(k), dtype=torch.int64,
                                                                  device=dev)))
                if k == 0:
                    flat = lg.reshape(B, -1).argmax(dim=1)
                    iy, ix = flat // 128, flat % 128                       # (B,) clean argmax cell
            conf_mine[s:s + B] = window_sum(back_maps[0], iy, ix, 2).cpu().numpy()
            for K, codes in SUBSETS.items():
                M = torch.stack([back_maps[c] for c in codes]).mean(0)    # (B,128,128) mean map
                out[f"m5_k{K}"][s:s + B] = window_sum(M, iy, ix, 2).cpu().numpy()
                out[f"m9_k{K}"][s:s + B] = window_sum(M, iy, ix, 4).cpu().numpy()
                if K == 8:
                    tta_tip[s:s + B] = tv.decode_logits(torch.log(M.clamp_min(1e-300)).float()[:, None]).cpu().numpy()
    print(f"[forward] {n} images x 8 passes  {time.time() - t0:.0f}s")
    print(f"[sanity] max |my clean 5x5 mass - bundle conf| = {np.abs(conf_mine - val['conf']).max():.2e}")

    # 1. agreement with the agent's committed score files
    for name in ("m5_k8", "m9_k8", "m9_k2"):
        theirs = np.load(AGENT / f"scores_dih_meanmap_{name}.npz")["val"]
        print(f"[agree] {name}: max |mine - agent| = {np.abs(out[name] - theirs).max():.2e}, "
              f"corr {np.corrcoef(out[name], theirs)[0, 1]:.6f}")

    # 2. the frozen harness on my own scores
    ev.print_header()
    ev.evaluate("val", val["conf"], "conf (shipped)")
    ev.evaluate("val", val["m9"], "m9")
    for K in SUBSETS:
        for w in (5, 9):
            ev.evaluate("val", out[f"m{w}_k{K}"], f"MINE dih mean-map m{w} K={K}", passes=K)

    # 3. paired bootstrap of the gain over the baseline (resample images, both scores together)
    err = val["err"]
    bad = err > 3.0
    rng = np.random.default_rng(1)
    for new, base, bname in (("m5_k8", val["conf"], "conf"), ("m5_k4", val["conf"], "conf"),
                             ("m9_k8", val["m9"], "m9"), ("m9_k2", val["m9"], "m9")):
        d_auc, d_ap = [], []
        for _ in range(1000):
            i = rng.integers(0, n, n)
            d_auc.append(ev.auroc(out[new][i], bad[i]) - ev.auroc(base[i], bad[i]))
            d_ap.append(ev.average_precision(out[new][i], bad[i]) - ev.average_precision(base[i], bad[i]))
        lo, hi = np.percentile(d_auc, [2.5, 97.5])
        plo, phi = np.percentile(d_ap, [2.5, 97.5])
        print(f"[boot] {new} - {bname}: dAUROC3 {np.mean(d_auc):+.4f} [{lo:+.4f}, {hi:+.4f}]   "
              f"dAP3 {np.mean(d_ap):+.4f} [{plo:+.4f}, {phi:+.4f}]")

    # 4. the K=8 mean-map tip
    tta_err = np.linalg.norm(tta_tip - val["label"], axis=1)
    for nm, e in (("clean tip", err), ("K=8 mean-map tip", tta_err)):
        print(f"[tip] {nm:<18} p99 {np.percentile(e, 99):.3f}  >3px {int((e > 3).sum())}  >7px {int((e > 7).sum())}")
    np.savez(HERE / "verify_multi.npz", tta_err=tta_err, conf_mine=conf_mine, **out)


if __name__ == "__main__":
    main()
