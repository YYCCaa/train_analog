"""Stage 1 of 2: every confidence method from the three research agents on one val table, + one combination.

    scores (val, one float per image, higher = more trustworthy)
      baselines   conf, m9                          bundle meta
      single      bc T1.5, conf_sub*m9^16, kl T2    verify_single.npz (mine) / conf-single's npz
      multi       dih mean-map m5 / m9, K = 2, 4, 8 recomputed here (8 dihedral passes)
      learned     LR3 (mine), LR36, CNN             verify_learned.npz / conf-learned's npz
      combo       bc T1.5 on the dihedral mean map  NEW here: g at the CLEAN decoded tip, p_T ∝ M^(1/1.5)
                  (K = 1 is exactly single bc T1.5: a sanity check)
      │
      ├─▶ frozen harness rows            comparison_results.jsonl (overwritten)
      ├─▶ paired bootstrap vs conf       1000 shared resamples: ΔAUROC>3, ΔAP>3 per method
      └─▶ reliability (val, 2-fold OOF)  raw conf / iso(conf) / iso(LR3): predicted vs observed P(err > 3)
    ─▶ comparison_stats.npz  (read by comparison_plots.py)

Run from the repo root.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from verify_learned import pav  # noqa: E402
from verify_multi import inv_code  # noqa: E402
from verify_single import DEV, G, cb, ev, win_sum  # noqa: E402
import train_v9_tip as tv  # noqa: E402

# SINGLE / LEARNED: Path, the research folders whose committed score files are read here
SINGLE = HERE.parent / "single_pass"
LEARNED = HERE.parent / "learned"
# SUBSETS: dict K -> list[int] dihedral codes (conf-multi's train-chosen subsets; 0 = identity)
SUBSETS = {1: [0], 2: [0, 7], 4: [0, 4, 5, 7], 8: list(range(8))}
# T_BC: float, bc temperature (conf-single's train-tuned value; not re-tuned here)
T_BC = 1.5
# N_BOOT: int, paired bootstrap resamples (shared across methods)
N_BOOT = 1000


def clean_tip_target(L):
    """L (B,128,128) float64 clean logits -> iy, ix (B,) argmax; gy, gx (B,128) ideal σ=1 marginals at the tip."""
    B = len(L)
    a = L.reshape(B, -1).argmax(1)
    iy, ix = a // 128, a % 128
    pad = torch.nn.functional.pad(L, (2, 2, 2, 2), value=float("-inf"))
    d = torch.arange(5, device=DEV)
    b = torch.arange(B, device=DEV)[:, None, None]
    w = torch.softmax(pad[b, iy[:, None, None] + d[None, :, None], ix[:, None, None] + d[None, None, :]]
                      .reshape(B, -1), 1).reshape(B, 5, 5)
    off = (d - 2).double()
    v, u = iy.double() + (w.sum(2) * off).sum(1), ix.double() + (w.sum(1) * off).sum(1)
    gy = torch.exp(-(G[None] - v[:, None]) ** 2 / 2)
    gx = torch.exp(-(G[None] - u[:, None]) ** 2 / 2)
    return iy, ix, gy / gy.sum(1, keepdim=True), gx / gx.sum(1, keepdim=True)


def dihedral_scores(model, images):
    """images (n,256,256) uint8 memmap -> dict of (n,) float64: dm5_K, dm9_K, dbc_K for K in SUBSETS."""
    n = len(images)
    out = {f"{k}_{K}": np.empty(n) for k in ("dm5", "dm9", "dbc") for K in SUBSETS}
    with torch.no_grad():
        for s in range(0, n, 256):
            x = torch.from_numpy(np.array(images[s:s + 256])).to(DEV)              # (B,256,256) uint8
            B = len(x)
            back = []                                                               # 8 x (B,128,128) f64
            for k in range(8):
                lg = model(tv.to_input(tv.dihedral_images(x, torch.full((B,), k, device=DEV)))).float()[:, 0]
                lg = lg.double()
                if k == 0:
                    iy, ix, gy, gx = clean_tip_target(lg)
                p = torch.softmax(lg.reshape(B, -1), 1).reshape(B, 128, 128)
                back.append(tv.dihedral_images(p, torch.full((B,), inv_code(k), device=DEV)))
            sg = gy.sqrt()[:, :, None] * gx.sqrt()[:, None, :]                       # (B,128,128) √g
            for K, codes in SUBSETS.items():
                M = torch.stack([back[c] for c in codes]).mean(0)                    # (B,128,128) mean map
                pT = torch.softmax(M.clamp_min(1e-300).log().reshape(B, -1) / T_BC, 1).reshape(B, 128, 128)
                out[f"dm5_{K}"][s:s + B] = win_sum(M, iy, ix, 2).cpu().numpy()
                out[f"dm9_{K}"][s:s + B] = win_sum(M, iy, ix, 4).cpu().numpy()
                out[f"dbc_{K}"][s:s + B] = (pT.sqrt() * sg).sum((1, 2)).cpu().numpy()
    return out


def reliability(score, bad, fold):
    """score (n,), bad (n,) bool, fold (n,) 0/1 -> (n,) float64 2-fold out-of-fold isotonic P(bad)."""
    pred = np.empty(len(score))
    for a in (0, 1):
        kx, ky = pav(score[fold == a], (~bad)[fold == a])
        pred[fold != a] = 1.0 - np.interp(score[fold != a], kx, ky)
    return pred


def binned(pred_bad, bad):
    """Quantile bins of pred_bad (finer in the risky tail; bins < 20 images dropped) -> (mean pred, observed, n)."""
    qs = np.quantile(pred_bad, [0, .5, .75, .9, .95, .97, .98, .99, .995, .998, 1.0])
    idx = np.clip(np.searchsorted(qs, pred_bad, side="right") - 1, 0, len(qs) - 2)
    keep = [b for b in range(len(qs) - 1) if (idx == b).sum() >= 20]
    return (np.array([pred_bad[idx == b].mean() for b in keep]), np.array([bad[idx == b].mean() for b in keep]),
            np.array([(idx == b).sum() for b in keep]))


def main():
    t0 = time.time()
    model = cb.load_model()
    val = cb.load_split("val")
    dih = dihedral_scores(model, val["images"])
    print(f"[forward] val x 8 passes {time.time() - t0:.0f}s")
    vs, vl = np.load(HERE / "verify_single.npz"), np.load(HERE / "verify_learned.npz")
    print(f"[sanity] K=1 combo vs single bc: max diff {np.abs(dih['dbc_1'] - vs['val_bc15']).max():.2e}; "
          f"K=1 dm5 vs conf {np.abs(dih['dm5_1'] - val['conf']).max():.2e}")

    # METHODS: list of (key, label, family, passes, cpu_ms or nan, (n,) score)
    nan = float("nan")
    methods = [
        ("conf", "conf, shipped 5x5 mass", "baseline", 1, 0.09, val["conf"]),
        ("m9", "m9, 9x9 mass", "baseline", 1, 0.10, val["m9"]),
        ("bc", "bc T1.5 (map vs ideal shape)", "single", 1, 0.27, vs["val_bc15"]),
        ("csm", "conf_sub x m9^16", "single", 1, 0.12, vs["val_csm16"]),
        ("kl", "-KL(ideal || map) T2", "single", 1, 0.49, np.load(SINGLE / "scores_kl_T2.npz")["val"]),
        ("lr3", "LR3 (3 features, 4 weights)", "learned", 1, 0.8, vl["val"]),
        ("lr36", "LR36 (36 features)", "learned", 1, 6.7, np.load(LEARNED / "scores_LR36.npz")["val"]),
        ("cnn", "tiny CNN on the map", "learned", 1, 9.0, np.load(LEARNED / "scores_CNN.npz")["val"]),
        ("dm9_2", "dihedral mean-map m9, K=2", "multi", 2, 0.5, dih["dm9_2"]),
        ("dm5_4", "dihedral mean-map m5, K=4", "multi", 4, 0.9, dih["dm5_4"]),
        ("dm5_8", "dihedral mean-map m5, K=8", "multi", 8, 3.1, dih["dm5_8"]),
        ("dm9_8", "dihedral mean-map m9, K=8", "multi", 8, 3.1, dih["dm9_8"]),
        ("dbc_4", "bc T1.5 on dihedral mean map, K=4", "combo", 4, nan, dih["dbc_4"]),
        ("dbc_8", "bc T1.5 on dihedral mean map, K=8", "combo", 8, nan, dih["dbc_8"]),
    ]
    res_path = HERE / "comparison_results.jsonl"
    res_path.unlink(missing_ok=True)
    ev.print_header()
    rows = [ev.evaluate("val", s, lab, passes=p, out=res_path) for _, lab, _, p, _, s in methods]

    # paired bootstrap: the same N_BOOT resamples for every method
    bad, n = val["err"] > 3.0, len(val["err"])
    rng = np.random.default_rng(7)
    idx = rng.integers(0, n, (N_BOOT, n))                                   # (N_BOOT, n) int64
    base_auc = np.array([ev.auroc(val["conf"][i], bad[i]) for i in idx])
    base_ap = np.array([ev.average_precision(val["conf"][i], bad[i]) for i in idx])
    d_auc, d_ap = {}, {}
    for key, lab, _, _, _, s in methods:
        d_auc[key] = np.array([ev.auroc(s[i], bad[i]) for i in idx]) - base_auc
        d_ap[key] = np.array([ev.average_precision(s[i], bad[i]) for i in idx]) - base_ap
        print(f"[boot] {lab:<36} dAUROC3 {np.percentile(d_auc[key], 50):+.4f} "
              f"[{np.percentile(d_auc[key], 2.5):+.4f}, {np.percentile(d_auc[key], 97.5):+.4f}]  "
              f"dAP3 {np.percentile(d_ap[key], 50):+.4f} "
              f"[{np.percentile(d_ap[key], 2.5):+.4f}, {np.percentile(d_ap[key], 97.5):+.4f}]", flush=True)

    # reliability: raw conf read as P(good), and 2-fold OOF isotonic of conf and LR3
    fold = np.arange(n) % 2
    rel = {"raw_conf": binned(1.0 - val["conf"], bad), "iso_conf": binned(reliability(val["conf"], bad, fold), bad),
           "iso_lr3": binned(reliability(vl["val"], bad, fold), bad)}

    np.savez(HERE / "comparison_stats.npz",
             keys=np.array([m[0] for m in methods]), labels=np.array([m[1] for m in methods]),
             families=np.array([m[2] for m in methods]), passes=np.array([m[3] for m in methods]),
             cpu_ms=np.array([m[4] for m in methods]),
             auroc=np.array([r["auroc_err3"] for r in rows]), ap=np.array([r["ap_err3"] for r in rows]),
             flag_all=np.array([r["flag_all_miss"] for r in rows]),
             d_auc=np.stack([d_auc[m[0]] for m in methods]), d_ap=np.stack([d_ap[m[0]] for m in methods]),
             k_curve_K=np.array(list(SUBSETS)),
             k_curve=np.array([[ev.auroc(dih[f"{f}_{K}"], bad) for K in SUBSETS] for f in ("dm5", "dm9", "dbc")]),
             **{f"rel_{k}_{j}": v[j_i] for k, v in rel.items() for j_i, j in enumerate(("pred", "obs", "n"))})
    print(f"[done] {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
