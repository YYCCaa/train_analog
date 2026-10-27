"""Multi-pass consistency confidence scores for TipNet v9 (family 2), computed from mp_features.py output.

Every score is one float per image, HIGHER = MORE TRUSTWORTHY. K = forward passes per image, the clean
pass included; a family's K-pass set is the clean pass + its first K-1 extra transforms (mp_features.py).

    feat_<split>.npz ──score(f, fam, K, name)──▶ (n,) float64 ──eval_conf.evaluate──▶ results.jsonl

Score names (d_j = |tip_j - clean tip| px of extra pass j, mapped back to the clean frame):
    spread_max   -max_j d_j
    spread_rms   -RMS distance of all K tips to their coordinate-wise median
    agreeR       fraction of extras with d_j <= R px (R = 1, 2, 3), ties broken by 1e-3 * conf
    avgm5c       5x5 mass of the K-pass MEAN probability map at the clean argmax cell
                 (= mean of the per-pass masses there: the window sum is linear in the map)
    avgm9c       same with the 9x9 window
    minm5c/minm9c  min over the K passes of that 5x5 / 9x9 mass
    avgown5/9    5x5 / 9x9 mass of the mean map at ITS OWN argmax (exact families only)
    combo        base * exp(-spread_max / tau), base in conf | m9 | avgm9c

Final mode (evaluates the final methods on both splits, appends results.jsonl, writes scores_<method>.npz):
    python research/confidence/multi_pass/mp_scores.py --final
CPU cost of the post-processing (numpy port, per image, given the K logit maps):
    python research/confidence/multi_pass/mp_scores.py --time
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# HERE / CACHE: Path, this folder and its (uncommitted) feature cache
HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
sys.path.insert(0, str(HERE))
from mp_features import BUNDLE, DIH_ORDER, FAMILIES, KS  # noqa: E402,F401
sys.path.insert(0, str(BUNDLE))
import eval_conf as ev  # noqa: E402

# RESULTS: Path, the harness rows of every final method (both splits)
RESULTS = HERE / "results.jsonl"


def load_feat(path):
    """npz from mp_features.extract -> dict name -> numpy array (float64 for the float features)."""
    d = dict(np.load(path))
    return {k: (v.astype(np.float64) if v.dtype == np.float32 else v) for k, v in d.items()}


def load_meta(split, n=None):
    """Bundle meta of `split` (err (n,), label (n, 2), conf, m9 ...), first n rows."""
    m = dict(np.load(BUNDLE / f"{split}_meta.npz"))
    return {k: v[:n] for k, v in m.items()}


# PICKS: dict (fam, K) -> list[int] indices (into the family's 7 extras) of the K-1 extra passes used.
#   Chosen on TRAIN by mp_sweep.py --stage 2 (AUROC>3px of dih avgm9c): K=2 -> code 7, K=4 -> codes 4, 5, 7.
#   Families / budgets not listed use the prefix order of mp_features.py (first K-1 extras).
PICKS = {("dih", 2): [DIH_ORDER.index(7)], ("dih", 4): [DIH_ORDER.index(c) for c in (4, 5, 7)]}


def extras(fam, k):
    """-> list[int] indices of the k-1 extra passes of `fam` used at a k-pass budget."""
    return PICKS.get((fam, k), list(range(k - 1)))


def dists(f, fam, k):
    """(n, k-1) float64: distance px of the k-1 extra tips of `fam` to the clean tip."""
    return np.linalg.norm(f[f"{fam}_tip"][:, extras(fam, k)] - f["clean_tip"][:, None], axis=2)


def all_tips(f, fam, k):
    """(n, k, 2) float64: the clean tip + the k-1 extra tips (clean frame)."""
    return np.concatenate([f["clean_tip"][:, None], f[f"{fam}_tip"][:, extras(fam, k)]], axis=1)


def median_tip(f, fam, k):
    """(n, 2) float64 coordinate-wise median of the k tips (a TTA prediction)."""
    return np.median(all_tips(f, fam, k), axis=1)


def masses(f, fam, k, key):
    """(n, k) float64: the clean pass's mass + the k-1 extras' `key` mass (key m5c | m9c)."""
    clean = f["clean_conf"] if key == "m5c" else f["clean_m9"]
    return np.concatenate([clean[:, None], f[f"{fam}_{key}"][:, extras(fam, k)]], axis=1)


def score(f, fam, k, name):
    """One named score (see module doc) for family `fam` at k passes -> (n,) float64, or None if n/a."""
    if name == "spread_max":
        return -dists(f, fam, k).max(1)
    if name == "spread_rms":
        t = all_tips(f, fam, k)
        return -np.sqrt((np.linalg.norm(t - np.median(t, axis=1, keepdims=True), axis=2) ** 2).mean(1))
    if name.startswith("agree"):
        r = float(name[5:])
        return (dists(f, fam, k) <= r).mean(1) + 1e-3 * f["clean_conf"]
    if name in ("avgm5c", "avgm9c"):
        return masses(f, fam, k, name[3:]).mean(1)
    if name in ("minm5c", "minm9c"):
        return masses(f, fam, k, name[3:]).min(1)
    if name in ("avgown5", "avgown9"):           # stored for the prefix pass sets only
        key = f"{fam}_avgm{name[-1]}"
        return f[key][:, KS.index(k)] if key in f and (fam, k) not in PICKS else None
    raise KeyError(name)


def set_masses(f, passes, key):
    """Mixed pass set: passes list[(fam, j)] extra passes -> (n, 1 + len) float64 masses (clean first)."""
    clean = f["clean_conf"] if key == "m5c" else f["clean_m9"]
    return np.column_stack([clean] + [f[f"{fam}_{key}"][:, j] for fam, j in passes])


def set_dists(f, passes):
    """Mixed pass set -> (n, len) float64 distance px of each extra tip to the clean tip."""
    return np.column_stack([np.linalg.norm(f[f"{fam}_tip"][:, j] - f["clean_tip"], axis=1) for fam, j in passes])


def combo(f, fam, k, base, tau):
    """base * exp(-spread_max / tau): base conf | m9 | avgm9c; tau float px -> (n,) float64."""
    b = {"conf": f["clean_conf"], "m9": f["clean_m9"]}.get(base)
    if b is None:
        b = score(f, fam, k, base)
    return b * np.exp(score(f, fam, k, "spread_max") / tau)


# FINAL: list[(method name, family, K, builder(f) -> (n,) score)] evaluated by --final; filled by final_methods()
def final_methods():
    """-> list of (name, passes, fn(f) -> score) for every final method, all hyperparameters chosen on train."""
    # SPECS: list[(label, fam, score name)]; per family the mean-map mass window that won on train + spread
    specs = [("mean-map m9", "dih", "avgm9c"), ("mean-map m5", "dih", "avgm5c"), ("log m5 - rms/32", "dih", None),
             ("spread_max", "dih", "spread_max"), ("agree3", "dih", "agree3"),
             ("spread_max", "noise", "spread_max"), ("mean-map m9", "noise", "avgm9c"),
             ("spread_max", "shift", "spread_max"), ("mean-map m5", "shift", "avgm5c"),
             ("spread_max", "scale", "spread_max"), ("mean-map m9", "scale", "avgm9c")]
    meths = []
    for label, fam, sname in specs:
        for k in KS:
            if sname is None:       # log(5x5 mean-map mass) - RMS tip spread / 32 px (tau chosen on train)
                fn = (lambda f, k=k: np.log(score(f, "dih", k, "avgm5c")) + score(f, "dih", k, "spread_rms") / 32.0)
            else:
                fn = (lambda f, k=k, fam=fam, sname=sname: score(f, fam, k, sname))
            meths.append((f"{fam} {label} K={k}", k, fn))
    return meths


# SAVE: dict method name -> npz file stem for the best <= 3 methods (scores_<stem>.npz, keys val / train)
SAVE = {"dih mean-map m9 K=8": "dih_meanmap_m9_k8", "dih mean-map m5 K=8": "dih_meanmap_m5_k8",
        "dih mean-map m9 K=2": "dih_meanmap_m9_k2"}


def run_final():
    """Evaluate every final method on train and val with the frozen harness; append rows to RESULTS."""
    feats = {s: load_feat(CACHE / f"feat_{s}.npz") for s in ("train", "val")}
    ev.print_header()
    for name, passes, fn in final_methods():
        sc = {}
        for split in ("train", "val"):
            sc[split] = fn(feats[split]).astype(np.float64)
            ev.evaluate(split, sc[split], name, passes=passes, out=str(RESULTS),
                        notes="multi_pass; hyperparameters chosen on train")
        if name in SAVE:
            np.savez(HERE / f"scores_{SAVE[name]}.npz", val=sc["val"].astype(np.float32),
                     train=sc["train"].astype(np.float32))


# ── numpy port of the dihedral post-processing (what a webapp would run), for the CPU cost ─────────────

def np_dihedral_points(xy, k, size):
    """numpy twin of train_v9_tip.dihedral_points: xy (..., 2) float, k int, size int (256 image px, 128 cells)."""
    m = float(size - 1)
    x, y = xy[..., 0], xy[..., 1]
    if k >= 4:
        x = m - x
    r = k % 4
    nx, ny = [(x, y), (y, m - x), (m - x, m - y), (m - y, x)][r]
    return np.stack([nx, ny], -1)


def np_dih_masses(maps, codes):
    """Mean-map masses of ONE image from its K dihedral logit maps, each in its own (variant) frame.

    maps: (K, 128, 128) float32 logits, maps[0] = the clean pass; codes: list[int] K dihedral codes, codes[0] = 0.
    Returns (avgm5c, avgm9c) floats, higher = more trustworthy. No map is resampled: the clean argmax cell is
    mapped into each variant frame and the window summed there (window sums are linear, so this IS the mass
    of the back-transformed mean map).
    """
    z = maps.reshape(len(maps), -1).astype(np.float64)                  # (K, 16384)
    p = np.exp(z - z.max(1, keepdims=True))
    p = (p / p.sum(1, keepdims=True)).reshape(maps.shape)               # (K, 128, 128) probs
    cy, cx = divmod(int(np.argmax(maps[0])), 128)                        # clean argmax cell
    m5 = m9 = 0.0
    for j, k in enumerate(codes):
        vx, vy = np_dihedral_points(np.array([cx, cy], np.float64), k, 128).round().astype(int)
        m5 += float(p[j, max(0, vy - 2):vy + 3, max(0, vx - 2):vx + 3].sum())
        m9 += float(p[j, max(0, vy - 4):vy + 5, max(0, vx - 4):vx + 5].sum())
    return m5 / len(codes), m9 / len(codes)


def np_dih_spread(maps, codes):
    """RMS distance (px) of the K back-mapped tips to their coordinate-wise median; maps / codes as above."""
    tips = []
    for j, k in enumerate(codes):
        iy, ix = divmod(int(np.argmax(maps[j])), 128)
        y0, x0 = max(0, iy - 2), max(0, ix - 2)
        win = maps[j, y0:iy + 3, x0:ix + 3].astype(np.float64)
        w = np.exp(win - win.max())
        w /= w.sum()
        u = (w.sum(0) * np.arange(x0, x0 + w.shape[1])).sum()
        v = (w.sum(1) * np.arange(y0, y0 + w.shape[0])).sum()
        kinv = (4 - k) % 4 if k < 4 else k
        tips.append(np_dihedral_points(np.array([2 * u + 0.5, 2 * v + 0.5]), kinv, 256))
    t = np.array(tips)                                                   # (K, 2)
    return float(np.sqrt((np.linalg.norm(t - np.median(t, 0), axis=1) ** 2).mean()))


def time_post(n=256):
    """Run the 8 dihedral passes on n val images, time the numpy post-processing per image, cross-check vs the cache."""
    import torch
    import confbundle as cb
    import mp_features as mf
    val = cb.load_split("val")
    model = cb.load_model()
    f = load_feat(CACHE / "feat_val.npz")
    x0 = torch.from_numpy(np.ascontiguousarray(val["images"][:n])).to(mf.DEV).float().div(255.0)[:, None]
    all_codes = [0] + mf.DIH_ORDER                                       # list[int] 8 codes, storage order
    with torch.no_grad():
        maps = np.stack([mf.run_pass(model, tv_img)[0].cpu().numpy() for tv_img in
                         [x0] + [mf.Dihedral(k).img(x0, None) for k in mf.DIH_ORDER]], 1)   # (n, 8, 128, 128)
    for k in KS:
        cols = [0] + [1 + j for j in extras("dih", k)]                   # list[int] the k passes used
        codes = [all_codes[c] for c in cols]
        t = time.perf_counter()
        mm = np.array([np_dih_masses(maps[i, cols], codes) for i in range(n)])    # (n, 2)
        t_mass = (time.perf_counter() - t) / n * 1e3
        t = time.perf_counter()
        sp = np.array([np_dih_spread(maps[i, cols], codes) for i in range(n)])    # (n,)
        t_spread = (time.perf_counter() - t) / n * 1e3
        gap5 = np.abs(mm[:, 0] - score(f, "dih", k, "avgm5c")[:n]).max()
        gap9 = np.abs(mm[:, 1] - score(f, "dih", k, "avgm9c")[:n]).max()
        gaps = np.abs(-sp - score(f, "dih", k, "spread_rms")[:n]).max()
        print(f"K={k} codes {codes}: mean-map m5+m9 {t_mass:.2f} ms/image (max |numpy - cache| {gap5:.1e}, "
              f"{gap9:.1e});  + tip spread {t_spread:.2f} ms/image (gap {gaps:.1e} px)")


def report_accuracy():
    """Does averaging over variants change ACCURACY? p99 / p99.9 error vs label, >3 px and >7 px counts per split."""
    for split in ("train", "val"):
        f = load_feat(CACHE / f"feat_{split}.npz")
        lab = load_meta(split)["label"].astype(np.float64)              # (n, 2) ground-truth tip px

        def acc(name, tip):
            e = np.linalg.norm(tip - lab, axis=1)
            print(f"{split:<6}{name:<34} p99 {np.percentile(e, 99):.3f}  p99.9 {np.percentile(e, 99.9):.3f}  "
                  f">3px {(e > 3).sum():4d}  >7px {(e > 7).sum():3d}")
        acc("clean (shipped decode)", f["clean_tip"])
        for k in KS:
            acc(f"dih K={k} median of tips", median_tip(f, "dih", k))
        acc("dih K=8 mean-map decode", f["dih_avgtip"][:, KS.index(8)])
        for fam in ("noise", "shift", "scale"):
            acc(f"{fam} K=8 median of tips", median_tip(f, fam, 8))


def paired_bootstrap(n_boot=1000, seed=0):
    """Val: paired bootstrap (images resampled) of AUROC>3px / AP>3px differences, saved methods minus conf / m9."""
    f = load_feat(CACHE / "feat_val.npz")
    meta = load_meta("val")
    bad = meta["err"] > 3.0                                              # (n,) bool
    base = {"conf": meta["conf"], "m9": meta["m9"]}
    cand = {"dih m9 K=2": score(f, "dih", 2, "avgm9c"), "dih m9 K=8": score(f, "dih", 8, "avgm9c"),
            "dih m5 K=4": score(f, "dih", 4, "avgm5c"), "dih m5 K=8": score(f, "dih", 8, "avgm5c")}
    rng = np.random.default_rng(seed)
    idx = [rng.integers(0, len(bad), len(bad)) for _ in range(n_boot)]  # list of (n,) int64 resamples
    for cn, cs in cand.items():
        for bn, bs in base.items():
            d_au = [ev.auroc(cs[i], bad[i]) - ev.auroc(bs[i], bad[i]) for i in idx]
            d_ap = [ev.average_precision(cs[i], bad[i]) - ev.average_precision(bs[i], bad[i]) for i in idx]
            lo, hi = np.percentile(d_au, [2.5, 97.5])
            alo, ahi = np.percentile(d_ap, [2.5, 97.5])
            print(f"val {cn:<11} - {bn:<5}: dAUROC3 {np.mean(d_au):+.4f} [{lo:+.4f},{hi:+.4f}]   "
                  f"dAP3 {np.mean(d_ap):+.3f} [{alo:+.3f},{ahi:+.3f}]")


def tta_side():
    """Side note (NOT harness rows): if the K=8 dihedral mean-map decode were shipped as the prediction,
    how well do its own masses rank ITS errors? AUROC / AP for err > 3 px and AUROC for err > 7 px."""
    for split in ("train", "val"):
        f = load_feat(CACHE / f"feat_{split}.npz")
        lab = load_meta(split)["label"].astype(np.float64)
        e = np.linalg.norm(f["dih_avgtip"][:, KS.index(8)] - lab, axis=1)  # (n,) TTA-prediction error px
        b3, b7 = e > 3.0, e > 7.0
        for name, s in (("mean-map own m5", f["dih_avgm5"][:, KS.index(8)]),
                        ("mean-map own m9", f["dih_avgm9"][:, KS.index(8)]),
                        ("mean-map m5 at clean argmax", score(f, "dih", 8, "avgm5c")),
                        ("clean conf", f["clean_conf"])):
            a7 = ev.auroc(s, b7) if b7.any() else float("nan")
            print(f"{split:<6}TTA-err K=8  {name:<28} AUROC3 {ev.auroc(s, b3):.4f}  AP3 "
                  f"{ev.average_precision(s, b3):.3f}  AUROC7 {a7:.3f}  (bad3 {b3.sum()}, miss {b7.sum()})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--accuracy", action="store_true")
    ap.add_argument("--paired", action="store_true")
    ap.add_argument("--tta", action="store_true")
    a = ap.parse_args()
    if a.final:
        run_final()
    if a.accuracy:
        report_accuracy()
    if a.paired:
        paired_bootstrap()
    if a.tta:
        tta_side()
    if a.time:
        time_post()
