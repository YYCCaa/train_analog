"""Shared pieces of the learned-confidence scripts: cache loading, fold layout, isotonic calibration, ECE.

Folds (README rule 2):
    val:   VAL_FOLD = arange(22500) % 2. Cross-fit = fit on train (+ noisy train) + val fold a, score val fold 1-a.
    train: TRAIN_FOLD = source-image index % 5. Fold-out train scores: fit on the other 4 folds (clean + noisy
           copies of the same source images are always held out together).

Calibration: isotonic regression of y = 1[err <= 3 px] on a score, stored as knots (block-mean score, block
probability); new scores are mapped with np.interp between the knots (monotone, keeps the ranking within
the knot range, flat beyond the ends).
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import isotonic_regression

HERE = Path(__file__).resolve().parent
# BUNDLE / CACHE: Path, the read-only bundle and the scratchpad cache written by extract.py
BUNDLE = Path(r"C:/Users/ADMINI~1/AppData/Local/Temp/claude/C--Users-Administrator-PycharmProjects-train-dial/"
              r"f292a1d6-f81e-4d2c-b353-1ee00d6e9308/scratchpad/confbundle")
CACHE = BUNDLE.parent / "learned_cache"
sys.path.insert(0, str(BUNDLE))
sys.path.insert(0, str(HERE))
# RESULTS: Path, the jsonl every final eval_conf.evaluate row is appended to
RESULTS = HERE / "results.jsonl"
# N_VAL / N_TRAIN: int, split sizes
N_VAL, N_TRAIN = 22500, 20000
# VAL_FOLD: (22500,) int64, the README's 2-fold split of val
VAL_FOLD = np.arange(N_VAL) % 2
# TRAIN_FOLD: (20000,) int64, 5-fold split of the train sample by source image
TRAIN_FOLD = np.arange(N_TRAIN) % 5
# GOOD_PX: float, the calibrated event is err <= GOOD_PX image px
GOOD_PX = 3.0


def load_set(name, what=("feats", "err")):
    """Cached arrays of one set ("train", "val", "trainnoisy") -> dict key -> array (logp / crop memmapped)."""
    out = {}
    for k in what:
        mm = "r" if k in ("logp", "crop") else None
        out[k] = np.load(CACHE / f"{name}_{k}.npy", mmap_mode=mm)
    if name == "trainnoisy":
        out["src"] = np.load(CACHE / "trainnoisy_src.npy")
    return out


def iso_fit(score, good):
    """Isotonic P(good | score). score (n,) float, good (n,) bool -> knots dict {x: list, y: list}."""
    order = np.argsort(score, kind="stable")
    xs, ys = np.asarray(score, np.float64)[order], good[order].astype(np.float64)
    fit = isotonic_regression(ys, increasing=True).x                          # (n,) float64 block values
    # blocks: runs of equal fitted value -> knot at the block's mean score
    edges = np.flatnonzero(np.diff(fit) != 0) + 1
    starts = np.concatenate([[0], edges])
    ends = np.concatenate([edges, [len(fit)]])
    kx = np.array([xs[s:e].mean() for s, e in zip(starts, ends)])
    kn = (ends - starts).astype(np.float64)                                    # (n_knots,) images per block
    # shrink each block's rate by half an image each way ((good + 0.5) / (n + 1)): a 15000-image block with no
    # bad image reads 0.99997, not a literal 100 %; a cumulative max keeps the map monotone
    ky = np.maximum.accumulate((np.array([fit[s] for s in starts]) * kn + 0.5) / (kn + 1.0))
    return {"x": kx.tolist(), "y": ky.tolist(), "n": kn.astype(int).tolist(), "n_fit": int(len(xs))}


def iso_apply(knots, score):
    """Map scores (n,) through isotonic knots -> (n,) float64 probabilities P(err <= 3 px)."""
    return np.interp(np.asarray(score, np.float64), np.asarray(knots["x"]), np.asarray(knots["y"]))


def iso_crossfit(score_val, good_val):
    """Out-of-fold calibrated val probabilities: fold b mapped by the isotonic fit on fold 1-b -> (22500,)."""
    q = np.empty(N_VAL)
    for b in (0, 1):
        k = iso_fit(score_val[VAL_FOLD != b], good_val[VAL_FOLD != b])
        q[VAL_FOLD == b] = iso_apply(k, score_val[VAL_FOLD == b])
    return q


# REL_EDGES: (9,) float, reliability-table bin edges of the predicted P(err <= 3 px)
REL_EDGES = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0 + 1e-9])


def calib_report(q, good, name):
    """Reliability of predicted P(good) q (n,) vs outcomes good (n,) bool -> dict (table + ECE / Brier / NLL)."""
    rows = []
    for lo, hi in zip(REL_EDGES[:-1], REL_EDGES[1:]):
        m = (q >= lo) & (q < hi)
        if m.any():
            rows.append({"bin": f"[{lo:.3f},{min(hi, 1.0):.3f}]", "n": int(m.sum()),
                         "pred": float(q[m].mean()), "obs": float(good[m].mean())})
    ece_tab = sum(r["n"] * abs(r["pred"] - r["obs"]) for r in rows) / len(q)
    # standard 15 equal-width bins and 20 equal-count bins
    wb = np.minimum((q * 15).astype(int), 14)
    ece15 = sum(abs(q[wb == b].mean() - good[wb == b].mean()) * (wb == b).sum() for b in range(15) if (wb == b).any()) / len(q)
    order = np.argsort(q, kind="stable")
    chunks = np.array_split(order, 20)
    ece_eq = sum(abs(q[c].mean() - good[c].mean()) * len(c) for c in chunks) / len(q)
    qc = np.clip(q, 1e-6, 1 - 1e-6)
    nll = float(-np.mean(good * np.log(qc) + (~good) * np.log(1 - qc)))
    brier = float(np.mean((q - good) ** 2))
    # tail: the 2 % least confident images, predicted vs observed bad rate
    low = order[: int(0.02 * len(q))]
    return {"name": name, "ece_table": float(ece_tab), "ece15": float(ece15), "ece_eqmass20": float(ece_eq),
            "brier": brier, "nll": nll, "base_rate_bad": float(1 - good.mean()),
            "low2pct_pred_bad": float(1 - q[low].mean()), "low2pct_obs_bad": float(1 - good[low].mean()),
            "table": rows}


def print_calib(rep):
    print(f"{rep['name']}: ECE(table) {rep['ece_table']:.4f}  ECE15 {rep['ece15']:.4f}  ECE(20 eq-mass) "
          f"{rep['ece_eqmass20']:.4f}  Brier {rep['brier']:.5f}  NLL {rep['nll']:.4f}  "
          f"low-2% bad pred {rep['low2pct_pred_bad']:.3f} obs {rep['low2pct_obs_bad']:.3f}")
    for r in rep["table"]:
        print(f"    {r['bin']:>15}  n={r['n']:>6}  pred {r['pred']:.4f}  obs {r['obs']:.4f}")


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)
