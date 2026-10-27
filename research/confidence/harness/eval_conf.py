"""Frozen evaluation harness for TipNet confidence scores. Score every method with evaluate(); never edit this file.

A score is one float per image of a split, HIGHER = MORE TRUSTWORTHY (negate an uncertainty first).

    score (n,) float  ──▶  evaluate(split, score, name)  ──▶  dict of metrics (+ one printed row, + JSON line)

Metrics ("bad" = an image whose clean prediction is off; ranking flags the LOWEST scores first):
    auroc_err3     AUROC for bad = err > 3 px (val: 238 bad of 22500)          primary
    ap_err3        average precision for the same bad set (rank bad-first)
    auroc_miss     AUROC for bad = err > 7 px (val: 13 misses), with a stratified bootstrap 95 % CI
    flag_all_miss  % of images flagged (lowest scores first) to catch ALL misses       lower is better
    flag_75_miss   % flagged to catch 75 % of the misses (val: 10 of 13)              lower is better
    rho_err        Spearman rank correlation of score with -err (higher is better)
    p99_keep95     p99 error of the 95 % highest-scored images (selective prediction)
    miss_keep95    misses left among those 95 %
    auroc_jump     val only: AUROC for the 12 "jumpers" (tip moves > 7 px under some training-style
                   noise mask) among the 2000 best / worst val images of the stability probe
    passes         forward passes per image the method needs (1 = the normal prediction alone)

CLI:  python eval_conf.py baselines     prints the current confidence and simple map statistics
"""

import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

# BUNDLE: Path, this folder (meta files live here)
BUNDLE = Path(__file__).resolve().parent
# MISS_PX / BAD_PX: float, error thresholds (image px) of a miss and of a "bad" image
MISS_PX, BAD_PX = 7.0, 3.0
# N_BOOT: int, bootstrap resamples for the miss-AUROC confidence interval
N_BOOT = 2000
# COLUMNS: list[(key, header, format)] of the printed row
COLUMNS = [("auroc_err3", "AUROC>3px", "{:.4f}"), ("ap_err3", "AP>3px", "{:.3f}"),
           ("auroc_miss", "AUROC>7px", "{:.3f}"), ("auroc_miss_ci", "95% CI", "[{:.3f},{:.3f}]"),
           ("flag_all_miss", "flag%all", "{:.2f}"), ("flag_75_miss", "flag%75", "{:.2f}"),
           ("rho_err", "rho", "{:.3f}"), ("p99_keep95", "p99@95", "{:.2f}"), ("miss_keep95", "miss@95", "{:d}"),
           ("auroc_jump", "AUROCjump", "{:.3f}"), ("passes", "passes", "{:d}")]


def auroc(score, bad):
    """P(score of a bad image < score of a good image), ties count 1/2. score (n,) float, bad (n,) bool -> float."""
    r = rankdata(score)                                     # (n,) float64 average ranks, ascending
    n_b, n_g = int(bad.sum()), int((~bad).sum())
    return float((r[~bad].sum() - n_g * (n_g + 1) / 2) / (n_b * n_g))


def auroc_ci(score, bad, n_boot=N_BOOT, seed=0):
    """Stratified bootstrap 95 % interval of auroc -> (lo, hi) floats."""
    rng = np.random.default_rng(seed)
    ib, ig = np.flatnonzero(bad), np.flatnonzero(~bad)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ib, len(ib)), rng.choice(ig, len(ig))])
        vals.append(auroc(score[idx], bad[idx]))
    return tuple(float(v) for v in np.percentile(vals, [2.5, 97.5]))


def flag_to_catch(score, bad, frac):
    """% of all images with score <= the score that catches ceil(frac * n_bad) bad images -> float."""
    k = max(1, math.ceil(frac * int(bad.sum())))
    thr = np.sort(score[bad])[k - 1]
    return float(100.0 * (score <= thr).mean())


def average_precision(score, bad):
    """Average precision when images are flagged lowest score first (ties broken pessimistically) -> float."""
    order = np.lexsort((bad, score))                        # ascending score; among ties, good (False) first
    hits = bad[order]
    prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(prec[hits].mean())


def evaluate(split, score, name, passes=1, notes="", out=None):
    """Score one method on one split -> dict of the metrics above (also printed; appended to `out` as JSON).

    split: "val" or "train"; score: (n,) float, higher = more trustworthy; name: str; passes: int forward
    passes per image; notes: str; out: path of a .jsonl file to append to (or None).
    """
    meta = np.load(BUNDLE / f"{split}_meta.npz")
    err = meta["err"]                                       # (n,) float64 image px
    score = np.asarray(score, dtype=np.float64)
    if score.shape != err.shape or not np.all(np.isfinite(score)):
        raise ValueError(f"score must be {err.shape} finite floats, got {score.shape}")
    miss, bad3 = err > MISS_PX, err > BAD_PX
    keep = score >= np.quantile(score, 0.05)                 # the 95 % highest-scored images (ties kept)
    res = {"name": name, "split": split, "passes": int(passes), "notes": notes,
           "n": int(len(err)), "n_miss": int(miss.sum()), "n_err3": int(bad3.sum()),
           "auroc_err3": auroc(score, bad3), "ap_err3": average_precision(score, bad3),
           "auroc_miss": auroc(score, miss), "auroc_miss_ci": auroc_ci(score, miss),
           "flag_all_miss": flag_to_catch(score, miss, 1.0), "flag_75_miss": flag_to_catch(score, miss, 0.75),
           "rho_err": float(spearmanr(score, -err).statistic),
           "p99_keep95": float(np.percentile(err[keep], 99)), "miss_keep95": int(miss[keep].sum()),
           "auroc_jump": float("nan")}
    if split == "val":
        st = np.load(BUNDLE / "stab_stats.npz")
        jumper = st["shift"].max(axis=1) > MISS_PX           # (2000,) bool
        res["auroc_jump"] = auroc(score[st["pos"]], jumper)
    print_row(res)
    if out is not None:
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
    return res


def print_header():
    print(f"{'method':<34}{'split':<6}" + "".join(f"{h:>{max(len(h), 9) + 2}}" for _, h, _ in COLUMNS))


def print_row(res):
    cells = []
    for key, head, fmt in COLUMNS:
        v = res[key]
        txt = fmt.format(*v) if isinstance(v, (tuple, list)) else ("nan" if isinstance(v, float) and v != v
                                                                  else fmt.format(v))
        cells.append(f"{txt:>{max(len(head), 9) + 2}}")
    print(f"{res['name'][:33]:<34}{res['split']:<6}" + "".join(cells), flush=True)


def baselines(out=None):
    """The current confidence and three one-pass map statistics on both splits -> list of dicts."""
    rows = []
    print_header()
    for split in ("train", "val"):
        m = np.load(BUNDLE / f"{split}_meta.npz")
        for name, s in (("conf (5x5 mass, shipped)", m["conf"]), ("m9 (9x9 mass)", m["m9"]),
                        ("peak (argmax cell prob)", m["peak"]), ("-runner/conf (webapp ratio)", -m["runner"] / m["conf"])):
            rows.append(evaluate(split, s, name, out=out))
    return rows


if __name__ == "__main__":
    if sys.argv[1:] == ["baselines"]:
        baselines()
    else:
        print(__doc__)
