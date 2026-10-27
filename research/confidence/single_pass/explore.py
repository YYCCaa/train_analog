"""Screen every single-pass candidate score on the TRAIN split (README rule 2: tune on train, never on val).

    <cache>/features_train.npz  {name: (20000,) float64}     (features.py)
    bundle train_meta.npz       err (20000,) float64 image px
        |  candidates(f): {score name: (20000,) float64, higher = more trustworthy}
        v
    table sorted by train AUROC>3px:  name | AUROC>3px | AP>3px | rho      (eval_conf's own metric functions)

Only train numbers are printed here; val is touched solely through eval_conf.evaluate in scores.py for the
final methods.

Usage (from the worktree root):
    python research/confidence/single_pass/explore.py --bundle <bundle dir> --cache <cache dir> [--grep m9]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def candidates(f):
    """Feature dict {name: (n,) float64} -> {score name: (n,) float64}, every score oriented higher = trustworthy."""
    s = {}
    eps = 1e-300
    for k in sorted(f):
        if k[0] == "m" and k[1:].isdigit():
            s[k] = f[k]                                           # square-window mass
            s[f"{k}/im"] = f[k] / f["i" + k]                      # ... over its sub-cell ideal ceiling
            s[f"-log(1-{k})"] = -np.log(np.maximum(1 - f[k], eps))   # far mass beyond the window (log)
        if k[0] == "c" and k[1:].replace(".", "").isdigit():
            s[k] = f[k]                                           # disc mass around the decoded tip
            s[f"{k}/ic"] = f[k] / f["i" + k]
            s[f"-log(1-{k})"] = -np.log(np.maximum(1 - f[k], eps))
            s[f"-(ic-{k})"] = f[k] - f["i" + k]                    # deficit against the ideal
        if k.startswith("m5_T") or k.startswith("m9_T"):
            s[k] = f[k]
    # second peak
    s["lmax-lm2"] = f["lmax"] - f["lm2"]                         # logit margin to the runner-up local max
    s["-lm2_m5/m5"] = -f["lm2_m5"] / f["m5"]
    s["-(lm2-lse)"] = -(f["lm2"] - f["lse"])                      # -log prob of the runner-up peak cell
    s["-(lm2-lm_med)"] = -(f["lm2"] - f["lm_med"])                # runner-up over the floor
    for R in (2, 3, 4, 6, 8, 12):
        s[f"lmax-out{R}"] = f["lmax"] - f[f"out{R}"]
        s[f"lse-out{R}"] = f["lse"] - f[f"out{R}"]
    # background floor (coordinator's probe): median local-max height
    s["-(lm_med-lse)"] = -(f["lm_med"] - f["lse"])                # -log median local-max prob
    s["lmax-lm_med"] = f["lmax"] - f["lm_med"]                    # -log(floor / main peak)
    s["-lm_med"] = -f["lm_med"]
    s["-lm_n"] = -f["lm_n"]
    for q in ("q10", "q50", "q90"):
        s[f"lse-{q}"] = f["lse"] - f[q]
        s[f"lmax-{q}"] = f["lmax"] - f[q]
    # shape / entropy / conformity
    s["-ent"] = -f["ent"]
    s["-mom2"] = -f["mom2"]
    for R in ("2", "3", "5"):
        s[f"-mom2_{R}"] = -f[f"mom2_{R}"]
        s[f"-(mom2_{R}-ideal)"] = -(f[f"mom2_{R}"] - f[f"imom2_{R}"])
    s["-lam1"], s["-lam2"] = -f["lam1"], -f["lam2"]
    s["-(lam1-lam2)"] = -(f["lam1"] - f["lam2"])
    s["-(lam1*lam2)"] = -(f["lam1"] * f["lam2"])
    s["-kl_gp"], s["-tv"], s["bc"] = -f["kl_gp"], -f["tv"], f["bc"]
    # logit space
    s["lmax"], s["lse"], s["lmax-lse"] = f["lmax"], f["lse"], f["lmax"] - f["lse"]
    return s


def screen(scores, err, ev, grep=None):
    """Print train AUROC>3px / AP>3px / rho for each score, best first -> list of (name, auroc, ap, rho)."""
    bad3 = err > ev.BAD_PX
    rows = []
    for name, sc in scores.items():
        if grep and grep not in name:
            continue
        sc = np.nan_to_num(sc, nan=-1e300, posinf=1e300, neginf=-1e300)
        rows.append((name, ev.auroc(sc, bad3), ev.average_precision(sc, bad3), spearmanr(sc, -err).statistic))
    rows.sort(key=lambda r: -r[1])
    print(f"{'score':<26}{'AUROC>3':>9}{'AP>3':>8}{'rho':>8}")
    for r in rows:
        print(f"{r[0]:<26}{r[1]:>9.4f}{r[2]:>8.3f}{r[3]:>8.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--grep", default=None, help="only scores whose name contains this text")
    args = ap.parse_args()
    sys.path.insert(0, args.bundle)
    import eval_conf as ev
    f = dict(np.load(Path(args.cache) / "features_train.npz"))
    err = np.load(Path(args.bundle) / "train_meta.npz")["err"]   # (20000,) float64 image px
    screen(candidates(f), err, ev, args.grep)


if __name__ == "__main__":
    main()
