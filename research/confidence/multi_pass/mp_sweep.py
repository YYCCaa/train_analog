"""Train-only sweep of multi-pass consistency scores (hyperparameters chosen here, never on val).

Reads research/confidence/multi_pass/_cache/feat_train.npz (mp_features.py) and prints, for every
family x K x score, the train AUROC>3px and AP>3px of the harness (eval_conf.auroc / average_precision),
plus the TTA accuracy (p99 error vs label of the mean-map and median tips).

    python research/confidence/multi_pass/mp_sweep.py [--feat <npz>]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_scores as ms  # noqa: E402


def stage2(f, row):
    """Second train sweep: which dihedral extras for K = 2 / 4, and log-mass minus spread combinations."""
    from itertools import combinations
    print("-- dih K=2: each single extra code (avgm9c / avgm5c / spread_max) --")
    for j, k in enumerate(ms.DIH_ORDER):
        ps = [("dih", j)]
        row(f"K2 code {k} avgm9c", ms.set_masses(f, ps, "m9c").mean(1))
        row(f"K2 code {k} avgm5c", ms.set_masses(f, ps, "m5c").mean(1))
    print("-- dih K=4: all 35 triples, top 5 and the prefix (codes 2,4,6) by AUROC3 of avgm9c --")
    res = []
    for tri in combinations(range(7), 3):
        ps = [("dih", j) for j in tri]
        s = ms.set_masses(f, ps, "m9c").mean(1)
        res.append((ms.ev.auroc(s, f["_bad"]), tri))
    res.sort(reverse=True)
    for au, tri in res[:5] + [r for r in res if r[1] == (0, 1, 2)]:
        print(f"K4 codes {[ms.DIH_ORDER[j] for j in tri]} avgm9c AUROC3 {au:.4f}")
    print(f"K4 triples AUROC3 range {res[-1][0]:.4f} .. {res[0][0]:.4f}")
    print("-- log(mean-map mass) - spread_rms / tau --")
    for k in ms.KS:
        rms = -ms.score(f, "dih", k, "spread_rms")
        for w in ("avgm9c", "avgm5c"):
            lm = np.log(ms.score(f, "dih", k, w))
            for tau in (4.0, 8.0, 16.0, 32.0, 64.0):
                row(f"dih K={k} log {w} - rms/{tau:g}", lm - rms / tau)
        g = 0.5 * (np.log(ms.score(f, "dih", k, "avgm9c")) + np.log(ms.score(f, "dih", k, "avgm5c")))
        row(f"dih K={k} mean(log avgm5c, log avgm9c)", g)
        for tau in (8.0, 16.0, 32.0):
            row(f"dih K={k} mean(log m5c, log m9c) - rms/{tau:g}", g - rms / tau)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--feat", default=str(ms.CACHE / "feat_train.npz"))
    p.add_argument("--stage", type=int, default=1, help="1 = full grid, 2 = dihedral subsets + log combos")
    a = p.parse_args(argv)
    f = ms.load_feat(a.feat)
    meta = ms.load_meta("train", len(f["clean_conf"]))
    err, lab = meta["err"], meta["label"]
    bad = err > 3.0
    f["_bad"] = bad
    print(f"n {len(err)}  bad>3px {bad.sum()}  miss>7px {(err > 7).sum()}")

    def row(name, s):
        print(f"{name:<44} AUROC3 {ms.ev.auroc(s, bad):.4f}  AP3 {ms.ev.average_precision(s, bad):.3f}", flush=True)
    if a.stage == 1:
        ms.PICKS = {}            # stage 1 ran before the picks existed: every family uses its prefix order
    if a.stage == 2:
        row("conf (clean)", f["clean_conf"])
        row("m9 (clean)", f["clean_m9"])
        stage2(f, row)
        return

    row("conf (clean)", f["clean_conf"])
    row("m9 (clean)", f["clean_m9"])
    for fam in ms.FAMILIES:
        for k in ms.KS:
            for sname in ("spread_max", "spread_rms", "agree1", "agree2", "agree3", "avgm5c", "avgm9c", "minm9c",
                          "minm5c", "avgown5", "avgown9"):
                s = ms.score(f, fam, k, sname)
                if s is not None:
                    row(f"{fam} K={k} {sname}", s)
    print("\n-- combos conf/m9/avgm9c x exp(-spread_max / tau), tau sweep --")
    for fam in ms.FAMILIES:
        for k in ms.KS:
            for base in ("conf", "m9", "avgm9c"):
                for tau in (0.5, 1.0, 2.0, 4.0, 8.0):
                    row(f"{fam} K={k} {base}*exp(-spread/{tau})", ms.combo(f, fam, k, base, tau))
    print("\n-- mixed pass sets (avgm9c / minm9c / m9*exp(-spread/2)) --")
    # MIXES: dict name -> list[(fam, j)] extra passes (clean pass implied)
    mixes = {"K4 dih+shift+scale": [("dih", 0), ("shift", 0), ("scale", 0)],
             "K4 dih+dih+noise": [("dih", 0), ("dih", 1), ("noise", 0)],
             "K8 dih3+shift2+scale2": [("dih", 0), ("dih", 1), ("dih", 2), ("shift", 0), ("shift", 1),
                                       ("scale", 0), ("scale", 1)],
             "K8 dih7 (ref)": [("dih", j) for j in range(7)],
             "K15 dih7+shift4+scale3": [("dih", j) for j in range(7)] + [("shift", j) for j in range(4)]
                                       + [("scale", j) for j in range(3)],
             "K29 all": [(fam, j) for fam in ms.FAMILIES for j in range(7)]}
    for mname, ps in mixes.items():
        m9s = ms.set_masses(f, ps, "m9c")
        row(f"{mname} avgm9c", m9s.mean(1))
        row(f"{mname} minm9c", m9s.min(1))
        row(f"{mname} m9*exp(-spread/2)", f["clean_m9"] * np.exp(-ms.set_dists(f, ps).max(1) / 2.0))
    print("\n-- accuracy: p99 / p99.9 / max error vs label --")
    def acc(name, tip):
        e = np.linalg.norm(tip - lab, axis=1)
        print(f"{name:<44} p99 {np.percentile(e, 99):.3f}  p99.9 {np.percentile(e, 99.9):.3f}  "
              f">3px {(e > 3).sum()}  >7px {(e > 7).sum()}")
    acc("clean", f["clean_tip"])
    for fam in ms.FAMILIES:
        for ki, k in enumerate(ms.KS):
            if f"{fam}_avgtip" in f:
                acc(f"{fam} K={k} mean-map tip", f[f"{fam}_avgtip"][:, ki])
            acc(f"{fam} K={k} median tip", ms.median_tip(f, fam, k))


if __name__ == "__main__":
    main()
