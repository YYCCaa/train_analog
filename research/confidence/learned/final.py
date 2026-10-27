"""Step 4: harness rows, calibration and deliverables for the final learned-confidence methods.

    CACHE/raw_<tag>_<protocol>.npz (val OOF (22500,), train (20000,))  ──▶  eval_conf.evaluate rows
                                                                         ──▶  results.jsonl
    scores (any, incl. the shipped conf / m9) ─isotonic, val 2-fold cross-fit─▶ P(err <= 3 px) ──▶ ECE, reliability
                                                                         ──▶  calibration.json
    full-val isotonic knots of the recommended scores                    ──▶  isotonic_<name>.json
    best <= 3 methods                                                    ──▶  scores_<method>.npz (float32)

Run from research/confidence/learned:  python final.py
"""

import json

import numpy as np

import common as C
import eval_conf as ev
import features as ft

# METHODS: list of (results name, raw npz tag, notes) evaluated with the harness (filled from what exists)
METHODS = [
    ("LR3 trainfit", "lr3_trainfit", "LR on 3 map features (logit m9, entropy, log floor/peak), 5 ordinal heads, "
     "l2 1e-3, score=-logit P(err>3); fit train+4x noisy train; val never seen; train rows 5-fold fold-out"),
    ("LR3 xfit", "lr3_xfit", "same LR3, fit train+noisy+val fold a -> val fold 1-a; train rows IN-SAMPLE"),
    ("LR36 trainfit", "lr36_trainfit", "LR on 36 map features (all but tip position / raw logit scale), 5 "
     "ordinal heads, l2 1e-3; fit train+4x noisy; val never seen; train rows 5-fold fold-out"),
    ("LR36 xfit", "lr36_xfit", "same LR36, val 2-fold cross-fit; train rows IN-SAMPLE"),
    ("MLP36 trainfit", "mlp36_trainfit", "3 x MLP 36-32-32-5 (wd 1e-2) on the LR36 features; fit train+4x "
     "noisy; train rows 5-fold fold-out"),
    ("MLP36 xfit", "mlp36_xfit", "same MLP, val 2-fold cross-fit; train rows IN-SAMPLE"),
    ("CNN map trainfit", "cnn_trainfit", "tiny 2-branch CNN (12.9k params) on the 128x128 log-prob map, 8x "
     "dihedral TTA, 2 seeds; fit train+4x noisy; train rows 5-fold fold-out (1 seed, TTA)"),
    ("CNN map xfit", "cnn_xfit", "same CNN, val 2-fold cross-fit; train rows IN-SAMPLE"),
]
# SHIP: list of (file stem, raw tag) for scores_<stem>.npz (best <= 3, chosen by the train CV and simplicity)
SHIP = [("LR3", "lr3_trainfit"), ("LR36", "lr36_trainfit"), ("CNN", "cnn_trainfit")]
# CALIB: list of (name, raw tag or meta key) whose OOF isotonic calibration is reported and whose full-val
# knots are written to isotonic_<name>.json (LR3 / LR36 = the trainfit models, fit without val)
CALIB = [("conf", "conf"), ("m9", "m9"), ("LR3", "lr3_trainfit"), ("LR36", "lr36_trainfit")]
# N_BOOT_PAIR: int, paired stratified bootstrap resamples for the AUROC>3 / AP>3 differences vs the baselines
N_BOOT_PAIR = 1000


def paired_delta(s_new, s_ref, bad, n_boot=N_BOOT_PAIR, seed=0):
    """Paired stratified bootstrap of AUROC and AP differences (new - ref) -> dict of point + 95 % intervals."""
    rng = np.random.default_rng(seed)
    ib, ig = np.flatnonzero(bad), np.flatnonzero(~bad)
    da, dp = [], []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(ib, len(ib)), rng.choice(ig, len(ig))])
        b = bad[idx]
        da.append(ev.auroc(s_new[idx], b) - ev.auroc(s_ref[idx], b))
        dp.append(ev.average_precision(s_new[idx], b) - ev.average_precision(s_ref[idx], b))
    pt_a = ev.auroc(s_new, bad) - ev.auroc(s_ref, bad)
    pt_p = ev.average_precision(s_new, bad) - ev.average_precision(s_ref, bad)
    return {"d_auroc3": pt_a, "d_auroc3_ci": np.percentile(da, [2.5, 97.5]).tolist(),
            "d_ap3": pt_p, "d_ap3_ci": np.percentile(dp, [2.5, 97.5]).tolist()}


def meta(split):
    return np.load(C.BUNDLE / f"{split}_meta.npz")


def raw(tag):
    """Raw score dict of a method tag -> {"val": (22500,), "train": (20000,)} float64 (or None)."""
    if tag in ("conf", "m9"):
        return {s: meta(s)[tag].astype(np.float64) for s in ("val", "train")}
    p = C.CACHE / f"raw_{tag}.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return {"val": d["val"], "train": d["train"]}


def main():
    C.RESULTS.unlink(missing_ok=True)
    good_v = meta("val")["err"] <= C.GOOD_PX
    good_t = meta("train")["err"] <= C.GOOD_PX
    ev.print_header()
    for name, tag in (("conf (5x5 mass, shipped)", "conf"), ("m9 (9x9 mass)", "m9")):
        r = raw(tag)
        for split in ("train", "val"):
            ev.evaluate(split, r[split], name, passes=1, notes="baseline", out=C.RESULTS)
    done = []
    for name, tag, notes in METHODS:
        r = raw(tag)
        if r is None:
            print(f"(skip {name}: no raw scores)")
            continue
        for split in ("train", "val"):
            ev.evaluate(split, r[split], name, passes=1, notes=notes, out=C.RESULTS)
        done.append((name, tag))

    # paired bootstrap of the val differences vs the shipped conf and m9 (OOF scores; same resamples)
    bad_v = ~good_v
    deltas = {}
    print("\nPAIRED BOOTSTRAP on val (new - baseline), 95 % intervals")
    for name, tag in done:
        for ref in ("conf", "m9"):
            d = paired_delta(raw(tag)["val"], raw(ref)["val"], bad_v)
            deltas[f"{name} - {ref}"] = d
            print(f"  {name:<18} - {ref:<5} dAUROC>3 {d['d_auroc3']:+.4f} [{d['d_auroc3_ci'][0]:+.4f},"
                  f"{d['d_auroc3_ci'][1]:+.4f}]   dAP>3 {d['d_ap3']:+.3f} [{d['d_ap3_ci'][0]:+.3f},{d['d_ap3_ci'][1]:+.3f}]")
    C.save_json(C.HERE / "paired_bootstrap.json", deltas)

    # calibration: isotonic P(err <= 3) cross-fitted on val (fold b mapped by fold 1-b), plus two references
    reps = []
    print("\nCALIBRATION (val, out-of-fold)")
    for name, tag in CALIB + [("CNN", "cnn_trainfit")]:
        r = raw(tag)
        q = C.iso_crossfit(r["val"], good_v)
        rep = C.calib_report(q, good_v, f"iso({name}) val-xfit")
        C.print_calib(rep)
        reps.append(rep)
        # a mapping fitted on the TRAIN sample, applied to val: shows the train -> val shift
        kt = C.iso_fit(r["train"], good_t)
        rep_t = C.calib_report(C.iso_apply(kt, r["val"]), good_v, f"iso({name}) fit on train -> val")
        C.print_calib(rep_t)
        reps.append(rep_t)
        if tag in [t for _, t in CALIB]:
            kv = C.iso_fit(r["val"], good_v)
            kv["what"] = (f"P(err <= {C.GOOD_PX} px) = np.interp(score, x, y); score = {name}; isotonic fit on "
                          f"all {C.N_VAL} val images (out-of-fold ECE in calibration.json)")
            C.save_json(C.HERE / f"isotonic_{name}.json", kv)
            # harness row of the calibrated probability itself (OOF): what ranking does calibration cost?
            ev.evaluate("val", q, f"iso({name}) OOF prob", passes=1,
                        notes="isotonic P(err<=3) of the score, val 2-fold cross-fit", out=C.RESULTS)
    # the raw shipped conf read directly as a probability (what a webapp user sees today)
    rep = C.calib_report(meta("val")["conf"], good_v, "raw conf read as P(err<=3)")
    C.print_calib(rep)
    reps.append(rep)
    C.save_json(C.HERE / "calibration.json", reps)

    for stem, tag in SHIP:
        r = raw(tag)
        np.savez(C.HERE / f"scores_{stem}.npz", val=r["val"].astype(np.float32), train=r["train"].astype(np.float32))
        print("wrote", f"scores_{stem}.npz")


if __name__ == "__main__":
    main()
