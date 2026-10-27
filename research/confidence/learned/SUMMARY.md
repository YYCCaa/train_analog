# Learned / calibrated confidence for TipNet v9 (family 3)

> Written by the coordinating session from the research agent's report: the subagent guard blocked the agent's
> own write of this file. The text is the agent's; the last section adds the coordinator's verification notes.

The pipeline is one pass; every number comes from the frozen `eval_conf.evaluate`. Val scores are out-of-fold: either val was never used for fitting, or it was 2-fold cross-fitted.
```
logits ─▶ map features (features.py) ─▶ LR / MLP   or   tiny CNN on the log-prob map ─▶ score ─▶ isotonic ─▶ P(err<=3px)
fit:  train sample (20k) + 4 noisy copies per image (cb.noise_mask; 80k rows, 648 > 3 px, 25 > 7 px)
hyper-parameters: 5-fold CV on train, grouped by source image (never on val)
trainfit = fit on train + noisy only (val unseen);  xfit = also val fold a -> score val fold 1-a
```

## What was tried
- **features.py:** 41 cheap map statistics.
  - window masses 3–25 as logit(m), m5 minus its sub-cell ceiling, far mass;
  - top-3 runner masses, distance and logit margin, candidate counts;
  - background floor: the median height of the 9×9 local maxima, and floor / peak;
  - entropy, spread, log-prob quantiles, peak curvature, decode-window entropy and offset, tip position, raw logit scale.
  - The log-softmax runs in float64: 1 − m9 is about 1e-5, and float32 rounding moved logit(m9) by up to 0.1.
- **Logistic regression (LR):** 5 ordinal heads (err > 1.5, 2, 3, 4, 5 px); score = −logit P(err > 3).
  - Train-CV grid: l2 ∈ {1e-4, 1e-3, 1e-2}, one head vs five heads, with or without noisy rows, and three feature subsets.
  - Every LR reaches 0.950–0.952 in CV (`conf` 0.9405). The noisy rows add about +0.003.
  - CV best: **LR36**, all features except position and raw logit scale, at 0.9517 / AP 0.160.
  - **LR3** (logit m9, entropy, log floor/peak) reaches 0.9508 with 4 weights.
- **MLP (failed):** 3 × (36-32-32-5) scored 0.943 in CV, below LR, and also below LR on val (0.9586).
- **Tiny CNN:** 12.9k params. One branch sees the whole 128² log-prob map plus an argmax bump; the other a 17² crop. Dihedral augmentation, 8× TTA.
  - Train-CV 0.9476; val 0.957 on err > 3 px.
  - **Bad on misses:** AUROC>7 0.94, and it flags 32 % to catch all of them. Trained on err > 3 it learns blur, not decoys.
- **Not done:**
  - boosted stumps: LR already saturates CV;
  - image-crop CNN: 13 misses cannot train a decoy detector;
  - a learned P(err > 7): too few misses to calibrate.

## Val table
```
method              split AUROC>3px AP>3px AUROC>7px 95% CI        flag%all flag%75 rho   p99@95 miss@95 AUROCjump passes
conf (5x5, shipped) val   0.9458    0.243  0.989[0.978,0.996]      6.33     1.26    0.371 2.23   1       0.990     1
m9 (9x9 mass)       val   0.9497    0.201  0.996[0.994,0.998]      1.47     0.44    0.409 2.32   0       0.996     1
LR3 trainfit        val   0.9610    0.234  0.992[0.987,0.996]      2.57     1.56    0.471 2.18   0       0.990     1
LR36 trainfit       val   0.9601    0.233  0.990[0.984,0.995]      3.29     1.29    0.462 2.18   0       0.976     1
CNN map trainfit    val   0.9567    0.216  0.940[0.877,0.986]     31.96     4.87    0.476 2.21   3       0.931     1
```
Two more:
- MLP36 trainfit: 0.9586 / 0.219 / 0.988, flag%all 4.53.
- The xfit rows equal the trainfit rows within 0.001: adding val folds does not help.

Paired stratified bootstrap on val, ΔAUROC>3 with 95 % CI (`paired_bootstrap.json`):

| | vs `conf` | vs `m9` |
|---|---|---|
| LR3 | +0.015 [+0.007, +0.026] | +0.011 [+0.006, +0.017] |
| LR36 | +0.014 [+0.007, +0.024] | +0.010 [+0.005, +0.016] |
| CNN | +0.011 [−0.000, +0.023] (not significant) | — |

- **ΔAP:** vs `conf` −0.009 [−0.032, +0.012], within noise; vs `m9` +0.033 [+0.007, +0.057].
- **Other columns:** rho 0.37/0.41 → 0.47; p99@95 2.23/2.32 → 2.18.
- **Misses (13):** LR3 sits between `conf` and `m9`. AUROC>7 0.992 (`conf` 0.989, `m9` 0.996); flag%all 2.6 % (`conf` 6.3 %, `m9` 1.5 %).

## Calibration
Out-of-fold isotonic on val (fit on fold a, apply to fold b); full tables are in calibration.json and reliability.csv.

| calibrator | ECE15 | Brier | NLL | least-confident 2 %: predicted bad vs observed bad |
|---|---|---|---|---|
| raw `conf` read as a probability | 0.029 | 0.00984 | 0.060 | 0.14 vs 0.24 |
| iso(`conf`) | 0.0016 | 0.00887 | 0.036 | 0.24 vs 0.24 |
| iso(`m9`) | 0.0018 | 0.00913 | 0.037 | — |
| iso(LR3) | 0.0017 | 0.00896 | 0.035 | 0.25 vs 0.23 |
| iso(LR36) | 0.0020 | 0.00890 | 0.035 | — |
| iso(CNN) | 0.0018 | 0.00904 | 0.036 | — |
| iso fitted on train → val | about 0.003 | — | — | 0.16 vs 0.24 (over-confident) |

iso(LR3) reliability, predicted vs observed P(err ≤ 3):

| predicted bin | n | mean predicted | observed |
|---|---|---|---|
| 0.5–0.8 | 265 | 0.726 | 0.747 |
| 0.8–0.9 | 471 | 0.847 | 0.845 |
| 0.9–0.95 | 538 | 0.933 | 0.928 |
| 0.95–0.98 | 831 | 0.968 | 0.978 |
| 0.98–0.99 | 1547 | 0.987 | 0.985 |
| ≥ 0.995 | 17924 | 0.999 | 0.9996 |

- **Raw `conf` is not a probability.** Its 0.95–0.98 bulk is right 99.8 % of the time, while its 0.5–0.8 bin is right only 56 %.
- **Isotonic costs almost no ranking:** `conf` 0.9458 → 0.9414, LR3 0.9610 → 0.9607.
- **Knot files** (`isotonic_*.json`): x = block-mean score, y = P(err ≤ 3), n = images per block. Apply with `np.interp`.
  - Each block is shrunk to (good + ½)/(n + 1), so no knot reads exactly 0 or 1 (top 0.99997).
  - They are fit on all val. That is clean because the LR3 and LR36 models never saw val.

## Cost (CPU, 1 thread, per image, on top of the normal forward pass; passes = 1)

| method | params | ms per image |
|---|---|---|
| LR3, numpy (`lean.py`) | 4 + 6 standardisation constants + 26 knots | 0.5–0.8, including softmax and the 9×9 local maxima the webapp already computes |
| LR36 (torch `map_features`, unoptimised) | 37 | 6.7 |
| MLP36 | 7.2k | 6.8 |
| CNN | 12.9k | 1.15 without TTA, about 9 with the 8× TTA used |
| isotonic map | 23–27 knots | ~0.01 |

## Recommendation
- **Ship LR3 + isotonic:** `model_lr3.json` + `isotonic_LR3.json`, about 0.8 ms of numpy. It has the best val AUROC>3 among the one-pass methods here, significantly better than both baselines, with almost nothing to overfit. The webapp should display P(err ≤ 3 px).
- **Minimal option:** `isotonic_conf.json` alone gives an honest probability with no new model.
- **For decoy misses, keep `m9` / the floor features in view.** No learned model beat `m9` on flag%all, and the 13 misses cannot train or select one.

## Caveats
- **Small counts:** 126 clean train positives, 235 val positives and 13 misses. The miss metrics and the calibration bins with n < 20 are noise.
- **Train rows:** trainfit = 5-fold fold-out; xfit = in-sample (optimistic).
- **Shift:** TipNet has memorised the train sample (fewer errors, sharper maps), so a calibration fitted on train is over-confident on val. Refit the isotonic map on held-out data, ideally real photos, before trusting its probabilities.
- **Correlated rows:** the noisy rows share source images, so CV is grouped by source image.

The map / feature cache (about 4 GB) lived in the session scratchpad and is not kept.

## Coordinator notes (independent verification)
- **Reproduced LR3 in torch** from `model_lr3.json` alone by `../comparison/verify_learned.py`, without this folder's code.
  - Val scores match `scores_LR3.npz` to 9e-7.
  - The train scores in that file are the 5-fold out-of-fold predictions (see Caveats), so they differ from the final model's by up to 0.09 by design. The final model scores train AUROC>3 0.9513 in-sample.
- **Paired bootstrap (val, 1000 resamples):**
  - LR3 − conf: ΔAUROC3 +0.015 [+0.007, +0.025], ΔAP3 −0.009 [−0.030, +0.011].
  - LR3 − m9: +0.011 [+0.007, +0.016] / +0.032 [+0.007, +0.057].
- **Out-of-fold isotonic with an independent pool-adjacent-violators fit:** ECE15 drops from 0.029 (raw conf) to 0.0014 for iso(conf), 0.0012 for iso(m9) and 0.0012 for iso(LR3). Among the least-confident 2 % by conf, conf predicts 0.141 bad; 0.242 are.
- **Multi-pass scores rank better than LR3.** bc on the dihedral mean map reaches val AUROC>3 0.9665 with 4 passes; see `../comparison/SUMMARY.md`.
