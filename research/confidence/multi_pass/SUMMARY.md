# TipNet v9 confidence: multi-pass consistency (family 2)

> Written by the coordinating session from the research agent's report: the subagent guard blocked the agent's
> own write of this file. The text is the agent's; the last section adds the coordinator's verification notes.

**What ran.** `mp_features.py` runs the clean pass plus 7 extras from each of 4 families on all 42 500 images and maps every pass back to the clean frame:
- `dih`: the 8 dihedral transforms, as in training;
- `noise`: `cb.noise_mask` applied inside `cb.inside_dial`, from a seeded bank of 512;
- `shift`: ±6 px shifts, diagonals included, black fill;
- `scale`: 0.95–1.05 about 127.5, bilinear.

It keeps small per-pass features in `_cache/` (36 MB, not committed; rebuilding takes about 5 min per split). `mp_scores.py` scores them with the frozen harness. `mp_sweep.py` makes every choice on TRAIN only (`logs/sweep_train_stage{1,2}.txt`).

**Unit check** (`--check`, `logs/unit_check.txt`):
- On 128 confident val images, every dihedral, shift and scale transform maps the tip back within a median of 0.16–0.23 px, |bias| < 0.05 px.
- A deliberately wrong flip (256−x) gives 1.03 px.
- `decode(map_inv(logits))` equals `inv(decode(logits))` to 1.5e-5 px.
- The fast hull equals `cb.inside_dial` bit for bit.

**Scores tried:**
- spread: max and RMS distance of the back-mapped tips; fraction agreeing within 1, 2 or 3 px;
- mean-map 5×5 / 9×9 mass at the clean argmax (a window sum is linear, so no map is ever resampled);
- the minimum over passes; the mean map's mass at its own argmax;
- conf / m9 × exp(−spread/τ), and log m5 − rms/τ;
- mixed-family pass sets.

**Chosen on train:**
- K=2 uses code 7 (anti-transpose);
- K=4 uses codes 4, 5, 7 (hflip, transpose, anti-transpose);
- τ = 32.

## Val table
```
method                            split   AUROC>3px     AP>3px  AUROC>7px     95% CI   flag%all    flag%75        rho     p99@95    miss@95  AUROCjump     passes
dih mean-map m9 K=2               val        0.9544      0.209      0.997[0.995,0.998]       1.16       0.64      0.429       2.31          0      0.994          2
dih mean-map m5 K=4               val        0.9604      0.285      0.995[0.990,0.999]       3.20       0.43      0.416       2.13          0      0.977          4
dih mean-map m9 K=8               val        0.9605      0.233      0.997[0.994,0.999]       1.85       0.50      0.445       2.26          0      0.993          8
dih mean-map m5 K=8               val        0.9644      0.311      0.995[0.989,0.999]       3.12       0.66      0.424       2.13          0      0.972          8
```
Baselines: conf 0.9458 / 0.243 / 0.989, flag%all 6.33; m9 0.9497 / 0.201 / 0.996, flag%all 1.47. All 66 rows are in `results.jsonl`, and `logs/final_eval.txt` holds both splits. Selected other val rows (AUROC3 / AP3 / flag%all):

| method | AUROC3 | AP3 | flag%all |
|---|---|---|---|
| dih mean-map m9 K=4 | 0.9573 | 0.219 | 1.65 |
| dih mean-map m5 K=2 | 0.9576 | 0.251 | 4.25 |
| dih log m5 − rms/32 K=8 | 0.9653 | 0.296 | 3.48 |
| dih spread_max K=8 | 0.9472 | 0.250 | 24.69 |
| noise mean-map m9 K=8 | 0.9472 | 0.186 | 1.13 |
| shift mean-map m5 K=8 | 0.9597 | 0.265 | 5.33 |
| scale mean-map m9 K=8 | 0.9555 | 0.213 | 1.86 |

Paired bootstrap (val, 1000 resamples):
- m5 K=8 − conf: ΔAUROC3 +0.019 [+0.010, +0.028], ΔAP3 +0.067 [+0.029, +0.101].
- m5 K=4 − conf: +0.015 [+0.008, +0.022] / +0.043 [+0.009, +0.076].
- m9 K=8 − m9: +0.011 [+0.006, +0.016] / +0.030 [+0.007, +0.053].
- m9 K=2 − m9: +0.005 [+0.001, +0.009]; the AP change is not significant.

## Cost vs K (dihedral, val)
| K | AUROC3 m9 / m5 | AP3 m9 / m5 | CPU post (ms/img) | p99 err of the TTA tip |
|---|---|---|---|---|
| 1 | 0.9497 / 0.9458 (m9 / conf) | 0.201 / 0.243 | — | 3.073 (clean) |
| 2 | 0.9544 / 0.9576 | 0.209 / 0.251 | 0.5 | 2.897 |
| 4 | 0.9573 / 0.9604 | 0.219 / 0.285 | 0.9 | 2.797 |
| 8 | 0.9605 / 0.9644 | 0.233 / 0.311 | 3.1 | 2.723 (median of tips) / 2.690 (mean-map decode) |

- The CPU figure is the numpy post-processing only; it matches the cache to 1e-7. The forward passes dominate the real cost.
- Val >3 px errors drop from 235 to 164 with the K=8 mean-map decode.
- Noise, shift and scale TTA leave accuracy flat: p99 3.10, 2.87, 2.93.
- Side note, not a harness row: if the TTA tip were shipped, its own mean-map 5×5 mass ranks its errors at val AUROC3 0.957 / AP3 0.201, against 0.934 / 0.142 for clean conf.

## What failed
- **Spread alone** loses to conf at every K: val AUROC3 0.947 at K=8, 0.847 at K=2. The 0.2–0.5 px jitter of good images decides the ranking. agree3 ≈ conf.
- **Spread penalties** on conf / m9 hurt AUROC3 at every τ. log m5 − rms/32 equals the mean-map mass within noise: a pass that moves already takes its mass out of the clean window.
- **Noise TTA:** no gain (train 0.936 vs m9 0.938). Noise spread flags misses badly (flag%all 19–84 %).
- **Shift / scale:** below dihedral at equal K. Mixing them in hurts on train: dih3+shift2+scale2 0.9468 vs dih K=8 0.9556; all 29 passes 0.9489.
- **Other:** min < mean over passes; rot180 as the single extra pass is the worst choice for K=2.

## The 13 val misses (`logs/probe_misses.txt`)
- **The two confident decoys move under dihedral TTA:**
  - the 61 px miss is kept by only 1 of 7 variants (mean-map m5 0.26, flagged first);
  - the 20.7 px miss at conf 0.94 is dropped by 1 of 7 variants, landing 19 px away (m9 rank 0.08 %, vs m9 0.44 % and conf 6.33 %). This is why flag%all falls from 6.33 to 1.2–1.9.
- **Four 7–9 px misses** (positions 5271, 14971, 6324, 16013) are stable in all 8 variants (spread ≤ 1.4 px, mass ≥ 0.82). They are invisible to consistency and set m5's flag%all of about 3 %.

## Recommendation
**Use dihedral mean-map mass:**
- K=4 with the 5×5 window gives the best cost/gain.
- K=8 gives the best scores and a better tip.
- Use the 9×9 window if catching misses (flag%all) matters more than ranking 3–7 px errors.

Skip spread scores and noise / shift / scale TTA.

## Caveats
- **Misses:** there are 13 (1 on train). The AUROC>7px CIs overlap for all mass methods, flag%all hinges on 1–2 images, and no multi-pass score beats m9 on misses significantly.
- **Subset selection:** the K=2 / K=4 subsets were picked on train from 7 / 35 options; their train spread (0.950–0.956) is within noise.
- **Test conditions:** noise masks are not the stability probe's. Shift was tested only at ±6 px and scale at ±5 %, bilinear rather than the webapp's LANCZOS.

## Coordinator notes (independent verification)
- **Reproduced from scratch** by `../comparison/verify_multi.py`, which does not use this folder's code.
  - The mean-map scores match `scores_*.npz` to ≤ 2.5e-7, with identical harness rows and bootstrap intervals.
  - The K=8 mean-map tip gives p99 2.690 px with 7 misses, as above.
- **This mean map stacks with `single_pass`'s shape score.** bc T1.5 computed on the dihedral mean map reaches val AUROC>3 0.9665 at K=4 and 0.9687 at K=8, the best of all methods. Nothing was re-tuned. The plain m5 mass here keeps the best AP (0.311). See `../comparison/SUMMARY.md`.
