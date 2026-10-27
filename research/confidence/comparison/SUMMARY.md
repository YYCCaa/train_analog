# TipNet v9 confidence: all three families compared, verified, and combined

This folder adds three things on top of `single_pass/`, `multi_pass/` and `learned/`:
- an independent re-implementation of each family's recommended scores;
- one val table of every method with shared bootstrap resamples;
- one new method: the single-pass shape score computed on the multi-pass mean map.

Every number comes from the frozen harness in `../harness/eval_conf.py`.

## Pipeline (run from the repo root, in this order)
```
 verify_multi.py    8 dihedral passes, mean map → m5 / m9 at K = 2, 4, 8; mean-map tip      ─▶ verify_multi.npz
 verify_single.py   bc T1.5, conf_sub·m9^16 (torch, both splits) + the "floor = lse" test    ─▶ verify_single.npz
 verify_learned.py  LR3 from ../learned/model_lr3.json + out-of-fold isotonic calibration     ─▶ verify_learned.npz
        │   each prints max |mine − agent's committed scores| and the harness rows (logs/verify_*.log)
        ▼
 comparison_compute.py   all 14 methods + "bc on the mean map" → harness rows, paired bootstrap (1000
                         resamples shared by every method), reliability bins     ─▶ comparison_stats.npz,
                                                                                    comparison_results.jsonl
 combo_train_check.py    the combination on the 20k TRAIN sample (it was chosen after seeing val)
 comparison_plots.py     ─▶ figC1_methods.png, figC2_cost_calibration.png
```
The `verify_*` scripts read the agents' `scores_*.npz` only to compare. They never import the agents' code. All of
them need the data bundle (`BUNDLE` path, described in `../harness/README.md`) and a CUDA GPU. A full run takes
about 9 min.

## Result, by compute budget (val, 22 500 images)
| budget | score | AUROC >3 px | AP >3 px | ΔAUROC vs conf, 95 % CI | % flagged to catch all 13 misses |
|---|---|---|---|---|---|
| today | conf (shipped 5×5 mass) | 0.9458 | 0.243 | — | 6.33 |
| 1 pass | bc T1.5 (`single_pass`) | 0.9589 | 0.243 | +0.013 [+0.006, +0.021] | 2.18 |
| 1 pass | LR3 (`learned`) | 0.9610 | 0.234 | +0.015 [+0.007, +0.025] | 2.57 |
| 4 passes | dihedral mean-map m5 (`multi_pass`) | 0.9604 | 0.285 | +0.015 [+0.008, +0.023] | 3.20 |
| 4 passes | **bc on the dihedral mean map (new)** | **0.9665** | 0.267 | **+0.021 [+0.011, +0.032]** | 2.43 |
| 8 passes | dihedral mean-map m5 | 0.9644 | **0.311** | +0.019 [+0.010, +0.028] | 3.12 |
| 8 passes | **bc on the dihedral mean map (new)** | **0.9687** | 0.289 | **+0.023 [+0.013, +0.035]** | 2.23 |

All 14 rows, with every harness column, are in `comparison_results.jsonl` and `logs/comparison_compute.log`.

## The new method: shape score on the averaged map
```
 image ─┬─ K dihedral views ─▶ TipNet × K ─▶ undo each view ─▶ mean probability map M (128×128)
        │                                                            │
        └─ clean view ─▶ decoded tip (u, v) ─▶ ideal σ = 1 cell bump g ┤
                                                                     ▼
                         score = Σ over all cells √( p_T · g ),   p_T = M^(1/1.5) / Σ M^(1/1.5)
```
- **Why the two stack:** they catch different things.
  - The mean map catches a fragile tip, which wanders between views, so the averaged peak smears or splits.
  - bc catches a map whose shape is wrong: too wide, off-centre, or leaking to a decoy.
- **Nothing was tuned here.** T = 1.5 is `single_pass`'s train choice, and the K=4 views {identity, hflip, transpose, anti-transpose} are `multi_pass`'s train choice.
- **At K = 1 it equals bc T1.5 exactly** (max difference 4e-16), which checks the implementation.
- **Train check:** the combination was picked after the val results, so `combo_train_check.py` re-ran it on the train sample. It leads there too: 0.9582 at K=4, against 0.9535 for mean-map m5 and 0.9511 for single-pass bc. On both splits the plain mean-map m5 keeps the higher AP.

## Figures
- `figC1_methods.png`: every method against conf, showing ΔAUROC and ΔAP with paired-bootstrap 95 % intervals, and the % of images flagged to catch all 13 misses.
- `figC2_cost_calibration.png`:
  - left: AUROC against forward passes (K = 1, 2, 4, 8);
  - right: reliability of P(err > 3 px). Raw conf is far off; isotonic(conf) and isotonic(LR3) sit on the diagonal.

## Verification of the three families
| family | re-implemented | agreement with the agent's scores | harness rows and CIs |
|---|---|---|---|
| multi_pass | mean-map m5 / m9, K = 2, 4, 8; mean-map tip | ≤ 2.5e-7 | identical; tip p99 2.690 px, 7 misses |
| single_pass | bc T1.5, conf_sub·m9^16 | ≤ 2.2e-7 (both splits) | identical |
| learned | LR3 from its JSON | 9e-7 on val (train in the npz is out-of-fold, by design) | identical; isotonic ECE 0.029 → 0.0012 |

**One correction to an earlier coordinator claim.** The "background floor" (the median 9×9 local-max probability) is about 36× higher on the worst 1000 images. That is not a louder background: the floor logit is the same in both groups (P = 0.53). The softmax normaliser is what moves (P = 0.96), so the floor is the peak height in disguise and adds nothing as a score (val AUROC>3 0.929).

## Caveats
- **Misses:** only 13 on val. The miss columns hinge on 1–2 images; their intervals overlap for every good method.
- **Four stable misses:** four 7–9 px misses look identical in all 8 views and every statistic. No confidence score here flags them.
- **Calibration data:** the isotonic probabilities were fit on synthetic val. Refit them on real photos before showing them as percentages.
- **Cost:** K passes cost K × the model time. CPU post-processing per image, as the agents measured it:
  - bc 0.27 ms, LR3 0.5–0.8 ms, mean-map masses 0.9 ms at K=4 and 3.1 ms at K=8;
  - LR36 6.7 ms and the CNN about 9 ms, both unoptimised;
  - the new method was not timed. Estimate: the mean-map averaging plus one bc.
