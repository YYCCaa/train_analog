# TipNet v9 confidence: single-pass map statistics (family 1)

> Written by the coordinating session from the research agent's report: the subagent guard blocked the agent's
> own write of this file. The text is the agent's; the last section adds the coordinator's verification notes.

All scores read only the logit map of the normal prediction (passes = 1). Hyperparameters are the train-split argmax of AUROC>3px (rule 2); val was touched only through `eval_conf.evaluate`. `results.jsonl` holds 14 methods × 2 splits.

Pipeline: `dump_logits.py` (logit cache, outside the repo) → `features.py` (80 per-map statistics) → `explore.py` (train screen of ~150 scores) → `tune_grid.py` (train grid) → `scores.py` (final one-map numpy functions, harness run, timing via `--time-only`).

## What I tried (train AUROC>3px; conf 0.9405, m9 0.9384)
- **Window masses:** squares 3..31 around the argmax and discs r = 1..15 around the tip. The best are 7×7 (0.9446) and disc r4 (0.9476); m31 drops to 0.909.
- **Sub-cell normalisation (works):** mass ÷ the mass a perfect σ = 1 target would hold at the same (u, v). conf_sub scores 0.9459; disc r3 at T 0.85 scores 0.9501. The ceiling spread (0.962–0.982) is as wide as the good images' conf IQR.
- **Shape / target conformity:**
  - Bhattacharyya Σ√(p·g) 0.9505, −KL(g‖p) 0.9501, −E_p[d²] 0.9501.
  - −entropy 0.9477, −TV 0.9482, −λ1 of the local covariance 0.9444.
- **Temperature:** T 1.25–2 helps a little (conf at T2: 0.9491). T < 1 hurts; bc and kl collapse at T ≤ 0.85.
- **Grid:** flat. mom(a 1–2, T 1–1.5), bc(T 1–2) and kl(T 1–2) all land at 0.950–0.951. The SE is about 0.005 with 126 bad images.
- **Second peak (fails on err3):** lmax − best logit outside Chebyshev 4 scores 0.9305; runner mass ratio 0.877; runner ÷ median local max 0.479. On val the margin catches misses well (flag%all 1.24) but is the worst on err3 (0.9452).
- **Floor (coordinator probe, fails):** train 0.912; val AUROC>7px 0.911 with flag%all 31.8. −log(median local-max prob) has rank correlation 0.988 with −logsumexp, and the raw floor logit hardly varies (sd 0.18, against 1.6 for logsumexp): the "floor ÷ peak" signal is the logsumexp normaliser, i.e. the peak height in disguise, not the background.
- **Logit space:** lmax 0.9186, logsumexp 0.912, log peak prob 0.908.
- **Combination** log conf_sub + k·log m9: far mass (outside 9×9) counts 1 + k times. Train is flat (0.9459–0.9469); its argmax is k = 16.

## Val
```
method                            split   AUROC>3px     AP>3px  AUROC>7px     95% CI   flag%all    flag%75        rho     p99@95    miss@95  AUROCjump     passes
conf (shipped, ref)               val        0.9458      0.243      0.989[0.978,0.996]       6.33       1.26      0.371       2.23          1      0.990          1
m9 (ref)                          val        0.9497      0.201      0.996[0.994,0.998]       1.47       0.44      0.409       2.32          0      0.996          1
sp_bc T1.5                        val        0.9589      0.243      0.994[0.989,0.997]       2.18       1.27      0.438       2.16          0      0.992          1
sp_conf_sub*m9^16                 val        0.9545      0.222      0.996[0.993,0.998]       1.35       0.68      0.419       2.21          0      0.996          1
sp_kl T2                          val        0.9599      0.217      0.990[0.983,0.995]       3.59       1.88      0.457       2.17          0      0.984          1
sp_floor -log(med lm prob)        val        0.9288      0.148      0.911[0.845,0.965]      31.75      10.18      0.445       2.36          6      0.917          1
```
Plus (AUROC>3px / AP / flag%all):
- conf_sub: 0.9527 / 0.247 / 5.50
- disc r3 T0.85: 0.9559 / 0.243 / 5.30
- mom2: 0.9547 / 0.200 / 1.99
- conf T2: 0.9579 / 0.211 / 3.04

On train every final method scores 0.947–0.951 and keeps no miss in the top 95 %.

## Cost
CPU numpy ms per map: best of 7 × 1000 val maps, minimum of 3 runs. The CPU is shared, so expect up to 2× noise.

| method | ms per map |
|---|---|
| shared prep (softmax + argmax + decode) | 0.095 |
| conf | 0.09 |
| m9 | 0.10 |
| conf_sub·m9^16 | 0.12 |
| bc T1.5 | 0.27 |
| mom2 | 0.31 |
| kl T2 | 0.49 |
| floor | 1.3 |

## Recommendation
- **Ship bc T1.5**, the overlap Σ√(p_T·g) between the T = 1.5 softmax and the ideal σ = 1 cell target at the decoded tip. Against conf: AUROC>3px 0.9458 → 0.9589, rho 0.371 → 0.438, p99@95 2.23 → 2.16 px, flag%all 6.33 → 2.18, AP unchanged.
- **Use conf_sub·m9^16 if decoys matter most.** flag%all 1.35, flag%75 0.68, AUROCjump 0.996, at the price of a lower AP (0.222).
- **Re-derive the webapp thresholds** (conf < 0.5) on whichever new scale ships.

## Caveats
- **13 misses:** all good methods' AUROC>7px intervals overlap, and flag%all hangs on a single image. For conf_sub·m9^16 it is the 8.5 px miss (1.35 %); for bc the 9.4 px miss (2.18 %). conf's 6.33 % comes from the confident 20.7 px miss, which bc ranks at 1.33 % and conf_sub·m9^16 at 0.68 %.
- **Train cannot see the far-mass trade-off:** it has 1 miss, and m9 loses to conf on train err3 but wins on val.
- **k is not identified by err3:** a 2-fold val cross-fit on err3 only picks k ≈ 64 on fold 0 and k ≈ 0–4 on fold 1.
- **The 3 saved methods were picked after the val run;** their hyperparameters were not. My pick before the run was mom a1 T1.25, which bc dominates.
- **The gains are small:** AUROC>3px improves by about 0.01, only about twice the SE. The ordering of bc, kl and disc is not significant.

The logit, feature and score caches (about 2.8 GB of logits) lived in the session scratchpad and are not kept.

## Coordinator notes (independent verification)
- **Reproduced in torch** by `../comparison/verify_single.py`, which does not use this folder's code. bc T1.5 and conf_sub·m9^16 match `scores_*.npz` to ≤ 2.2e-7 on both splits.
- **Paired bootstrap (val, 1000 resamples):**
  - bc − conf: ΔAUROC3 +0.013 [+0.006, +0.021], ΔAP3 −0.000 [−0.018, +0.016].
  - conf_sub·m9^16 − conf: +0.009 [+0.003, +0.015]; its AP loss is significant: −0.020 [−0.040, −0.002].
- **The floor finding holds on the stability probe's best / worst 1000:**
  - sd of the floor logit 0.18, against 2.46 for logsumexp.
  - P(worst > best) is 0.530 for the floor logit alone and 0.960 for −logsumexp alone, the same as the median-peak probability (0.959).
- **Picking after val is a small effect.** The pre-run pick, mom a1 T1.25, still scores 0.9548 on val (conf 0.9458).
- **bc stacks with `multi_pass`.** bc T1.5 on the dihedral mean map reaches val AUROC>3 0.9665 at K=4 and 0.9687 at K=8; see `../comparison/SUMMARY.md`.
