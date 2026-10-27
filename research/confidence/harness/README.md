# TipNet v9 confidence research bundle (read-only)

> **In the repo:** this folder holds only the code and README of the bundle every number in
> `research/confidence/*` was measured with: `eval_conf.py` defines the metrics, and `confbundle.py` holds the loaders.
> The data files listed below were not committed: about 2.8 GB of image arrays, plus meta files and weights.
> They were built from `sv9_150k` and `runs/v9_tipnet_a/tip.pth` into a session scratchpad folder, and the
> scripts reference that path as `BUNDLE`.

Goal: a per-image **confidence score** that flags wrong tip predictions better than the shipped one.
Everything here is read-only. Never write into this folder; your outputs live in your own worktree.

## The model and the shipped confidence

```
gray 256x256 uint8 ─/255─▶ TipNet (100k params) ─▶ logits (128,128) float32 ─softmax over all cells─▶ probs, Σ=1
decode:      argmax cell (iy, ix) ─▶ softmax of the 5x5 LOGITS around it ─▶ centroid (u, v) ─▶ x = 2u+0.5, y = 2v+0.5
confidence:  Σ probs[iy-2:iy+3, ix-2:ix+3]   (global-softmax mass of the 5x5 cells around the argmax)
```
- **Training:** cross-entropy against a Gaussian (σ = 1 cell) normalised over the map, plus 0.1 × an L1 penalty on the soft-argmax coordinates, with dihedral augmentation (8 flips and rotations). The code is `train_v9_tip.py` in your worktree.
- **Ceiling:** a perfect model scores 0.962–0.982 on the shipped confidence, depending on where the tip sits inside its cell. The median on val is 0.970.
- **Webapp signals** (`tip_webapp/app/inference.py`):
  - **Candidates:** local maxima of probs over a 9×9 neighbourhood, with 5×5 mass ≥ 0.001.
  - **Ambiguous:** runner-up mass > 0.2 × the winner's, or confidence < 0.5.

## Known so far
- **Misses:** only 13 val images are off by more than 7 px. Most come from decoys (a triangle artifact, a shadow edge, a mark on the rim) being taken for the pointer.
  - Some misses are confident and stable, e.g. 61 px at confidence 0.72 and 20.7 px at 0.94: noise does not move them at all.
- **9×9 mass already beats the 5×5 confidence on misses** (table below). Blur inside the ring between the two windows is mostly harmless; mass far from the peak is what signals trouble.
- **Jumpers:** under training-style noise masks, 12 of the best/worst 1000 val images jump more than 7 px. All 12 have clean confidence < 0.9.

## Files
| file | content |
|---|---|
| `val_images.npy` | (22500, 256, 256) uint8, the trainer's validation split. **Report here.** |
| `train_images.npy` | (20000, 256, 256) uint8, a fixed sample of the training split. **Tune / fit here.** It has 126 images > 3 px and only 1 > 7 px. |
| `val_meta.npz`, `train_meta.npz` | rows, label, tip, err, conf, m9, peak, runner. See `confbundle.load_split`. |
| `stab_stats.npz` | The noise-stability probe: 10 masks × the best/worst 1000 val images. See `confbundle.load_stability`. |
| `tip.pth` | The model weights. |
| `confbundle.py` | Loaders, `load_model()` (CUDA, full float32 with TF32 off, matches the shipped TFLite within 1e-4), `logits()`, and the training-style noise masks. |
| `eval_conf.py` | **The frozen evaluation harness.** Every number you report comes from `evaluate()`. |

Run from your worktree root:
```python
import sys; sys.path.insert(0, r"<bundle path>")
import confbundle as cb, eval_conf as ev
val = cb.load_split("val"); model = cb.load_model(); lg = cb.logits(model, val["images"][:512])
ev.evaluate("val", my_score_val, "my method", passes=1, out="research/confidence/<family>/results.jsonl")
```

## Rules
1. **Score format:** one float per image, **higher = more trustworthy**.
2. **Choose hyperparameters and fit models without peeking at val.** Two options:
   - Use the **train** sample and its `auroc_err3` / `ap_err3`.
   - Use **2-fold cross-fitting on val** with `fold = np.arange(22500) % 2`: fit on fold 0 and score fold 1, then the reverse, and report the out-of-fold scores.

   Never tune on val miss metrics. With 13 misses they are noise; read the 95 % CI.
3. **Report each final method on both splits** with `evaluate(..., out=...)`, next to the baselines below.
4. **Report the cost:** forward passes per image (`passes=`), and the CPU ms per image of the post-processing.
5. **The hardware is shared** with other agents (one RTX 4090, about 18 GB free RAM):
   - memmap the image arrays;
   - keep GPU batches ≤ 512;
   - put a timeout on every command.

## Baselines (from `python eval_conf.py baselines`)
```
method                            split   AUROC>3px     AP>3px  AUROC>7px     95% CI   flag%all    flag%75        rho     p99@95    miss@95  AUROCjump     passes
conf (5x5 mass, shipped)          train      0.9405      0.143      0.998[0.997,0.998]       0.24       0.24      0.363       2.06          0        nan          1
m9 (9x9 mass)                     train      0.9384      0.116      0.999[0.999,0.999]       0.10       0.10      0.398       2.17          0        nan          1
peak (argmax cell prob)           train      0.9080      0.139      0.998[0.997,0.999]       0.19       0.19      0.271       2.04          0        nan          1
-runner/conf (webapp ratio)       train      0.5072      0.004      0.499[0.499,0.499]     100.00     100.00      0.033       2.61          1        nan          1
conf (5x5 mass, shipped)          val        0.9458      0.243      0.989[0.978,0.996]       6.33       1.26      0.371       2.23          1      0.990          1
m9 (9x9 mass)                     val        0.9497      0.201      0.996[0.994,0.998]       1.47       0.44      0.409       2.32          0      0.996          1
peak (argmax cell prob)           val        0.9192      0.201      0.916[0.797,0.985]      65.13       6.85      0.282       2.29          4      0.950          1
-runner/conf (webapp ratio)       val        0.5120      0.023      0.653[0.538,0.769]     100.00     100.00      0.027       3.04          9      0.707          1
```
