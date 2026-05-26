# v4 generator — tick-only ablation against Places365

**Date:** 2026-05-21
**Status:** Active (design)

## Why

The unified `autocal` model and its decoupled successors ([[2026-05-16-decouple-ellipse-mark0]]) localize the dial primarily from the bezel and dial-face silhouette — `val_center_mean` lands at ~5–10 px almost regardless of head architecture. The 10 scale-mark positions over-determine the ellipse on their own (≥5 points fix a conic), but it's not been measured how well the network can actually solve the task when forced to rely on tick + digit evidence alone.

v4 is an **ablation/diagnostic dataset.** Train Model J (and optionally Model E for a regression-head counterpart) on v4 and measure the gap to the same architecture trained on v3. The gap quantifies how much of v3's center accuracy came from bezel cues vs. tick-structure learning. After this single experiment v4 may not be used again; v5 is a separate decision.

This is **not** a deployment-target distribution. Real dials almost always have a visible bezel; a v4-only model would underperform on real images.

## Scope

A new generator `generate_synthetic_v4.py`, forked from `generate_synthetic_v3.py`. Renders **only scale marks + digits** at canonical angles `t_i = φ₀ + i · 36°` on top of a Places365 photo background. Every other dial element is stripped.

| v3 element | v4 |
|---|---|
| Procedural background fill | **Removed.** Always Places365 photo. |
| Background distractor text | **Removed.** |
| Intruder-dial augmentation | **Removed.** |
| Outer canvas / bezel + rim | **Removed.** |
| Dial-boundary polyline | **Removed.** |
| Scale marks | **Kept.** Same canonical-angle placement, same per-tick augmentations (curl, length jitter, blur, grey jitter, `TICK_COLORED_PROB` corner case). |
| Digits | **Kept.** Same placement, same `DIGIT_FONT_SIZE_RANGE = (18, 24)` — v3's 18 px minimum is already comfortably legible at Places365 resolutions. |
| Pointer + hub circle | **Removed.** |
| Post-processing (color/exposure jitter, triangular adversarial artifacts) | **Kept**, applied to whole canvas — identical to v3 so post-render noise is comparable. |

No changes to geometric sampling. `CX_DISP_RANGE`, `CY_DISP_RANGE`, `A_RANGE = (75, 90)`, `B_RATIO_RANGE = (0.8, 1.0)`, `THETA_RANGE = [-π/2, π/2]`, `PHI_0_RANGE = [0, 2π)` all carry over verbatim from v3. The only inter-version variable is visual content.

## Per-image ink color

The single new mechanism. With the bezel and dial face removed, scale marks and digits sit directly on an arbitrary photo; without adaptation, a black tick on a black wall vanishes and a white digit on a white sky vanishes. Mechanism: **one ink color per image, picked from the local photo luminance.**

Computed *after* all photo augmentation (flip + brightness/color/contrast jitter), because the model sees the post-jitter pixels and the ink choice should match:

```python
xs = max(0, int(cx - a))
ys = max(0, int(cy - a))
xe = min(IMG_SIZE, int(cx + a))
ye = min(IMG_SIZE, int(cy + a))
patch = np.asarray(bg.crop((xs, ys, xe, ye)))
if patch.size == 0:
    mean_lum = 128.0
else:
    mean_lum = (0.299 * patch[..., 0]
              + 0.587 * patch[..., 1]
              + 0.114 * patch[..., 2]).mean()
ink_color = (0, 0, 0) if mean_lum >= 128 else (255, 255, 255)
```

`ink_color` then replaces every tick/digit RGB triple v3 would otherwise compute from its grey/saturation ranges. **Exception:** v3's `TICK_COLORED_PROB = 0.10` branch (saturated-hue ticks instead of grey) is preserved; in 10% of images those ticks override the auto-color. This keeps the model from memorizing "ticks are pure {0,0,0} or {255,255,255}" — it still gets auto-color treatment 90% of the time.

The threshold (`128`) is a starting default. If smoke generation reveals systematic illegibility on one side (e.g., too many borderline photos picking the wrong ink), bias to 100 or 150.

### Why per-image, not per-tick or per-pixel

Realism analog: a real meter manufacturer picks one contrasting paint for the dial's ticks; the ticks of one dial share a color. Per-tick auto-color violates this and gives the model an unrealistic "each tick is whatever's locally most contrasted" cue. Per-pixel adaptation adds a v4-specific code path with no equivalent in v3, conflating the ablation result.

### Why no halo, no faint dial-face wash

Rejected during brainstorming:

- **Halo / 1-px stroke outline.** A natural mechanism (paint-on-substrate edge) but adds a contrast cue independent of background — could be too easy. Auto-color alone is sufficient on most photos; halo is a v4.1 follow-up if smoke generation shows residual illegibility.
- **Faint dial-face wash** (low-alpha circular fill at the dial size). Reintroduces a center-anchored substrate — a weaker version of the very cue the ablation removes. Strongly defeats the experiment's purpose.
- **Photo sampling bias** toward flat backgrounds. Changes what the ablation measures from "ticks vs. arbitrary photos" to "ticks vs. flat photos."

## Digit size (no change)

v3 already uses `DIGIT_FONT_SIZE_RANGE = (18, 24)` for scale digits — the 8-px minimum that motivated raising the floor came from `DISTRACTOR_FONT_SIZE_RANGE = (8, 32)`, which v4 drops with the rest of the distractor-text pipeline. v3's existing range carries over verbatim. All other digit augmentation (rotation, font choice; tick/digit color was previously palette-derived and is now `ink_color`) unchanged.

## Colored-ticks corner case

v3's `TICK_COLORED_PROB = 0.10` branch picks per-image saturated-hue ticks with HSV value `v ∈ [0.10, 0.30]` — i.e. dark saturated tones, tuned for a *light* dial face. Carried over verbatim to v4, this means ~10% of images get dark-coloured ticks regardless of the photo background; on a dark photo these will be poorly visible. Accepted as designed (the auto-color branch covers the other 90%), but worth re-evaluating if smoke generation shows colored-tick images concentrating in the failure tail of v4-J's val errors.

## CLI surface

Inherits v3's argparse. Differences:

| Flag | v3 default | v4 default |
|---|---|---|
| `--output-dir` | `sv3` | `sv4` |
| `--num-images` | 25000 | 25000 |
| `--bg-source` | `procedural` or `photo` | removed (or hard-validated to `photo`-only) |
| `--bg-photo-dir` | `~/places365` | `~/places365` |
| All other flags | unchanged | unchanged |

No changes to BG prefetching (`BG_PREFETCH`, `BG_PREFETCH_QUEUE`), reuse-ratio gate (`BG_PHOTO_MAX_REUSE_RATIO`), or photo augmentation ranges. All inherited from v3.

## Downstream code: zero changes required

- `AutoCalibDataset` in `autocal_utils.py` reads `<data-dir>/labels.csv` — schema identical to v3.
- Every `train_ellipse_*.py` and `train_mark0_*.py` accepts `--data-dir`. Train on v4 with `python train_ellipse_j.py --data-dir sv4 --save-onnx ellipse_model_j_v4.onnx`.
- `continue_ellipse.py` / `continue_mark0.py` likewise work via `--data-dir`.
- `infer_decoupled.py` is unchanged — it doesn't know about training data.

## Validation plan

1. **Smoke generation.** Generate 50 images to `sv4_smoke/`. Eyeball every one:
   - Are ticks visible against the photo?
   - Are digits readable?
   - Any leftover dial-structure leakage (stray bezel polyline, pointer, etc.)?
   - Any photo where auto-color picked the wrong direction?
2. **Geometry parity.** Generate 5 v4 images with a fixed seed; generate 5 v3 images with the same seed; verify the geometric labels (`cx, cy, a, b, theta, phi_0`, all 10 marks) match to within FP noise. Only visual content should differ.
3. **Full run.** 25k images to `sv4/` once smoke passes.
4. **First training experiment.** `train_ellipse_j.py --data-dir sv4 --save-onnx ellipse_model_j_v4.onnx` for 200 epochs. Compare:
   - v4-trained J on v4 val vs. v3-trained J on v3 val (each on its own distribution)
   - v4-trained J on v3 val (cross-distribution check)

   Optional second run on `EllipseModelE` for a regression-head counterpart.

## Success criteria

This is a *measurement* experiment, not a target-meeting one. There's no v4 val-error threshold to hit. The result is the comparison table from step 4 of the validation plan. Possible outcomes and what they'd mean:

- **v4-J ≈ v3-J on the marks metric.** Tick + digit structure carries the task; the bezel was a redundant cue. Worth promoting the v4 distribution to a curriculum stage (pretrain on v4 → fine-tune on v3) for robustness, or extending toward a v5 with mixed bezel-present / bezel-absent samples.
- **v4-J substantially worse than v3-J.** The bezel was load-bearing for shape learning. Phase 1 plan's "tick supervision alone is enough" assumption is weak; should re-evaluate the loss design or pretrain regimens. No promotion to curriculum.
- **v4-J center error fine, shape (a, b, θ) error explodes.** Ticks fix center well but the model can't recover orientation from 10 short line segments alone. Suggests adding a θ-specific auxiliary signal (e.g., explicit tick-orientation supervision).

## Out of scope for v4

- Halo / outline rendering
- Per-tick or per-pixel color adaptation
- Faint dial-face substrate wash
- Photo sampling bias / pre-filter
- A `--render-mode` flag on v3
- Refactor of v3 into composable renderers
- v5 design (separate decision after v4 results)
