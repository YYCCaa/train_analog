# Autocalibration Plan

Predict dial geometry (center, ellipse, scale-mark positions) from a box-selected dial region, so the downstream pointer-reading pipeline can be calibrated without per-installation manual work.

## Decision

**Phase 1: 6-DOF affine model.** Predict `(cx, cy, a, b, θ, φ₀)` — ellipse + rotational anchor — and derive the 10 scale-mark positions deterministically.

Why:
- **Direct parametric rendering.** Each synthetic image is drawn from sampled `(cx, cy, a, b, θ, φ₀)` via a canonical-circle → ellipse map, so labels are exact by construction. No back-fitting from stretch ratios.
- 6 scalars are easier to debug and regularize than 11 heatmaps.
- Wall/panel-mounted dials are typically viewed near head-on; the aspect-ratio constraint `b/a ∈ [0.8, 1.0]` covers that regime without wasting capacity on implausible tilts.
- Extends cleanly to 8-DOF later (sample 2 vanishing-line params) without throwing the model away.

Loss is direct param regression + differentiable mark-position regression (the marks are computed from the params inside the loss). No heatmap head, no decoder.

**Defer perspective (homography), radial distortion, and non-standard dial topologies until real-image evaluation shows they're needed.**

---

## Quick Start

End-to-end Phase 1 invocation:

```bash
# 1. Generate synthetic training data (defaults: 15k images, sv3/).
python generate_synthetic_v3.py --output-dir sv3 --num-images 15000

# 2. Train both backbones in parallel (run in separate shells).
python train_autocal_e.py --data-dir sv3 --save-onnx autocal_model_e.onnx
python train_autocal_f.py --data-dir sv3 --save-onnx autocal_model_f.onnx

# 3. Inference on one image (writes an overlay PNG).
python infer_autocal.py --model autocal_model_e.onnx --image path/to/dial.png --output overlay.png
```

For a custom training loop, import `AutoCalibDataset` and `autocal_loss` from `autocal_utils`. Architecture, loss, normalization, and label schema are all documented below.

---

## Phase 1 — Affine 6-DOF Model (do now)

### 1.1 Generator: `generate_synthetic_v3.py`

Replace v2's "draw circle, then affine-warp the canvas" pipeline with **direct parametric rendering.** Sample geometry, then draw the dial face by mapping canonical-unit-circle positions onto the parameterized ellipse. v2 is forked for textures (pointer style, noise, ticks, artifacts) but the geometric pipeline is rebuilt.

**Sampling per image:**
- `cx, cy` — uniform within image bounds, displacement up to ±40 px from center. Dial center stays inside the image by construction (the user's box-select contains the dial area), but the bezel can reach close to image edges.
- `a` — semi-major axis, uniform in `[30, 115]` px on a 256-canvas. Covers loose box (small `a`, dial ≈ half canvas) through tight box (large `a`, dial nearly fills bezel-and-all). Bias upper end if box-select UX defaults to tight crops.
- `b = a · r`, where `r ~ Uniform[0.8, 1.0]`. Enforces `a ≥ b`, breaking the a/b swap symmetry.
- `θ` — major-axis orientation, uniform in `[-π/2, π/2]` (period-π for an axis line).
- `φ₀` — canonical angle of the "0" mark, uniform in `[0, 2π)`. Real dials have varying "0" positions; if you only ever see "0 at top," shrink this range.
- `pointer_value` — uniform in `[0, 9]`. Pointer is a distractor, not a target.

**Canonical → image map.** For canonical point at radius `r_c ∈ [0, 1]` and angle `t`:

```
x = cx + r_c · (a·cos(t)·cosθ - b·sin(t)·sinθ)
y = cy + r_c · (a·cos(t)·sinθ + b·sin(t)·cosθ)
```

**Rendering order (image-space steps, then canonical → mapped steps):**

1. **Procedural background fill (full canvas).** Random HSV base color: hue uniform `[0, 1)`, saturation uniform `[0, 0.3]` (near-grey realistic surfaces), value uniform `[0.1, 0.95]` (dark panels through white walls). Apply hierarchical block + pixel noise (reuse v2's `HIER_NOISE_BLOCKS`, `HIER_NOISE_BLOCK_VARIANCE`, `HIER_NOISE_PIXEL_VARIANCE`) but across the entire canvas and at higher amplitude than v2's inside-dial pattern. Optional: one-sided gradient (directional wall lighting), per-image Gaussian blur (shallow DOF).

2. **Background distractor text.** `n ~ Uniform[0, 6]` strings rendered at random positions, **rejected if they overlap the outer bezel disc.** Each string: random font (arial, courier, times), size `Uniform[8, 32]` px, color contrasting against local background, rotation `Uniform[-15°, 15°]`. Content from a mixed pool — single digits `'0'-'9'`, multi-digit numbers `'12' '230' '0.5'`, alphanumeric codes `'A-50' 'V1.2' 'K12'`, short labels `'V' 'AC' 'kW' 'OFF'`. Teaches the model to ignore text outside the bezel.

2b. **Optional intruder dial (intrusion augmentation).** With probability 0.3, a complete neighbor dial is rendered whose center is placed *outside* the canvas edge, so only a random depth-fraction (10–50 %) of its bezel is visible at the frame edge. The intruder is drawn *before* the target so the target visually dominates. Labels are unchanged — they describe only the target. This makes the Phase 2 stage-2 per-dial autocal model robust to slivers of adjacent dials that appear at the edges of tight two-stage detector crops.

3. **Outer canvas (housing / bezel).** Always rendered. Annular fill + rim band, per v2's `OUTER_SCALE_RANGE = (1.25, 1.32)` machinery. Bezel ellipse = dial ellipse scaled by `outer_scale`.

4. **Dial boundary.** Sample `t` densely over `[0, 2π]`, map via the canonical formula, draw the faint dial circle as polyline.

5. **Scale marks.** Mark `i` at canonical angle `t_i = φ₀ + i · 36°`. Inner endpoint at `r_c = 1 - tick_len_frac`, outer at `r_c = 1.0`. Map both endpoints, draw line.

6. **Digits.** Text at canonical `t_i`, `r_c` slightly inside or outside ticks. `DIGIT_SHOW_PROB = 1.0` — without a visible anchor `φ₀` is unidentifiable.

7. **Pointer.** Hub circle + isosceles triangle in canonical space, apex at canonical angle `φ₀ + pointer_value · 36°`. All vertices mapped.

8. **Post-processing.** Color/exposure jitter, triangular adversarial artifacts. Applied across the whole canvas (not just inside the bezel) so the background gets jittered too.

**Output labels per image (`labels.csv`):**

```
filename, cx, cy, a, b, theta, phi_0,
mark_0_x, mark_0_y, ..., mark_9_x, mark_9_y
```

The 10 mark positions are computed in the generator from the canonical → image map at `r_c = 1.0`, `t = t_i`, and stored for direct supervision.

**Kept from v2 (texture pipeline):** pointer rendering style (gradient, edges, mounting hole), hierarchical noise machinery (block + pixel), color/exposure jitter, tick augmentation (counts, lengths, widths, jitter), triangular adversarial artifacts, outer canvas / bezel rendering (always visible — box-select is guaranteed to be more lenient than the bezel).

**Dropped from v2:**
- `STRETCH_H_RANGE`, `STRETCH_V_RANGE`, `SCALE_RANGE` — replaced by direct `(a, b)` sampling.
- The full-canvas affine warp at `generate_synthetic_v2.py:807-824`.
- `CUT_TIP` — set to `False`; pointer isn't the target.
- `DIGIT_SHOW_PROB < 1.0`.
- **Absolute-black outside the outer canvas** at `generate_synthetic_v2.py:948, 973` — replaced by procedural textured background filling the full canvas (rendering step 1).

**Shadow caveat.** v2's `draw_rim_shadows` (`generate_synthetic_v2.py:464`) assumes a circular dial; on an ellipse it renders incorrectly. Set `SHADOW_PROB = 0` initially. Revisit an ellipse-aware shadow model once Phase 1 trains and we know whether the absence of shadows is a sim-to-real gap.

### 1.2 Training: `train_autocal_e.py` / `train_autocal_f.py`

**Backbones:** train two variants in parallel — one forked from `model_e_inv_res_final`, one from `model_f_se_dwsep_final`. Compare on validation; pick the winner for production. Replace the heatmap output head on each with an **8-scalar regression head** (the 6 params packed as 8 outputs — see normalization below). No decoder, no auxiliary heatmap output.

**Loss:**

```
L = w_param · SmoothL1(pred_params_norm, true_params_norm)
  + w_mark  · SmoothL1(pred_marks_norm,  true_marks_norm)
```

- `pred_params_norm`: 8 scalars — `cx/IMG_SIZE`, `cy/IMG_SIZE`, `a/(IMG_SIZE/2)`, `b/(IMG_SIZE/2)`, `sin(2θ)`, `cos(2θ)` (note: `2θ` because θ has period π under the `a ≥ b` convention), `sin(φ₀)`, `cos(φ₀)`.
- `pred_marks_norm`: 10 marks × 2 coords ÷ `IMG_SIZE`. Computed inside the loss as a differentiable function of `pred_params` via the canonical → image map. The network does *not* output marks directly — they're derived.
- **Weights:** `w_param = 0.1`, `w_mark = 1.0`. Mark loss carries the main signal (task-direct gradient); param loss is a regularizer breaking parameter-space degeneracies (sign of θ, drift in φ₀, etc.). Retune by 3–5× empirically.
- **Loss type:** SmoothL1 (Huber) for robustness against the label-noise outliers we'll see when fine-tuning on real images later.

**Recovering raw params** at inference / inside the mark formula:
- `θ = ½ · atan2(sin2θ, cos2θ)`
- `φ₀ = atan2(sinφ₀, cosφ₀)`

**Data volume:** start at 15k images (shared across both backbones — single dataset, two training runs). Expand to 30k if validation underfits.

### 1.3 Inference: `infer_autocal.py`

Input: a box-selected (user-cropped) dial image, resized + padded to `IMG_SIZE`.
Output: 6 params + 10 derived mark positions, drawn on the input for visual validation.

### 1.4 Evaluation

On a synthetic held-out set (1k–2k images):
- Mean Euclidean error on `(cx, cy)`, in pixels.
- Mean relative error on `(a, b)`.
- Mean angular error on `θ, φ₀`, in degrees.
- **Headline metric:** mean Euclidean error per scale mark.

**Target:** < 2 px mean mark error on synthetic. This sets the calibration accuracy ceiling for downstream pointer reading.

---

## Phase 2 — Real-World Validation & Conditional Upgrades

Run only after Phase 1 ships and we have real images to evaluate against.

### 2.A Real-image fine-tuning  *(always do this)*

Collect 20–50 labeled real dial images. Label by manually marking 10 scale-mark positions; back-fit the 6 params from the marks (least-squares ellipse fit + tick-order anchoring).

If mean mark error on real > 5 px:
- Fine-tune Phase 1 weights with mixed synthetic + real batches (e.g., 4:1 synth:real).
- Texture domain gap is usually the dominant error source, and 5–20 labeled real images beats any amount of additional synthetic data for closing it.

### 2.B Texture realism upgrade  *(if 2.A residual is still high)*

Add to the generator:
- **Real photo backgrounds.** Composite the meter onto a random crop from a real-photo dataset (COCO / ImageNet / Unsplash, ~10k images). Strongest single sim-to-real bridge — replaces Phase 1's procedural background. Adds a dataset dependency.
- **Bezel text.** Model numbers, unit indicators, brand names rendered on the bezel ring (e.g., "Cal 25°C", "0–10 A", "AC ONLY"). Teaches the model that digits on the bezel ≠ scale digits.
- **Partial other-dials in background.** A second meter visible in the box, with its own ellipse + digits. Hardest distractor case. Only if real images show this commonly.
- Glass-cover specular reflections.
- Engraving highlights (paint-in-recess vs. raised-print appearance).
- Dust, scratches, sunlight glare patches.
- Non-uniform lighting (LED-strip rolloff, side-lit shadows).
- Digit font diversity beyond `arial.ttf`; include non-Latin numerals if production targets non-English markets.

### 2.C Homography augmentation  *(only if real tilts > 15–20°)*

Diagnostic: measure `b/a` on real images. If common values are `< 0.85`, real perspective is non-trivial.

- Fork `generate_synthetic_v4.py`. Replace the stretch step (`generate_synthetic_v2.py:807-824` analog) with a 4-point `Image.PERSPECTIVE` warp. Parameterize physically by tilt angle β and azimuth γ.
- Render the dial face on a canonical (un-warped) canvas first, *then* apply the perspective transform. This fixes "near-side ticks should appear larger than far-side ticks."
- Add 2 outputs `(v1, v2)` — vanishing-line params — to the model. Expect ~1.5–2× more synthetic images and ~1.5× more backbone parameters.
- **Critical:** bias β toward the observed real-world range, not uniform across `[0°, 45°]`. Weak-perspective identifiability means uniform sampling wastes capacity on ill-conditioned regimes (mild β is hard to distinguish from affine, and the gradient on `(v1, v2)` is near zero there).
- Two-stage option: keep the Phase 1 6-DOF head, add a small `(v1, v2)` refinement head conditioned on ellipse output + image features. Adds 10–20% params, degrades gracefully if perspective prediction is noisy.

### 2.D Radial distortion  *(only if production uses wide-angle cameras)*

- Apply mild barrel distortion *after* homography in the generator.
- If camera intrinsics are known per installation, undistort at inference and skip this.

### 2.E Non-standard dial topologies  *(only if real demand)*

- Predict scale count and starting value as additional outputs, OR train per-dial-family models.
- Probably out of scope until customer demand surfaces.

---

## Open questions to revisit during Phase 1

1. **Box-select robustness.** How tightly must the user crop? Test by varying displacement and `a` range in the synthetic distribution — too-tight crops should still calibrate correctly. The wider the training distribution, the more capacity is spent on rare crop styles.
2. **Dials without visible digits.** If production includes worn / missing digits, `φ₀` is fundamentally ambiguous from a symmetric tick pattern. Options:
   - Accept unreliable orientation; ask the user to click one known tick manually as a fallback UX.
   - Train explicitly on asymmetric tick patterns (major vs. minor lengths) as the orientation anchor.
3. **Pointer occlusion of scale marks.** The direct mark loss forces the model to predict marks that may sit behind the pointer in the image — exactly what we want for calibration (the calibration is the *dial's* geometry, not the visible-mark set). Sanity-check on synthetic that mark prediction stays accurate across all pointer positions.
4. **`a` sampling range vs. final image size.** `[40, 100]` px is a starting guess on a 256-px canvas. Match this to actual deployment: if the box-selected crop is always resized to fill the model input, real `a` will cluster near the maximum. Bias sampling accordingly.
