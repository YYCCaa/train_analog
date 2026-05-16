# Stage 1 Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CenterNet-style dial detector — Stage 1 of the two-stage multi-dial pipeline. Given a multi-meter image (1–4 dials), output a list of `(cx, cy, size)` square boxes, each suitable as the input crop for the existing Phase 1 single-dial autocal model (Stage 2).

**Architecture:**
- Backbone: reuse Model E and Model F encoders + decoders (same hourglass shapes as the original tip-prediction models — we keep the decoder this time since the output is a spatial heatmap, not a regression vector).
- Head: two separate 1×1 convs on the decoder output — one sigmoid heatmap channel (dial centers), one raw size channel (square crop side length, normalized to `[0, 1]` of input size).
- Loss: weighted MSE on the heatmap (reusing Phase 1's tip-task pattern) + L1 on the size channel at ground-truth peak locations only.
- Post-processing: 3×3 local-max + threshold + center-distance NMS.

**Tech Stack:** PyTorch, PIL, NumPy, ONNX Runtime, pytest. Same toolchain as Phase 1.

**Spec (inline):**
- Input image size: `512 × 512` (called `CANVAS_SIZE` to disambiguate from Phase 1's `IMG_SIZE = 256`).
- Heatmap output size: `256 × 256` (half-res, matches existing `half_res=True` backbone pattern).
- Dial count per training scene: uniform in `{1, 2, 3, 4}`.
- Placement: random, non-overlapping bezels (rejection sampling).
- Label format: one CSV row per dial — `filename, cx, cy, size` — so a 4-dial image produces 4 rows. Dataset groups by filename at init.
- The Phase 1 autocal model is unchanged. The v3 intrusion augmentation (commit `31b0a8a`) already covers adjacent-dial slivers in Stage 2 crops.

---

## File Structure

**New files:**
- `stage1_utils.py` — math (heatmap construction, peak picking), MultiDialDataset, model variants, loss, training utilities, ONNX export.
- `generate_multidial_v1.py` — multi-dial scene generator. Reuses v3's `_hsv_rgb`, `make_procedural_background`, `draw_distractor_text`, `draw_outer_canvas`, `draw_dial_boundary`, `draw_scale_ticks`, `draw_scale_digits`, `draw_pointer`, `add_triangle_artifacts` via import.
- `train_stage1_e.py` — Model E variant trainer.
- `train_stage1_f.py` — Model F variant trainer.
- `infer_stage1.py` — stage-1 only: run detector, draw `(cx, cy, size)` boxes.
- `infer_pipeline.py` — full pipeline: stage 1 detects → for each box crop → stage 2 (Phase 1 autocal) produces ellipse params → composite overlay.
- `tests/test_stage1_math.py`
- `tests/test_stage1_dataset.py`
- `tests/test_stage1_models.py`

**Unchanged but referenced:**
- `autocal_utils.py` — used by the pipeline inference for Stage 2.
- `generate_synthetic_v3.py` — its rendering helpers are imported by the new multi-dial generator.

---

## Conventions

- Tests use `pytest`. Run from project root.
- One commit per task.
- Commit prefix: `stage1: <what>`.
- `CANVAS_SIZE = 512` throughout stage 1.

---

## Task 1: Heatmap construction + peak picking

**Files:**
- Create: `tests/test_stage1_math.py`
- Create: `stage1_utils.py`

- [ ] **Step 1.1: Write failing tests**

Create `tests/test_stage1_math.py`:

```python
"""Tests for stage1_utils geometry math."""

import math

import numpy as np
import pytest

from stage1_utils import (
    CANVAS_SIZE,
    OUTPUT_SIZE,
    make_multi_peak_heatmap,
    find_peaks,
)


# ── make_multi_peak_heatmap ──────────────────────────────────────────────

def test_heatmap_empty_centers():
    hm = make_multi_peak_heatmap([], size=64, sigma=2.0)
    assert hm.shape == (64, 64)
    assert hm.dtype == np.float32
    assert hm.max() == 0.0


def test_heatmap_single_peak_at_center():
    hm = make_multi_peak_heatmap([(32, 32)], size=64, sigma=2.0)
    assert hm[32, 32] == pytest.approx(1.0, abs=1e-5)


def test_heatmap_falls_off_with_distance():
    hm = make_multi_peak_heatmap([(32, 32)], size=64, sigma=2.0)
    assert hm[32, 32] > hm[34, 32] > hm[40, 32]


def test_heatmap_multiple_peaks_via_elementwise_max():
    """Two non-overlapping peaks both reach 1.0."""
    hm = make_multi_peak_heatmap([(10, 10), (50, 50)], size=64, sigma=2.0)
    assert hm[10, 10] == pytest.approx(1.0, abs=1e-5)
    assert hm[50, 50] == pytest.approx(1.0, abs=1e-5)


def test_heatmap_overlapping_peaks_clipped_at_1():
    """Overlapping peaks combine via max, not sum — value stays ≤ 1."""
    hm = make_multi_peak_heatmap([(32, 32), (32, 33)], size=64, sigma=2.0)
    assert hm.max() == pytest.approx(1.0, abs=1e-5)


# ── find_peaks ───────────────────────────────────────────────────────────

def test_find_peaks_empty():
    hm = np.zeros((32, 32), dtype=np.float32)
    assert find_peaks(hm, threshold=0.3) == []


def test_find_peaks_single_peak():
    hm = make_multi_peak_heatmap([(16, 16)], size=32, sigma=2.0)
    peaks = find_peaks(hm, threshold=0.3, min_distance=4)
    assert len(peaks) == 1
    y, x, score = peaks[0]
    assert (y, x) == (16, 16)
    assert score == pytest.approx(1.0, abs=1e-5)


def test_find_peaks_two_separate_peaks():
    hm = make_multi_peak_heatmap([(10, 10), (25, 25)], size=32, sigma=2.0)
    peaks = find_peaks(hm, threshold=0.3, min_distance=4)
    assert len(peaks) == 2
    coords = sorted([(y, x) for y, x, _ in peaks])
    assert coords == [(10, 10), (25, 25)]


def test_find_peaks_nms_suppresses_nearby():
    """Two peaks within min_distance — only the stronger survives."""
    hm = np.zeros((32, 32), dtype=np.float32)
    hm[16, 16] = 1.0
    hm[16, 18] = 0.9  # 2 px away, weaker
    peaks = find_peaks(hm, threshold=0.3, min_distance=5)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (16, 16)


def test_find_peaks_threshold_filters():
    hm = np.zeros((16, 16), dtype=np.float32)
    hm[5, 5] = 0.2  # below threshold
    hm[10, 10] = 0.9  # above
    peaks = find_peaks(hm, threshold=0.3, min_distance=4)
    assert len(peaks) == 1
    assert (peaks[0][0], peaks[0][1]) == (10, 10)


def test_constants_present():
    assert CANVAS_SIZE == 512
    assert OUTPUT_SIZE == 256
```

- [ ] **Step 1.2: Run tests, verify they fail**

Run: `pytest tests/test_stage1_math.py -v`
Expected: `ModuleNotFoundError: No module named 'stage1_utils'`.

- [ ] **Step 1.3: Implement `stage1_utils.py` (math portion)**

Create `stage1_utils.py`:

```python
"""Stage 1 (CenterNet-style dial detector) shared utilities.

Contents:
  - Constants: CANVAS_SIZE, OUTPUT_SIZE, etc.
  - Math: make_multi_peak_heatmap, find_peaks.

Future tasks will add: MultiDialDataset, Stage1ModelE/F, loss, training loop.
"""

import numpy as np


CANVAS_SIZE = 512        # detector input H = W = 512
OUTPUT_SIZE = 256        # heatmap output H = W = 256 (half-res decoder)
N_DIALS_MAX = 4          # cap on dials per training scene


# ── Heatmap construction ────────────────────────────────────────────────

def make_multi_peak_heatmap(centers, size, sigma):
    """Build a (size, size) Gaussian heatmap with peaks at each (x, y) in centers.

    Peaks combine via element-wise max, so overlapping peaks stay ≤ 1.0.
    Empty centers → all-zero heatmap.

    Args:
        centers: iterable of (x_px, y_px) tuples in heatmap-pixel coordinates.
        size: int, output side length.
        sigma: float, Gaussian standard deviation in heatmap pixels.

    Returns:
        float32 numpy array of shape (size, size), values in [0, 1].
    """
    hm = np.zeros((size, size), dtype=np.float32)
    if not centers:
        return hm
    x = np.arange(size, dtype=np.float32)
    y = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    for cx, cy in centers:
        peak = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
        np.maximum(hm, peak, out=hm)
    return hm


# ── Peak picking ────────────────────────────────────────────────────────

def find_peaks(heatmap, threshold=0.3, min_distance=10):
    """Find local maxima above threshold, then center-distance NMS.

    Args:
        heatmap: (H, W) numpy array of float scores in [0, 1].
        threshold: only consider pixels at least this hot.
        min_distance: NMS radius (in heatmap pixels); only one peak per disc.

    Returns:
        list of (y_px, x_px, score_float) tuples, sorted by score descending.
        Empty list if no peaks pass threshold.
    """
    H, W = heatmap.shape
    # 3x3 local max via 9-way shifted reduction with edge padding.
    padded = np.pad(heatmap, 1, mode="edge")
    views = [padded[dy:dy + H, dx:dx + W]
             for dy in range(3) for dx in range(3)]
    local_max = np.maximum.reduce(views)
    is_peak = (heatmap == local_max) & (heatmap >= threshold)

    ys, xs = np.where(is_peak)
    if len(ys) == 0:
        return []
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)

    selected = []
    min_d2 = min_distance ** 2
    for idx in order:
        y, x = int(ys[idx]), int(xs[idx])
        v = float(scores[idx])
        if all((y - sy) ** 2 + (x - sx) ** 2 >= min_d2
               for sy, sx, _ in selected):
            selected.append((y, x, v))
    return selected
```

- [ ] **Step 1.4: Run tests, verify they pass**

Run: `pytest tests/test_stage1_math.py -v`
Expected: all 11 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add tests/test_stage1_math.py stage1_utils.py
git commit -m "stage1: heatmap construction and peak picking with tests"
```

---

## Task 2: Loss function

**Files:**
- Modify: `tests/test_stage1_math.py` (append)
- Modify: `stage1_utils.py` (append)

- [ ] **Step 2.1: Append failing tests**

Append to `tests/test_stage1_math.py`:

```python
# ── Loss ─────────────────────────────────────────────────────────────────

import torch

from stage1_utils import stage1_loss


def test_loss_zero_at_truth():
    """Loss is zero when prediction matches ground truth exactly."""
    B, H, W = 1, 32, 32
    gt_hm = torch.zeros(B, 1, H, W)
    gt_hm[0, 0, 16, 16] = 1.0
    gt_size = torch.zeros(B, 1, H, W)
    gt_size[0, 0, 16, 16] = 0.5  # normalized size = 0.5
    gt_mask = torch.zeros(B, 1, H, W)
    gt_mask[0, 0, 16, 16] = 1.0

    pred = torch.cat([gt_hm, gt_size], dim=1)  # (B, 2, H, W)
    total, l_hm, l_size = stage1_loss(pred, gt_hm, gt_size, gt_mask)
    assert total.item() == pytest.approx(0.0, abs=1e-6)
    assert l_hm.item() == pytest.approx(0.0, abs=1e-6)
    assert l_size.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_heatmap_penalised():
    """Loss > 0 when heatmap prediction is wrong."""
    B, H, W = 1, 32, 32
    gt_hm = torch.zeros(B, 1, H, W)
    gt_hm[0, 0, 16, 16] = 1.0
    gt_size = torch.zeros(B, 1, H, W)
    gt_mask = torch.zeros(B, 1, H, W)
    gt_mask[0, 0, 16, 16] = 1.0

    # Predict heatmap of all zeros — wrong.
    pred_hm = torch.zeros(B, 1, H, W)
    pred_size = gt_size.clone()
    pred = torch.cat([pred_hm, pred_size], dim=1)
    total, l_hm, _ = stage1_loss(pred, gt_hm, gt_size, gt_mask)
    assert l_hm.item() > 0
    assert total.item() > 0


def test_loss_size_only_at_mask():
    """Size loss is masked: only evaluated at peak locations (gt_mask == 1)."""
    B, H, W = 1, 16, 16
    gt_hm = torch.zeros(B, 1, H, W)
    gt_hm[0, 0, 8, 8] = 1.0
    gt_size = torch.zeros(B, 1, H, W)
    gt_size[0, 0, 8, 8] = 0.4
    gt_mask = torch.zeros(B, 1, H, W)
    gt_mask[0, 0, 8, 8] = 1.0

    pred_hm = gt_hm.clone()
    # Garbage size predictions everywhere EXCEPT the peak.
    pred_size = torch.full((B, 1, H, W), 99.0)
    pred_size[0, 0, 8, 8] = 0.4  # correct at peak
    pred = torch.cat([pred_hm, pred_size], dim=1)
    total, _, l_size = stage1_loss(pred, gt_hm, gt_size, gt_mask)
    # Non-peak garbage is masked out; loss should be near zero.
    assert l_size.item() == pytest.approx(0.0, abs=1e-6)


def test_loss_backprop():
    """Gradient flows back through both heads."""
    pred = torch.zeros(1, 2, 16, 16, requires_grad=True)
    gt_hm = torch.zeros(1, 1, 16, 16)
    gt_hm[0, 0, 8, 8] = 1.0
    gt_size = torch.zeros(1, 1, 16, 16)
    gt_size[0, 0, 8, 8] = 0.4
    gt_mask = torch.zeros(1, 1, 16, 16)
    gt_mask[0, 0, 8, 8] = 1.0
    total, _, _ = stage1_loss(pred, gt_hm, gt_size, gt_mask)
    total.backward()
    assert pred.grad is not None
    assert torch.isfinite(pred.grad).all()
    assert (pred.grad.abs() > 0).any()
```

- [ ] **Step 2.2: Run tests, verify they fail**

Run: `pytest tests/test_stage1_math.py -v`
Expected: new tests FAIL — `stage1_loss` not defined.

- [ ] **Step 2.3: Append loss to `stage1_utils.py`**

Append to `stage1_utils.py`:

```python
# ── Loss ─────────────────────────────────────────────────────────────────

import torch
import torch.nn.functional as F


def stage1_loss(pred, gt_heatmap, gt_size, gt_mask,
                w_pos=10.0, w_size=1.0):
    """Combined heatmap + masked-size loss.

    Args:
        pred: (B, 2, H, W) — channel 0 = predicted heatmap (already sigmoid'd
              by the model), channel 1 = predicted normalized size.
        gt_heatmap: (B, 1, H, W) — Gaussian target heatmap in [0, 1].
        gt_size: (B, 1, H, W) — normalized target size; only valid where mask=1.
        gt_mask: (B, 1, H, W) — 1 at peak pixels, 0 elsewhere.
        w_pos: weight multiplier on heatmap MSE near peak (target > 0.1).
        w_size: weight on the size loss in the total.

    Returns:
        (total_loss, heatmap_loss, size_loss)
    """
    pred_hm = pred[:, 0:1]
    pred_size = pred[:, 1:2]

    # Weighted MSE on heatmap (Phase 1 tip pattern).
    weight = torch.ones_like(gt_heatmap)
    weight[gt_heatmap > 0.1] = w_pos
    l_hm = (weight * (pred_hm - gt_heatmap) ** 2).mean()

    # L1 on size at masked locations only.
    n_peaks = gt_mask.sum().clamp(min=1.0)  # avoid divide-by-zero
    l_size = (gt_mask * (pred_size - gt_size).abs()).sum() / n_peaks

    total = l_hm + w_size * l_size
    return total, l_hm, l_size
```

- [ ] **Step 2.4: Run tests, verify they pass**

Run: `pytest tests/test_stage1_math.py -v`
Expected: all 15 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add tests/test_stage1_math.py stage1_utils.py
git commit -m "stage1: heatmap + masked-size loss with tests"
```

---

## Task 3: Multi-dial generator — parameter sampling and non-overlap placement

**Files:**
- Create: `generate_multidial_v1.py`

- [ ] **Step 3.1: Create the generator skeleton with sampling logic**

Create `generate_multidial_v1.py`:

```python
"""Generate synthetic multi-dial training images for the Stage 1 detector.

Each image contains 1–4 dials placed at non-overlapping positions on a
CANVAS_SIZE × CANVAS_SIZE canvas. Each dial is rendered via the v3
single-dial helpers (background, bezel, ticks, digits, pointer, etc.) —
this generator only adds the multi-dial composition layer.

Labels: one CSV row per dial — `filename, cx, cy, size` — so a 4-dial image
produces 4 rows. The "size" is the side length (in pixels) of the square
crop centered on the dial that comfortably contains the bezel; downstream
this is what Stage 2 will crop and resize.
"""

import argparse
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

import generate_synthetic_v3 as v3
from stage1_utils import CANVAS_SIZE, N_DIALS_MAX


OUTPUT_DIR = "md1"
NUM_IMAGES = 15000

# ── Scene composition ───────────────────────────────────────────────────
N_DIALS_RANGE = (1, N_DIALS_MAX)            # uniform over {1, 2, 3, 4}
PLACEMENT_MAX_ATTEMPTS = 50                  # rejection sampling tries per dial
SIZE_MARGIN_FRAC = 0.08                      # extra padding around bezel for crop size

# ── Per-dial geometry (override v3 ranges for the larger canvas) ────────
DIAL_A_RANGE = (40, 130)                     # semi-major axis, pixels
DIAL_B_RATIO_RANGE = (0.8, 1.0)
DIAL_THETA_RANGE = (-math.pi / 2, math.pi / 2)
DIAL_PHI_0_RANGE = (0.0, 2 * math.pi)
DIAL_POINTER_VALUE_RANGE = (0.0, 9.0)

# ── Color/exposure jitter (canvas-level, post composition) ──────────────
BRIGHTNESS_RANGE = (0.85, 1.30)
CONTRAST_RANGE = (0.80, 1.20)
SATURATION_RANGE = (0.75, 1.25)
SHARPNESS_RANGE = (0.7, 1.5)


# ── Single-dial sampling (adapted for variable canvas) ──────────────────

def _sample_dial_params():
    """Sample one dial's geometric params, WITHOUT placing it on canvas.

    Caller assigns cx, cy after rejection sampling. Returns dict with the
    same shape as v3.sample_params() except cx/cy are placeholders (None).
    """
    a = random.uniform(*DIAL_A_RANGE)
    b = a * random.uniform(*DIAL_B_RATIO_RANGE)
    theta = random.uniform(*DIAL_THETA_RANGE)
    phi_0 = random.uniform(*DIAL_PHI_0_RANGE)
    pointer_value = random.uniform(*DIAL_POINTER_VALUE_RANGE)
    return {
        "cx": None, "cy": None, "a": a, "b": b,
        "theta": theta, "phi_0": phi_0,
        "pointer_value": pointer_value,
    }


def _bezel_radius(dial, outer_scale):
    """Conservative pixel radius enclosing the entire bezel (any orientation)."""
    return dial["a"] * outer_scale  # a ≥ b, so a*outer is the max extent


def _bezels_overlap(d1, s1, d2, s2):
    """True iff the bezel discs of dial1 and dial2 intersect (incl. margin)."""
    r1 = _bezel_radius(d1, s1)
    r2 = _bezel_radius(d2, s2)
    dx = d1["cx"] - d2["cx"]
    dy = d1["cy"] - d2["cy"]
    return dx * dx + dy * dy < (r1 + r2) ** 2


def sample_scene(canvas_size=CANVAS_SIZE):
    """Sample a complete scene: a list of (dial_params, dial_h, dial_s, outer_scale).

    Number of dials uniform in [1, N_DIALS_MAX]. Dials are placed at random
    positions via rejection sampling; bezels do not overlap. If placement
    fails after PLACEMENT_MAX_ATTEMPTS per dial, that dial is skipped (the
    scene ends up with fewer dials than originally requested).

    Returns a list of dicts:
        { "params": dial_param_dict (with cx, cy set),
          "dial_h": float, "dial_s": float, "outer_scale": float }
    """
    n_target = random.randint(*N_DIALS_RANGE)
    placed = []
    for _ in range(n_target):
        for _attempt in range(PLACEMENT_MAX_ATTEMPTS):
            d = _sample_dial_params()
            outer_scale = random.uniform(*v3.OUTER_SCALE_RANGE)
            r = _bezel_radius(d, outer_scale)
            d["cx"] = random.uniform(r, canvas_size - r)
            d["cy"] = random.uniform(r, canvas_size - r)
            if all(not _bezels_overlap(d, outer_scale,
                                       p["params"], p["outer_scale"])
                   for p in placed):
                dial_h = random.random()
                dial_s = random.uniform(*v3.DIAL_S_RANGE)
                placed.append({
                    "params": d,
                    "dial_h": dial_h,
                    "dial_s": dial_s,
                    "outer_scale": outer_scale,
                })
                break
        # If we couldn't place this dial in PLACEMENT_MAX_ATTEMPTS,
        # silently skip it. The scene just has fewer dials.
    return placed
```

- [ ] **Step 3.2: Smoke-test sampling**

Run:
```bash
python -c "
import random; random.seed(0)
from generate_multidial_v1 import sample_scene
for _ in range(5):
    s = sample_scene()
    print(f'scene with {len(s)} dial(s):')
    for d in s:
        p = d['params']
        print(f'  cx={p[\"cx\"]:.1f} cy={p[\"cy\"]:.1f} a={p[\"a\"]:.1f} b={p[\"b\"]:.1f}')
"
```
Expected: 5 scenes printed, each with 1–4 dials, each dial's center within `[0, 512]` and bezels non-overlapping by inspection (centers spaced by at least ~`a_i + a_j`).

- [ ] **Step 3.3: Commit**

```bash
git add generate_multidial_v1.py
git commit -m "stage1: multi-dial generator skeleton + non-overlap placement"
```

---

## Task 4: Multi-dial generator — rendering + main loop + label CSV

**Files:**
- Modify: `generate_multidial_v1.py` (append)

- [ ] **Step 4.1: Append rendering, top-level driver, and main**

Append to `generate_multidial_v1.py`:

```python
# ── Per-dial rendering on the shared canvas ─────────────────────────────

def _render_dial_on_canvas(img, dial_entry):
    """Render one dial onto img using v3's helpers.

    Mirrors v3.generate_image's per-dial steps 3–7 (outer canvas, dial
    boundary, ticks, digits, pointer) but ALL on the shared multi-dial
    canvas.
    """
    params = dial_entry["params"]
    dial_h = dial_entry["dial_h"]
    dial_s = dial_entry["dial_s"]
    outer_scale = dial_entry["outer_scale"]
    img, _ = v3.draw_outer_canvas(img, params, outer_scale, dial_h, dial_s)
    img = v3.draw_dial_boundary(img, params, dial_h, dial_s)
    img = v3.draw_scale_ticks(img, params)
    img = v3.draw_scale_digits(img, params)
    img = v3.draw_pointer(img, params)
    return img


def _crop_size_for(dial_entry):
    """Pixel side length of the square crop centered on the dial.

    Includes a margin so Stage 2 sees the full bezel plus a small border —
    matching the v3-trained Phase 1 distribution.
    """
    r = _bezel_radius(dial_entry["params"], dial_entry["outer_scale"])
    return 2.0 * r * (1.0 + SIZE_MARGIN_FRAC)


def generate_image(path, canvas_size=CANVAS_SIZE):
    """Generate one multi-dial synthetic image. Returns a list of label dicts.

    Each label dict has keys: filename, cx, cy, size — one entry per dial.
    """
    scene = sample_scene(canvas_size=canvas_size)

    # 1. Procedural background fill (v3 helper, parameterized by canvas size).
    img = v3.make_procedural_background(canvas_size)

    # 2. Background distractor text — use the FIRST dial's bezel info just
    # so the v3 helper has a reference disc to reject overlap against. If
    # there are no dials (extremely rare), skip distractor text.
    if scene:
        # Treat the BIGGEST bezel as the reference for distractor rejection.
        biggest = max(scene, key=lambda d: _bezel_radius(d["params"],
                                                        d["outer_scale"]))
        img = v3.draw_distractor_text(img, biggest["params"],
                                      biggest["outer_scale"])
        # Then reject text that overlaps any other dial too.
        # Simple two-pass: re-run distractor with each other dial as reference.
        # We accept some text near non-biggest dials; the model can still learn.

    # 3. Render each dial in scene order.
    for dial_entry in scene:
        img = _render_dial_on_canvas(img, dial_entry)

    # 4. Triangle artifacts across whole canvas.
    img = v3.add_triangle_artifacts(img)

    # 5. Colour / exposure jitter.
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
    img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # 6. Save.
    img.save(path)

    # 7. Build label rows: one per dial.
    fname = os.path.basename(path)
    rows = []
    for dial_entry in scene:
        p = dial_entry["params"]
        rows.append({
            "filename": fname,
            "cx": round(p["cx"], 3),
            "cy": round(p["cy"], 3),
            "size": round(_crop_size_for(dial_entry), 3),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic multi-dial Stage 1 detector images")
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"directory to write PNGs and labels.csv (default: {OUTPUT_DIR})")
    parser.add_argument(
        "--num-images", type=int, default=NUM_IMAGES,
        help=f"number of synthetic images to generate (default: {NUM_IMAGES})")
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be a positive integer")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "labels.csv")
    fieldnames = ["filename", "cx", "cy", "size"]

    total_rows = 0
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.num_images):
            fname = f"scene_{i:05d}.png"
            fpath = os.path.join(args.output_dir, fname)
            rows = generate_image(fpath)
            for row in rows:
                writer.writerow(row)
            total_rows += len(rows)

    print(f"Generated {args.num_images} images in '{args.output_dir}/'")
    print(f"Labels written to '{csv_path}' ({total_rows} dial rows)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4.2: Generate a tiny dataset and sanity-check**

Run:
```bash
python generate_multidial_v1.py --output-dir md1_smoke --num-images 10
```
Expected: `Generated 10 images in 'md1_smoke/'`. The directory contains 10 PNGs + `labels.csv` with somewhere between 10 and 40 rows.

- [ ] **Step 4.3: Verify labels match rendered geometry**

Create `check_md1_labels.py` at the project root:

```python
"""Overlay center-and-size boxes on the first 3 generated scenes."""

import csv
import os
from collections import defaultdict
from PIL import Image, ImageDraw

DATA_DIR = "md1_smoke"

rows_by_file = defaultdict(list)
with open(os.path.join(DATA_DIR, "labels.csv"), newline="") as f:
    for row in csv.DictReader(f):
        rows_by_file[row["filename"]].append(row)

for fname in list(rows_by_file)[:3]:
    img = Image.open(os.path.join(DATA_DIR, fname)).convert("RGB")
    d = ImageDraw.Draw(img)
    for row in rows_by_file[fname]:
        cx, cy = float(row["cx"]), float(row["cy"])
        s = float(row["size"]) / 2
        d.rectangle([cx - s, cy - s, cx + s, cy + s],
                    outline=(0, 255, 0), width=3)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 0, 0))
    img.save(os.path.join(DATA_DIR, "check_" + fname))
    print(f"check_{fname}: {len(rows_by_file[fname])} dial(s)")
```

Run: `python check_md1_labels.py`

Inspect (programmatically — you may not have a viewer):
- The 3 `check_*.png` files exist and have non-trivial size.
- Open one with PIL, count green pixels (rectangle outlines) and red pixels (center dots):
  - Number of red dots should equal the number of dials reported (1–4).
  - Each green rectangle should enclose a real dial bezel (i.e., the rectangle's interior should contain dial-face / bezel pixels, not solid background).

- [ ] **Step 4.4: Cleanup and commit**

```bash
python -c "import shutil, os; shutil.rmtree('md1_smoke', ignore_errors=True); os.remove('check_md1_labels.py') if os.path.exists('check_md1_labels.py') else None"
git add generate_multidial_v1.py
git commit -m "stage1: multi-dial generator main loop with per-dial CSV labels"
```

---

## Task 5: MultiDialDataset

**Files:**
- Modify: `tests/test_stage1_math.py` (append) OR create `tests/test_stage1_dataset.py`
- Modify: `stage1_utils.py` (append)

Create a new test file to keep concerns separated.

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_stage1_dataset.py`:

```python
"""Tests for MultiDialDataset."""

import csv
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from stage1_utils import (
    MultiDialDataset,
    CANVAS_SIZE,
    OUTPUT_SIZE,
)


def _write_tiny_md_dataset(tmp_dir):
    """Write 3 blank canvases and a labels.csv with varying dial counts."""
    rows = [
        # img_0.png: 2 dials
        {"filename": "img_0.png", "cx": 128.0, "cy": 128.0, "size": 180.0},
        {"filename": "img_0.png", "cx": 384.0, "cy": 384.0, "size": 200.0},
        # img_1.png: 1 dial
        {"filename": "img_1.png", "cx": 256.0, "cy": 256.0, "size": 240.0},
        # img_2.png: 4 dials
        {"filename": "img_2.png", "cx": 128.0, "cy": 128.0, "size": 160.0},
        {"filename": "img_2.png", "cx": 384.0, "cy": 128.0, "size": 160.0},
        {"filename": "img_2.png", "cx": 128.0, "cy": 384.0, "size": 160.0},
        {"filename": "img_2.png", "cx": 384.0, "cy": 384.0, "size": 160.0},
    ]
    for fname in ("img_0.png", "img_1.png", "img_2.png"):
        Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE),
                  (100, 100, 100)).save(os.path.join(tmp_dir, fname))
    with open(os.path.join(tmp_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["filename", "cx", "cy", "size"])
        w.writeheader()
        w.writerows(rows)


def test_dataset_groups_rows_by_filename():
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_md_dataset(tmp)
        ds = MultiDialDataset(tmp)
        assert len(ds) == 3  # 3 distinct images


def test_dataset_returns_correct_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_md_dataset(tmp)
        ds = MultiDialDataset(tmp)
        img, hm, size_map, mask, fname = ds[0]
        assert img.shape == (3, CANVAS_SIZE, CANVAS_SIZE)
        assert img.dtype == torch.float32
        assert hm.shape == (1, OUTPUT_SIZE, OUTPUT_SIZE)
        assert size_map.shape == (1, OUTPUT_SIZE, OUTPUT_SIZE)
        assert mask.shape == (1, OUTPUT_SIZE, OUTPUT_SIZE)
        assert isinstance(fname, str)


def test_dataset_heatmap_has_correct_peak_count():
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_md_dataset(tmp)
        ds = MultiDialDataset(tmp)
        # Find the 4-dial image (img_2.png) and verify heatmap has 4 peaks.
        for i in range(len(ds)):
            _, hm, _, mask, fname = ds[i]
            if fname == "img_2.png":
                # Count mask pixels — should be exactly 4 (one per dial).
                assert mask.sum().item() == pytest.approx(4.0)
                # Heatmap should have 4 high spots.
                assert hm.max().item() == pytest.approx(1.0, abs=1e-4)
                return
        pytest.fail("img_2.png not found in dataset")


def test_dataset_size_map_normalized():
    """size_map values at peak locations equal raw_size / CANVAS_SIZE."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_md_dataset(tmp)
        ds = MultiDialDataset(tmp)
        for i in range(len(ds)):
            _, _, size_map, mask, fname = ds[i]
            if fname == "img_1.png":
                # Single dial, size=240. Mask peak should be at (128, 128) in
                # half-res coords (256/512 * 256 = 128).
                # Mean size value at masked pixels should be 240/512 ≈ 0.469.
                masked_sizes = size_map[mask > 0]
                assert masked_sizes.numel() >= 1
                assert masked_sizes.mean().item() == pytest.approx(
                    240.0 / CANVAS_SIZE, abs=1e-3)
                return
        pytest.fail("img_1.png not found")
```

- [ ] **Step 5.2: Run tests, verify they fail**

Run: `pytest tests/test_stage1_dataset.py -v`
Expected: FAIL — `MultiDialDataset` not defined.

- [ ] **Step 5.3: Append dataset to `stage1_utils.py`**

Append to `stage1_utils.py`:

```python
# ── Dataset ──────────────────────────────────────────────────────────────

import csv
import os
from collections import defaultdict

from PIL import Image
from torch.utils.data import Dataset

HEATMAP_SIGMA = 4.0  # Gaussian sigma in OUTPUT_SIZE pixels


class MultiDialDataset(Dataset):
    """Multi-dial scenes for Stage 1.

    __getitem__ returns:
        img:       (3, CANVAS_SIZE, CANVAS_SIZE) float32 in [0, 1]
        heatmap:   (1, OUTPUT_SIZE, OUTPUT_SIZE) float32 — multi-peak Gaussian
        size_map:  (1, OUTPUT_SIZE, OUTPUT_SIZE) float32 — normalized size at peak pixels
        mask:      (1, OUTPUT_SIZE, OUTPUT_SIZE) float32 — 1 at peak pixels, 0 elsewhere
        filename:  str
    """

    def __init__(self, data_dir,
                 canvas_size=CANVAS_SIZE,
                 output_size=OUTPUT_SIZE,
                 sigma=HEATMAP_SIGMA):
        self.data_dir = data_dir
        self.canvas_size = canvas_size
        self.output_size = output_size
        self.sigma = sigma
        # Group rows by filename so __getitem__ has all dials for one image.
        grouped = defaultdict(list)
        with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                grouped[row["filename"]].append({
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "size": float(row["size"]),
                })
        self.filenames = sorted(grouped.keys())
        self.entries = {fn: grouped[fn] for fn in self.filenames}

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        dials = self.entries[fname]

        # Image.
        img = Image.open(os.path.join(self.data_dir, fname)).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        img_t = torch.from_numpy(arr)

        # Convert dial centers from CANVAS_SIZE coords to OUTPUT_SIZE coords.
        scale = self.output_size / self.canvas_size
        out_centers = [(d["cx"] * scale, d["cy"] * scale) for d in dials]

        # Heatmap.
        hm = make_multi_peak_heatmap(out_centers, self.output_size, self.sigma)
        hm_t = torch.from_numpy(hm).unsqueeze(0)  # (1, H, W)

        # Mask + size_map: zero everywhere, set to 1/size at integer peak px.
        size_map = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        mask = np.zeros((self.output_size, self.output_size), dtype=np.float32)
        for d, (ocx, ocy) in zip(dials, out_centers):
            ix = int(round(ocx))
            iy = int(round(ocy))
            if 0 <= ix < self.output_size and 0 <= iy < self.output_size:
                size_map[iy, ix] = d["size"] / self.canvas_size  # normalize
                mask[iy, ix] = 1.0
        size_t = torch.from_numpy(size_map).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)

        return img_t, hm_t, size_t, mask_t, fname
```

- [ ] **Step 5.4: Run tests, verify they pass**

Run: `pytest tests/test_stage1_dataset.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5.5: Commit**

```bash
git add stage1_utils.py tests/test_stage1_dataset.py
git commit -m "stage1: MultiDialDataset with tests"
```

---

## Task 6: Stage 1 backbones (Model E + F with 2-channel head)

**Files:**
- Create: `tests/test_stage1_models.py`
- Modify: `stage1_utils.py` (append)

The Stage 1 backbones preserve Model E and F's encoder + decoder (giving us a spatial output), and replace the single-channel heatmap head with two separate 1×1 convs (heatmap with sigmoid, size raw). We can reuse the helper blocks (`_InvRes`, `_ChannelSE`, `_DWSepSE`, `_DWSepResSE`) from `autocal_utils.py` rather than re-copying.

- [ ] **Step 6.1: Write failing tests**

Create `tests/test_stage1_models.py`:

```python
"""Smoke tests for the two Stage 1 backbones."""

import torch

from stage1_utils import Stage1ModelE, Stage1ModelF


def test_model_e_forward_shape():
    model = Stage1ModelE()
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 2, 256, 256)
    assert torch.isfinite(out).all()
    # Channel 0 (heatmap, sigmoid'd) is in [0, 1].
    assert (out[:, 0] >= 0).all() and (out[:, 0] <= 1).all()


def test_model_f_forward_shape():
    model = Stage1ModelF()
    model.eval()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    assert out.shape == (2, 2, 256, 256)
    assert torch.isfinite(out).all()
    assert (out[:, 0] >= 0).all() and (out[:, 0] <= 1).all()


def test_model_e_backprop():
    model = Stage1ModelE()
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    out.sum().backward()
    stem_w = model.stem[0].weight
    assert stem_w.grad is not None
    assert stem_w.grad.abs().sum() > 0


def test_model_f_backprop():
    model = Stage1ModelF()
    x = torch.randn(1, 3, 512, 512)
    out = model(x)
    out.sum().backward()
    enc1_w = model.enc1[0].weight
    assert enc1_w.grad is not None
    assert enc1_w.grad.abs().sum() > 0
```

- [ ] **Step 6.2: Run tests, verify they fail**

Run: `pytest tests/test_stage1_models.py -v`
Expected: FAIL — models not defined.

- [ ] **Step 6.3: Append model classes to `stage1_utils.py`**

Append to `stage1_utils.py`:

```python
# ── Backbones (heatmap decoder + 2-channel split head) ───────────────────

import torch.nn as nn
import torch.nn.functional as F

# Reuse the building blocks from autocal_utils (they're identical to the
# train_*_*.py originals; copying again would duplicate twice).
from autocal_utils import (
    _InvRes,
    _ChannelSE,
    _DWSepSE,
    _DWSepResSE,
)


class _SplitHead(nn.Module):
    """Two 1×1 convs producing (heatmap_sigmoid, size_raw) concatenated."""

    def __init__(self, in_ch):
        super().__init__()
        self.heatmap = nn.Conv2d(in_ch, 1, 1)
        self.size = nn.Conv2d(in_ch, 1, 1)

    def forward(self, x):
        hm = torch.sigmoid(self.heatmap(x))
        sz = self.size(x)
        return torch.cat([hm, sz], dim=1)  # (B, 2, H, W)


class Stage1ModelE(nn.Module):
    """Stage 1 detector: InvResNet encoder + decoder + split head.

    Identical to InvResNet (train_e_inv_res.py) up to (and including) the
    decoder; the final 1×1 conv heatmap head is replaced by _SplitHead
    which emits both a heatmap (sigmoid) channel and a raw size channel.
    """

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6(inplace=True))           # 256×256×24
        self.enc2 = _InvRes(24, 48, stride=2, expand_ratio=2)      # 128×128×48
        self.enc3 = _InvRes(48, 64, stride=2, expand_ratio=2)      # 64×64×64
        self.bn1 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.bn2 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.up3_proj = nn.Sequential(
            nn.Conv2d(64, 48, 1, bias=False), nn.BatchNorm2d(48))
        self.dec3 = _InvRes(48, 48, stride=1, expand_ratio=2)      # 128×128×48
        self.up2_proj = nn.Sequential(
            nn.Conv2d(48, 24, 1, bias=False), nn.BatchNorm2d(24))
        self.dec2 = _InvRes(24, 24, stride=1, expand_ratio=2)      # 256×256×24
        self.head = _SplitHead(24)

    def forward(self, x):
        e1 = self.stem(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bn2(self.bn1(e3))
        up3 = self.up3_proj(F.interpolate(b, size=e2.shape[2:],
                                          mode="bilinear",
                                          align_corners=False))
        d3 = self.dec3(up3 + e2)
        up2 = self.up2_proj(F.interpolate(d3, size=e1.shape[2:],
                                          mode="bilinear",
                                          align_corners=False))
        d2 = self.dec2(up2 + e1)
        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return self.head(d2)


class Stage1ModelF(nn.Module):
    """Stage 1 detector: SE-enhanced DWSep encoder + decoder + split head."""

    def __init__(self, half_res=True):
        super().__init__()
        self.half_res = half_res
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))            # 256×256×32
        self.enc2 = _DWSepSE(32, 64, stride=2)                     # 128×128×64
        self.enc3 = _DWSepSE(64, 96, stride=2)                     # 64×64×96
        self.bn1 = _DWSepResSE(96)
        self.bn2 = _DWSepResSE(96)
        self.up3_proj = nn.Sequential(
            nn.Conv2d(96, 64, 1, bias=False), nn.BatchNorm2d(64))
        self.dec3 = _DWSepSE(64, 64)                               # 128×128×64
        self.up2_proj = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32))
        self.dec2 = _DWSepSE(32, 32)                               # 256×256×32
        self.head = _SplitHead(32)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bn2(self.bn1(e3))
        up3 = self.up3_proj(F.interpolate(b, size=e2.shape[2:],
                                          mode="bilinear",
                                          align_corners=False))
        d3 = self.dec3(up3 + e2)
        up2 = self.up2_proj(F.interpolate(d3, size=e1.shape[2:],
                                          mode="bilinear",
                                          align_corners=False))
        d2 = self.dec2(up2 + e1)
        if not self.half_res:
            d2 = F.interpolate(d2, scale_factor=2, mode="bilinear",
                               align_corners=False)
        return self.head(d2)
```

- [ ] **Step 6.4: Run tests, verify they pass**

Run: `pytest tests/test_stage1_models.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add stage1_utils.py tests/test_stage1_models.py
git commit -m "stage1: Model E + F detector variants with split head"
```

---

## Task 7: Training utilities

**Files:**
- Modify: `stage1_utils.py` (append)

Mirrors `autocal_utils.run_autocal_training` in structure; key differences:
- Loss takes `(pred, gt_heatmap, gt_size, gt_mask)` — four tensors per batch.
- Best-checkpoint metric: per-image mean center error in **canvas pixels** (computed by peak-picking the predicted heatmap, matching peaks to GT by NMS, then averaging Euclidean distance).

- [ ] **Step 7.1: Append training utilities**

Append to `stage1_utils.py`:

```python
# ── Training utilities ───────────────────────────────────────────────────

import time

import torch.optim as optim
from torch.utils.data import DataLoader, random_split


def _predicted_centers_canvas_px(pred, threshold=0.3, min_distance=10):
    """For each batch sample, return the list of (cx, cy) predicted centers
    in CANVAS_SIZE pixel coordinates (sorted by score desc).
    """
    B = pred.shape[0]
    scale = CANVAS_SIZE / OUTPUT_SIZE
    out = []
    with torch.no_grad():
        for i in range(B):
            hm = pred[i, 0].cpu().numpy()
            peaks = find_peaks(hm, threshold=threshold,
                               min_distance=min_distance)
            out.append([(x * scale, y * scale) for (y, x, _) in peaks])
    return out


def _gt_centers_canvas_px(mask, canvas_size=CANVAS_SIZE,
                          output_size=OUTPUT_SIZE):
    """For each batch sample, list of (cx, cy) GT centers in canvas pixels.

    Uses the `mask` tensor which has 1 at each peak pixel.
    """
    scale = canvas_size / output_size
    out = []
    for i in range(mask.shape[0]):
        ys, xs = torch.where(mask[i, 0] > 0)
        out.append([(int(x) * scale, int(y) * scale)
                    for x, y in zip(xs.cpu(), ys.cpu())])
    return out


def _match_and_mean_error(pred_centers, gt_centers):
    """Greedy matching: for each GT center, find the nearest predicted center
    not already matched; record the distance; unmatched GTs contribute a
    penalty equal to CANVAS_SIZE/2 (so the loss penalises missed dials).

    Returns mean per-GT-dial Euclidean error in pixels.
    """
    if not gt_centers:
        return 0.0
    used = set()
    distances = []
    for gx, gy in gt_centers:
        best_d, best_i = None, None
        for i, (px, py) in enumerate(pred_centers):
            if i in used:
                continue
            d = math.hypot(px - gx, py - gy)
            if best_d is None or d < best_d:
                best_d, best_i = d, i
        if best_i is None:
            distances.append(CANVAS_SIZE / 2.0)
        else:
            used.add(best_i)
            distances.append(best_d)
    return float(np.mean(distances))


def train_one_epoch_stage1(model, loader, optimizer, device,
                           w_pos=10.0, w_size=1.0):
    """One training epoch. Returns (mean_loss, mean_center_err_px)."""
    model.train()
    total_loss = 0.0
    total_err = 0.0
    n = 0
    for imgs, gt_hm, gt_size, gt_mask, _fnames in loader:
        imgs = imgs.to(device)
        gt_hm = gt_hm.to(device)
        gt_size = gt_size.to(device)
        gt_mask = gt_mask.to(device)

        pred = model(imgs)
        loss, _lh, _ls = stage1_loss(pred, gt_hm, gt_size, gt_mask,
                                     w_pos=w_pos, w_size=w_size)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_centers = _predicted_centers_canvas_px(pred)
            gt_centers = _gt_centers_canvas_px(gt_mask.cpu())
            per_sample_err = [
                _match_and_mean_error(pc, gc)
                for pc, gc in zip(pred_centers, gt_centers)
            ]
            mean_err = float(np.mean(per_sample_err)) if per_sample_err else 0.0

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_err += mean_err * bs
        n += bs

    if n == 0:
        return 0.0, 0.0
    return total_loss / n, total_err / n


@torch.no_grad()
def evaluate_stage1(model, loader, device, w_pos=10.0, w_size=1.0):
    """Evaluate. Returns dict with loss, center_err_mean/p95/p99,
    plus per-sample count_diff (|pred_count - gt_count|) mean."""
    model.eval()
    total_loss = 0.0
    all_err = []
    all_count_diff = []
    n = 0
    for imgs, gt_hm, gt_size, gt_mask, _fnames in loader:
        imgs = imgs.to(device)
        gt_hm = gt_hm.to(device)
        gt_size = gt_size.to(device)
        gt_mask = gt_mask.to(device)

        pred = model(imgs)
        loss, _lh, _ls = stage1_loss(pred, gt_hm, gt_size, gt_mask,
                                     w_pos=w_pos, w_size=w_size)

        pred_centers = _predicted_centers_canvas_px(pred)
        gt_centers = _gt_centers_canvas_px(gt_mask.cpu())
        for pc, gc in zip(pred_centers, gt_centers):
            all_err.append(_match_and_mean_error(pc, gc))
            all_count_diff.append(abs(len(pc) - len(gc)))

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        n += bs

    if n == 0:
        return {"loss": 0.0, "center_err_mean": 0.0, "center_err_p95": 0.0,
                "center_err_p99": 0.0, "count_diff_mean": 0.0}

    err_arr = np.array(all_err)
    return {
        "loss": total_loss / n,
        "center_err_mean": float(err_arr.mean()),
        "center_err_p95": float(np.percentile(err_arr, 95)),
        "center_err_p99": float(np.percentile(err_arr, 99)),
        "count_diff_mean": float(np.mean(all_count_diff)),
    }


def run_stage1_training(model, args):
    """Full pipeline: train → save best by center_err_mean → export ONNX."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    ds = MultiDialDataset(args.data_dir)
    assert len(ds) >= 2, f"Dataset too small: {len(ds)} samples"
    assert args.epochs >= 1, f"--epochs must be >= 1 (got {args.epochs})"

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    nw = getattr(args, "num_workers", 4)
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=nw, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=nw, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pth_path = os.path.splitext(args.save_onnx)[0] + ".pth"
    best_err = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_err = train_one_epoch_stage1(
            model, train_ldr, optimizer, device,
            w_pos=args.w_pos, w_size=args.w_size)
        val = evaluate_stage1(model, val_ldr, device,
                              w_pos=args.w_pos, w_size=args.w_size)
        sched.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:3d}/{args.epochs}  "
              f"train_loss={tr_loss:.4f} train_err={tr_err:.2f}px  "
              f"val_loss={val['loss']:.4f} err_mean={val['center_err_mean']:.2f}px "
              f"p95={val['center_err_p95']:.2f} count_diff={val['count_diff_mean']:.2f}  "
              f"lr={lr:.2e}  ({dt:.1f}s)")

        if val["center_err_mean"] < best_err:
            best_err = val["center_err_mean"]
            torch.save(model.state_dict(), pth_path)
            print(f"  -> saved best center_err_mean={best_err:.2f}px")

    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    _export_onnx_stage1(model, args.save_onnx)


def _export_onnx_stage1(model, path, canvas_size=CANVAS_SIZE):
    import onnx
    model.eval().cpu()
    dummy = torch.randn(1, 3, canvas_size, canvas_size)
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"], output_names=["heatmap_size"])
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX exported: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
```

- [ ] **Step 7.2: Quick import sanity check**

Run:
```bash
python -c "import stage1_utils; print('ok')"
```
Expected: `ok`.

- [ ] **Step 7.3: Run all tests to confirm no regression**

Run: `pytest tests/ -v`
Expected: 23 (autocal) + 11 (stage1 math) + 4 (stage1 loss) + 4 (stage1 dataset) + 4 (stage1 models) = 46 passing.

- [ ] **Step 7.4: Commit**

```bash
git add stage1_utils.py
git commit -m "stage1: training utilities, evaluation, ONNX export"
```

---

## Task 8: Training entrypoints

**Files:**
- Create: `train_stage1_e.py`
- Create: `train_stage1_f.py`

- [ ] **Step 8.1: Create `train_stage1_e.py`**

```python
"""Train the Stage 1 detector — Model E variant."""

import argparse

from stage1_utils import Stage1ModelE, run_stage1_training


def main():
    p = argparse.ArgumentParser(description="Stage 1 detector — Model E trainer")
    p.add_argument("--data-dir", default="md1")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--w-pos", type=float, default=10.0,
                   help="weight multiplier on heatmap MSE near peak (target > 0.1)")
    p.add_argument("--w-size", type=float, default=1.0,
                   help="weight on the size head loss")
    p.add_argument("--save-onnx", default="stage1_model_e.onnx")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    run_stage1_training(Stage1ModelE(), args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.2: Create `train_stage1_f.py`**

```python
"""Train the Stage 1 detector — Model F variant."""

import argparse

from stage1_utils import Stage1ModelF, run_stage1_training


def main():
    p = argparse.ArgumentParser(description="Stage 1 detector — Model F trainer")
    p.add_argument("--data-dir", default="md1")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--w-pos", type=float, default=10.0)
    p.add_argument("--w-size", type=float, default=1.0)
    p.add_argument("--save-onnx", default="stage1_model_f.onnx")
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()
    run_stage1_training(Stage1ModelF(), args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 8.3: Smoke-check both `--help`**

Run:
```bash
python train_stage1_e.py --help
python train_stage1_f.py --help
```
Expected: both print usage with `--data-dir md1`, `--w-pos`, `--w-size`, `--num-workers`, etc.

- [ ] **Step 8.4: Commit**

```bash
git add train_stage1_e.py train_stage1_f.py
git commit -m "stage1: training entrypoints for Model E and F variants"
```

---

## Task 9: Stage 1 inference script

**Files:**
- Create: `infer_stage1.py`

- [ ] **Step 9.1: Create `infer_stage1.py`**

```python
"""Run Stage 1 inference on a single image and draw the detected boxes."""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from stage1_utils import CANVAS_SIZE, OUTPUT_SIZE, find_peaks


def preprocess(img_path, canvas_size=CANVAS_SIZE):
    """Load → resize longest-side to canvas_size → edge-mean pad to square.

    Returns (chw_array, vis_img, scale, pad_left, pad_top) so callers can
    map detector coords back to the original image space if needed.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = canvas_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.array(img, dtype=np.uint8)
    pad_left = (canvas_size - new_w) // 2
    pad_top = (canvas_size - new_h) // 2
    pad_right = canvas_size - new_w - pad_left
    pad_bottom = canvas_size - new_h - pad_top

    if pad_left or pad_right:
        edge = arr[:, [0, -1], :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((new_h, canvas_size, 3), dtype=np.uint8)
        new[:, :pad_left] = edge
        new[:, pad_left:pad_left + new_w] = arr
        new[:, pad_left + new_w:] = edge
        arr = new
    if pad_top or pad_bottom:
        edge = arr[[0, -1], :, :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((canvas_size, canvas_size, 3), dtype=np.uint8)
        new[:pad_top, :] = edge
        new[pad_top:pad_top + new_h, :] = arr
        new[pad_top + new_h:, :] = edge
        arr = new

    chw = arr.astype(np.float32) / 255.0
    chw = chw.transpose(2, 0, 1)[None]
    return chw, Image.fromarray(arr, "RGB"), scale, pad_left, pad_top


def detect_boxes(out, threshold=0.3, min_distance=10):
    """From the model's (1, 2, OUTPUT_SIZE, OUTPUT_SIZE) output, return
    a list of (cx_canvas_px, cy_canvas_px, size_canvas_px) tuples.
    """
    heatmap = out[0, 0]
    size_map = out[0, 1]
    peaks = find_peaks(heatmap, threshold=threshold, min_distance=min_distance)
    scale = CANVAS_SIZE / OUTPUT_SIZE
    boxes = []
    for y, x, _score in peaks:
        cx = x * scale
        cy = y * scale
        s = float(size_map[y, x]) * CANVAS_SIZE
        boxes.append((cx, cy, s))
    return boxes


def draw_boxes(img, boxes):
    d = ImageDraw.Draw(img)
    for i, (cx, cy, s) in enumerate(boxes):
        half = s / 2
        d.rectangle([cx - half, cy - half, cx + half, cy + half],
                    outline=(0, 255, 0), width=3)
        d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 0))
        d.text((cx - half + 4, cy - half + 4), str(i), fill=(0, 255, 0))
    return img


def main():
    p = argparse.ArgumentParser(description="Stage 1 dial detection inference")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="stage1_overlay.png")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--min-distance", type=int, default=10)
    args = p.parse_args()

    sess = ort.InferenceSession(args.model,
                                providers=["CPUExecutionProvider"])
    inp, vis_img, _, _, _ = preprocess(args.image)
    out = sess.run(None, {"input": inp})[0]  # (1, 2, OUTPUT_SIZE, OUTPUT_SIZE)
    boxes = detect_boxes(out, threshold=args.threshold,
                         min_distance=args.min_distance)
    print(f"Detected {len(boxes)} dial(s):")
    for i, (cx, cy, s) in enumerate(boxes):
        print(f"  [{i}] cx={cx:.1f} cy={cy:.1f} size={s:.1f}")

    vis_img = draw_boxes(vis_img, boxes)
    vis_img.save(args.output)
    print(f"Overlay saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 9.2: Smoke `--help`**

Run: `python infer_stage1.py --help`
Expected: usage with `--model`, `--image`, `--output`, `--threshold`, `--min-distance`.

- [ ] **Step 9.3: Commit**

```bash
git add infer_stage1.py
git commit -m "stage1: detection inference script with box overlay"
```

---

## Task 10: Full two-stage pipeline inference

**Files:**
- Create: `infer_pipeline.py`

This script chains Stage 1 + Stage 2:
1. Detect dial boxes via Stage 1.
2. For each box, crop the original image, resize to Phase 1's expected input, run the autocal model.
3. Map each per-crop ellipse back to original image coordinates.
4. Render boxes (green) + per-dial ellipses (cyan) + per-dial marks (red) + per-dial center (yellow) on a single overlay.

- [ ] **Step 10.1: Create `infer_pipeline.py`**

```python
"""Run the full two-stage detection + autocal pipeline on one image."""

import argparse
import math

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from autocal_utils import IMG_SIZE as AUTOCAL_INPUT_SIZE, derive_marks_np
from infer_stage1 import preprocess as stage1_preprocess, detect_boxes
from infer_autocal import unpack_8


def _crop_to_canvas(orig_arr, cx, cy, size):
    """Crop a square region of `orig_arr` centered on (cx, cy) with side
    `size`, edge-mean padding any portion outside the original image.

    All coords are in CANVAS_SIZE-pixel space (after stage 1 preprocess).
    Returns a (size, size, 3) uint8 array.
    """
    size_i = int(round(size))
    half = size_i // 2
    H, W = orig_arr.shape[:2]
    x0 = int(round(cx)) - half
    y0 = int(round(cy)) - half
    x1 = x0 + size_i
    y1 = y0 + size_i

    out = np.empty((size_i, size_i, 3), dtype=np.uint8)

    # Compute edge color from valid border of the original.
    edge = orig_arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    out[:] = edge

    sx0 = max(x0, 0)
    sy0 = max(y0, 0)
    sx1 = min(x1, W)
    sy1 = min(y1, H)
    if sx0 < sx1 and sy0 < sy1:
        dx0 = sx0 - x0
        dy0 = sy0 - y0
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)
        out[dy0:dy1, dx0:dx1] = orig_arr[sy0:sy1, sx0:sx1]
    return out


def _resize_to_autocal_input(crop_arr):
    """Resize a square crop to AUTOCAL_INPUT_SIZE × AUTOCAL_INPUT_SIZE."""
    img = Image.fromarray(crop_arr, "RGB").resize(
        (AUTOCAL_INPUT_SIZE, AUTOCAL_INPUT_SIZE), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.transpose(2, 0, 1)[None]  # (1, 3, H, W)


def _map_ellipse_to_canvas(cx_in_crop, cy_in_crop, a, b,
                           crop_cx, crop_cy, crop_size):
    """Map an ellipse from autocal-crop coordinates back to the canvas."""
    # The crop was a square of side `crop_size` centered at (crop_cx, crop_cy)
    # on the canvas, then resized to AUTOCAL_INPUT_SIZE × AUTOCAL_INPUT_SIZE.
    # So one autocal pixel = (crop_size / AUTOCAL_INPUT_SIZE) canvas pixels.
    s = crop_size / AUTOCAL_INPUT_SIZE
    canvas_cx = (cx_in_crop - AUTOCAL_INPUT_SIZE / 2) * s + crop_cx
    canvas_cy = (cy_in_crop - AUTOCAL_INPUT_SIZE / 2) * s + crop_cy
    canvas_a = a * s
    canvas_b = b * s
    return canvas_cx, canvas_cy, canvas_a, canvas_b


def main():
    p = argparse.ArgumentParser(
        description="Two-stage multi-dial detection + autocalibration")
    p.add_argument("--stage1-model", required=True, help="Stage 1 ONNX")
    p.add_argument("--stage2-model", required=True, help="Phase 1 autocal ONNX")
    p.add_argument("--image", required=True)
    p.add_argument("--output", default="pipeline_overlay.png")
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--min-distance", type=int, default=10)
    args = p.parse_args()

    stage1 = ort.InferenceSession(args.stage1_model,
                                  providers=["CPUExecutionProvider"])
    stage2 = ort.InferenceSession(args.stage2_model,
                                  providers=["CPUExecutionProvider"])

    inp_arr, vis_img, _, _, _ = stage1_preprocess(args.image)
    canvas_arr = np.array(vis_img)  # (CANVAS_SIZE, CANVAS_SIZE, 3) uint8

    out1 = stage1.run(None, {"input": inp_arr})[0]
    boxes = detect_boxes(out1, threshold=args.threshold,
                         min_distance=args.min_distance)
    print(f"Stage 1: detected {len(boxes)} dial(s)")

    draw = ImageDraw.Draw(vis_img)
    for i, (cx, cy, size) in enumerate(boxes):
        half = size / 2
        # Stage-1 box (green).
        draw.rectangle([cx - half, cy - half, cx + half, cy + half],
                       outline=(0, 255, 0), width=3)

        # Run Stage 2 on the crop.
        crop = _crop_to_canvas(canvas_arr, cx, cy, size)
        s2_inp = _resize_to_autocal_input(crop)
        out2 = stage2.run(None, {"input": s2_inp})[0][0]  # (8,)
        ecx, ecy, ea, eb, etheta, ephi0 = unpack_8(out2)

        # Map ellipse from crop coords back to canvas.
        ccx, ccy, ca, cb = _map_ellipse_to_canvas(
            ecx, ecy, ea, eb, cx, cy, size)
        print(f"  [{i}] center=({ccx:.1f},{ccy:.1f}) a={ca:.1f} b={cb:.1f} "
              f"theta={math.degrees(etheta):.1f}° phi0={math.degrees(ephi0):.1f}°")

        # Draw ellipse (cyan).
        pts = []
        for k in range(181):
            t = 2 * math.pi * k / 180
            cos_t, sin_t = math.cos(t), math.sin(t)
            ex = ccx + (ca * cos_t * math.cos(etheta) - cb * sin_t * math.sin(etheta))
            ey = ccy + (ca * cos_t * math.sin(etheta) + cb * sin_t * math.cos(etheta))
            pts.append((ex, ey))
        draw.line(pts, fill=(0, 200, 255), width=2)

        # Draw marks (red) — note: derive_marks_np works in autocal-crop coords,
        # but the formulas are independent of frame as long as we pass the
        # canvas-mapped (ccx, ccy, ca, cb, etheta, ephi0).
        marks = derive_marks_np(ccx, ccy, ca, cb, etheta, ephi0)
        for j in range(10):
            mx, my = marks[j]
            draw.ellipse([mx - 4, my - 4, mx + 4, my + 4],
                         outline=(255, 0, 0), width=2)

        # Center (yellow).
        draw.ellipse([ccx - 5, ccy - 5, ccx + 5, ccy + 5], fill=(255, 255, 0))

    vis_img.save(args.output)
    print(f"Overlay saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 10.2: Smoke `--help`**

Run: `python infer_pipeline.py --help`
Expected: usage with `--stage1-model`, `--stage2-model`, `--image`, `--output`, etc.

- [ ] **Step 10.3: Commit**

```bash
git add infer_pipeline.py
git commit -m "stage1: full two-stage pipeline inference (detect → autocal)"
```

---

## Task 11: End-to-end smoke test

This is a manual run that validates the entire pipeline.

- [ ] **Step 11.1: Generate a tiny multi-dial dataset (200 scenes)**

```bash
python generate_multidial_v1.py --output-dir md1_smoke --num-images 200
```
Expected: `Generated 200 images in 'md1_smoke/'`. The CSV should have somewhere between 200 and 800 dial rows.

- [ ] **Step 11.2: Train Stage 1 Model F for 5 epochs**

```bash
python train_stage1_f.py --data-dir md1_smoke --epochs 5 --batch-size 16 --save-onnx stage1_smoke_f.onnx --num-workers 0
```
Expected:
- `Parameters: ~65,000`.
- 5 epoch summaries; `val_loss` and `center_err_mean` trend downward.
- `center_err_mean` < 80 px by epoch 5 (loose; this is undertrained).
- `count_diff_mean` should fall (model gradually learns to predict the right number of dials).

If `val_loss` is NaN or `center_err_mean` is stuck near `CANVAS_SIZE/2 = 256`, stop and report. Likely a label-encoding bug.

- [ ] **Step 11.3: Verify Stage 1 alone on one image**

```bash
python infer_stage1.py --model stage1_smoke_f.onnx --image md1_smoke/scene_00000.png --output stage1_check.png --threshold 0.1
```
(Lower threshold because the model is undertrained.)

Expected: prints `Detected N dial(s)` with N ≥ 1; `stage1_check.png` is created. The boxes won't perfectly match ground truth at 5 epochs but the model should show some signal (at least one box near a real dial).

- [ ] **Step 11.4: Run the full pipeline (requires a Stage 2 model)**

If you've already trained a Phase 1 autocal model (commit `f2407eb` smoke or a real training run):

```bash
python infer_pipeline.py --stage1-model stage1_smoke_f.onnx --stage2-model <path-to-autocal_model_f.onnx> --image md1_smoke/scene_00000.png --output pipeline_check.png --threshold 0.1
```

Expected: prints `Stage 1: detected N dial(s)` followed by N per-dial geometry lines. `pipeline_check.png` shows green boxes + cyan ellipses + red marks + yellow centers.

If you DON'T have a Phase 1 model handy, skip this step — the Stage 1 smoke (Step 11.3) is enough to validate Stage 1 in isolation.

- [ ] **Step 11.5: Cleanup**

```bash
python -c "import shutil, os; shutil.rmtree('md1_smoke', ignore_errors=True); [os.remove(p) for p in ['stage1_smoke_f.onnx', 'stage1_smoke_f.pth', 'stage1_check.png', 'pipeline_check.png'] if os.path.exists(p)]"
```

- [ ] **Step 11.6: Marker commit**

```bash
git status
```
If clean, mark the milestone:
```bash
git commit --allow-empty -m "stage1: smoke test passed"
```

---

## Done

Stage 1 implementation complete. Next steps from the spec:

1. **Full training run.** Generate a 15k-image `md1/` dataset:
   ```bash
   python generate_multidial_v1.py --output-dir md1 --num-images 15000
   python train_stage1_e.py --data-dir md1 --epochs 200
   python train_stage1_f.py --data-dir md1 --epochs 200
   ```
   Target on synthetic held-out: `center_err_mean < 5 px`, `count_diff_mean < 0.1`.

2. **Pipeline benchmark.** Run `infer_pipeline.py` on a curated synthetic test set; measure end-to-end per-dial mark accuracy after stage 1 detection + stage 2 calibration.

3. **Real-image fine-tuning** (Phase 2 of stage 1). Same playbook as the Phase 1 plan:
   - Collect 20–50 labeled real multi-dial images.
   - Fine-tune the winning Stage 1 backbone with mixed synth + real batches.
   - If detection accuracy on real images < 80% recall, escalate to a stronger detector backbone.

4. **Two-stage joint optimization** (optional). If end-to-end performance is the bottleneck, consider joint training where stage 1 and stage 2 are trained together with a combined loss. Most teams don't need this until real-world deployment shows specific failure modes.
