# Autocalibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Phase 1 autocalibration pipeline — a 6-DOF affine model that predicts dial geometry (center, ellipse, scale-mark positions) from a user-box-selected dial region.

**Architecture:**
- New synthetic generator (`generate_synthetic_v3.py`) that samples ellipse parameters directly and renders the dial via a canonical-circle → ellipse map; includes procedural backgrounds, distractor text, and always-visible bezel.
- Two backbones (forks of Model E and Model F) — heatmap decoder replaced with a global-pool + 1×1-conv regression head outputting 8 scalars.
- SmoothL1 loss on both the normalized parameters and the 10 mark positions derived from them inside the loss (differentiable, task-direct gradient).
- ONNX + TFLite export, smoke-tested end-to-end.

**Tech Stack:** PyTorch, PIL, NumPy, ONNX Runtime, pytest. Same toolchain as the existing tip-prediction models.

**Spec:** `AUTOCALIBRATION_PLAN.md` at the project root.

---

## File Structure

**New files:**
- `autocal_utils.py` — Math (canonical→image map, mark derivation), parameter packing, loss, dataset, training utilities. Single-responsibility name, but bundled because all four pieces share the same math.
- `generate_synthetic_v3.py` — New generator. Forked from v2 for texture rendering (pointer, ticks, noise, artifacts) but with a rebuilt geometric pipeline.
- `train_autocal_e.py` — Trainer for Model E variant (regression head replaces heatmap).
- `train_autocal_f.py` — Trainer for Model F variant (regression head replaces heatmap).
- `infer_autocal.py` — Run trained model on a single image, visualize the 6 params + 10 derived marks.
- `tests/__init__.py` — Empty package marker.
- `tests/test_autocal_math.py` — Tests for canonical→image map, mark derivation, parameter normalization, loss.

**Unchanged files (referenced but not modified):**
- `train_utils_v2.py` — kept; the autocal trainers do not use its heatmap-specific pipeline.
- `generate_synthetic_v2.py` — kept as the reference for texture rendering patterns copied into v3.

---

## Conventions

- All tests assume `pytest`. If not installed: `pip install pytest`.
- Run tests from project root.
- Commit after each task. Use short prefixed messages like `autocal: <what>`.
- `IMG_SIZE = 256` throughout — matches existing project convention.

---

## Task 1: Geometry math — canonical→image map and mark derivation

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/test_autocal_math.py`
- Create: `autocal_utils.py`

- [ ] **Step 1.1: Create empty test package marker**

Create `tests/__init__.py` with empty content (the file just needs to exist).

```python
# tests/__init__.py
```

- [ ] **Step 1.2: Write failing tests for canonical→image map and mark derivation**

Create `tests/test_autocal_math.py`:

```python
"""Tests for autocal_utils geometry math."""

import math

import numpy as np
import pytest
import torch

from autocal_utils import (
    canonical_to_image_np,
    canonical_to_image_torch,
    derive_marks_np,
    derive_marks_torch,
)


# ── canonical_to_image_np ────────────────────────────────────────────────

def test_c2i_np_center_when_rc_zero():
    """At r_c = 0, the canonical point is the centre regardless of t."""
    x, y = canonical_to_image_np(t=1.234, r_c=0.0,
                                 cx=100.0, cy=80.0, a=50.0, b=40.0, theta=0.3)
    assert x == pytest.approx(100.0)
    assert y == pytest.approx(80.0)


def test_c2i_np_circle_when_a_eq_b_theta_zero():
    """When a = b and theta = 0, the ellipse is a circle of radius a."""
    for t in np.linspace(0, 2 * math.pi, 9):
        x, y = canonical_to_image_np(t=t, r_c=1.0,
                                     cx=0.0, cy=0.0, a=50.0, b=50.0, theta=0.0)
        assert math.hypot(x, y) == pytest.approx(50.0, abs=1e-6)


def test_c2i_np_axis_aligned_ellipse():
    """At theta = 0, t = 0 lands at (cx + a, cy); t = pi/2 lands at (cx, cy + b)."""
    x0, y0 = canonical_to_image_np(t=0.0, r_c=1.0,
                                   cx=10.0, cy=20.0, a=50.0, b=30.0, theta=0.0)
    assert (x0, y0) == pytest.approx((60.0, 20.0))

    x1, y1 = canonical_to_image_np(t=math.pi / 2, r_c=1.0,
                                   cx=10.0, cy=20.0, a=50.0, b=30.0, theta=0.0)
    assert (x1, y1) == pytest.approx((10.0, 50.0))


def test_c2i_np_rotated_ellipse():
    """At theta = pi/2, the major axis points along +y; t = 0 lands at (cx, cy + a)."""
    x, y = canonical_to_image_np(t=0.0, r_c=1.0,
                                 cx=10.0, cy=20.0, a=50.0, b=30.0,
                                 theta=math.pi / 2)
    assert (x, y) == pytest.approx((10.0, 70.0))


# ── derive_marks_np ──────────────────────────────────────────────────────

def test_derive_marks_count():
    marks = derive_marks_np(cx=128.0, cy=128.0, a=60.0, b=60.0,
                            theta=0.0, phi_0=0.0)
    assert marks.shape == (10, 2)


def test_derive_marks_on_circle_when_a_eq_b():
    """All 10 marks lie on a circle of radius a when a == b."""
    marks = derive_marks_np(cx=128.0, cy=128.0, a=60.0, b=60.0,
                            theta=0.0, phi_0=0.5)
    for (mx, my) in marks:
        assert math.hypot(mx - 128.0, my - 128.0) == pytest.approx(60.0, abs=1e-6)


def test_derive_marks_spacing_when_a_eq_b():
    """Marks are spaced exactly 36° apart on the circle when a == b."""
    marks = derive_marks_np(cx=0.0, cy=0.0, a=100.0, b=100.0,
                            theta=0.0, phi_0=0.0)
    for i in range(10):
        ang_i = math.atan2(marks[i][1], marks[i][0])
        ang_0 = math.atan2(marks[0][1], marks[0][0])
        expected = (math.radians(36 * i)) % (2 * math.pi)
        actual = (ang_i - ang_0) % (2 * math.pi)
        assert actual == pytest.approx(expected, abs=1e-6)


def test_derive_marks_mark0_at_phi0_on_ellipse():
    """Mark 0 lies at parametric angle phi_0 on the ellipse."""
    phi_0 = 0.7
    cx, cy, a, b, theta = 100.0, 200.0, 50.0, 30.0, 0.4
    marks = derive_marks_np(cx=cx, cy=cy, a=a, b=b, theta=theta, phi_0=phi_0)

    expected_x, expected_y = canonical_to_image_np(
        t=phi_0, r_c=1.0, cx=cx, cy=cy, a=a, b=b, theta=theta)
    assert marks[0][0] == pytest.approx(expected_x, abs=1e-6)
    assert marks[0][1] == pytest.approx(expected_y, abs=1e-6)


# ── torch parity ─────────────────────────────────────────────────────────

def test_c2i_torch_matches_numpy():
    """Torch implementation produces same output as numpy."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        cx, cy = rng.uniform(0, 256), rng.uniform(0, 256)
        a = rng.uniform(30, 110)
        b = a * rng.uniform(0.8, 1.0)
        theta = rng.uniform(-math.pi / 2, math.pi / 2)
        t = rng.uniform(0, 2 * math.pi)
        r_c = rng.uniform(0.5, 1.0)

        x_np, y_np = canonical_to_image_np(t, r_c, cx, cy, a, b, theta)
        out = canonical_to_image_torch(
            torch.tensor([t]), torch.tensor([r_c]),
            torch.tensor([cx]), torch.tensor([cy]),
            torch.tensor([a]), torch.tensor([b]), torch.tensor([theta]))
        assert out.shape == (1, 2)
        assert out[0, 0].item() == pytest.approx(x_np, abs=1e-5)
        assert out[0, 1].item() == pytest.approx(y_np, abs=1e-5)


def test_derive_marks_torch_matches_numpy():
    """Torch implementation produces same output as numpy."""
    cx, cy, a, b, theta, phi_0 = 128.0, 100.0, 60.0, 50.0, 0.3, 0.9
    np_marks = derive_marks_np(cx, cy, a, b, theta, phi_0)
    t_marks = derive_marks_torch(
        torch.tensor([cx]), torch.tensor([cy]),
        torch.tensor([a]), torch.tensor([b]),
        torch.tensor([theta]), torch.tensor([phi_0]))
    assert t_marks.shape == (1, 10, 2)
    for i in range(10):
        assert t_marks[0, i, 0].item() == pytest.approx(np_marks[i][0], abs=1e-5)
        assert t_marks[0, i, 1].item() == pytest.approx(np_marks[i][1], abs=1e-5)


def test_derive_marks_torch_is_differentiable():
    """Gradient flows from marks back to params."""
    cx = torch.tensor([128.0], requires_grad=True)
    cy = torch.tensor([128.0], requires_grad=True)
    a = torch.tensor([60.0], requires_grad=True)
    b = torch.tensor([55.0], requires_grad=True)
    theta = torch.tensor([0.3], requires_grad=True)
    phi_0 = torch.tensor([0.5], requires_grad=True)

    marks = derive_marks_torch(cx, cy, a, b, theta, phi_0)
    loss = marks.sum()
    loss.backward()

    for p in (cx, cy, a, b, theta, phi_0):
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
```

- [ ] **Step 1.3: Run tests to verify they fail**

Run: `pytest tests/test_autocal_math.py -v`
Expected: FAIL — `autocal_utils` module not found.

- [ ] **Step 1.4: Implement the math in autocal_utils.py**

Create `autocal_utils.py`:

```python
"""Autocalibration utilities: geometry math, parameter packing, loss, dataset, training loop.

Shared by `generate_synthetic_v3.py`, `train_autocal_e.py`, `train_autocal_f.py`,
and `infer_autocal.py`.
"""

import math

import numpy as np
import torch


IMG_SIZE = 256
N_MARKS = 10
DEG_PER_MARK = 360.0 / N_MARKS


# ── Geometry: canonical-circle → ellipse map ────────────────────────────

def canonical_to_image_np(t, r_c, cx, cy, a, b, theta):
    """Map a canonical-unit-circle point to ellipse image coords (numpy/float).

    For a canonical point at angle `t` and radius `r_c ∈ [0, 1]`, the image
    position is:
        x = cx + r_c · (a·cos(t)·cosθ - b·sin(t)·sinθ)
        y = cy + r_c · (a·cos(t)·sinθ + b·sin(t)·cosθ)

    Major axis is along (cosθ, sinθ); minor axis is perpendicular.
    """
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return x, y


def canonical_to_image_torch(t, r_c, cx, cy, a, b, theta):
    """Differentiable batched version of canonical_to_image.

    All inputs are 1-D tensors of shape (B,). Returns (B, 2).
    """
    cos_t = torch.cos(t)
    sin_t = torch.sin(t)
    cos_o = torch.cos(theta)
    sin_o = torch.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return torch.stack([x, y], dim=-1)


def derive_marks_np(cx, cy, a, b, theta, phi_0):
    """Compute the 10 scale-mark positions in image coords.

    Mark i lies on the ellipse at canonical angle t_i = phi_0 + i · 36°.
    Returns array of shape (10, 2).
    """
    marks = np.empty((N_MARKS, 2), dtype=np.float64)
    for i in range(N_MARKS):
        t_i = phi_0 + i * math.radians(DEG_PER_MARK)
        x, y = canonical_to_image_np(t=t_i, r_c=1.0,
                                     cx=cx, cy=cy, a=a, b=b, theta=theta)
        marks[i, 0] = x
        marks[i, 1] = y
    return marks


def derive_marks_torch(cx, cy, a, b, theta, phi_0):
    """Differentiable batched mark derivation.

    All inputs are 1-D tensors of shape (B,). Returns (B, 10, 2).
    """
    device = cx.device
    dtype = cx.dtype
    i = torch.arange(N_MARKS, device=device, dtype=dtype)
    step = 2 * math.pi / N_MARKS
    # Broadcast: phi_0 (B,1) + i*(10,) -> (B, 10)
    t_i = phi_0.unsqueeze(-1) + i * step
    cos_t = torch.cos(t_i)
    sin_t = torch.sin(t_i)
    cos_o = torch.cos(theta).unsqueeze(-1)
    sin_o = torch.sin(theta).unsqueeze(-1)
    a_b = a.unsqueeze(-1)
    b_b = b.unsqueeze(-1)
    cx_b = cx.unsqueeze(-1)
    cy_b = cy.unsqueeze(-1)
    x = cx_b + a_b * cos_t * cos_o - b_b * sin_t * sin_o
    y = cy_b + a_b * cos_t * sin_o + b_b * sin_t * cos_o
    return torch.stack([x, y], dim=-1)
```

- [ ] **Step 1.5: Run tests to verify they pass**

Run: `pytest tests/test_autocal_math.py -v`
Expected: All 9 tests PASS.

- [ ] **Step 1.6: Commit**

```bash
git add tests/__init__.py tests/test_autocal_math.py autocal_utils.py
git commit -m "autocal: canonical-to-image map and mark derivation with tests"
```

---

## Task 2: Parameter packing and loss

**Files:**
- Modify: `tests/test_autocal_math.py` (append)
- Modify: `autocal_utils.py` (append)

- [ ] **Step 2.1: Append failing tests for param packing and loss**

Append to `tests/test_autocal_math.py`:

```python
# ── Param packing/unpacking ──────────────────────────────────────────────

from autocal_utils import (
    pack_params_to_8,
    unpack_params_from_8,
    autocal_loss,
)


def test_pack_params_shape():
    cx = torch.tensor([128.0])
    cy = torch.tensor([128.0])
    a = torch.tensor([60.0])
    b = torch.tensor([55.0])
    theta = torch.tensor([0.3])
    phi_0 = torch.tensor([0.5])
    packed = pack_params_to_8(cx, cy, a, b, theta, phi_0, img_size=256)
    assert packed.shape == (1, 8)


def test_pack_params_normalization_ranges():
    """All 8 normalized values should sit in [-1, 1]."""
    cx = torch.tensor([128.0])
    cy = torch.tensor([0.0])
    a = torch.tensor([128.0])
    b = torch.tensor([100.0])
    theta = torch.tensor([0.0])
    phi_0 = torch.tensor([0.0])
    packed = pack_params_to_8(cx, cy, a, b, theta, phi_0, img_size=256)
    assert packed[0, 0].item() == pytest.approx(0.5)  # cx/256
    assert packed[0, 1].item() == pytest.approx(0.0)  # cy/256
    assert packed[0, 2].item() == pytest.approx(1.0)  # a/128
    assert packed[0, 3].item() == pytest.approx(100.0 / 128)
    assert packed[0, 4].item() == pytest.approx(0.0)  # sin(2*0)
    assert packed[0, 5].item() == pytest.approx(1.0)  # cos(2*0)
    assert packed[0, 6].item() == pytest.approx(0.0)  # sin(0)
    assert packed[0, 7].item() == pytest.approx(1.0)  # cos(0)


def test_unpack_inverse_of_pack():
    """unpack(pack(x)) ≈ x for valid inputs."""
    rng = np.random.default_rng(42)
    for _ in range(5):
        cx_v = rng.uniform(0, 256)
        cy_v = rng.uniform(0, 256)
        a_v = rng.uniform(30, 115)
        b_v = a_v * rng.uniform(0.8, 1.0)
        # Keep theta in (-pi/2, pi/2) so atan2(sin2θ, cos2θ)/2 recovers it.
        theta_v = rng.uniform(-math.pi / 2 + 0.01, math.pi / 2 - 0.01)
        # Keep phi_0 in (-pi, pi).
        phi_0_v = rng.uniform(-math.pi + 0.01, math.pi - 0.01)

        cx = torch.tensor([cx_v])
        cy = torch.tensor([cy_v])
        a = torch.tensor([a_v])
        b = torch.tensor([b_v])
        theta = torch.tensor([theta_v])
        phi_0 = torch.tensor([phi_0_v])

        packed = pack_params_to_8(cx, cy, a, b, theta, phi_0, img_size=256)
        cx2, cy2, a2, b2, theta2, phi_0_2 = unpack_params_from_8(packed, img_size=256)

        assert cx2.item() == pytest.approx(cx_v, abs=1e-4)
        assert cy2.item() == pytest.approx(cy_v, abs=1e-4)
        assert a2.item() == pytest.approx(a_v, abs=1e-4)
        assert b2.item() == pytest.approx(b_v, abs=1e-4)
        assert theta2.item() == pytest.approx(theta_v, abs=1e-4)
        assert phi_0_2.item() == pytest.approx(phi_0_v, abs=1e-4)


# ── Loss ──────────────────────────────────────────────────────────────────

def test_autocal_loss_zero_at_truth():
    """Loss is zero when prediction matches ground truth exactly."""
    gt_cx = torch.tensor([128.0])
    gt_cy = torch.tensor([100.0])
    gt_a = torch.tensor([60.0])
    gt_b = torch.tensor([55.0])
    gt_theta = torch.tensor([0.3])
    gt_phi_0 = torch.tensor([0.5])
    gt_marks = derive_marks_torch(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0)

    pred_8 = pack_params_to_8(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                              img_size=256)

    total, l_p, l_m = autocal_loss(
        pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0, gt_marks,
        img_size=256, w_param=0.1, w_mark=1.0)

    assert total.item() == pytest.approx(0.0, abs=1e-6)
    assert l_p.item() == pytest.approx(0.0, abs=1e-6)
    assert l_m.item() == pytest.approx(0.0, abs=1e-6)


def test_autocal_loss_positive_for_error():
    """Loss is positive when prediction differs from ground truth."""
    gt_cx = torch.tensor([128.0])
    gt_cy = torch.tensor([100.0])
    gt_a = torch.tensor([60.0])
    gt_b = torch.tensor([55.0])
    gt_theta = torch.tensor([0.3])
    gt_phi_0 = torch.tensor([0.5])
    gt_marks = derive_marks_torch(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0)

    # Perturb predicted center by 10 px
    pred_8 = pack_params_to_8(gt_cx + 10.0, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                              img_size=256)
    total, _, _ = autocal_loss(
        pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0, gt_marks,
        img_size=256)
    assert total.item() > 0.0


def test_autocal_loss_backprop():
    """Gradient flows from total loss back to predicted params."""
    pred_8 = torch.zeros((1, 8), requires_grad=True)
    gt_cx = torch.tensor([128.0])
    gt_cy = torch.tensor([100.0])
    gt_a = torch.tensor([60.0])
    gt_b = torch.tensor([55.0])
    gt_theta = torch.tensor([0.0])
    gt_phi_0 = torch.tensor([0.0])
    gt_marks = derive_marks_torch(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0)

    total, _, _ = autocal_loss(
        pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0, gt_marks)
    total.backward()
    assert pred_8.grad is not None
    assert torch.isfinite(pred_8.grad).all()
    assert (pred_8.grad.abs() > 0).any()
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `pytest tests/test_autocal_math.py -v`
Expected: New tests FAIL — `pack_params_to_8`, `unpack_params_from_8`, `autocal_loss` not yet defined.

- [ ] **Step 2.3: Append implementation to autocal_utils.py**

Append to `autocal_utils.py`:

```python
import torch.nn.functional as F


# ── Parameter packing ────────────────────────────────────────────────────

def pack_params_to_8(cx, cy, a, b, theta, phi_0, img_size=IMG_SIZE):
    """Pack 6 raw params into 8 normalized scalars suitable for regression.

    Layout: [cx_n, cy_n, a_n, b_n, sin(2θ), cos(2θ), sin(φ₀), cos(φ₀)]
      cx_n, cy_n: divided by IMG_SIZE
      a_n, b_n: divided by IMG_SIZE/2
      θ uses 2θ because, with a ≥ b enforced, θ has period π (an axis line).
    Returns tensor of shape (B, 8).
    """
    cx_n = cx / img_size
    cy_n = cy / img_size
    a_n = a / (img_size / 2)
    b_n = b / (img_size / 2)
    sin2 = torch.sin(2 * theta)
    cos2 = torch.cos(2 * theta)
    sin_p = torch.sin(phi_0)
    cos_p = torch.cos(phi_0)
    return torch.stack([cx_n, cy_n, a_n, b_n, sin2, cos2, sin_p, cos_p], dim=-1)


def unpack_params_from_8(packed, img_size=IMG_SIZE):
    """Inverse of pack_params_to_8.

    Returns (cx, cy, a, b, theta, phi_0), each shape (B,).
    """
    cx_n, cy_n, a_n, b_n, sin2, cos2, sin_p, cos_p = packed.unbind(-1)
    cx = cx_n * img_size
    cy = cy_n * img_size
    a = a_n * (img_size / 2)
    b = b_n * (img_size / 2)
    theta = 0.5 * torch.atan2(sin2, cos2)
    phi_0 = torch.atan2(sin_p, cos_p)
    return cx, cy, a, b, theta, phi_0


# ── Loss ──────────────────────────────────────────────────────────────────

def autocal_loss(pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                 gt_marks, img_size=IMG_SIZE, w_param=0.1, w_mark=1.0):
    """Combined SmoothL1 loss on packed params and derived marks.

    Args:
        pred_8: (B, 8) — network output (the packed prediction).
        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0: ground truth, each (B,).
        gt_marks: (B, 10, 2) — ground truth mark positions in pixel coords.
        img_size: used to normalize the mark loss to unit scale.
        w_param, w_mark: weighting; mark loss carries the main signal.

    Returns: (total_loss, param_loss, mark_loss)
    """
    gt_8 = pack_params_to_8(gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
                            img_size=img_size)

    l_param = F.smooth_l1_loss(pred_8, gt_8)

    cx_p, cy_p, a_p, b_p, theta_p, phi_0_p = unpack_params_from_8(
        pred_8, img_size=img_size)
    pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_0_p)

    pred_marks_n = pred_marks / img_size
    gt_marks_n = gt_marks / img_size
    l_mark = F.smooth_l1_loss(pred_marks_n, gt_marks_n)

    total = w_param * l_param + w_mark * l_mark
    return total, l_param, l_mark
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `pytest tests/test_autocal_math.py -v`
Expected: All tests PASS (15+ tests now).

- [ ] **Step 2.5: Commit**

```bash
git add tests/test_autocal_math.py autocal_utils.py
git commit -m "autocal: param packing and SmoothL1 loss with tests"
```

---

## Task 3: Generator — parameter sampling and canonical map (numpy)

**Files:**
- Create: `generate_synthetic_v3.py`

The generator is large; we build it bottom-up across Tasks 3–7. This task lays the file skeleton and parameter sampling. Visual rendering arrives in Task 4+.

- [ ] **Step 3.1: Create the generator file with constants and parameter sampling**

Create `generate_synthetic_v3.py`:

```python
"""Generate synthetic training images for autocalibration.

Each image:
  - Procedural textured background (replacing v2's absolute-black outside).
  - Background distractor text outside the bezel (random digits, numbers, labels).
  - Outer canvas / bezel (always rendered).
  - Dial face with scale ticks and digits at canonical angles φ₀ + i·36°.
  - Pointer (distractor; not a target).

Labels per image:
  filename, cx, cy, a, b, theta, phi_0, mark_0_x, mark_0_y, ..., mark_9_x, mark_9_y
"""

import argparse
import colorsys
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from autocal_utils import IMG_SIZE, N_MARKS, DEG_PER_MARK, derive_marks_np

OUTPUT_DIR = "sv3"
NUM_IMAGES = 15000

# ── Ellipse parameter sampling ranges ────────────────────────────────────
CX_DISP_RANGE = (-40, 40)        # px from image centre
CY_DISP_RANGE = (-40, 40)
A_RANGE = (30, 115)              # px (semi-major)
B_RATIO_RANGE = (0.8, 1.0)       # b / a
THETA_RANGE = (-math.pi / 2, math.pi / 2)
PHI_0_RANGE = (0.0, 2 * math.pi)
POINTER_VALUE_RANGE = (0.0, 9.0)

# ── Outer canvas (bezel) ─────────────────────────────────────────────────
OUTER_SCALE_RANGE = (1.20, 1.32)
RIM_WIDTH = 3

# ── Scale ticks ──────────────────────────────────────────────────────────
TICKS_PER_DIGIT_RANGE = (1, 5)
TICK_LEN_FRAC_RANGE = (0.06, 0.10)   # fraction of a (semi-major) for tick length
TICK_WIDTH_RANGE = (2, 4)
TICK_GREY_RANGE = (0, 30)

# ── Digits ───────────────────────────────────────────────────────────────
DIGIT_FONT_SIZE_RANGE = (12, 24)
DIGIT_GREY_RANGE = (0, 30)
DIGIT_INSIDE_PROB = 0.5

# ── Pointer ──────────────────────────────────────────────────────────────
HUB_RATIO_RANGE = (0.35, 0.45)       # hub_radius / a
PTR_S_RANGE = (0.5, 1.0)
HOLE_RATIO_RANGE = (1.05, 1.10)

# ── HSV palette ──────────────────────────────────────────────────────────
DIAL_S_RANGE = (0, 0.15)
BG_V_INSIDE_RANGE = (0.67, 0.71)
CIRCLE_V_RANGE = (0.84, 0.88)
OUTER_V_RANGE = (0.76, 0.86)
RIM_V_RANGE = (0.71, 0.76)

# ── Procedural background (full canvas) ──────────────────────────────────
BG_HUE_RANGE = (0.0, 1.0)
BG_S_RANGE = (0.0, 0.30)
BG_V_RANGE = (0.10, 0.95)
BG_LOWFREQ_SIZE = 8
BG_VARIATION = 25
BG_PIXEL_NOISE = 5
BG_BLUR_PROB = 0.3
BG_BLUR_RADIUS_RANGE = (1.0, 3.0)

# ── Distractor text ──────────────────────────────────────────────────────
DISTRACTOR_COUNT_RANGE = (0, 6)
DISTRACTOR_FONT_SIZE_RANGE = (8, 32)
DISTRACTOR_ROT_RANGE = (-15, 15)
DISTRACTOR_FONTS = ["arial.ttf", "cour.ttf", "times.ttf"]
DISTRACTOR_TEXTS_DIGITS = [str(d) for d in range(10)]
DISTRACTOR_TEXTS_NUMBERS = ["12", "230", "0.5", "5.0", "100", "60", "0", "3.7"]
DISTRACTOR_TEXTS_CODES = ["A-50", "V1.2", "K12", "M-30", "RPM", "Hz", "PSI"]
DISTRACTOR_TEXTS_LABELS = ["V", "A", "AC", "DC", "kW", "OFF", "ON", "MAX"]

# ── Colour / exposure jitter ─────────────────────────────────────────────
BRIGHTNESS_RANGE = (0.85, 1.30)
CONTRAST_RANGE = (0.80, 1.20)
SATURATION_RANGE = (0.75, 1.25)
SHARPNESS_RANGE = (0.7, 1.5)

# ── Adversarial triangle artifacts ───────────────────────────────────────
TRI_COUNT_RANGE = (0, 6)
TRI_SIZE_RANGE = (15, 40)
TRI_MEAN_RANGE = (50, 200)
TRI_PIXEL_VAR = 5
TRI_VERTEX_DIST_RANGE = (0.3, 1.0)


def _hsv_rgb(h, s, v):
    """HSV (0-1) → (R, G, B) ints 0-255."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── Parameter sampling ───────────────────────────────────────────────────

def sample_params(img_size=IMG_SIZE):
    """Sample one set of (cx, cy, a, b, theta, phi_0, pointer_value)."""
    cx = img_size / 2 + random.uniform(*CX_DISP_RANGE)
    cy = img_size / 2 + random.uniform(*CY_DISP_RANGE)
    a = random.uniform(*A_RANGE)
    b = a * random.uniform(*B_RATIO_RANGE)
    theta = random.uniform(*THETA_RANGE)
    phi_0 = random.uniform(*PHI_0_RANGE)
    pointer_value = random.uniform(*POINTER_VALUE_RANGE)
    return {
        "cx": cx, "cy": cy, "a": a, "b": b,
        "theta": theta, "phi_0": phi_0,
        "pointer_value": pointer_value,
    }
```

- [ ] **Step 3.2: Verify the file imports and `sample_params` runs**

Run:
```bash
python -c "from generate_synthetic_v3 import sample_params; import random; random.seed(0); print(sample_params())"
```
Expected: prints a dict with 7 keys and plausible numeric values.

- [ ] **Step 3.3: Commit**

```bash
git add generate_synthetic_v3.py
git commit -m "autocal: v3 generator skeleton, constants, sample_params"
```

---

## Task 4: Generator — procedural background and distractor text

**Files:**
- Modify: `generate_synthetic_v3.py` (append)

- [ ] **Step 4.1: Append background and distractor text rendering**

Append to `generate_synthetic_v3.py`:

```python
# ── Background fill ──────────────────────────────────────────────────────

def make_procedural_background(canvas_size):
    """Generate a full-canvas procedural background.

    Random HSV base color → low-frequency variation → per-pixel noise.
    Optionally Gaussian-blurred to simulate shallow depth of field.
    Returns an RGB PIL.Image.
    """
    h = random.uniform(*BG_HUE_RANGE)
    s = random.uniform(*BG_S_RANGE)
    v = random.uniform(*BG_V_RANGE)
    base_rgb = np.array(_hsv_rgb(h, s, v), dtype=np.int16)

    lf_var = np.random.randint(-BG_VARIATION, BG_VARIATION + 1,
                               (BG_LOWFREQ_SIZE, BG_LOWFREQ_SIZE, 3),
                               dtype=np.int16)
    lf_img = Image.fromarray(
        np.clip(base_rgb + lf_var, 0, 255).astype(np.uint8), "RGB"
    ).resize((canvas_size, canvas_size), Image.BICUBIC)

    arr = np.array(lf_img, dtype=np.int16)
    noise = np.random.randint(-BG_PIXEL_NOISE, BG_PIXEL_NOISE + 1,
                              (canvas_size, canvas_size, 3), dtype=np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB")

    if random.random() < BG_BLUR_PROB:
        r = random.uniform(*BG_BLUR_RADIUS_RANGE)
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return img


# ── Distractor text ──────────────────────────────────────────────────────

def _load_font(name, size):
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _random_distractor_text():
    pool = (DISTRACTOR_TEXTS_DIGITS + DISTRACTOR_TEXTS_NUMBERS
            + DISTRACTOR_TEXTS_CODES + DISTRACTOR_TEXTS_LABELS)
    return random.choice(pool)


def _bezel_disc_contains(x, y, cx, cy, a, b, theta, outer_scale):
    """Check if image point (x, y) lies inside the bezel (outer ellipse)."""
    # Transform (x, y) into the ellipse's local axis-aligned frame.
    dx = x - cx
    dy = y - cy
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    u = dx * cos_o + dy * sin_o
    v = -dx * sin_o + dy * cos_o
    return (u / (a * outer_scale)) ** 2 + (v / (b * outer_scale)) ** 2 <= 1.0


def draw_distractor_text(img, params, outer_scale):
    """Render `n` random text strings outside the bezel.

    Each string is rejected (re-sampled) if it would overlap the bezel disc.
    """
    canvas_size = img.size[0]
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    n = random.randint(*DISTRACTOR_COUNT_RANGE)
    for _ in range(n):
        for _attempt in range(8):
            text = _random_distractor_text()
            font_name = random.choice(DISTRACTOR_FONTS)
            size = random.randint(*DISTRACTOR_FONT_SIZE_RANGE)
            font = _load_font(font_name, size)
            x = random.randint(0, canvas_size - 1)
            y = random.randint(0, canvas_size - 1)
            if _bezel_disc_contains(x, y, params["cx"], params["cy"],
                                    params["a"], params["b"], params["theta"],
                                    outer_scale):
                continue
            rot = random.uniform(*DISTRACTOR_ROT_RANGE)
            # Choose contrasting color: sample any HSV, but bias toward opposite V.
            grey = random.randint(0, 255)
            color = (grey, grey, grey, 255)

            # Render to a temp layer, rotate, paste.
            tmp = Image.new("RGBA", (size * len(text) * 2, size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((0, 0), text, fill=color, font=font)
            tmp = tmp.rotate(rot, expand=True, resample=Image.BILINEAR)
            overlay.paste(tmp, (x, y), tmp)
            break
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
```

- [ ] **Step 4.2: Smoke-test background generation**

Run:
```bash
python -c "from generate_synthetic_v3 import make_procedural_background; make_procedural_background(256).save('bg_test.png'); print('saved')"
```
Expected: prints "saved". Open `bg_test.png` in the project root — should look like a textured wall/panel surface, not solid black. Delete after inspection: `python -c "import os; os.remove('bg_test.png')"`.

- [ ] **Step 4.3: Commit**

```bash
git add generate_synthetic_v3.py
git commit -m "autocal: v3 procedural background and distractor text"
```

---

## Task 5: Generator — outer canvas (bezel) and dial face

**Files:**
- Modify: `generate_synthetic_v3.py` (append)

- [ ] **Step 5.1: Append bezel/outer-canvas rendering**

Append to `generate_synthetic_v3.py`:

```python
# ── Outer canvas (bezel) and dial face ───────────────────────────────────

def draw_outer_canvas(img, params, outer_scale, dial_h, dial_s):
    """Render the bezel (annular fill + rim band) around the dial.

    The bezel is an ellipse with the same orientation as the dial, scaled by
    `outer_scale`. The annular region between the dial face and the bezel
    boundary is filled with `outer_rgb`; the rim band sits on the dial edge.
    """
    canvas_size = img.size[0]
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]

    outer_a = a * outer_scale
    outer_b = b * outer_scale

    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    u = (xs - cx) * cos_o + (ys - cy) * sin_o
    v = -(xs - cx) * sin_o + (ys - cy) * cos_o

    inner_r2 = (u / max(a, 1)) ** 2 + (v / max(b, 1)) ** 2
    outer_r2 = (u / max(outer_a, 1)) ** 2 + (v / max(outer_b, 1)) ** 2

    inside_outer = outer_r2 <= 1.0
    annular = (inner_r2 > 1.0) & inside_outer

    outer_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*OUTER_V_RANGE))
    rim_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*RIM_V_RANGE))

    arr = np.array(img)
    arr[annular] = outer_rgb

    # Rim band: thin shell straddling inner_r2 == 1
    rim_half = RIM_WIDTH / min(max(a, 1), max(b, 1))
    r_norm = np.sqrt(inner_r2)
    rim_band = (r_norm > 1.0 - rim_half) & (r_norm < 1.0 + rim_half)
    arr[rim_band] = rim_rgb

    return Image.fromarray(arr, img.mode), inside_outer


def draw_dial_boundary(img, params, dial_h, dial_s):
    """Draw the faint dial circle as a polyline mapped through canonical→image."""
    cx, cy, a, b, theta = params["cx"], params["cy"], params["a"], params["b"], params["theta"]
    circle_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*CIRCLE_V_RANGE))

    pts = []
    n_seg = 180
    for k in range(n_seg + 1):
        t = 2 * math.pi * k / n_seg
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = cx + (a * cos_t * math.cos(theta) - b * sin_t * math.sin(theta))
        y = cy + (a * cos_t * math.sin(theta) + b * sin_t * math.cos(theta))
        pts.append((x, y))

    draw = ImageDraw.Draw(img)
    draw.line(pts, fill=circle_rgb + (255,), width=1)
    return img
```

- [ ] **Step 5.2: Commit**

```bash
git add generate_synthetic_v3.py
git commit -m "autocal: v3 bezel and dial-boundary rendering"
```

---

## Task 6: Generator — scale ticks, digits, and pointer

**Files:**
- Modify: `generate_synthetic_v3.py` (append)

- [ ] **Step 6.1: Append tick, digit, and pointer rendering**

Append to `generate_synthetic_v3.py`:

```python
# ── Scale ticks ──────────────────────────────────────────────────────────

def _canonical_to_image(t, r_c, cx, cy, a, b, theta):
    """Inline copy of the canonical→image map for use in the generator."""
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return x, y


def draw_scale_ticks(img, params):
    """Render scale ticks at canonical angles t_i = phi_0 + i·36°.

    Sub-ticks evenly spaced between consecutive main ticks. Tick lengths and
    widths are jittered per image (shared values across all ticks in the image).
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]

    ticks_per_digit = random.randint(*TICKS_PER_DIGIT_RANGE)
    main_len_frac = random.uniform(*TICK_LEN_FRAC_RANGE)
    main_width = random.randint(*TICK_WIDTH_RANGE)
    minor_width = max(1, main_width - 1)
    grey = random.randint(*TICK_GREY_RANGE)
    color = (grey, grey, grey, 255)

    draw = ImageDraw.Draw(img)
    step_t = 2 * math.pi / N_MARKS / ticks_per_digit

    for i in range(N_MARKS):
        t_main = phi_0 + i * (2 * math.pi / N_MARKS)
        for s in range(ticks_per_digit):
            t = t_main + s * step_t
            if s == 0:
                len_frac = main_len_frac
                width = main_width
            else:
                len_frac = main_len_frac * 0.5
                width = minor_width
            r_inner = 1.0 - len_frac
            x0, y0 = _canonical_to_image(t, r_inner, cx, cy, a, b, theta)
            x1, y1 = _canonical_to_image(t, 1.0, cx, cy, a, b, theta)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    return img


# ── Digits ───────────────────────────────────────────────────────────────

def draw_scale_digits(img, params):
    """Render digits 0..9 at canonical t_i = phi_0 + i·36°.

    Random font size shared across the 10 digits. Half the time inside the
    ticks, half outside.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]

    font_size = random.randint(*DIGIT_FONT_SIZE_RANGE)
    font = _load_font("arial.ttf", font_size)
    grey = random.randint(*DIGIT_GREY_RANGE)
    color = (grey, grey, grey, 255)

    inside = random.random() < DIGIT_INSIDE_PROB
    if inside:
        r_c = 1.0 - random.uniform(0.18, 0.28)
    else:
        r_c = 1.0 + random.uniform(0.08, 0.18)

    draw = ImageDraw.Draw(img)
    for i in range(N_MARKS):
        t = phi_0 + i * (2 * math.pi / N_MARKS)
        x, y = _canonical_to_image(t, r_c, cx, cy, a, b, theta)
        text = str(i)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw / 2, y - th / 2), text, fill=color, font=font)
    return img


# ── Pointer ──────────────────────────────────────────────────────────────

def _random_pointer_rgb():
    h = random.random()
    s = random.uniform(*PTR_S_RANGE)
    v = random.uniform(0.4, 0.95)
    r, g, b_ = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b_ * 255))


def draw_pointer(img, params):
    """Draw the pointer (hub circle + isosceles triangle) at canonical angle.

    Apex at canonical angle phi_0 + pointer_value · 36°, r_c = 1.0 - small margin.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]
    pv = params["pointer_value"]

    hub_ratio = random.uniform(*HUB_RATIO_RANGE)
    hub_r = a * hub_ratio
    hole_r = hub_r * random.uniform(*HOLE_RATIO_RANGE)
    ptr_rgb = _random_pointer_rgb()
    hole_rgb = tuple(max(0, c - 60) for c in ptr_rgb)

    t_apex = phi_0 + pv * (2 * math.pi / N_MARKS)
    apex_x, apex_y = _canonical_to_image(t_apex, 0.92, cx, cy, a, b, theta)

    # Base of triangle: a chord at the hub edge.
    base_t_offset = math.pi  # opposite side
    # We render the pointer in image space as a triangle from hub to apex.
    dx = apex_x - cx
    dy = apex_y - cy
    norm = math.hypot(dx, dy) or 1.0
    ux = dx / norm
    uy = dy / norm
    perp_x = -uy
    perp_y = ux
    half_w = hub_r * 0.6
    base_mx = cx
    base_my = cy
    base_left = (base_mx + half_w * perp_x, base_my + half_w * perp_y)
    base_right = (base_mx - half_w * perp_x, base_my - half_w * perp_y)

    draw = ImageDraw.Draw(img)
    draw.polygon([(apex_x, apex_y), base_left, base_right],
                 fill=ptr_rgb + (255,))
    draw.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r],
                 fill=ptr_rgb + (255,))
    draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r],
                 fill=hole_rgb + (255,))
    return img
```

- [ ] **Step 6.2: Commit**

```bash
git add generate_synthetic_v3.py
git commit -m "autocal: v3 scale ticks, digits, and pointer rendering"
```

---

## Task 7: Generator — post-processing, main loop, CSV output

**Files:**
- Modify: `generate_synthetic_v3.py` (append)

- [ ] **Step 7.1: Append post-processing and the `generate_image` driver**

Append to `generate_synthetic_v3.py`:

```python
# ── Adversarial triangle artifacts ───────────────────────────────────────

def add_triangle_artifacts(img):
    """Random translucent triangles scattered on the canvas."""
    arr = np.array(img)
    canvas_size = arr.shape[0]
    n = random.randint(*TRI_COUNT_RANGE)
    for _ in range(n):
        size = random.randint(*TRI_SIZE_RANGE)
        cx = random.randint(0, canvas_size - 1)
        cy = random.randint(0, canvas_size - 1)
        verts = []
        for _v in range(3):
            ang = random.uniform(0, 2 * math.pi)
            r = random.uniform(*TRI_VERTEX_DIST_RANGE) * size
            verts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
        base = tuple(random.randint(*TRI_MEAN_RANGE) for _ in range(3))
        tri_mask = Image.new("L", (canvas_size, canvas_size), 0)
        ImageDraw.Draw(tri_mask).polygon(verts, fill=255)
        m = np.array(tri_mask) > 127
        if not m.any():
            continue
        for c in range(3):
            noise = np.random.randint(-TRI_PIXEL_VAR, TRI_PIXEL_VAR + 1,
                                      size=m.sum(), dtype=np.int16)
            arr[:, :, c][m] = np.clip(base[c] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, img.mode)


# ── Top-level image generation ───────────────────────────────────────────

def generate_image(path, img_size=IMG_SIZE):
    """Generate one synthetic dial image. Returns dict of label fields."""
    params = sample_params(img_size=img_size)

    # 1. Procedural background fill (full canvas).
    img = make_procedural_background(img_size)

    # Shared HSV dial palette for bezel/circle/rim.
    dial_h = random.random()
    dial_s = random.uniform(*DIAL_S_RANGE)

    outer_scale = random.uniform(*OUTER_SCALE_RANGE)

    # 2. Distractor text (outside bezel disc only).
    img = draw_distractor_text(img, params, outer_scale)

    # 3. Outer canvas (bezel + rim).
    img, _inside_outer = draw_outer_canvas(img, params, outer_scale,
                                           dial_h, dial_s)

    # 4. Dial boundary, ticks, digits.
    img = draw_dial_boundary(img, params, dial_h, dial_s)
    img = draw_scale_ticks(img, params)
    img = draw_scale_digits(img, params)

    # 5. Pointer.
    img = draw_pointer(img, params)

    # 6. Triangle artifacts (across whole canvas).
    img = add_triangle_artifacts(img)

    # 7. Colour/exposure jitter.
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
    img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    img.save(path)

    # Derive marks from the same params we rendered.
    marks = derive_marks_np(params["cx"], params["cy"], params["a"], params["b"],
                            params["theta"], params["phi_0"])

    label = {
        "filename": os.path.basename(path),
        "cx": round(params["cx"], 3),
        "cy": round(params["cy"], 3),
        "a": round(params["a"], 3),
        "b": round(params["b"], 3),
        "theta": round(params["theta"], 6),
        "phi_0": round(params["phi_0"], 6),
    }
    for i in range(N_MARKS):
        label[f"mark_{i}_x"] = round(float(marks[i, 0]), 3)
        label[f"mark_{i}_y"] = round(float(marks[i, 1]), 3)
    return label


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic autocal images")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "labels.csv")

    fieldnames = ["filename", "cx", "cy", "a", "b", "theta", "phi_0"]
    for i in range(N_MARKS):
        fieldnames += [f"mark_{i}_x", f"mark_{i}_y"]

    rows = []
    for i in range(args.num_images):
        fname = f"dial_{i:05d}.png"
        fpath = os.path.join(args.output_dir, fname)
        rows.append(generate_image(fpath))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {args.num_images} images in '{args.output_dir}/'")
    print(f"Labels written to '{csv_path}'")


if __name__ == "__main__":
    main()
```

- [ ] **Step 7.2: Generate a tiny dataset and visually inspect**

Run:
```bash
python generate_synthetic_v3.py --output-dir sv3_smoke --num-images 20
```
Expected: prints `Generated 20 images in 'sv3_smoke/'`. Open 3-5 PNGs in `sv3_smoke/`; verify:
- Each shows a dial with bezel, on a textured background (not black).
- Some have distractor digits/labels in the background.
- The pointer is present at varied positions.
- The dial ellipse is rotated / non-circular in some images.

- [ ] **Step 7.3: Sanity-check label/image consistency**

Create a one-off check script `check_v3_labels.py` at the project root:

```python
"""Overlay derived marks on the first 5 generated images for visual verification."""

import csv
import os

from PIL import Image, ImageDraw

DATA_DIR = "sv3_smoke"

with open(os.path.join(DATA_DIR, "labels.csv"), newline="") as f:
    rows = list(csv.DictReader(f))[:5]

for row in rows:
    img = Image.open(os.path.join(DATA_DIR, row["filename"])).convert("RGB")
    d = ImageDraw.Draw(img)
    for i in range(10):
        mx = float(row[f"mark_{i}_x"])
        my = float(row[f"mark_{i}_y"])
        d.ellipse([mx - 4, my - 4, mx + 4, my + 4],
                  outline=(255, 0, 0), width=2)
    cx = float(row["cx"])
    cy = float(row["cy"])
    d.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], fill=(0, 255, 0))
    img.save(os.path.join(DATA_DIR, "check_" + row["filename"]))
    print(f"check_{row['filename']}")
```

Run: `python check_v3_labels.py`
Expected: 5 `check_*.png` files in `sv3_smoke/`. Open them; each red dot should land on a scale tick / digit position, the green dot on the dial centre. If any red dot misses, the math/labels are inconsistent — debug before continuing.

- [ ] **Step 7.4: Remove smoke artifacts and commit the generator**

Portable cleanup (works on Windows PowerShell and Unix shells):
```bash
python -c "import shutil, os; shutil.rmtree('sv3_smoke', ignore_errors=True); os.remove('check_v3_labels.py') if os.path.exists('check_v3_labels.py') else None"
git add generate_synthetic_v3.py
git commit -m "autocal: v3 generator main loop with CSV labels"
```

---

## Task 8: Dataset class

**Files:**
- Modify: `autocal_utils.py` (append)
- Create: `tests/test_autocal_dataset.py`

- [ ] **Step 8.1: Write failing test for the dataset**

Create `tests/test_autocal_dataset.py`:

```python
"""Tests for the AutoCalibDataset."""

import csv
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from autocal_utils import AutoCalibDataset


def _write_tiny_dataset(tmp_dir, n=3):
    """Write `n` blank 256x256 images and a labels.csv with hand-set values."""
    fieldnames = ["filename", "cx", "cy", "a", "b", "theta", "phi_0"]
    for i in range(10):
        fieldnames += [f"mark_{i}_x", f"mark_{i}_y"]
    rows = []
    for k in range(n):
        fname = f"img_{k}.png"
        Image.new("RGB", (256, 256), (k * 20, 0, 0)).save(
            os.path.join(tmp_dir, fname))
        row = {"filename": fname, "cx": 128.0, "cy": 100.0 + k,
               "a": 60.0, "b": 55.0, "theta": 0.1, "phi_0": 0.5}
        for i in range(10):
            row[f"mark_{i}_x"] = 100.0 + i
            row[f"mark_{i}_y"] = 200.0 + i
        rows.append(row)
    with open(os.path.join(tmp_dir, "labels.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_dataset_loads_and_returns_correct_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_dataset(tmp, n=3)
        ds = AutoCalibDataset(tmp)
        assert len(ds) == 3
        img, params_raw, marks_raw, fname = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 256, 256)
        assert img.dtype == torch.float32
        assert (img >= 0).all() and (img <= 1).all()
        assert params_raw.shape == (6,)
        assert marks_raw.shape == (10, 2)
        assert fname == "img_0.png"


def test_dataset_param_ordering():
    """Returned params are [cx, cy, a, b, theta, phi_0] in that order."""
    with tempfile.TemporaryDirectory() as tmp:
        _write_tiny_dataset(tmp, n=1)
        ds = AutoCalibDataset(tmp)
        _, params_raw, _, _ = ds[0]
        assert params_raw[0].item() == pytest.approx(128.0)
        assert params_raw[1].item() == pytest.approx(100.0)
        assert params_raw[2].item() == pytest.approx(60.0)
        assert params_raw[3].item() == pytest.approx(55.0)
        assert params_raw[4].item() == pytest.approx(0.1)
        assert params_raw[5].item() == pytest.approx(0.5)
```

- [ ] **Step 8.2: Run tests to verify they fail**

Run: `pytest tests/test_autocal_dataset.py -v`
Expected: FAIL — `AutoCalibDataset` not yet defined.

- [ ] **Step 8.3: Implement the dataset in autocal_utils.py**

Append to `autocal_utils.py`:

```python
# ── Dataset ──────────────────────────────────────────────────────────────

import csv
import os

from PIL import Image
from torch.utils.data import Dataset


class AutoCalibDataset(Dataset):
    """Synthetic autocal images with regression targets.

    __getitem__ returns:
        img:          (3, IMG_SIZE, IMG_SIZE) float32 in [0, 1]
        params_raw:   (6,) float32 — [cx, cy, a, b, theta, phi_0]
        marks_raw:    (10, 2) float32 — pixel coords of 10 marks
        filename:     str
    """

    def __init__(self, data_dir, img_size=IMG_SIZE):
        self.data_dir = data_dir
        self.img_size = img_size
        self.samples = []
        with open(os.path.join(data_dir, "labels.csv"), newline="") as f:
            for row in csv.DictReader(f):
                marks = []
                for i in range(N_MARKS):
                    marks.append((float(row[f"mark_{i}_x"]),
                                  float(row[f"mark_{i}_y"])))
                self.samples.append({
                    "filename": row["filename"],
                    "cx": float(row["cx"]),
                    "cy": float(row["cy"]),
                    "a": float(row["a"]),
                    "b": float(row["b"]),
                    "theta": float(row["theta"]),
                    "phi_0": float(row["phi_0"]),
                    "marks": marks,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(os.path.join(self.data_dir, s["filename"])).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC → CHW

        params_raw = torch.tensor([s["cx"], s["cy"], s["a"], s["b"],
                                   s["theta"], s["phi_0"]], dtype=torch.float32)
        marks_raw = torch.tensor(s["marks"], dtype=torch.float32)
        return torch.from_numpy(arr), params_raw, marks_raw, s["filename"]
```

- [ ] **Step 8.4: Run tests to verify they pass**

Run: `pytest tests/test_autocal_dataset.py -v`
Expected: Both tests PASS.

- [ ] **Step 8.5: Commit**

```bash
git add autocal_utils.py tests/test_autocal_dataset.py
git commit -m "autocal: AutoCalibDataset with tests"
```

---

## Task 9: Backbone architectures with regression heads

**Files:**
- Modify: `autocal_utils.py` (append)
- Create: `tests/test_autocal_models.py`

The two backbones reuse the encoders/bottlenecks of `train_e_inv_res.py` and `train_f_se_dwsep.py` but drop the decoder. They use a 1×1-conv regression head on the bottleneck features.

- [ ] **Step 9.1: Write failing tests for both backbones**

Create `tests/test_autocal_models.py`:

```python
"""Smoke tests for the two autocal backbones."""

import pytest
import torch

from autocal_utils import AutocalModelE, AutocalModelF


def test_model_e_forward_shape():
    model = AutocalModelE()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_model_f_forward_shape():
    model = AutocalModelF()
    model.eval()
    x = torch.randn(2, 3, 256, 256)
    out = model(x)
    assert out.shape == (2, 8)
    assert torch.isfinite(out).all()


def test_model_e_backprop():
    model = AutocalModelE()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    out.sum().backward()
    # Spot-check one parameter has a gradient.
    found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            found = True
            break
    assert found


def test_model_f_backprop():
    model = AutocalModelF()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    out.sum().backward()
    found = False
    for p in model.parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            found = True
            break
    assert found
```

- [ ] **Step 9.2: Run tests to verify they fail**

Run: `pytest tests/test_autocal_models.py -v`
Expected: FAIL — `AutocalModelE`, `AutocalModelF` not yet defined.

- [ ] **Step 9.3: Implement both backbones**

Append to `autocal_utils.py`:

```python
# ── Backbone architectures (regression heads) ────────────────────────────

import torch.nn as nn
import torch.nn.functional as F


# Building blocks copied from train_e_inv_res.py and train_f_se_dwsep.py.

class _InvRes(nn.Module):
    """Inverted residual block (MobileNetV2 style)."""

    def __init__(self, in_ch, out_ch, stride=1, expand_ratio=2):
        super().__init__()
        mid = in_ch * expand_ratio
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid, 1, bias=False),
                       nn.BatchNorm2d(mid), nn.ReLU6(inplace=True)]
        layers += [nn.Conv2d(mid, mid, 3, stride=stride, padding=1,
                             groups=mid, bias=False),
                   nn.BatchNorm2d(mid), nn.ReLU6(inplace=True),
                   nn.Conv2d(mid, out_ch, 1, bias=False),
                   nn.BatchNorm2d(out_ch)]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_res:
            out = out + x
        return out


class _ChannelSE(nn.Module):
    """Squeeze-and-Excitation channel attention."""

    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False), nn.Sigmoid())

    def forward(self, x):
        return x * self.fc(self.pool(x))


class _DWSepSE(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, stride=stride, padding=1,
                      groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch), nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
        self.se = _ChannelSE(out_ch, reduction=4)

    def forward(self, x):
        return self.se(self.block(x))


class _DWSepResSE(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, groups=ch, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True))
        self.se = _ChannelSE(ch, reduction=4)

    def forward(self, x):
        return x + self.se(self.block(x))


class _RegressionHead(nn.Module):
    """Global avg pool → 1x1 conv ReLU → 1x1 conv to 8 outputs.

    Conv-only design exports cleanly to ONNX.
    """

    def __init__(self, in_ch, hidden=64, out=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_ch, hidden, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, out, 1, bias=True)

    def forward(self, x):
        x = self.pool(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.flatten(1)  # (B, 8)


class AutocalModelE(nn.Module):
    """Autocal variant of Model E (Inverted Residual encoder + regression head).

    Encoder identical to InvResNet up to the bottleneck; decoder removed.
    """

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 24, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24), nn.ReLU6(inplace=True))         # 128x128x24
        self.enc2 = _InvRes(24, 48, stride=2, expand_ratio=2)    # 64x64x48
        self.enc3 = _InvRes(48, 64, stride=2, expand_ratio=2)    # 32x32x64
        self.bn1 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.bn2 = _InvRes(64, 64, stride=1, expand_ratio=2)
        self.head = _RegressionHead(64)

    def forward(self, x):
        x = self.stem(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bn2(self.bn1(x))
        return self.head(x)


class AutocalModelF(nn.Module):
    """Autocal variant of Model F (SE-enhanced DWSep encoder + regression head)."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True))           # 128x128x32
        self.enc2 = _DWSepSE(32, 64, stride=2)                    # 64x64x64
        self.enc3 = _DWSepSE(64, 96, stride=2)                    # 32x32x96
        self.bn1 = _DWSepResSE(96)
        self.bn2 = _DWSepResSE(96)
        self.head = _RegressionHead(96)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.bn2(self.bn1(x))
        return self.head(x)
```

- [ ] **Step 9.4: Run tests to verify they pass**

Run: `pytest tests/test_autocal_models.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 9.5: Commit**

```bash
git add autocal_utils.py tests/test_autocal_models.py
git commit -m "autocal: Model E and F backbones with regression head"
```

---

## Task 10: Training utilities (epoch loop, eval, mark-error metric)

**Files:**
- Modify: `autocal_utils.py` (append)

- [ ] **Step 10.1: Append training utilities**

Append to `autocal_utils.py`:

```python
# ── Training utilities ───────────────────────────────────────────────────

import time

import torch.optim as optim
from torch.utils.data import DataLoader, random_split


def train_one_epoch_autocal(model, loader, optimizer, device,
                            img_size=IMG_SIZE, w_param=0.1, w_mark=1.0):
    """One training epoch. Returns (mean_loss, mean_mark_error_px)."""
    model.train()
    total_loss = 0.0
    total_mark_err = 0.0
    n = 0
    for imgs, params_raw, marks_raw, _fnames in loader:
        imgs = imgs.to(device)
        params_raw = params_raw.to(device)
        marks_raw = marks_raw.to(device)

        pred_8 = model(imgs)

        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0 = params_raw.unbind(-1)
        loss, _lp, _lm = autocal_loss(
            pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
            marks_raw, img_size=img_size, w_param=w_param, w_mark=w_mark)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            cx_p, cy_p, a_p, b_p, theta_p, phi_p = unpack_params_from_8(
                pred_8, img_size=img_size)
            pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_p)
            mark_err = torch.norm(pred_marks - marks_raw, dim=-1).mean()

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        total_mark_err += mark_err.item() * bs
        n += bs

    return total_loss / n, total_mark_err / n


@torch.no_grad()
def evaluate_autocal(model, loader, device, img_size=IMG_SIZE,
                     w_param=0.1, w_mark=1.0):
    """Evaluate on a validation loader. Returns dict of metrics."""
    model.eval()
    total_loss = 0.0
    all_mark_err = []
    all_center_err = []
    n = 0
    for imgs, params_raw, marks_raw, _fnames in loader:
        imgs = imgs.to(device)
        params_raw = params_raw.to(device)
        marks_raw = marks_raw.to(device)

        pred_8 = model(imgs)

        gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0 = params_raw.unbind(-1)
        loss, _lp, _lm = autocal_loss(
            pred_8, gt_cx, gt_cy, gt_a, gt_b, gt_theta, gt_phi_0,
            marks_raw, img_size=img_size, w_param=w_param, w_mark=w_mark)

        cx_p, cy_p, a_p, b_p, theta_p, phi_p = unpack_params_from_8(
            pred_8, img_size=img_size)
        pred_marks = derive_marks_torch(cx_p, cy_p, a_p, b_p, theta_p, phi_p)
        per_sample_mark_err = torch.norm(pred_marks - marks_raw, dim=-1).mean(dim=-1)
        per_sample_center_err = torch.sqrt(
            (cx_p - gt_cx) ** 2 + (cy_p - gt_cy) ** 2)

        bs = imgs.size(0)
        total_loss += loss.item() * bs
        all_mark_err.extend(per_sample_mark_err.cpu().tolist())
        all_center_err.extend(per_sample_center_err.cpu().tolist())
        n += bs

    mark_arr = np.array(all_mark_err)
    center_arr = np.array(all_center_err)
    return {
        "loss": total_loss / n,
        "mark_err_mean": float(mark_arr.mean()),
        "mark_err_p95": float(np.percentile(mark_arr, 95)),
        "mark_err_p99": float(np.percentile(mark_arr, 99)),
        "center_err_mean": float(center_arr.mean()),
    }


def run_autocal_training(model, args):
    """Full training pipeline: train → save best → export ONNX → benchmark."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    ds = AutoCalibDataset(args.data_dir)
    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(
        ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_ldr = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, pin_memory=True)
    val_ldr = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=4, pin_memory=True)
    print(f"Train: {n_train}  Val: {n_val}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    pth_path = args.save_onnx.replace(".onnx", ".pth")
    best_mark_err = float("inf")

    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_mark_err = train_one_epoch_autocal(
            model, train_ldr, optimizer, device,
            w_param=args.w_param, w_mark=args.w_mark)
        val = evaluate_autocal(
            model, val_ldr, device,
            w_param=args.w_param, w_mark=args.w_mark)
        sched.step()
        dt = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {ep:3d}/{args.epochs}  "
              f"train_loss={train_loss:.4f} train_mark={train_mark_err:.2f}px  "
              f"val_loss={val['loss']:.4f} mark_mean={val['mark_err_mean']:.2f}px "
              f"p95={val['mark_err_p95']:.2f} p99={val['mark_err_p99']:.2f} "
              f"center_mean={val['center_err_mean']:.2f}px  "
              f"lr={lr:.2e}  ({dt:.1f}s)")

        if val["mark_err_mean"] < best_mark_err:
            best_mark_err = val["mark_err_mean"]
            torch.save(model.state_dict(), pth_path)
            print(f"  -> saved best mark_err_mean={best_mark_err:.2f}px")

    # Export best.
    model.load_state_dict(torch.load(pth_path, map_location="cpu", weights_only=True))
    _export_onnx_autocal(model, args.save_onnx)


def _export_onnx_autocal(model, path, img_size=IMG_SIZE):
    import onnx
    model.eval().cpu()
    dummy = torch.randn(1, 3, img_size, img_size)
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"], output_names=["params_8"])
    onnx.checker.check_model(onnx.load(path))
    print(f"ONNX exported: {path}  ({os.path.getsize(path) / 1024:.1f} KB)")
```

- [ ] **Step 10.2: Commit**

```bash
git add autocal_utils.py
git commit -m "autocal: training loop, evaluation, and ONNX export"
```

---

## Task 11: Training entrypoints (Model E and Model F)

**Files:**
- Create: `train_autocal_e.py`
- Create: `train_autocal_f.py`

- [ ] **Step 11.1: Create `train_autocal_e.py`**

Create `train_autocal_e.py`:

```python
"""Train the autocal Model E variant (Inverted Residual encoder + regression head)."""

import argparse

from autocal_utils import AutocalModelE, run_autocal_training


def main():
    p = argparse.ArgumentParser(description="Autocal Model E trainer")
    p.add_argument("--data-dir", default="sv3")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--w-param", type=float, default=0.1)
    p.add_argument("--w-mark", type=float, default=1.0)
    p.add_argument("--save-onnx", default="autocal_model_e.onnx")
    args = p.parse_args()

    run_autocal_training(AutocalModelE(), args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.2: Create `train_autocal_f.py`**

Create `train_autocal_f.py`:

```python
"""Train the autocal Model F variant (SE-enhanced DWSep encoder + regression head)."""

import argparse

from autocal_utils import AutocalModelF, run_autocal_training


def main():
    p = argparse.ArgumentParser(description="Autocal Model F trainer")
    p.add_argument("--data-dir", default="sv3")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-split", type=float, default=0.15)
    p.add_argument("--w-param", type=float, default=0.1)
    p.add_argument("--w-mark", type=float, default=1.0)
    p.add_argument("--save-onnx", default="autocal_model_f.onnx")
    args = p.parse_args()

    run_autocal_training(AutocalModelF(), args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 11.3: Smoke-test that both entrypoints parse args without error**

Run:
```bash
python train_autocal_e.py --help
python train_autocal_f.py --help
```
Expected: both print usage text including `--data-dir`, `--epochs`, `--save-onnx`, `--w-param`, `--w-mark`.

- [ ] **Step 11.4: Commit**

```bash
git add train_autocal_e.py train_autocal_f.py
git commit -m "autocal: training entrypoints for Model E and F variants"
```

---

## Task 12: Inference + visualization script

**Files:**
- Create: `infer_autocal.py`

- [ ] **Step 12.1: Create `infer_autocal.py`**

Create `infer_autocal.py`:

```python
"""Run autocal inference on a single image and draw the predicted geometry."""

import argparse
import math
import os

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

from autocal_utils import IMG_SIZE, derive_marks_np


def preprocess(img_path, img_size=IMG_SIZE):
    """Load and resize/pad an image to (img_size, img_size).

    Resizes the longer side to fit, then edge-mean pads the rest.
    Returns (chw_float_array, original_image_resized).
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    scale = img_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = img.resize((new_w, new_h), Image.BICUBIC)

    arr = np.array(img, dtype=np.uint8)
    pad_left = (img_size - new_w) // 2
    pad_top = (img_size - new_h) // 2
    pad_right = img_size - new_w - pad_left
    pad_bottom = img_size - new_h - pad_top

    if pad_left or pad_right:
        edge = arr[:, [0, -1], :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((new_h, img_size, 3), dtype=np.uint8)
        new[:, :pad_left] = edge
        new[:, pad_left:pad_left + new_w] = arr
        new[:, pad_left + new_w:] = edge
        arr = new
    if pad_top or pad_bottom:
        edge = arr[[0, -1], :, :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((img_size, img_size, 3), dtype=np.uint8)
        new[:pad_top, :] = edge
        new[pad_top:pad_top + new_h, :] = arr
        new[pad_top + new_h:, :] = edge
        arr = new

    chw = arr.astype(np.float32) / 255.0
    chw = chw.transpose(2, 0, 1)[None]  # (1, 3, H, W)
    return chw, Image.fromarray(arr, "RGB")


def unpack_8(p):
    """Unpack the 8-vector network output back to 6 raw params (numpy)."""
    cx_n, cy_n, a_n, b_n, sin2, cos2, sin_p, cos_p = p
    cx = cx_n * IMG_SIZE
    cy = cy_n * IMG_SIZE
    a = a_n * (IMG_SIZE / 2)
    b = b_n * (IMG_SIZE / 2)
    theta = 0.5 * math.atan2(sin2, cos2)
    phi_0 = math.atan2(sin_p, cos_p)
    return cx, cy, a, b, theta, phi_0


def draw_geometry(img, cx, cy, a, b, theta, phi_0):
    """Overlay predicted ellipse and 10 marks on the image."""
    d = ImageDraw.Draw(img)
    # Ellipse as polyline
    pts = []
    for k in range(181):
        t = 2 * math.pi * k / 180
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = cx + (a * cos_t * math.cos(theta) - b * sin_t * math.sin(theta))
        y = cy + (a * cos_t * math.sin(theta) + b * sin_t * math.cos(theta))
        pts.append((x, y))
    d.line(pts, fill=(0, 255, 0), width=2)
    # 10 marks
    marks = derive_marks_np(cx, cy, a, b, theta, phi_0)
    for i in range(10):
        mx, my = marks[i]
        d.ellipse([mx - 5, my - 5, mx + 5, my + 5],
                  outline=(255, 0, 0), width=2)
        d.text((mx + 6, my + 6), str(i), fill=(255, 0, 0))
    # Center
    d.ellipse([cx - 6, cy - 6, cx + 6, cy + 6], fill=(255, 255, 0))
    return img


def main():
    p = argparse.ArgumentParser(description="Autocal inference")
    p.add_argument("--model", required=True, help="ONNX model path")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--output", default="autocal_overlay.png")
    args = p.parse_args()

    sess = ort.InferenceSession(args.model,
                                providers=["CPUExecutionProvider"])
    inp_arr, vis_img = preprocess(args.image)
    out = sess.run(None, {"input": inp_arr})[0][0]  # (8,)
    cx, cy, a, b, theta, phi_0 = unpack_8(out)
    print(f"cx={cx:.2f} cy={cy:.2f} a={a:.2f} b={b:.2f} "
          f"theta={math.degrees(theta):.2f}° phi_0={math.degrees(phi_0):.2f}°")
    vis_img = draw_geometry(vis_img, cx, cy, a, b, theta, phi_0)
    vis_img.save(args.output)
    print(f"Overlay saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 12.2: Verify the script parses arguments**

Run: `python infer_autocal.py --help`
Expected: prints usage text including `--model`, `--image`, `--output`.

- [ ] **Step 12.3: Commit**

```bash
git add infer_autocal.py
git commit -m "autocal: inference script with geometry overlay"
```

---

## Task 13: End-to-end smoke test

This task verifies the entire pipeline by generating a small dataset, training for a few epochs, and running inference. It is not run by `pytest` — it's a manual sanity check before scaling up.

**Files:**
- No code files created. This task is a script-level smoke test.

- [ ] **Step 13.1: Generate a tiny dataset (500 images)**

Run:
```bash
python generate_synthetic_v3.py --output-dir sv3_smoke500 --num-images 500
```
Expected: prints `Generated 500 images in 'sv3_smoke500/'`. Takes 1-5 minutes depending on machine.

- [ ] **Step 13.2: Train Model F for 5 epochs on the tiny dataset**

Run:
```bash
python train_autocal_f.py --data-dir sv3_smoke500 --epochs 5 --batch-size 32 --save-onnx autocal_smoke_f.onnx
```
Expected:
- Prints `Parameters: ~65,000`.
- 5 epoch summaries.
- `train_loss` and `val_loss` should decrease across epochs.
- `mark_err_mean` should be < 30 px by epoch 5 (loose bar; this is undertrained).
- `autocal_smoke_f.onnx` and `autocal_smoke_f.pth` are written.

If `mark_err_mean` stays > 80 px or loss is NaN: stop and debug. Likely a normalization mismatch between dataset and loss.

- [ ] **Step 13.3: Run inference on a held-out synthetic image**

```bash
python infer_autocal.py --model autocal_smoke_f.onnx --image sv3_smoke500/dial_00000.png --output check_overlay.png
```
Expected:
- Prints `cx`, `cy`, `a`, `b`, `theta`, `phi_0` values.
- `check_overlay.png` is created. Open it and verify:
  - Green ellipse roughly matches the dial boundary.
  - Red mark dots roughly land on tick / digit positions.
  - Yellow center dot is roughly inside the dial.
- "Roughly" because the model was only trained 5 epochs — alignment will be visibly imperfect but recognizable as a fit.

- [ ] **Step 13.4: Clean up smoke artifacts**

Portable cleanup (works on Windows PowerShell and Unix shells):
```bash
python -c "import shutil, os; shutil.rmtree('sv3_smoke500', ignore_errors=True); [os.remove(p) for p in ['autocal_smoke_f.onnx', 'autocal_smoke_f.pth', 'check_overlay.png'] if os.path.exists(p)]"
```

- [ ] **Step 13.5: Final commit (no code change — marker commit)**

If git status shows nothing to commit, skip. Otherwise:

```bash
git commit --allow-empty -m "autocal: phase 1 smoke test passed"
```

---

## Done

Phase 1 implementation complete. Next steps from the spec:

1. **Real training run.** Generate a 15k-image `sv3/` dataset and train both Model E and Model F for the full 200 epochs each. Target: `mark_err_mean` < 2 px on synthetic held-out.

   ```bash
   python generate_synthetic_v3.py --output-dir sv3 --num-images 15000
   python train_autocal_e.py --data-dir sv3 --epochs 200 --save-onnx autocal_model_e.onnx
   python train_autocal_f.py --data-dir sv3 --epochs 200 --save-onnx autocal_model_f.onnx
   ```

2. **Compare backbones.** Pick the winner on `mark_err_mean` for production.

3. **Phase 2 entry conditions** (see `AUTOCALIBRATION_PLAN.md`):
   - Collect 20-50 real-image labels.
   - Fine-tune the winning backbone on synth + real.
   - Only escalate to homography / radial distortion / bezel text if real-image evaluation shows the need.
