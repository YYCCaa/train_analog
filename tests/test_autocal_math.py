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
    """derive_marks_np returns exactly 10 (x, y) mark positions."""
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
    """Marks are spaced exactly 36° apart on the circle when a == b.

    Precondition: cx = cy = 0 and phi_0 = 0 so that atan2(y, x) measures
    image-frame angle directly from the origin without offset ambiguity.
    """
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
            torch.tensor([t], dtype=torch.float64),
            torch.tensor([r_c], dtype=torch.float64),
            torch.tensor([cx], dtype=torch.float64),
            torch.tensor([cy], dtype=torch.float64),
            torch.tensor([a], dtype=torch.float64),
            torch.tensor([b], dtype=torch.float64),
            torch.tensor([theta], dtype=torch.float64))
        assert out.shape == (1, 2)
        assert out[0, 0].item() == pytest.approx(x_np, abs=1e-5)
        assert out[0, 1].item() == pytest.approx(y_np, abs=1e-5)


def test_derive_marks_torch_matches_numpy():
    """Torch implementation produces same output as numpy."""
    cx, cy, a, b, theta, phi_0 = 128.0, 100.0, 60.0, 50.0, 0.3, 0.9
    np_marks = derive_marks_np(cx, cy, a, b, theta, phi_0)
    t_marks = derive_marks_torch(
        torch.tensor([cx], dtype=torch.float64),
        torch.tensor([cy], dtype=torch.float64),
        torch.tensor([a], dtype=torch.float64),
        torch.tensor([b], dtype=torch.float64),
        torch.tensor([theta], dtype=torch.float64),
        torch.tensor([phi_0], dtype=torch.float64))
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
