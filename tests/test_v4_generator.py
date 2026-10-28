"""Tests for generate_synthetic_v4.

Covers:
  - pick_viable_ink_color picks dark on bright BG, bright on dark BG
  - pick_viable_ink_color avoids dominant hues
  - sample_params stays inside v4's own ranges (2000 seeds)
  - sample_params keeps the true bezel inside the image where the clamp box fits;
    the full promise (every seed) is a strict xfail documenting a clamp bug
  - the ranges v4 shares with v3 are equal; v4's A_RANGE is narrower
"""

import math
import random

import numpy as np
import pytest

import generate_synthetic_v4 as v4


def test_pick_viable_ink_color_returns_dark_on_uniformly_bright_bg():
    """When every BG pixel is bright, viable inks must be dark."""
    bg = np.full((200, 3), 230, dtype=np.uint8)
    for _ in range(20):
        color = v4.pick_viable_ink_color(bg)
        lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        assert lum < 200, f"got bright ink {color} (lum={lum}) on bright BG"


def test_pick_viable_ink_color_returns_bright_on_uniformly_dark_bg():
    """When every BG pixel is dark, viable inks must be far from dark in
    RGB Euclidean distance — bright or colorful candidates outcompete
    dark ones because they're farther from (20,20,20)."""
    bg = np.full((200, 3), 20, dtype=np.uint8)
    for _ in range(20):
        color = v4.pick_viable_ink_color(bg)
        dist = math.sqrt((color[0]-20)**2 + (color[1]-20)**2 + (color[2]-20)**2)
        assert dist > 100, f"got color {color} too close to dark BG (dist={dist:.1f})"


def test_pick_viable_ink_color_avoids_extremes_on_bimodal_bg():
    """Bimodal BG (black + white) → ink must avoid both pure black and
    pure white (each is dist 0 from one of the BG pixel sets)."""
    bg = np.array([[0, 0, 0]] * 100 + [[255, 255, 255]] * 100, dtype=np.uint8)
    for _ in range(20):
        color = v4.pick_viable_ink_color(bg)
        assert color != (0, 0, 0)
        assert color != (255, 255, 255)


def test_pick_viable_ink_color_avoids_dominant_hue():
    """Uniformly yellow BG → ink should be far from yellow in RGB."""
    bg = np.zeros((200, 3), dtype=np.uint8)
    bg[:, 0] = 240
    bg[:, 1] = 220
    bg[:, 2] = 50
    for _ in range(20):
        color = v4.pick_viable_ink_color(bg)
        dr = color[0] - 240
        dg = color[1] - 220
        db = color[2] - 50
        dist = math.sqrt(dr * dr + dg * dg + db * db)
        assert dist > 100, f"ink {color} too close to BG yellow (dist={dist:.1f})"


def test_pick_viable_ink_color_empty_falls_back():
    """Empty union → returns mid-grey fallback."""
    bg = np.zeros((0, 3), dtype=np.uint8)
    color = v4.pick_viable_ink_color(bg)
    assert color == (128, 128, 128)


# Number of seeds swept by the sample_params tests.  int.
N_SEEDS = 2000
# Float slack for range / containment comparisons (px or rad).  float.
EPS = 1e-6


def _samples():
    """Draw one sample_params() dict per seed in range(N_SEEDS).

    Returns: list[dict[str, float]] of length N_SEEDS, keys cx, cy, a, b,
    theta, phi_0, pointer_value, outer_scale.
    """
    out = []
    for seed in range(N_SEEDS):
        random.seed(seed)
        np.random.seed(seed)
        out.append(v4.sample_params())
    return out


def test_sample_params_within_v4_ranges():
    """Every drawn parameter lies in v4's own *_RANGE constants; the centre
    clamp only moves cx / cy toward the image middle, so the displacement
    stays inside CX_DISP_RANGE / CY_DISP_RANGE."""
    half = v4.IMG_SIZE / 2  # float, image centre coordinate (px)
    for p in _samples():
        assert v4.A_RANGE[0] - EPS <= p["a"] <= v4.A_RANGE[1] + EPS
        ratio = p["b"] / p["a"]  # float, b / a
        assert v4.B_RATIO_RANGE[0] - EPS <= ratio <= v4.B_RATIO_RANGE[1] + EPS
        for key, rng in (("theta", v4.THETA_RANGE),
                         ("phi_0", v4.PHI_0_RANGE),
                         ("pointer_value", v4.POINTER_VALUE_RANGE),
                         ("outer_scale", v4.OUTER_SCALE_RANGE)):
            assert rng[0] - EPS <= p[key] <= rng[1] + EPS, (key, p[key])
        assert v4.CX_DISP_RANGE[0] - EPS <= p["cx"] - half <= v4.CX_DISP_RANGE[1] + EPS
        assert v4.CY_DISP_RANGE[0] - EPS <= p["cy"] - half <= v4.CY_DISP_RANGE[1] + EPS


def _outline(p, t):
    """True rotated-bezel outline of one sample.

    p: dict from sample_params; t: (K,) float angles.
    Returns (xs, ys, hx, hy, bx, by): xs/ys (K,) float outline coords (px);
    hx/hy float exact ellipse half-extents (px); bx/by float half-extents of
    the |A cos|+|B sin| box the generator's clamp uses (px).
    """
    A = p["a"] * p["outer_scale"]  # float, bezel semi-major (px)
    B = p["b"] * p["outer_scale"]  # float, bezel semi-minor (px)
    c, s = math.cos(p["theta"]), math.sin(p["theta"])
    xs = p["cx"] + A * np.cos(t) * c - B * np.sin(t) * s
    ys = p["cy"] + A * np.cos(t) * s + B * np.sin(t) * c
    hx = math.hypot(A * c, B * s)
    hy = math.hypot(A * s, B * c)
    bx = abs(A * c) + abs(B * s)
    by = abs(A * s) + abs(B * c)
    return xs, ys, hx, hy, bx, by


def test_sample_params_bezel_fits_inside_image_when_clamp_box_fits():
    """Where the clamp's box fits the image on both axes (2*box <= IMG_SIZE),
    the true bezel outline (exact extents and 64 outline points) lies inside
    [0, IMG_SIZE].  The sweep must contain such samples."""
    size = v4.IMG_SIZE  # int, image side (px)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)  # (64,) outline angles
    n_fit = 0  # int, samples whose clamp box fits on both axes
    for p in _samples():
        xs, ys, hx, hy, bx, by = _outline(p, t)
        if 2 * bx > size or 2 * by > size:
            continue
        n_fit += 1
        assert p["cx"] - hx >= -EPS and p["cx"] + hx <= size + EPS, p
        assert p["cy"] - hy >= -EPS and p["cy"] + hy <= size + EPS, p
        assert xs.min() >= -EPS and xs.max() <= size + EPS, p
        assert ys.min() >= -EPS and ys.max() <= size + EPS, p
    assert n_fit > 0, n_fit


@pytest.mark.xfail(strict=True, reason="v4 bug: sample_params clamps with the over-wide |A cos|+|B sin| box; when it exceeds the image the centre is pushed off-middle and the true bezel overflows (131/2000 seeds, up to 21.5 px), though every such dial would fit centred")
def test_sample_params_bezel_always_inside_image():
    """The full documented promise: for EVERY seed the true bezel outline is
    inside [0, IMG_SIZE].  Currently violated; strict xfail, so fixing the
    clamp makes this XPASS-fail and prompts removing the marker."""
    size = v4.IMG_SIZE  # int, image side (px)
    t = np.linspace(0, 2 * np.pi, 64, endpoint=False)  # (64,) outline angles
    for p in _samples():
        xs, ys, *_ = _outline(p, t)
        assert xs.min() >= -EPS and xs.max() <= size + EPS, p
        assert ys.min() >= -EPS and ys.max() <= size + EPS, p


def test_geometry_ranges_shared_with_v3():
    """The seven ranges v4 did not narrow are identical to v3's; v4's A_RANGE
    is deliberately narrower and sits inside v3's."""
    import generate_synthetic_v3 as v3

    for name in ("CX_DISP_RANGE", "CY_DISP_RANGE", "B_RATIO_RANGE",
                 "THETA_RANGE", "PHI_0_RANGE", "POINTER_VALUE_RANGE",
                 "OUTER_SCALE_RANGE"):
        assert getattr(v4, name) == getattr(v3, name), name
    assert v3.A_RANGE[0] <= v4.A_RANGE[0] < v4.A_RANGE[1] <= v3.A_RANGE[1]
