"""Tests for generate_synthetic_v4.

Covers:
  - pick_viable_ink_color picks dark on bright BG, bright on dark BG
  - pick_viable_ink_color avoids dominant hues
  - sample_params matches v3 byte-for-byte under the same seed
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


def test_sample_params_geometry_matches_v3():
    """v4's sample_params is verbatim from v3 by design.  This regression
    test fails if anyone touches the geometric sampler in either copy."""
    import generate_synthetic_v3 as v3
    import generate_synthetic_v4 as v4_mod

    SEED = 42
    GEOMETRIC_KEYS = ["cx", "cy", "a", "b", "theta", "phi_0", "outer_scale"]

    random.seed(SEED)
    np.random.seed(SEED)
    v3_params = v3.sample_params()

    random.seed(SEED)
    np.random.seed(SEED)
    v4_params = v4_mod.sample_params()

    for key in GEOMETRIC_KEYS:
        assert v4_params[key] == pytest.approx(v3_params[key]), (
            f"v4 sample_params drifted from v3 on key {key!r}: "
            f"v3={v3_params[key]} v4={v4_params[key]}")
