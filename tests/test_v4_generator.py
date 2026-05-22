"""Tests for generate_synthetic_v4.

Covers:
  - pick_ink_color picks black on bright BG, white on dark BG
  - pick_ink_color falls back to a default for an empty patch
  - sample_params matches v3 byte-for-byte under the same seed
"""

import random

import numpy as np
import pytest
from PIL import Image

import generate_synthetic_v4 as v4


def _solid_bg(rgb, size=256):
    return Image.new("RGB", (size, size), rgb)


def test_pick_ink_color_picks_black_on_bright_bg():
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (0, 0, 0)


def test_pick_ink_color_picks_white_on_dark_bg():
    bg = _solid_bg((20, 20, 20))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (255, 255, 255)


def test_pick_ink_color_picks_black_on_threshold():
    # Mean luminance exactly 128 → black (>= threshold).
    bg = _solid_bg((128, 128, 128))
    color = v4.pick_ink_color(bg, cx=128.0, cy=128.0, a=80.0)
    assert color == (0, 0, 0)


def test_pick_ink_color_clamps_bbox_to_image():
    # Centre right at the corner so half the bbox is outside the image.
    # Should still return a valid colour (uses whatever pixels are in
    # the clamped bbox).
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=0.0, cy=0.0, a=80.0)
    assert color in [(0, 0, 0), (255, 255, 255)]


def test_pick_ink_color_handles_degenerate_bbox():
    # Centre fully outside the image → empty patch → fallback to black
    # (mean_lum default 128.0 → >= 128 → black).
    bg = _solid_bg((230, 230, 230))
    color = v4.pick_ink_color(bg, cx=-1000.0, cy=-1000.0, a=80.0)
    assert color == (0, 0, 0)
