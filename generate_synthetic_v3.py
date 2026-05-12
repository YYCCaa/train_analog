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
THETA_RANGE = (-math.pi / 2, math.pi / 2)  # rad
PHI_0_RANGE = (0.0, 2 * math.pi)           # rad
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
DIGIT_INSIDE_PROB = 0.5              # probability digits are inside ticks (vs outside)

# ── Pointer ──────────────────────────────────────────────────────────────
HUB_RATIO_RANGE = (0.35, 0.45)       # hub_radius / a
PTR_S_RANGE = (0.5, 1.0)             # pointer HSV saturation; matching V is sampled per-image in Task 6
HOLE_RATIO_RANGE = (1.05, 1.10)

# ── HSV palette ──────────────────────────────────────────────────────────
DIAL_S_RANGE = (0, 0.15)
BG_V_INSIDE_RANGE = (0.67, 0.71)     # HSV V for the dial face background (inside the ellipse)
CIRCLE_V_RANGE = (0.84, 0.88)
OUTER_V_RANGE = (0.76, 0.86)
RIM_V_RANGE = (0.71, 0.76)

# ── Procedural background (full canvas) ──────────────────────────────────
BG_HUE_RANGE = (0.0, 1.0)
BG_S_RANGE = (0.0, 0.30)
BG_V_RANGE = (0.10, 0.95)
BG_LOWFREQ_SIZE = 8                  # noise tile size (px), upsampled to IMG_SIZE
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
TRI_VERTEX_DIST_RANGE = (0.3, 1.0)   # fraction of TRI_SIZE_RANGE sampled `size`


def _hsv_rgb(h, s, v):
    """HSV (0-1) → (R, G, B) ints 0-255."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── Parameter sampling ───────────────────────────────────────────────────

def sample_params(img_size=IMG_SIZE):
    """Sample one set of (cx, cy, a, b, theta [rad], phi_0 [rad], pointer_value)."""
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
