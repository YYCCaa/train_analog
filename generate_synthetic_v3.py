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
DISTRACTOR_MAX_ATTEMPTS = 8       # rejection-sampling cap per text string

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
    # u along major axis, v along minor.
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
    n = random.randint(*DISTRACTOR_COUNT_RANGE)
    for _ in range(n):
        for _attempt in range(DISTRACTOR_MAX_ATTEMPTS):
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
            # greyscale text at random luminance
            grey = random.randint(0, 255)
            color = (grey, grey, grey, 255)

            # Render to a temp layer, rotate, paste.
            # generous: ~2 px per glyph at chosen font size
            tmp = Image.new("RGBA", (size * len(text) * 2, size * 2), (0, 0, 0, 0))
            ImageDraw.Draw(tmp).text((0, 0), text, fill=color, font=font)
            tmp = tmp.rotate(rot, expand=True, resample=Image.BILINEAR)
            overlay.paste(tmp, (x, y), tmp)
            break
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
