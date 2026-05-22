"""Generate synthetic training images for the tick-only ablation (v4).

Forked from generate_synthetic_v3.py with every render step except scale
ticks and digits removed, on top of Places365 photo backgrounds.  Ink
color (tick + digit) is picked once per image from the mean luminance of
the photo patch under the implicit dial ellipse.

This is an ablation dataset for measuring how much of v3's center
accuracy came from bezel cues vs. tick-structure learning.  Not a
deployment-target distribution.

See docs/superpowers/plans/2026-05-21-v4-generator-ablation.md for
the design rationale.

Labels per image (identical to v3):
  filename, cx, cy, a, b, theta, phi_0, mark_0_x, mark_0_y, ..., mark_9_x, mark_9_y
"""

import argparse
import colorsys
import csv
import math
import os
import random
import re

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

from autocal_utils import IMG_SIZE, N_MARKS, DEG_PER_MARK, derive_marks_np

OUTPUT_DIR = "sv4"
NUM_IMAGES = 25000

# ── Background source ────────────────────────────────────────────────────
# 'procedural' (default) — palette-matched synthetic BG with soft bezel
#                          outpaint.
# 'photo'                — real photo background.  Read directly from
# `BG_PHOTO_DIR` (a folder of *.jpg files, scanned recursively at startup).
# The bezel outpaint is disabled in this mode so the bezel edge stays
# crisp against the photo.
#
# Populating BG_PHOTO_DIR
#   Use the standalone `download_places365.py` script to fetch the
#   Places365 validation split (~36,500 photos, ~2 GB) into a directory
#   of your choice, then point BG_PHOTO_DIR at it.  Or populate that
#   directory by any other means — the generator only requires *.jpg
#   files, no specific layout, no torchvision dependency at runtime.
#
# Reuse-ratio gate
#   We sanity-check at startup that
#       NUM_IMAGES <= len(photo_pool) * BG_PHOTO_MAX_REUSE_RATIO.
#   Augmentation (flip + color jitter) absorbs moderate reuse; the
#   ratio cap catches accidental over-requests like 10M dials from
#   36k photos.  Exceeding the cap raises with an explicit error.
BG_SOURCE = "photo"                  # v4 requires photo; procedural removed
BG_PHOTO_DIR = "~/places365"         # folder of *.jpg / *.npy files (recursive scan)
BG_PHOTO_MAX_REUSE_RATIO = 10        # max NUM_IMAGES / num_photos; raises if exceeded
# Prefetcher — runs a single worker thread that pre-loads the next BG
# image while the main thread renders the current dial.  Overlaps
# ~10-15 ms of JPEG decode (or ~0.5 ms of .npy load) with the ~30 ms
# dial-render pass.  Set BG_PREFETCH=False to disable (useful while
# debugging BG loading; matches the simpler synchronous behavior).
BG_PREFETCH = True
BG_PREFETCH_QUEUE = 3                # how many images to keep pre-loaded ahead
# Per-image augmentation when sampling from the photo pool.
PHOTO_BRIGHTNESS_RANGE = (0.7, 1.3)
PHOTO_COLOR_RANGE = (0.6, 1.4)
PHOTO_CONTRAST_RANGE = (0.8, 1.2)
PHOTO_FLIP_PROB = 0.5
PHOTO_JITTER_PROB = 0.8


# ── Ellipse parameter sampling ranges ────────────────────────────────────
CX_DISP_RANGE = (-40, 40)        # px from image centre (clamped if bezel would overflow)
CY_DISP_RANGE = (-40, 40)
A_RANGE = (75, 90)               # px (semi-major); upper bound chosen so the
                                 # bezel (a · OUTER_SCALE max) always fits in
                                 # a 256-px image even when centred (90·1.32=119<128)
B_RATIO_RANGE = (0.8, 1.0)       # b / a
THETA_RANGE = (-math.pi / 2, math.pi / 2)  # rad
PHI_0_RANGE = (0.0, 2 * math.pi)           # rad
POINTER_VALUE_RANGE = (0.0, 9.0)

# ── Outer canvas (bezel) ─────────────────────────────────────────────────
OUTER_SCALE_RANGE = (1.20, 1.32)
RIM_WIDTH = 3
DIAL_POLY_SEGS = 180     # number of segments for the dial-boundary polyline
# Outer-bezel edge feathering: a soft alpha ramp around outer_r2 = 1.0
# whose half-width varies by angle around the dial center.  Combined with
# the outpaint background (same hue/sat as the dial palette), this
# makes the meter's silhouette dissolve gently into the background
# instead of having a sharp pixel-level edge.  Half-widths are in
# canonical r_norm units (1.0 = bezel boundary), so 0.10 ≈ 10% of the
# bezel radius on each side of the ramp center.
EDGE_FEATHER_RANGE = (0.16, 0.24)
EDGE_FEATHER_CONTROL_POINTS = 8

# ── Scale ticks ──────────────────────────────────────────────────────────
# Positions are rigid (every tick at canonical t = phi_0 + i·36° + s·step_t)
# so the model can rely on them for calibration.  Appearance is varied per
# tick (length jitter, gray jitter) and per image (optional blur, occasional
# fully-colored ticks) so the model can't overfit to a single visual style.
TICKS_PER_DIGIT_RANGE = (1, 3)
TICK_LEN_FRAC_RANGE = (0.1, 0.15)   # fraction of a (semi-major) for tick length
TICK_LEN_JITTER_RANGE = (0.80, 1.20) # per-IMAGE multiplier on main_len_frac (shared by all ticks)
TICK_WIDTH_RANGE = (2, 3)
TICK_GREY_RANGE = (0, 30)
TICK_GREY_PER_TICK_JITTER = (-8, 8)  # per-tick offset around the image-wide base gray
TICK_BLUR_PROB = 0.4                 # probability of applying Gaussian blur to the tick layer
TICK_BLUR_RADIUS_RANGE = (0.4, 1.4)  # blur radius (px) when applied
TICK_COLORED_PROB = 0.10             # probability of saturated-hue ticks instead of gray
TICK_COLORED_S_RANGE = (0.35, 0.75)  # HSV saturation for colored ticks
TICK_COLORED_V_RANGE = (0.10, 0.30)  # HSV value for colored ticks (kept dark)
# Tick curl: approximates the slight curvature real ticks pick up under
# camera perspective when the mark has some physical depth.  The 6-DOF
# affine model can't simulate true perspective, so we approximate with a
# 3-point polyline (outer → mid+curl → inner) where the mid point is
# offset perpendicular to the radial direction by `curl_signed`.
# Per-image: one `curl_strength` magnitude (px) and one `tilt_axis`
# direction.  Per-tick: curl_signed = curl_strength · sin(t - tilt_axis),
# so ticks on opposite sides of the tilt axis curl in opposite directions.
CURL_RANGE = (0.0, 3.0)              # per-image max curl magnitude in pixels

# ── Digits ───────────────────────────────────────────────────────────────
DIGIT_FONT_SIZE_RANGE = (18, 24)
DIGIT_GREY_RANGE = (0, 30)
DIGIT_INSIDE_PROB = 0.3              # probability digits are inside ticks (vs outside)
# Digit position is a SIGNED canonical r_c offset measured from the
# corresponding (un-curled) major tick's CENTROID:
#   inside  → digit_rc = tick_centroid_rc − offset  (toward dial center)
#   outside → digit_rc = tick_centroid_rc + offset  (away from center)
# where tick_centroid_rc = 1 − main_len/2 is the midpoint of the straight
# tick body BEFORE curl displacement.  The range below is the magnitude;
# sign comes from the inside/outside roll (DIGIT_INSIDE_PROB).  The
# bbox-safety floor in draw_scale_digits grows offsets up to the minimum
# that keeps the digit bbox clear of the tick body.  Outside placement
# has no upper bound — digits may spill past the bezel into the
# procedural background.
DIGIT_OFFSET_FROM_TICK_RANGE = (0.12, 0.2)
# Per-digit angular jitter (canonical-t radians).  Each of the 10 digits
# gets an independent sample, so positions look slightly hand-placed
# around their canonical anchor (phi_0 + i·36°) rather than perfectly
# stamped.  ±3° ≈ 5-6 px of arc displacement at r_c·a ≈ 70 px — visible
# but well under the 36° main-tick spacing.
DIGIT_ANGULAR_JITTER_RANGE = (-math.radians(3), math.radians(3))
DIGIT_FONTS = [                      # one font chosen per image (Windows-bundled)
    "arial.ttf", "ariblk.ttf", "calibri.ttf", "cour.ttf", "consola.ttf",
    "georgia.ttf", "tahoma.ttf", "times.ttf", "trebuc.ttf", "verdana.ttf",
]

# ── Pointer ──────────────────────────────────────────────────────────────
HUB_RATIO_RANGE = (0.35, 0.45)       # hub_radius / a (canonical: hub_r_c = HUB_RATIO * 1.0)
PTR_S_RANGE = (0.5, 1.0)             # pointer HSV saturation range
PTR_V_RANGE = (0.40, 0.95)           # pointer HSV value range
HOLE_RATIO_RANGE = (1.05, 1.10)      # mounting hole canonical radius / hub canonical radius
HOLE_DISP_RANGE = (0, 0)             # mounting hole canonical offset from hub centre
PTR_APEX_RC = 0.92                   # apex canonical radius (fraction of semi-axis)
TIP_VARIATION = 0.02                 # fractional jitter on PTR_APEX_RC
CHORD_RATIO_RANGE = (0.99, 1.0)      # base chord length / hub canonical diameter
PTR_DISPLACEMENT_RANGE = (-5, 5)     # whole-pointer image-space offset from dial centre (px)
PTR_GRADIENT_RANGE = (-25, 25)       # hub→tip brightness shift (additive)
PTR_NOISE_RANGE = 25                 # per-pixel pointer noise amplitude
PTR_EDGE_THICKNESS_RANGE = (1, 3)    # pointer edge band width (px)
PTR_EDGE_NOISE_RANGE = 5             # per-pixel random variation on pointer edge
PTR_DARKER_V_FRAC = (0.4, 0.6)       # hole / edge colour: fraction of pointer's V

# ── Pointer tip erasure ──────────────────────────────────────────────────
CUT_TIP = True
CUT_TIP_RADIUS_RANGE = (7, 10)       # radius (px) of erased disc at apex

# ── HSV palette ──────────────────────────────────────────────────────────
DIAL_S_RANGE = (0, 0.15)
BG_V_INSIDE_RANGE = (0.67, 0.71)     # HSV V for the dial face background (inside the ellipse)
CIRCLE_V_RANGE = (0.84, 0.88)
OUTER_V_RANGE = (0.76, 0.86)
RIM_V_RANGE = (0.71, 0.76)

# ── Procedural background (full canvas) ──────────────────────────────────
# When `make_procedural_background` is called with `dial_h`/`dial_s`, it
# samples the background in the dial palette (BG_V_OUTPAINT_RANGE) so the
# bezel-against-background edge has near-zero color contrast — the outer
# dial boundary becomes effectively invisible without needing to draw a
# faint boundary line.  Fallback HSV ranges (BG_HUE/BG_S/BG_V_RANGE) are
# still used if no dial palette is passed.
BG_HUE_RANGE = (0.0, 1.0)
BG_S_RANGE = (0.0, 0.30)
BG_V_RANGE = (0.10, 0.95)
BG_V_OUTPAINT_RANGE = (0.65, 0.88)   # dial-palette V range (centered on bezel V ~0.81)
BG_LOWFREQ_SIZE = 8                  # noise tile size (px), upsampled to IMG_SIZE
BG_VARIATION = 25
BG_PIXEL_NOISE = 5
BG_BLUR_PROB = 0.3
BG_BLUR_RADIUS_RANGE = (3.0, 4.0)

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
TRI_SIZE_RANGE = (15, 20)
TRI_MEAN_RANGE = (50, 200)
TRI_PIXEL_VAR = 5
TRI_VERTEX_DIST_RANGE = (0.3, 1.0)   # fraction of TRI_SIZE_RANGE sampled `size`

# ── Dial-boundary arc gap ────────────────────────────────────────────────
ARC_GAP_RANGE = (360,360)                      # arc gap span in DEGREES of canonical t

# ── Shadows ──────────────────────────────────────────────────────────────
SHADOW_PROB = 0.73
SHADOW_OFFSET_RANGE = (15, 25)         # pointer shadow displacement (px)
SHADOW_OPACITY_RANGE = (160, 180)      # pointer shadow alpha
SHADOW_BLUR_RADIUS = 2.5
SHADOW_S_RANGE = (0.05, 0.2)           # shadow HSV saturation
SHADOW_V_RANGE = (0.05, 0.25)          # shadow HSV value
RIM_SHADOW_OFFSET_FRAC = (0.15, 0.35)  # rim cutout offset / dial radius
RIM_SHADOW_CUTOUT_FRAC = (0.80, 0.95)  # rim cutout radius / dial radius
RIM_SHADOW_PRIMARY_OPACITY = (200, 220)
RIM_SHADOW_BLUR_RANGE = (4.0, 8.0)

# ── Hierarchical noise (inside bezel) ────────────────────────────────────
HIER_NOISE_BLOCKS = 5
HIER_NOISE_BLOCK_VARIANCE = 15
HIER_NOISE_PIXEL_VARIANCE = 3

# ── Dial-region Gaussian blur ────────────────────────────────────────────
DIAL_BLUR_PROB = 1.0
DIAL_BLUR_RADIUS_RANGE = (1.0, 2.0)

# ── Intrusion augmentation ───────────────────────────────────────────────
INTRUSION_PROB = 0.3                   # probability of rendering an intruder dial
INTRUSION_DEPTH_RANGE = (0.10, 0.50)   # fraction of intruder's outer bezel inside the canvas


def pick_ink_color(bg_img, cx, cy, a, threshold=128.0):
    """Pick a per-image tick/digit ink color based on the photo BG patch.

    Computes the mean luminance of the photo region inside the
    bounding box `[cx - a, cy - a, cx + a, cy + a]` (clamped to image
    bounds) using Rec. 601 weights (0.299/0.587/0.114).  If the patch
    is empty (centre fully outside the image), falls back to a 128.0
    mean luminance.  Returns pure black for mean >= threshold,
    pure white otherwise.

    Called once per image after photo augmentation; the returned
    triple is then used for every tick line and every digit stroke
    drawn on this image (except for the rare TICK_COLORED_PROB
    branch, which keeps its own saturated-hue logic).
    """
    w, h = bg_img.size
    xs = max(0, int(cx - a))
    ys = max(0, int(cy - a))
    xe = min(w, int(cx + a))
    ye = min(h, int(cy + a))
    if xe <= xs or ye <= ys:
        mean_lum = 128.0
    else:
        patch = np.asarray(bg_img.crop((xs, ys, xe, ye)))
        if patch.size == 0:
            mean_lum = 128.0
        else:
            mean_lum = round(float(
                (0.299 * patch[..., 0]
                 + 0.587 * patch[..., 1]
                 + 0.114 * patch[..., 2]).mean()), 6)
    return (0, 0, 0) if mean_lum >= threshold else (255, 255, 255)


def _hsv_rgb(h, s, v):
    """HSV (0-1) → (R, G, B) ints 0-255."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── Parameter sampling ───────────────────────────────────────────────────

def sample_params(img_size=IMG_SIZE):
    """Sample one set of (cx, cy, a, b, theta, phi_0, pointer_value, outer_scale).

    The center displacement is clamped so the rotated bezel AABB fits
    entirely inside the image — guarantees the entire dial (out to the
    bezel rim) is visible regardless of which a / theta we drew.
    """
    a = random.uniform(*A_RANGE)
    b = a * random.uniform(*B_RATIO_RANGE)
    theta = random.uniform(*THETA_RANGE)
    phi_0 = random.uniform(*PHI_0_RANGE)
    pointer_value = random.uniform(*POINTER_VALUE_RANGE)
    outer_scale = random.uniform(*OUTER_SCALE_RANGE)

    # Axis-aligned bounding box of the rotated bezel ellipse.  For an
    # ellipse with semi-axes (A, B) rotated by theta, the AABB half-widths
    # are |A·cos θ| + |B·sin θ| (x) and |A·sin θ| + |B·cos θ| (y).
    bezel_a = a * outer_scale
    bezel_b = b * outer_scale
    cos_o = abs(math.cos(theta))
    sin_o = abs(math.sin(theta))
    half_x = bezel_a * cos_o + bezel_b * sin_o
    half_y = bezel_a * sin_o + bezel_b * cos_o

    cx_min = half_x
    cx_max = img_size - half_x
    cy_min = half_y
    cy_max = img_size - half_y

    cx = img_size / 2 + random.uniform(*CX_DISP_RANGE)
    cy = img_size / 2 + random.uniform(*CY_DISP_RANGE)
    cx = max(cx_min, min(cx_max, cx))
    cy = max(cy_min, min(cy_max, cy))

    return {
        "cx": cx, "cy": cy, "a": a, "b": b,
        "theta": theta, "phi_0": phi_0,
        "pointer_value": pointer_value,
        "outer_scale": outer_scale,
    }


# ── Background fill ──────────────────────────────────────────────────────

def make_procedural_background(canvas_size, dial_h=None, dial_s=None):
    """Generate a full-canvas procedural background.

    When `dial_h` and `dial_s` are provided, the background is sampled in
    the dial palette (same hue/saturation, V in `BG_V_OUTPAINT_RANGE`) so
    the bezel-against-background pixel edge has near-zero color contrast.
    This mimics meters photographed against a same-colored mounting
    panel, where the dial outline is barely visible to a human observer
    and the model has to use scale marks for calibration.

    With `dial_h`/`dial_s` omitted, falls back to fully random HSV (the
    earlier high-contrast behavior, useful if you ever want to compare).

    Returns an RGB PIL.Image.
    """
    if dial_h is not None and dial_s is not None:
        h = dial_h
        s = dial_s
        v = random.uniform(*BG_V_OUTPAINT_RANGE)
    else:
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


# ── Photo background (read from a populated folder) ──────────────────────

# Module-level cache of (absolute folder path → sorted list of file paths).
# `_load_bg_photo_paths` is called once when main() validates the folder
# at startup, but is also re-callable safely if generate_image is invoked
# directly from another module — the os.walk only runs on first miss.
_BG_PHOTO_PATHS_CACHE = {}


def _load_bg_photo_paths(folder):
    """Recursively scan `folder` for *.jpg / *.jpeg / *.npy; return sorted list.

    Both JPEG sources (from `download_places365.py`) and pre-decoded
    numpy caches (from `prep_bg_cache.py`) are picked up.  When the
    pool contains both for the same image, `_load_bg_image` uses the
    extension on the chosen path to decide which to read — the
    fast .npy path is taken whenever it's selected.

    Cached on absolute-path so repeat calls from the generation loop
    don't re-walk the directory.  First walk on a 36k-photo tree takes
    <1 s; cached calls are O(1).
    """
    key = os.path.abspath(os.path.expanduser(folder))
    if key in _BG_PHOTO_PATHS_CACHE:
        return _BG_PHOTO_PATHS_CACHE[key]
    if not os.path.isdir(key):
        raise FileNotFoundError(
            f"BG_PHOTO_DIR '{folder}' (resolved to '{key}') does not exist. "
            f"Run `python download_places365.py --root {folder}` first, "
            f"or set BG_PHOTO_DIR to a folder of *.jpg / *.npy files.")
    paths = []
    for dp, _, files in os.walk(key):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".npy")):
                paths.append(os.path.join(dp, f))
    paths.sort()                       # deterministic order under fixed seed
    _BG_PHOTO_PATHS_CACHE[key] = paths
    return paths


def _load_bg_image(path, canvas_size):
    """Load a background image from `path` (.jpg / .jpeg / .npy).

    Returns a PIL.Image RGB at canvas_size × canvas_size, NOT yet
    augmented (caller applies flip + jitter so they stay on the main
    thread under prefetching).

    Loaders:
      .npy           → np.load (already (H, W, 3) uint8 at canvas_size);
                       ~0.5 ms.  Pre-decoded by `prep_bg_cache.py`.
      .jpg / .jpeg   → PIL.Image.open + decode + resize-if-needed;
                       ~5-15 ms (full JPEG decode every call).
    """
    if path.lower().endswith(".npy"):
        arr = np.load(path)
        return Image.fromarray(arr, "RGB")
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    if img.size != (canvas_size, canvas_size):
        img = img.resize((canvas_size, canvas_size), Image.BICUBIC)
    return img


def _apply_photo_augmentation(img):
    """Random flip + brightness/color/contrast jitter applied to every
    photo before it leaves the pool — gives reuse variety so the same
    photo doesn't appear pixel-identical across all the dials it backs.

    Always runs on the main thread (in `generate_image`), even when the
    prefetcher loads the image off-thread — keeps the augmentation RNG
    aligned with the rest of the generator's stochastic choices.
    """
    if random.random() < PHOTO_FLIP_PROB:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if random.random() < PHOTO_JITTER_PROB:
        img = ImageEnhance.Brightness(img).enhance(
            random.uniform(*PHOTO_BRIGHTNESS_RANGE))
        img = ImageEnhance.Color(img).enhance(
            random.uniform(*PHOTO_COLOR_RANGE))
        img = ImageEnhance.Contrast(img).enhance(
            random.uniform(*PHOTO_CONTRAST_RANGE))
    return img


def make_photo_background(canvas_size, photo_paths):
    """Synchronous per-call BG loader.  Used when prefetching is off
    or when `generate_image` is invoked standalone.

    Loads one photo from `photo_paths` (.jpg or .npy auto-detected),
    applies augmentation, returns the PIL.Image.
    """
    path = photo_paths[random.randint(0, len(photo_paths) - 1)]
    img = _load_bg_image(path, canvas_size)
    return _apply_photo_augmentation(img)


class _BGPrefetcher:
    """Pre-loads background images on a worker thread.

    Overlaps disk I/O + JPEG decode (or .npy mmap) with the dial-render
    pass.  Conceptually a one-element pipeline:

        main loop:    render(dial_i)        render(dial_{i+1})
        worker:       load(bg_{i+1})        load(bg_{i+2})

    The decoded image is returned to the main thread via `get()`; the
    main thread applies the augmentation itself so that flip + jitter
    don't race with other stochastic choices in `generate_image`.

    Errors raised by the worker (missing file, corrupt JPEG, etc.) are
    surfaced to the main thread on the next `get()` call.

    Thread-safety note: path selection in the worker uses Python's
    module-level `random` which is thread-safe via an internal lock,
    but the resulting RNG sequence is non-deterministic between runs
    even with the same seed.  This matches the generator's existing
    behavior (no `random.seed()` call elsewhere) — the prefetcher
    doesn't introduce a NEW non-determinism axis, just inherits it.
    """

    _STOP_SENTINEL = object()

    def __init__(self, photo_paths, canvas_size, queue_size=3):
        import queue as _queue_mod
        import threading

        self.photo_paths = photo_paths
        self.canvas_size = canvas_size
        self._queue = _queue_mod.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True)
        self._thread.start()

    def _worker_loop(self):
        import queue as _queue_mod
        n = len(self.photo_paths)
        while not self._stop.is_set():
            try:
                idx = random.randint(0, n - 1)
                img = _load_bg_image(self.photo_paths[idx], self.canvas_size)
            except Exception as e:                              # noqa: BLE001
                self._queue.put(("error", e))
                return
            # Block until space — bounded by queue_size.  Re-check stop
            # flag every 0.5 s so we don't hang on shutdown if the
            # main thread has stopped draining.
            while not self._stop.is_set():
                try:
                    self._queue.put(img, timeout=0.5)
                    break
                except _queue_mod.Full:
                    continue

    def get(self):
        """Return the next pre-loaded BG image (PIL.Image, un-augmented)."""
        item = self._queue.get()
        if isinstance(item, tuple) and item[0] == "error":
            raise RuntimeError(f"BG prefetch failed: {item[1]}") from item[1]
        return item

    def stop(self):
        """Signal the worker to exit and drain any remaining queued items."""
        import queue as _queue_mod
        self._stop.set()
        # Drain so worker can exit its put() block.
        try:
            while True:
                self._queue.get_nowait()
        except _queue_mod.Empty:
            pass
        self._thread.join(timeout=2)


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


# ── Outer canvas (bezel) and dial face ───────────────────────────────────

def draw_outer_canvas(img, params, outer_scale, dial_h, dial_s,
                      outpaint=True):
    """Render the bezel (dial-face fill + annular ring + rim band) with a
    soft, angle-dependent feather on the outer edge.

    The bezel is an ellipse with the same orientation as the dial, scaled
    by `outer_scale`.  Three concentric regions are painted from inside out:
      - inside_dial (inner_r2 ≤ 1.0)    → `bg_inside_rgb` (dial face)
      - annular (inner_r2 > 1.0, in bezel) → `outer_rgb` (bezel ring)
      - rim_band (thin shell at r_norm = 1.0) → `rim_rgb` (raised rim)
    Painting the dial face inside this pass prevents the procedural
    background from showing through the interior.

    The outer bezel boundary (outer_r2 = 1) is then softened: a per-angle
    feather half-width is sampled at `EDGE_FEATHER_CONTROL_POINTS` random
    points around the dial, interpolated to each pixel's local-frame
    angle, and used as an alpha ramp between the freshly painted bezel
    and a pre-paint snapshot of the background.

    `outpaint`
        True (default) — extend the bezel color OUTWARD into the
        feather ring (the `ext_bezel` step).  Combined with a
        procedural background sampled in the dial palette, this
        dissolves the meter silhouette into the surrounding scene.
        False — skip the outward color spread; the feather ramps
        directly to the unmodified background.  Use when the
        background is a real photo (Places365) whose color
        statistics don't match the bezel palette — preserves a
        clean bezel-vs-photo boundary while keeping anti-aliasing.
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
    inside_dial = inner_r2 <= 1.0
    annular = (inner_r2 > 1.0) & inside_outer

    bg_inside_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*BG_V_INSIDE_RANGE))
    outer_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*OUTER_V_RANGE))
    rim_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*RIM_V_RANGE))

    # Snapshot the background BEFORE bezel painting — needed for the
    # feathered alpha blend below.
    bg_snapshot = np.array(img)

    arr = np.array(img)
    arr[inside_dial] = bg_inside_rgb
    arr[annular] = outer_rgb

    # Rim band: thin shell straddling inner_r2 == 1 (overwrites both the
    # inside-dial edge and the inner annular edge so the rim wins on the
    # dial boundary).
    rim_half = RIM_WIDTH / min(max(a, 1), max(b, 1))
    r_norm_inner = np.sqrt(inner_r2)
    rim_band = (r_norm_inner > 1.0 - rim_half) & (r_norm_inner < 1.0 + rim_half)
    arr[rim_band] = rim_rgb

    # ── Soft outer-edge feather ──────────────────────────────────────
    n_cp = EDGE_FEATHER_CONTROL_POINTS
    feather_cp = np.random.uniform(
        EDGE_FEATHER_RANGE[0], EDGE_FEATHER_RANGE[1],
        size=n_cp).astype(np.float32)

    # Local-frame angle for each pixel (dial's own coordinate system).
    angle_local = np.arctan2(v, u).astype(np.float32)
    sector_f = (angle_local + np.pi) / (2.0 * np.pi) * n_cp
    sec_lo = np.floor(sector_f).astype(np.int32) % n_cp
    sec_hi = (sec_lo + 1) % n_cp
    t_frac = (sector_f - np.floor(sector_f)).astype(np.float32)
    # Interpolate feather half-width along the angular control points.
    half_w = ((1.0 - t_frac) * feather_cp[sec_lo]
              + t_frac * feather_cp[sec_hi])

    # Alpha: 1 inside (r_norm < 1 - half_w), 0 outside (r_norm > 1 + half_w),
    # linear ramp through the transition zone.
    r_norm_outer = np.sqrt(outer_r2).astype(np.float32)
    alpha = np.clip(
        (1.0 + half_w - r_norm_outer) / (2.0 * half_w), 0.0, 1.0)

    alpha_3 = alpha[..., None]
    if outpaint:
        # Give the bezel color something to fade FROM in the "outside-but-in-
        # ring" zone (currently bg in `arr`, but we need bezel color there
        # for the blend to produce a visible transition).
        ext_bezel = arr.copy()
        outside_in_ring = (outer_r2 > 1.0) & (alpha > 0.0)
        ext_bezel[outside_in_ring] = outer_rgb
        blended = (bg_snapshot.astype(np.float32) * (1.0 - alpha_3)
                   + ext_bezel.astype(np.float32) * alpha_3)
    else:
        # No outpaint — `arr` outside the bezel still holds the original
        # background.  Alpha ramps from bezel-painted interior to that
        # raw background, giving anti-aliasing without color spread.
        blended = (bg_snapshot.astype(np.float32) * (1.0 - alpha_3)
                   + arr.astype(np.float32) * alpha_3)
    arr = np.clip(blended, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, img.mode), inside_outer


def draw_dial_boundary(img, params, dial_h, dial_s,
                       gap_start_t=None, gap_span_t=0.0):
    """Draw the faint dial circle as a polyline mapped through canonical→image.

    If `gap_span_t > 0`, vertices whose canonical angle lies in the half-open
    range [gap_start_t, gap_start_t + gap_span_t) (mod 2π) are skipped, so
    the ellipse renders as an open arc.  The gap is specified in CANONICAL
    angular space, which under the affine projection appears as a missing
    arc on the projected ellipse.
    """
    cx, cy, a, b, theta = params["cx"], params["cy"], params["a"], params["b"], params["theta"]
    circle_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*CIRCLE_V_RANGE))

    n_seg = DIAL_POLY_SEGS
    draw = ImageDraw.Draw(img)

    if gap_span_t <= 0.0 or gap_start_t is None:
        pts = []
        for k in range(n_seg + 1):  # +1 to close the polyline
            t = 2 * math.pi * k / n_seg
            pts.append(_canonical_to_image(t, 1.0, cx, cy, a, b, theta))
        draw.line(pts, fill=circle_rgb, width=1)
        return img

    # With a gap: stream points into the current segment, flush when we
    # cross into the gap, then start a new segment when we cross out.
    segment = []
    for k in range(n_seg + 1):
        t = 2 * math.pi * k / n_seg
        t_rel = (t - gap_start_t) % (2 * math.pi)
        if t_rel < gap_span_t:
            if len(segment) >= 2:
                draw.line(segment, fill=circle_rgb, width=1)
            segment = []
            continue
        segment.append(_canonical_to_image(t, 1.0, cx, cy, a, b, theta))
    if len(segment) >= 2:
        draw.line(segment, fill=circle_rgb, width=1)
    return img


# ── Scale ticks ──────────────────────────────────────────────────────────

def _canonical_xy_to_image(xc, yc, cx, cy, a, b, theta):
    """Map a canonical-Cartesian point (xc, yc) to image space.

    The canonical frame has the dial as a unit circle centred at origin;
    image space stretches by (a, b) along axes rotated by theta and
    translates by (cx, cy).
    """
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + a * xc * cos_o - b * yc * sin_o
    y = cy + a * xc * sin_o + b * yc * cos_o
    return x, y


def _canonical_to_image(t, r_c, cx, cy, a, b, theta):
    """Polar canonical (t, r_c) → image. Thin wrapper around `_canonical_xy_to_image`.

    Mirror of `autocal_utils.canonical_to_image_np` — keep in sync.
    """
    return _canonical_xy_to_image(r_c * math.cos(t), r_c * math.sin(t),
                                  cx, cy, a, b, theta)


TICK_ARC_SEGMENTS = 8                # polyline segments along the curl arc


def _arc_polyline(x0, y0, x1, y1, mid_x, mid_y, n_segments=TICK_ARC_SEGMENTS):
    """Return a polyline approximating the (unique) circular arc through
    (x0, y0) → (mid_x, mid_y) → (x1, y1).

    Used to draw curved tick marks (perspective-distortion approximation).
    Three non-collinear points define a unique circle; we compute its
    circumcenter and walk the short arc between the endpoints with
    `n_segments` linear segments.  When the three points are (near-)
    collinear — i.e. the curl is essentially zero — the function falls
    back to a straight 2-point line, avoiding a divide-by-zero in the
    circumcenter formula.

    Output order: starts at (x0, y0), ends at (x1, y1), passing through
    a point arbitrarily close to (mid_x, mid_y).
    """
    # Circumcenter via the standard 3-point determinant formula.
    D = 2.0 * (x0 * (mid_y - y1) + mid_x * (y1 - y0) + x1 * (y0 - mid_y))
    if abs(D) < 1e-3:
        return [(x0, y0), (x1, y1)]

    s0 = x0 * x0 + y0 * y0
    sm = mid_x * mid_x + mid_y * mid_y
    s1 = x1 * x1 + y1 * y1
    ux = (s0 * (mid_y - y1) + sm * (y1 - y0) + s1 * (y0 - mid_y)) / D
    uy = (s0 * (x1 - mid_x) + sm * (x0 - x1) + s1 * (mid_x - x0)) / D

    r = math.hypot(x0 - ux, y0 - uy)
    a_start = math.atan2(y0 - uy, x0 - ux)
    a_end = math.atan2(y1 - uy, x1 - ux)
    # Short arc in (-π, π]; this is the one that actually passes through
    # (mid_x, mid_y) because the three points form a thin triangle with
    # the mid offset perpendicular to the chord.
    da = ((a_end - a_start) + math.pi) % (2.0 * math.pi) - math.pi

    pts = []
    for i in range(n_segments + 1):
        t = i / n_segments
        ang = a_start + da * t
        pts.append((ux + r * math.cos(ang), uy + r * math.sin(ang)))
    return pts


def draw_scale_ticks(img, params, protect_draw=None):
    """Render scale ticks at canonical angles t_i = phi_0 + i·36°.

    Positions are deterministic — no angular jitter, ever.  All length
    variability is sampled per image so the 10 main ticks (and their
    subdivisions) keep their bezel-radial inner endpoints aligned within
    a single image; the model can rely on a consistent "tick length"
    cue per image even as the cue varies across the training set.

    Per-image variation:
      - `ticks_per_digit ∈ TICKS_PER_DIGIT_RANGE`
      - `main_len_frac ∈ TICK_LEN_FRAC_RANGE` (base length)
      - `len_jitter ∈ TICK_LEN_JITTER_RANGE` (one multiplier shared by
        all ticks — main and minor scale together)
      - `main_width ∈ TICK_WIDTH_RANGE`
      - `base_grey ∈ TICK_GREY_RANGE`
      - optional Gaussian blur on the tick layer (TICK_BLUR_PROB)
      - rare full hue shift (TICK_COLORED_PROB)

    Per-tick variation:
      - small gray offset only (TICK_GREY_PER_TICK_JITTER) so each tick
        isn't visually identical to the others — this is luminance
        noise, not geometric noise; the endpoints stay rigid.

    Ticks are rasterized onto a separate RGBA layer, optionally blurred,
    then alpha-composited.  The protect mask gets the un-blurred tick
    footprint.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]

    ticks_per_digit = random.randint(*TICKS_PER_DIGIT_RANGE)
    main_len_frac = random.uniform(*TICK_LEN_FRAC_RANGE)
    len_jitter = random.uniform(*TICK_LEN_JITTER_RANGE)  # one per image
    main_width = random.randint(*TICK_WIDTH_RANGE)
    minor_width = max(1, main_width - 1)
    base_grey = random.randint(*TICK_GREY_RANGE)

    # Per-image color mode: usually gray, occasionally a saturated hue
    # (dark red / dark blue / etc.) so the model sees colored-tick variants.
    colored_mode = random.random() < TICK_COLORED_PROB
    if colored_mode:
        col_h = random.random()
        col_s = random.uniform(*TICK_COLORED_S_RANGE)
        col_v = random.uniform(*TICK_COLORED_V_RANGE)
        tick_base_rgb = _hsv_rgb(col_h, col_s, col_v)
    else:
        tick_base_rgb = (base_grey, base_grey, base_grey)

    main_len = main_len_frac * len_jitter            # uniform across all main ticks
    minor_len = main_len_frac * 0.5 * len_jitter     # uniform across all minor ticks

    # Per-image perspective-approximation curl: one strength + one tilt
    # axis define a coherent curl pattern over all ticks (ticks on
    # opposite sides of the tilt axis curl in opposite directions, ticks
    # aligned with the axis stay nearly straight).  curl_strength = 0
    # is common and renders straight ticks.
    curl_strength = random.uniform(*CURL_RANGE)
    tilt_axis_t = random.uniform(0, 2 * math.pi)

    canvas_size = img.size[0]
    tick_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    tick_draw = ImageDraw.Draw(tick_layer)

    step_t = 2 * math.pi / N_MARKS / ticks_per_digit

    for i in range(N_MARKS):
        t_main = phi_0 + i * (2 * math.pi / N_MARKS)
        for s in range(ticks_per_digit):
            t = t_main + s * step_t  # rigid canonical angle
            if s == 0:
                len_frac = main_len
                width = main_width
            else:
                len_frac = minor_len
                width = minor_width
            r_inner = 1.0 - len_frac
            x0, y0 = _canonical_to_image(t, r_inner, cx, cy, a, b, theta)
            x1, y1 = _canonical_to_image(t, 1.0, cx, cy, a, b, theta)

            # Tangent direction in IMAGE space (perpendicular to the
            # tick's drawn radial direction).  90° CCW of (dx, dy).
            dx = x1 - x0
            dy = y1 - y0
            d = math.hypot(dx, dy) or 1.0
            tx_dir = -dy / d
            ty_dir = dx / d

            # Per-tick curl magnitude: signed by tick's angular position
            # relative to the per-image tilt axis.  The mid point is the
            # geodesic midpoint of the radial segment offset
            # perpendicular by `curl_signed`; together with the endpoints
            # it defines a unique circumscribed circle.  We then walk the
            # short arc of that circle — smooth curved tick instead of a
            # kinked 3-point polyline.
            curl_signed = curl_strength * math.sin(t - tilt_axis_t)
            mid_x = (x0 + x1) / 2.0 + tx_dir * curl_signed
            mid_y = (y0 + y1) / 2.0 + ty_dir * curl_signed
            polyline = _arc_polyline(x0, y0, x1, y1, mid_x, mid_y)

            # Per-tick gray offset (luminance jitter only — geometry stays rigid).
            if colored_mode:
                offset = random.randint(*TICK_GREY_PER_TICK_JITTER)
                color_rgb = tuple(
                    max(0, min(255, int(c) + offset)) for c in tick_base_rgb)
            else:
                g = max(0, min(255,
                               base_grey + random.randint(*TICK_GREY_PER_TICK_JITTER)))
                color_rgb = (g, g, g)
            color_rgba = color_rgb + (255,)

            tick_draw.line(polyline, fill=color_rgba, width=width,
                           joint="curve")
            if protect_draw is not None:
                protect_draw.line(polyline, fill=255, width=width,
                                  joint="curve")

    if random.random() < TICK_BLUR_PROB:
        blur_r = random.uniform(*TICK_BLUR_RADIUS_RANGE)
        tick_layer = tick_layer.filter(ImageFilter.GaussianBlur(radius=blur_r))

    img = Image.alpha_composite(img.convert("RGBA"), tick_layer).convert("RGB")
    # Inner endpoint of the (main) ticks in canonical r_norm.  Returned so
    # downstream callers (e.g. draw_scale_digits) can place themselves
    # clear of the tick zone without re-sampling the same RNG state.
    tick_inner_rc = 1.0 - main_len
    return img, tick_inner_rc


# ── Digits ───────────────────────────────────────────────────────────────

DIGIT_TICK_GAP = 0.02      # canonical r_norm safety margin between digit bbox and tick


def draw_scale_digits(img, params, tick_inner_rc, protect_draw=None):
    """Render digits 0..9 at canonical t_i = phi_0 + i·36°.

    Position encoding
    -----------------
    Each digit's radial position is a SIGNED canonical r_c offset
    measured from the corresponding (un-curled) major tick's CENTROID:

        anchor  : tick_centroid_rc = 1.0 − main_len / 2  (midpoint of
                  the straight radial tick body, before any curl
                  displacement is applied)
        inside  : r_c = tick_centroid_rc − offset   (toward dial center)
        outside : r_c = tick_centroid_rc + offset   (away from center)

    Sign comes from the inside/outside roll (DIGIT_INSIDE_PROB); the
    magnitude is sampled from DIGIT_OFFSET_FROM_TICK_RANGE.  Anchoring
    on the centroid makes the encoding symmetric — same offset magnitude
    has the same geometric meaning on either side.

    Safety
    ------
    The worst-case digit half-extent in canonical r_norm is the maximum
    bbox half-diagonal across digits 0..9, divided by the dial's
    semi-major axis `a`.  Half-diagonal is the correct projection bound
    because the bbox is image-axis-aligned but the radial direction at
    the digit's angle is arbitrary.

    Bbox-aware floor:
      offset ≥ (main_len / 2) + digit_half_rc + GAP
    The half-tick term lets the digit clear the OPPOSITE end of the
    tick body (the one farther from the centroid in the chosen
    direction), and the bbox term + gap keeps the bbox off the tick
    surface.  The user-specified range bounds are grown up to this
    floor when needed.

    Inside placement also enforces r_c ≥ digit_half_rc + GAP so the
    digit doesn't pass the dial center.  Outside placement has NO
    upper bound — digits may extend past the bezel outer ring into the
    procedural background, matching real meters where panel labels
    spill outside the bezel.

    If the preferred side has no feasible offset, we fall back to the
    other side.  If neither side fits, we tuck the digits well below the
    tick inner edge as a last resort.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]
    outer_scale = params["outer_scale"]

    font_size = random.randint(*DIGIT_FONT_SIZE_RANGE)
    font_name = random.choice(DIGIT_FONTS)
    font = _load_font(font_name, font_size)
    grey = random.randint(*DIGIT_GREY_RANGE)
    color = (grey, grey, grey)

    # Pre-render each digit and measure its actual ink bbox.  We use the
    # ink bbox (not font.getbbox) for two reasons:
    #
    # 1. Half-diagonal for the safe-range calc — ink bbox is the
    #    tightest envelope, so the offset_floor doesn't reserve dead
    #    margin from font metrics (ascender/descender slots).
    # 2. Per-digit anchor correction — PIL's anchor="mm" centers the
    #    FONT bbox at (x, y), but old-style figures (e.g. georgia's
    #    descenders on 3/4/5/7/9 and ascenders on 6/8) make that bbox
    #    asymmetric.  The visible ink center can be up to ~5 px off
    #    from the canonical position, which on a circular dial reads
    #    as the digit ring drifting away from a perfect circle.
    #
    # `ink_offset[i] = (dx, dy)` is the displacement from the "mm"
    # anchor point to the ink bbox center.  At draw time we pass
    # `(x - dx, y - dy)` as the anchor so the visible ink lands at
    # exactly (x, y) regardless of font.
    PROBE = 256
    PROBE_C = PROBE / 2.0
    INK_THRESH = 64
    ink_offset = []
    max_half_diag_px = 0.0
    for i in range(N_MARKS):
        probe = Image.new("L", (PROBE, PROBE), 0)
        pdraw = ImageDraw.Draw(probe)
        pdraw.text((PROBE_C, PROBE_C), str(i), fill=255, font=font, anchor="mm")
        parr = np.asarray(probe)
        ys, xs = np.where(parr > INK_THRESH)
        if len(xs) > 0:
            ix_min, ix_max = xs.min(), xs.max()
            iy_min, iy_max = ys.min(), ys.max()
            ink_cx = (ix_min + ix_max) / 2.0
            ink_cy = (iy_min + iy_max) / 2.0
            ink_offset.append((ink_cx - PROBE_C, ink_cy - PROBE_C))
            tw = ix_max - ix_min
            th = iy_max - iy_min
            half_diag = math.sqrt(tw * tw + th * th) / 2.0
            if half_diag > max_half_diag_px:
                max_half_diag_px = half_diag
        else:
            # Glyph rendered nothing visible (font missing this codepoint
            # at this size); fall back to font.getbbox so safety floor
            # still has a sensible value.
            ink_offset.append((0.0, 0.0))
            bb = font.getbbox(str(i))
            tw = bb[2] - bb[0]
            th = bb[3] - bb[1]
            half_diag = math.sqrt(tw * tw + th * th) / 2.0
            if half_diag > max_half_diag_px:
                max_half_diag_px = half_diag
    digit_half_rc = max_half_diag_px / max(a, 1.0)

    # Anchor: canonical r_c of the un-curled tick's centroid (midpoint of
    # the straight radial segment that the rendered tick traces before
    # the curl displacement is applied).  Using the centroid gives a
    # symmetric offset semantics — `+offset` away from center, `-offset`
    # toward center — independent of the asymmetry between the tick's
    # inner and outer endpoints.
    main_len = 1.0 - tick_inner_rc
    tick_centroid_rc = 1.0 - main_len / 2.0
    tick_half_len = main_len / 2.0

    # Bbox-aware floor: starting from the centroid, the offset has to
    # clear (a) the remaining half-tick toward the relevant endpoint and
    # (b) the digit bbox half-extent + safety gap.  Both range bounds
    # are grown up to this floor when needed, so a user-specified range
    # is honored when feasible and silently widened otherwise (avoids
    # the silent fall-through to the pathological fallback).
    offset_floor = tick_half_len + digit_half_rc + DIGIT_TICK_GAP
    offset_lo = max(DIGIT_OFFSET_FROM_TICK_RANGE[0], offset_floor)
    offset_hi = max(DIGIT_OFFSET_FROM_TICK_RANGE[1], offset_floor)

    def _safe_rc(want_inside):
        """Return r_c by anchoring on the un-curled tick centroid and
        stepping the sampled offset toward (inside) or away from
        (outside) the dial center.  Returns None only if even the floor
        offset doesn't fit on the chosen side."""
        offset = random.uniform(offset_lo, offset_hi)
        if want_inside:
            rc = tick_centroid_rc - offset
            # Don't pass the dial center.
            if rc < digit_half_rc + DIGIT_TICK_GAP:
                return None
            return rc
        # Outside: digit is free to extend past the bezel outer ring
        # into the procedural background (matches real meters where
        # labels sometimes spill onto the surrounding panel).
        return tick_centroid_rc + offset

    inside_pref = random.random() < DIGIT_INSIDE_PROB
    r_c = _safe_rc(inside_pref)
    if r_c is None:
        r_c = _safe_rc(not inside_pref)
    if r_c is None:
        # Pathological case (tiny dial + huge font): tuck the digits
        # well below the tick inner edge so at least no overlap occurs,
        # even if the result looks visually cramped.
        r_c = max(digit_half_rc + DIGIT_TICK_GAP,
                  tick_inner_rc - digit_half_rc - DIGIT_TICK_GAP)

    draw = ImageDraw.Draw(img)
    for i in range(N_MARKS):
        t_jitter = random.uniform(*DIGIT_ANGULAR_JITTER_RANGE)
        t = phi_0 + i * (2 * math.pi / N_MARKS) + t_jitter
        x, y = _canonical_to_image(t, r_c, cx, cy, a, b, theta)
        # `ink_offset[i]` is the displacement from anchor="mm" to the
        # visible ink center for this glyph.  Shifting the anchor by
        # `-ink_offset` makes the ink center land exactly at (x, y).
        dx, dy = ink_offset[i]
        draw.text((x - dx, y - dy), str(i),
                  fill=color, font=font, anchor="mm")
        if protect_draw is not None:
            protect_draw.text((x - dx, y - dy), str(i),
                              fill=255, font=font, anchor="mm")
    return img


# ── Pointer ──────────────────────────────────────────────────────────────

POINTER_HUB_SEGS = 64                  # vertex count for the hub polygon


def _random_pointer_rgb():
    h = random.random()
    s = random.uniform(*PTR_S_RANGE)
    v = random.uniform(*PTR_V_RANGE)
    return _hsv_rgb(h, s, v)


def _luminance_strip_color(base_rgb, darker_only=False):
    """Pick a colour from the luminance strip of *base_rgb* (same hue/sat)."""
    r, g, b = [c / 255.0 for c in base_rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    if darker_only:
        new_v = random.uniform(PTR_DARKER_V_FRAC[0] * v, PTR_DARKER_V_FRAC[1] * v)
    else:
        new_v = random.random()
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, new_v)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


def _pointer_color_variation(img, mask, hub_cx, hub_cy, apex_x, apex_y):
    """Apply lengthwise gradient + per-pixel noise to the pointer body."""
    arr = np.array(img).astype(np.int16)
    m = np.array(mask)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return img

    dx = apex_x - hub_cx
    dy = apex_y - hub_cy
    norm = math.hypot(dx, dy) or 1.0
    axis_dx = dx / norm
    axis_dy = dy / norm

    rel_x = (xs - hub_cx).astype(np.float32)
    rel_y = (ys - hub_cy).astype(np.float32)
    along = rel_x * axis_dx + rel_y * axis_dy
    along_norm = along / norm  # ~-? .. +1 (hub .. tip; back of hub can go slightly negative)

    grad_strength = random.uniform(*PTR_GRADIENT_RANGE)
    grad_shift = (along_norm * grad_strength).astype(np.int16)
    noise = np.random.randint(-PTR_NOISE_RANGE, PTR_NOISE_RANGE + 1,
                              size=len(ys), dtype=np.int16)
    total_shift = grad_shift + noise

    for c in range(3):
        arr[ys, xs, c] = np.clip(arr[ys, xs, c] + total_shift, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), img.mode)


def _draw_pointer_edge(img, mask, edge_color_rgb, bg_arr):
    """Variable-thickness edge band on the pointer mask boundary.

    Outer ring blends mostly toward the background snapshot; inner ring
    is mostly the edge colour. Per-pixel noise is layered on top.
    """
    thickness = random.randint(*PTR_EDGE_THICKNESS_RANGE)
    m = np.array(mask)
    arr = np.array(img)

    dist = np.full(m.shape, thickness, dtype=np.float32)
    current = m.copy()
    for step in range(thickness):
        e1 = np.minimum(np.roll(current, 1, 0), np.roll(current, -1, 0))
        e2 = np.minimum(np.roll(current, 1, 1), np.roll(current, -1, 1))
        eroded = np.minimum(e1, e2) & current
        ring = (current > 0) & (eroded == 0)
        dist[ring] = step
        current = eroded

    edge_band = (m > 0) & (dist < thickness)
    ys, xs = np.where(edge_band)
    if len(ys) == 0:
        return img

    blend = dist[ys, xs] / max(thickness - 1, 1)
    noise = np.random.randint(-PTR_EDGE_NOISE_RANGE, PTR_EDGE_NOISE_RANGE + 1,
                              size=(len(ys), 3), dtype=np.int16)
    bg = bg_arr[ys, xs, :3].astype(np.int16)
    ec = np.array(edge_color_rgb, dtype=np.int16)
    blended = (bg + blend[:, None] * (ec - bg) + noise).astype(np.int16)
    arr[ys, xs, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    if arr.shape[2] == 4:
        arr[ys, xs, 3] = 255
    return Image.fromarray(arr, img.mode)


def _sample_pointer_geometry(params):
    """Sample per-image pointer parameters.  Returns a dict with all
    geometry constants needed to render either the pointer or its shadow.
    """
    a = params["a"]
    phi_0 = params["phi_0"]
    pv = params["pointer_value"]

    hub_r_c = random.uniform(*HUB_RATIO_RANGE)               # canonical units
    hole_r_c = hub_r_c * random.uniform(*HOLE_RATIO_RANGE)
    hole_off_mag = random.uniform(*HOLE_DISP_RANGE)
    hole_off_ang = random.uniform(0, 2 * math.pi)
    hole_off_x_c = hole_off_mag * math.cos(hole_off_ang) / max(a, 1)
    hole_off_y_c = hole_off_mag * math.sin(hole_off_ang) / max(a, 1)

    apex_rc = PTR_APEX_RC * (1 + random.uniform(-TIP_VARIATION, TIP_VARIATION))

    chord_ratio = random.uniform(*CHORD_RATIO_RANGE)
    chord_half_c = chord_ratio * hub_r_c                      # = chord_length / 2
    max_dist_c = math.sqrt(max(hub_r_c ** 2 - chord_half_c ** 2, 0.0))
    chord_dist_c = random.uniform(0, max_dist_c) if max_dist_c > 0 else 0.0

    t_apex = phi_0 + pv * (2 * math.pi / N_MARKS)

    disp_mag = random.uniform(*PTR_DISPLACEMENT_RANGE)
    disp_ang = random.uniform(0, 2 * math.pi)
    ptr_dx = disp_mag * math.cos(disp_ang)
    ptr_dy = disp_mag * math.sin(disp_ang)

    ptr_rgb = _random_pointer_rgb()
    hole_rgb = _luminance_strip_color(ptr_rgb, darker_only=True)
    edge_rgb = _luminance_strip_color(ptr_rgb, darker_only=True)
    cut_tip_radius = random.randint(*CUT_TIP_RADIUS_RANGE)

    return {
        "hub_r_c": hub_r_c, "hole_r_c": hole_r_c,
        "hole_off_xc": hole_off_x_c, "hole_off_yc": hole_off_y_c,
        "apex_rc": apex_rc, "t_apex": t_apex,
        "chord_half_c": chord_half_c, "chord_dist_c": chord_dist_c,
        "ptr_dx": ptr_dx, "ptr_dy": ptr_dy,
        "ptr_rgb": ptr_rgb, "hole_rgb": hole_rgb, "edge_rgb": edge_rgb,
        "cut_tip_radius": cut_tip_radius,
    }


def _pointer_image_vertices(geom, params):
    """Map canonical pointer geometry → image-space vertices, with
    image-space PTR_DISPLACEMENT applied uniformly to every vertex.
    Returns hub_pts (list of (x,y)), apex, base_left, base_right,
    hub_centre, hole_pts, hole_centre.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    dx, dy = geom["ptr_dx"], geom["ptr_dy"]

    def _to_img(xc, yc):
        x, y = _canonical_xy_to_image(xc, yc, cx, cy, a, b, theta)
        return (x + dx, y + dy)

    # Hub: sample canonical circle of radius hub_r_c at origin
    hub_r_c = geom["hub_r_c"]
    hub_pts = []
    for k in range(POINTER_HUB_SEGS):
        t = 2 * math.pi * k / POINTER_HUB_SEGS
        hub_pts.append(_to_img(hub_r_c * math.cos(t), hub_r_c * math.sin(t)))
    hub_centre = _to_img(0.0, 0.0)

    # Apex along canonical axis
    t_apex = geom["t_apex"]
    apex = _to_img(geom["apex_rc"] * math.cos(t_apex),
                   geom["apex_rc"] * math.sin(t_apex))

    # Base midpoint behind hub centre along the axis; perpendicular for L/R
    ax_x = math.cos(t_apex)
    ax_y = math.sin(t_apex)
    perp_x, perp_y = -ax_y, ax_x
    cd = geom["chord_dist_c"]
    ch = geom["chord_half_c"]
    base_mid_xc = -cd * ax_x
    base_mid_yc = -cd * ax_y
    base_left = _to_img(base_mid_xc + ch * perp_x, base_mid_yc + ch * perp_y)
    base_right = _to_img(base_mid_xc - ch * perp_x, base_mid_yc - ch * perp_y)

    # Hole: canonical circle of radius hole_r_c at (hole_off_xc, hole_off_yc)
    hole_r_c = geom["hole_r_c"]
    hox = geom["hole_off_xc"]
    hoy = geom["hole_off_yc"]
    hole_pts = []
    for k in range(POINTER_HUB_SEGS):
        t = 2 * math.pi * k / POINTER_HUB_SEGS
        hole_pts.append(_to_img(hox + hole_r_c * math.cos(t),
                                hoy + hole_r_c * math.sin(t)))
    hole_centre = _to_img(hox, hoy)

    return {
        "hub_pts": hub_pts, "hub_centre": hub_centre,
        "apex": apex, "base_left": base_left, "base_right": base_right,
        "hole_pts": hole_pts, "hole_centre": hole_centre,
    }


def draw_pointer(img, geom, verts):
    """Render pointer body (hub + triangle), apply colour variation, edge
    feathering, mounting hole, and optional tip cutoff.  Returns (img, mask).
    """
    canvas_size = img.size[0]
    apex = verts["apex"]
    base_left = verts["base_left"]
    base_right = verts["base_right"]
    hub_pts = verts["hub_pts"]
    hole_pts = verts["hole_pts"]
    hub_cx, hub_cy = verts["hub_centre"]

    # Paint the mounting hole FIRST.  Because HOLE_RATIO > 1, the hole is
    # slightly oversized vs the hub: the pointer body painted next covers
    # the interior, leaving only a thin darker ring around the hub rim
    # (matches v2's recessed-collar look).
    ImageDraw.Draw(img).polygon(hole_pts, fill=geom["hole_rgb"])

    # Snapshot the background AFTER the hole so edge feathering and tip
    # cutoff blend toward whatever is actually behind the pointer (dial
    # face away from the hub, hole ring at the hub edge).
    bg_arr = np.array(img)

    # Build the pointer mask (triangle + hub).
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon([apex, base_left, base_right], fill=255)
    mdraw.polygon(hub_pts, fill=255)

    # Fill pointer region with base colour.
    draw = ImageDraw.Draw(img)
    ptr_rgb = geom["ptr_rgb"]
    draw.polygon([apex, base_left, base_right], fill=ptr_rgb, outline=ptr_rgb)
    draw.polygon(hub_pts, fill=ptr_rgb, outline=ptr_rgb)

    # Lengthwise gradient + per-pixel noise.
    img = _pointer_color_variation(img, mask, hub_cx, hub_cy, apex[0], apex[1])

    # Edge feathering blends boundary pixels with the snapshot background.
    img = _draw_pointer_edge(img, mask, geom["edge_rgb"], bg_arr)

    # Optional tip erasure: restore the snapshotted background within a
    # disc around the apex so the pointer "ends" before the actual tip.
    if CUT_TIP:
        r = geom["cut_tip_radius"]
        mask_arr = np.array(mask)
        yy, xx = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1]]
        tip_circle = (xx - apex[0]) ** 2 + (yy - apex[1]) ** 2 <= r ** 2
        cut_pixels = tip_circle & (mask_arr > 0)
        if cut_pixels.any():
            arr = np.array(img)
            arr[cut_pixels] = bg_arr[cut_pixels]
            img = Image.fromarray(arr, img.mode)
            mask_arr[cut_pixels] = 0
            mask = Image.fromarray(mask_arr, "L")

    return img, mask


def draw_pointer_shadow(layer_draw, verts, offset_dx, offset_dy, opacity,
                        base_rgb=(40, 40, 40)):
    """Render the pointer's cast shadow as offset polygons.  Drawn into a
    separate RGBA layer; caller is responsible for blurring + compositing.
    """
    shadow_color = (*base_rgb, opacity)

    apex = verts["apex"]
    base_left = verts["base_left"]
    base_right = verts["base_right"]
    hub_pts = verts["hub_pts"]

    s_apex = (apex[0] + offset_dx, apex[1] + offset_dy)
    s_bl = (base_left[0] + offset_dx, base_left[1] + offset_dy)
    s_br = (base_right[0] + offset_dx, base_right[1] + offset_dy)
    s_hub = [(x + offset_dx, y + offset_dy) for x, y in hub_pts]

    layer_draw.polygon([s_apex, s_bl, s_br], fill=shadow_color)
    layer_draw.polygon(s_hub, fill=shadow_color)


# ── Rim crescent shadow (recessed-bowl effect) ───────────────────────────

def _sample_shadow_rgb():
    h = random.random()
    s = random.uniform(*SHADOW_S_RANGE)
    v = random.uniform(*SHADOW_V_RANGE)
    return _hsv_rgb(h, s, v)


def draw_rim_shadows(params, canvas_size, base_rgb):
    """Build an RGBA layer with a crescent shadow inside the dial ellipse.

    Physical model: the dial is a recessed bowl.  One side is lit by an
    external light; the opposite rim arc blocks light and casts a crescent
    onto the dial face.  In image space the crescent is the dial-ellipse
    region MINUS a smaller, shifted ellipse with the same orientation —
    the cutout "bites" from the lit side.

    Returns (layer, light_angle_deg).  Caller alpha-composites the layer.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]

    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    u = (xs - cx) * cos_o + (ys - cy) * sin_o
    v = -(xs - cx) * sin_o + (ys - cy) * cos_o
    dial_mask = (u / max(a, 1)) ** 2 + (v / max(b, 1)) ** 2 <= 1.0

    light_angle = random.uniform(0, 360)
    light_rad = math.radians(light_angle - 90)

    offset_frac = random.uniform(*RIM_SHADOW_OFFSET_FRAC)
    cutout_frac = random.uniform(*RIM_SHADOW_CUTOUT_FRAC)
    # Cutout offset in image space, biased toward the light direction.  The
    # cutout shape itself is an ellipse with the same θ as the dial, scaled
    # by `cutout_frac` along the dial's own axes.
    co_dx = offset_frac * a * math.cos(light_rad)
    co_dy = offset_frac * b * math.sin(light_rad)
    co_cx = cx + co_dx
    co_cy = cy + co_dy
    co_a = cutout_frac * a
    co_b = cutout_frac * b

    u2 = (xs - co_cx) * cos_o + (ys - co_cy) * sin_o
    v2 = -(xs - co_cx) * sin_o + (ys - co_cy) * cos_o
    cutout_mask = (u2 / max(co_a, 1)) ** 2 + (v2 / max(co_b, 1)) ** 2 <= 1.0

    crescent = dial_mask & (~cutout_mask)

    opacity = random.randint(*RIM_SHADOW_PRIMARY_OPACITY)
    rgba = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    rgba[..., 0] = base_rgb[0]
    rgba[..., 1] = base_rgb[1]
    rgba[..., 2] = base_rgb[2]
    alpha = np.zeros((canvas_size, canvas_size), dtype=np.uint8)
    alpha[crescent] = opacity
    rgba[..., 3] = alpha

    layer = Image.fromarray(rgba, "RGBA")
    blur_r = random.uniform(*RIM_SHADOW_BLUR_RANGE)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return layer, light_angle


# ── Hierarchical noise inside bezel ──────────────────────────────────────

def apply_hierarchical_noise(img, inside_outer):
    """Apply block-level + per-pixel noise to pixels inside the bezel.

    Mirrors v2's HIER_NOISE_* pass: a coarse N×N noise grid is upsampled
    via bicubic interpolation, then a fine per-pixel noise field is added,
    and the sum is applied only to the `inside_outer` region.
    """
    canvas_size = img.size[0]
    block_vals = np.random.randint(
        -HIER_NOISE_BLOCK_VARIANCE, HIER_NOISE_BLOCK_VARIANCE + 1,
        (HIER_NOISE_BLOCKS, HIER_NOISE_BLOCKS), dtype=np.int16)
    block_u8 = np.clip(block_vals + 128, 0, 255).astype(np.uint8)
    block_up = np.array(
        Image.fromarray(block_u8, "L").resize(
            (canvas_size, canvas_size), Image.BICUBIC),
        dtype=np.int16) - 128

    pixel_noise = np.random.randint(
        -HIER_NOISE_PIXEL_VARIANCE, HIER_NOISE_PIXEL_VARIANCE + 1,
        (canvas_size, canvas_size), dtype=np.int16)
    combined = block_up + pixel_noise

    arr = np.array(img)
    for c in range(3):
        ch = arr[:, :, c].astype(np.int16)
        ch[inside_outer] += combined[inside_outer]
        arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, img.mode)


# ── Whole-image Gaussian blur ────────────────────────────────────────────

def apply_global_blur(img):
    """Apply a Gaussian blur to the entire image with probability
    DIAL_BLUR_PROB.  Simulates camera defocus across the whole scene —
    target dial, bezel, intruder, distractor text, and background all
    softened together, so the model can't tell foreground from background
    by sharpness alone.
    """
    if random.random() >= DIAL_BLUR_PROB:
        return img
    blur_r = random.uniform(*DIAL_BLUR_RADIUS_RANGE)
    return img.filter(ImageFilter.GaussianBlur(radius=blur_r))


# ── Adversarial triangle artifacts ───────────────────────────────────────

def add_triangle_artifacts(img, protect_mask=None):
    """Opaque random-colour triangles scattered on the canvas.

    If `protect_mask` is provided (L-mode, 255 = protected), only the
    triangle pixels falling OUTSIDE the protected region are painted.
    Triangles whose paintable area shrinks to zero after masking are
    skipped silently — keeps the model from learning to ignore obscured
    ticks/digits/pointer pixels.
    """
    arr = np.array(img)
    canvas_size = arr.shape[0]
    protected = (np.array(protect_mask) > 127) if protect_mask is not None else None
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
        if protected is not None:
            m = m & (~protected)
        if not m.any():
            continue
        for c in range(3):
            noise = np.random.randint(-TRI_PIXEL_VAR, TRI_PIXEL_VAR + 1,
                                      size=m.sum(), dtype=np.int16)
            arr[:, :, c][m] = np.clip(base[c] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, img.mode)


# ── Intrusion augmentation ───────────────────────────────────────────────

INTRUSION_MAX_ATTEMPTS = 20            # rejection-sampling cap


def _sample_intruder_params(img_size=IMG_SIZE, target_params=None):
    """Sample parameters for an off-canvas intruder dial.

    Returns (params_dict, dial_h, dial_s, outer_scale) where params_dict
    has the same keys as sample_params() but with (cx, cy) placed outside
    the canvas so that only a depth-fraction of the bezel is visible.

    When `target_params` is supplied, the intruder is rejection-sampled
    until its bezel bounding circle does NOT intersect the target's bezel
    bounding circle.  Two circles (c1, r1) and (c2, r2) are disjoint when
    `dist(c1, c2) >= r1 + r2`.  Using `a * outer_scale` as the radius is a
    safe upper bound on the rotated ellipse's extent from its center.

    Returns None if no non-overlapping placement is found within
    `INTRUSION_MAX_ATTEMPTS` tries — caller should skip the intrusion.
    """
    target_circle = None
    if target_params is not None:
        target_circle = (
            target_params["cx"],
            target_params["cy"],
            target_params["a"] * target_params["outer_scale"],
        )

    for _attempt in range(INTRUSION_MAX_ATTEMPTS):
        a = random.uniform(*A_RANGE)
        b = a * random.uniform(*B_RATIO_RANGE)
        theta = random.uniform(*THETA_RANGE)
        phi_0 = random.uniform(*PHI_0_RANGE)
        pointer_value = random.uniform(*POINTER_VALUE_RANGE)
        outer_scale = random.uniform(*OUTER_SCALE_RANGE)

        # Conservative bezel radius for placement; use a (un-rotated upper bound).
        bezel_radius = a * outer_scale

        depth = random.uniform(*INTRUSION_DEPTH_RANGE)
        offset = bezel_radius * (1 - depth)

        edge = random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            cx = random.uniform(0, img_size)
            cy = -offset
        elif edge == 'bottom':
            cx = random.uniform(0, img_size)
            cy = img_size + offset
        elif edge == 'left':
            cx = -offset
            cy = random.uniform(0, img_size)
        else:  # right
            cx = img_size + offset
            cy = random.uniform(0, img_size)

        if target_circle is not None:
            tcx, tcy, tr = target_circle
            if math.hypot(cx - tcx, cy - tcy) < tr + bezel_radius:
                continue  # bezels overlap; try again
        break
    else:
        # All attempts overlapped — no valid placement.
        return None

    params = {
        "cx": cx, "cy": cy, "a": a, "b": b,
        "theta": theta, "phi_0": phi_0,
        "pointer_value": pointer_value,
        "outer_scale": outer_scale,
    }
    dial_h = random.random()
    dial_s = random.uniform(*DIAL_S_RANGE)
    return params, dial_h, dial_s, outer_scale


def _render_intruder(img, intruder_params, dial_h, dial_s, outer_scale,
                     outpaint=True):
    """Render a complete intruder dial onto img (in-place via successive calls).

    Mirrors the target rendering sequence but omits distractor text (canvas-
    global) and the expensive shadow / hierarchical-noise / dial-blur passes
    that the target gets.  Intruders are cheap edge distractors.

    `outpaint` is forwarded to `draw_outer_canvas` so an intruder over a
    Places365 photo background keeps the same crisp bezel-vs-photo edge
    as the target.
    """
    img, _ = draw_outer_canvas(img, intruder_params, outer_scale,
                               dial_h, dial_s, outpaint=outpaint)
    img = draw_dial_boundary(img, intruder_params, dial_h, dial_s)
    img, intr_tick_inner_rc = draw_scale_ticks(img, intruder_params)
    img = draw_scale_digits(img, intruder_params, intr_tick_inner_rc)
    intr_geom = _sample_pointer_geometry(intruder_params)
    intr_verts = _pointer_image_vertices(intr_geom, intruder_params)
    img, _ = draw_pointer(img, intr_geom, intr_verts)
    return img


# ── Top-level image generation ───────────────────────────────────────────

def generate_image(path, img_size=IMG_SIZE, photo_paths=None,
                   bg_prefetcher=None):
    """Generate one v4 synthetic dial image.  Returns dict of label fields.

    Renders ONLY scale ticks and digits on top of a Places365 photo
    background.  No bezel, no dial face, no pointer, no distractor
    text, no intruder dial, no rim shadows.  Per-image tick/digit ink
    color is picked from the mean luminance of the photo patch under
    the implicit dial ellipse (see pick_ink_color).

    `bg_prefetcher` is the same `_BGPrefetcher` interface as v3 — when
    provided, photo decoding has already happened on a worker thread.
    """
    if BG_SOURCE != "photo":
        raise ValueError(
            f"v4 requires BG_SOURCE='photo'; got {BG_SOURCE!r}.  v4 has "
            f"no procedural-background code path.")

    params = sample_params(img_size=img_size)

    # 1. Photo background.
    if bg_prefetcher is not None:
        img = _apply_photo_augmentation(bg_prefetcher.get())
    else:
        if photo_paths is None:
            photo_paths = _load_bg_photo_paths(BG_PHOTO_DIR)
        img = make_photo_background(img_size, photo_paths)

    # 2. Per-image ink color from the photo patch under the dial area.
    #    Computed AFTER photo augmentation so the colour matches what the
    #    model actually sees.
    ink_color = pick_ink_color(img, params["cx"], params["cy"], params["a"])

    # 3. Scale ticks + digits.  Each tick line and glyph is mirrored
    #    into `protect_mask` so the triangle-artifact pass can avoid them.
    protect_mask = Image.new("L", (img_size, img_size), 0)
    protect_draw = ImageDraw.Draw(protect_mask)
    img, tick_inner_rc = draw_scale_ticks(img, params, ink_color,
                                          protect_draw=protect_draw)
    img = draw_scale_digits(img, params, tick_inner_rc, ink_color,
                            protect_draw=protect_draw)

    # 4. Whole-image Gaussian blur (camera-defocus simulation).
    img = apply_global_blur(img)

    # 5. Adversarial triangle artifacts — skip pixels in protect_mask.
    img = add_triangle_artifacts(img, protect_mask=protect_mask)

    # 6. Colour / exposure jitter.
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
    img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # 7. Save.
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


_DIAL_FNAME_RE = re.compile(r"^dial_(\d+)\.png$")


def _scan_max_dial_index(output_dir):
    """Return the highest N in any existing `dial_NNNNN.png` under
    *output_dir*, or -1 if none.  Used by additive mode to start the
    next image's filename above any existing one (so we never silently
    overwrite a previously generated frame)."""
    if not os.path.isdir(output_dir):
        return -1
    best = -1
    for name in os.listdir(output_dir):
        m = _DIAL_FNAME_RE.match(name)
        if m:
            idx = int(m.group(1))
            if idx > best:
                best = idx
    return best


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic autocal images")
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help=f"directory to write PNGs and labels.csv (default: {OUTPUT_DIR})")
    parser.add_argument(
        "--num-images", type=int, default=NUM_IMAGES,
        help=f"number of synthetic images to generate (default: {NUM_IMAGES})")
    parser.add_argument(
        "--additive", action="store_true",
        help="Append to an existing dataset instead of overwriting. New "
             "images are numbered above the highest existing "
             "dial_NNNNN.png in --output-dir; new rows are appended to "
             "labels.csv (header is reused, schema is verified to match).")
    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error("--num-images must be a positive integer")

    # Background source — driven by the module-level BG_SOURCE constant.
    if BG_SOURCE not in ("procedural", "photo"):
        raise ValueError(
            f"BG_SOURCE must be 'procedural' or 'photo', got {BG_SOURCE!r}")
    photo_paths = None
    if BG_SOURCE == "photo":
        photo_paths = _load_bg_photo_paths(BG_PHOTO_DIR)
        n_photos = len(photo_paths)
        n_npy = sum(1 for p in photo_paths if p.lower().endswith(".npy"))
        n_jpg = n_photos - n_npy
        resolved = os.path.abspath(os.path.expanduser(BG_PHOTO_DIR))
        print(f"Photo BG: {n_photos} images found under '{resolved}' "
              f"({n_npy} pre-decoded .npy, {n_jpg} raw .jpg)")
        if n_photos == 0:
            raise SystemExit(
                f"No *.jpg / *.npy files under '{resolved}'.\n"
                f"Run `python download_places365.py --root {BG_PHOTO_DIR}` "
                f"first, or populate that folder with background images.\n"
                f"For faster loading, also run "
                f"`python prep_bg_cache.py --input-dir {BG_PHOTO_DIR} "
                f"--output-dir <cache_dir>` and point BG_PHOTO_DIR there.")
        if n_jpg > 0 and n_npy == 0:
            print(f"  tip: run `python prep_bg_cache.py --input-dir "
                  f"{BG_PHOTO_DIR} --output-dir {BG_PHOTO_DIR}_npy` and "
                  f"point BG_PHOTO_DIR at the .npy cache for ~10× faster "
                  f"loading.")
        # Reuse-ratio gate.  Augmentation (flip + jitter) absorbs
        # moderate reuse, but we refuse to silently over-recycle.
        max_dials = n_photos * BG_PHOTO_MAX_REUSE_RATIO
        if args.num_images > max_dials:
            raise SystemExit(
                f"Not enough background photos for {args.num_images} dials.\n"
                f"  Available: {n_photos} unique photos in '{resolved}'\n"
                f"  Max reuse ratio: {BG_PHOTO_MAX_REUSE_RATIO}× per photo\n"
                f"  → can generate up to {max_dials} dials.\n"
                f"Options:\n"
                f"  - raise BG_PHOTO_MAX_REUSE_RATIO "
                f"(cushioned by flip+jitter)\n"
                f"  - or add more photos to BG_PHOTO_DIR "
                f"(e.g. `python download_places365.py "
                f"--split train-standard`).")
        reuse_ratio = args.num_images / n_photos
        if reuse_ratio > 1.0:
            print(f"  reuse ratio: {reuse_ratio:.2f}× per photo "
                  f"(cap {BG_PHOTO_MAX_REUSE_RATIO}×); flip + color "
                  f"jitter applied per use")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "labels.csv")

    fieldnames = ["filename", "cx", "cy", "a", "b", "theta", "phi_0"]
    for i in range(N_MARKS):
        fieldnames += [f"mark_{i}_x", f"mark_{i}_y"]

    # Decide write mode + starting filename index.
    csv_exists = os.path.isfile(csv_path)
    if args.additive and csv_exists:
        # Verify the existing CSV's header matches what we're about to
        # write.  Mismatch usually means the user changed N_MARKS or
        # the label schema between runs — refuse to silently mix.
        with open(csv_path, "r", newline="") as f:
            existing_header = next(csv.reader(f), None)
        if existing_header != fieldnames:
            parser.error(
                f"Existing labels.csv header doesn't match current schema.\n"
                f"  existing: {existing_header}\n"
                f"  current:  {fieldnames}\n"
                f"Refusing to append a mismatched row layout.")
        file_mode = "a"
        write_header = False
    else:
        # Either non-additive (overwrite) or additive on a fresh dir
        # (no CSV yet) — write the header.
        file_mode = "w"
        write_header = True

    if args.additive:
        start_idx = _scan_max_dial_index(args.output_dir) + 1
    else:
        start_idx = 0

    # Optional BG prefetcher — only meaningful when actually using
    # photos and explicitly enabled.  Created here (after the
    # reuse-ratio gate) so the worker thread only runs when generation
    # will actually happen.
    bg_prefetcher = None
    if BG_SOURCE == "photo" and BG_PREFETCH and photo_paths:
        bg_prefetcher = _BGPrefetcher(
            photo_paths, IMG_SIZE, queue_size=BG_PREFETCH_QUEUE)
        print(f"  BG prefetcher: 1 worker, queue size {BG_PREFETCH_QUEUE}")

    try:
        with open(csv_path, file_mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            for i in range(args.num_images):
                idx = start_idx + i
                fname = f"dial_{idx:05d}.png"
                fpath = os.path.join(args.output_dir, fname)
                writer.writerow(generate_image(
                    fpath, photo_paths=photo_paths,
                    bg_prefetcher=bg_prefetcher))
    finally:
        if bg_prefetcher is not None:
            bg_prefetcher.stop()

    end_idx = start_idx + args.num_images - 1
    if args.additive:
        print(f"Appended {args.num_images} images "
              f"(indices {start_idx:05d}..{end_idx:05d}) to '{args.output_dir}/'")
    else:
        print(f"Generated {args.num_images} images in '{args.output_dir}/'")
    print(f"Labels {'appended to' if file_mode == 'a' else 'written to'} '{csv_path}'")


if __name__ == "__main__":
    main()
