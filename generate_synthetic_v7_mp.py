"""
v7: native-grayscale, YAML-driven synthetic analog-dial generator.

Every image is drawn directly on single-channel PIL "L" images: each
surface colour is one int gray level (0-255) drawn from a `*_gray_range`
in the config, and composited layers are an "L" value image plus an "L"
alpha mask (see `_composite_gray`).  No colour image exists anywhere in
the pipeline.  Output: 256x256 uint8 single-channel PNG + tip label.

Config: `cfg` is the nested dict from `v7_config` (section -> key ->
value), defaults in configs/v7_default.yaml, bounds in v7_schema.yaml.

Randomness: one independent stream per section (`section_rngs`), seeded
from (image_seed, crc32(section name)) through np.random.SeedSequence.
Each stream is a pair (random.Random, np.random.Generator).  A parameter
only ever draws from the stream of the section that owns it, so changing
e.g. a tick parameter cannot change the angle, the pointer, the noise or
the triangles of the same image.  Values one section needs from another
(dial radius, arc gap, pointer mask, show-digits flag ...) are passed as
data, never re-drawn.  Within a stream, the draws other sections consume
come first, so variable-count draws later in that stream cannot shift
them.  The global `random` / `np.random` state is never touched.

Render pipeline (order and semantics unchanged from the old luma-port v7):

    angle ─┐   geometry (arc gap, scale, hub, chord, tip, displacements,
           │             hole, stretch) ── dial_radius, canvas_size ...
           v
  ┌──────────────────────────────────────────────────────────────────┐
  │ 1  background: low-freq tiles (bicubic) + pixel noise   [canvas] │
  │ 2  faint dial circle with arc gap                                │
  │ 3  ticks -> separate overlay (value + alpha)          [td layer] │
  │ 4  rim crescent shadow mask m_r    ─┐ only if has_shadow;        │
  │ 5  pointer + circle shadow mask m_p ┘ masks only, img untouched; │
  │    d = max(s_p*m_p, s_r*m_r)  float32 darkening  [d layer]       │
  │ 6  digits -> overlay                                  [td layer] │
  │ 7  mounting hole, pointer body (gradient + noise), pointer edge, │
  │    cut-tip (restore pre-pointer pixels near the apex)            │
  │ 8  stretch h/v (canvas + overlay + d + apex)                     │
  │ 9  brightness -> contrast -> sharpness jitter                    │
  │ 10 outer annulus fill, rim band                                  │
  │ 11 hierarchical (block + pixel) noise inside outer ring          │
  │ 12 triangle artifacts (avoid pointer + hole)                     │
  │ 13 black outside outer ring; erase overlay over pointer;         │
  │    composite overlay; apply shadow out = round(in * (1 - d))     │
  │    (d zeroed outside the dial face, on the rim, pointer, hole)   │
  │ 13c cover glass, inside B: reflections out = in - sum(dk * m_k), │
  │    then glares out = lerp(out, V, m) one by one (glare wins);    │
  │    glare m zeroed within clear_distance of the visible tip       │
  │ 14 Gaussian blur inside outer ring; black outside again          │
  │ 15 place canvas on a black img_size^2 at centred offset + d;     │
  │    boundary B stays centred (everything inside B shifts by d)    │
  └──────────────────────────────────────────────────────────────────┘
           v
    RenderResult(image (256, 256) uint8, tip_x, tip_y, angle_deg)

The pointer: a hub circle + an isosceles triangle whose base is a chord
of the hub, apex pointing outward.  Label = the visible tip in pixel
coordinates: the apex, or, when cut_tip is on, the centre of the cut arc on
the pointer axis (apex - cut_r * unit pointer direction).

Shadows are physical: a shadow blocks the direct light and keeps only the
ambient light, so it scales the surface by an albedo-independent factor,
out = in * (1 - d).  One shared strength s per image, jittered separately
for the pointer (s_p) and rim (s_r) shadows; overlapping shadows of the
one light do not compound (max, not product).  Ticks, digits, noise and
triangles under a shadow are darkened by the same factor.

Reflections and glares sit on the cover glass.  Each item cuts a random
square patch (side ~ patch_size_range) from a random Places365 photo
(PLACES_DIR / $V7_PLACES_DIR, sorted list cached per folder), randomly
flipped / rotated, keeps the pixels at or above a random brightness
percentile as its shape m (0-1, edge softened by edge_soft), and lands at a
centre uniform in the disc of B:

    photo ──crop s x s──> patch ──lum >= p-th pct──> m ──blur──> placed m
    reflection: out = in - darken * m     (values, overlaps add)
    glare:      out = in*(1-m) + V*m      (after all reflections; not near tip)

The folder is only read when an image gets at least one item, so prob 0
(the default) needs no dataset.
"""

import argparse
import csv
import datetime
import glob
import math
import multiprocessing as mp
import os
import random
import sys
import zlib
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

import v7_config

OUTPUT_DIR = "sv7"  # default output folder (str) for PNGs + labels.csv
NUM_IMAGES = 30000  # default number of images (int) for the CLI

# Scale: 0-9 clockwise, 36 deg per division, 0 at top (fixed dial layout)
NUM_DIGITS = 10  # digits on the dial face (int)
DEG_PER_DIGIT = 360.0 / NUM_DIGITS  # angular pitch of the digits (float, deg)
SHADOW_ARC_LEVEL = 128  # mask level (int, of 255) of the dial-circle shadow arc: 0.5 x the pointer

# RNG stream names (tuple[str, ...]): every config section that draws random
# values, plus "angle" for the pointer angle (canvas/dataset draw nothing).
SECTIONS = ("angle", "geometry", "outer", "ticks", "digits", "palette",
            "pointer", "background", "cut_tip", "shadows", "jitter",
            "hier_noise", "tri_artifacts", "reflections", "glares")

# Places365 photo folder (str, may hold "~") that reflection / glare shapes are
# cut from; scanned recursively.  V7_PLACES_DIR overrides it, read again at
# every lookup (see places_dir) so tests and spawned workers can repoint it.
PLACES_DIR = os.environ.get("V7_PLACES_DIR", "~/places365")
_PLACES_EXTS = (".jpg", ".jpeg", ".png")  # photo file suffixes (tuple[str], lower case)
# sorted photo paths per resolved folder: dict[str, list[str]]; only
# non-empty scans are cached, so a dataset downloaded later is picked up
_PLACES_CACHE = {}


def section_rngs(image_seed):
    """Per-section RNG streams for one image.

    Returns dict[str, tuple[random.Random, np.random.Generator]] keyed by
    SECTIONS.  Each pair is seeded from SeedSequence([image_seed,
    crc32(name)]), whose two spawned children seed the Python and NumPy
    generators, so every stream is independent of every other.
    """
    out = {}
    for name in SECTIONS:
        ss = np.random.SeedSequence([int(image_seed) & 0xFFFFFFFF,
                                     zlib.crc32(name.encode("ascii"))])
        py_ss, np_ss = ss.spawn(2)
        # 128-bit int seed for random.Random, from 4 uint32 words
        py_seed = int.from_bytes(py_ss.generate_state(4, np.uint32).tobytes(), "little")
        out[name] = (random.Random(py_seed), np.random.default_rng(np_ss))
    return out


@dataclass
class RenderResult:
    """One rendered dial and its label."""
    image: np.ndarray  # gray dial image, (img_size, img_size) uint8
    tip_x: float  # visible pointer tip x in image pixels (float; apex or cut-arc centre)
    tip_y: float  # visible pointer tip y in image pixels (float; apex or cut-arc centre)
    angle_deg: int  # pointer angle, whole degrees (0 = up, clockwise)


# ── gray compositing ───────────────────────────────────────────────────

def _composite_gray(base, value, alpha):
    """Alpha-composite a gray layer onto an opaque gray image.

    base  : PIL "L" image (H, W), the opaque background
    value : PIL "L" image (H, W) or int 0-255, the layer's gray level
    alpha : PIL "L" image (H, W), the layer's opacity 0-255
    Returns a PIL "L" image, round(base*(255-a)/255 + value*a/255).
    """
    b = np.asarray(base, dtype=np.int32)  # (H, W) int32
    a = np.asarray(alpha, dtype=np.int32)  # (H, W) int32
    v = np.asarray(value, dtype=np.int32) if isinstance(value, Image.Image) else int(value)
    out = (b * (255 - a) + v * a + 127) // 255  # (H, W) int32, rounded
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


def _new_layer(size):
    """Empty gray layer: (value, alpha) pair of black "L" images (size x size)."""
    return Image.new("L", (size, size), 0), Image.new("L", (size, size), 0)


# ── dial face ───────────────────────────────────────────────────────────

def _arc_or_circle(draw, bbox, gap_start_deg, gap_span_deg, fill, width):
    """Draw a circle outline on `draw`, leaving an arc gap if span > 0.

    gap_start_deg : where the gap begins (PIL arc degrees: 0 = 3-o'clock, CW)
    gap_span_deg  : angular width of the gap in degrees
    """
    if gap_span_deg <= 0:
        draw.ellipse(bbox, outline=fill, width=width)
    else:
        arc_start = (gap_start_deg + gap_span_deg) % 360
        arc_end = gap_start_deg % 360
        draw.arc(bbox, start=arc_start, end=arc_end, fill=fill, width=width)


def draw_ticks(td_value, td_alpha, cx, cy, dial_radius, tcfg, rng):
    """Draw the 0-9 scale ticks (+ subdivisions) on the overlay layer.

    td_value / td_alpha : overlay "L" images (canvas, canvas)
    tcfg                : cfg["ticks"]; rng: the "ticks" random.Random
    Returns the inner radius (float, px) of the longest major tick.
    """
    vdraw = ImageDraw.Draw(td_value)  # draws tick gray into the overlay value
    adraw = ImageDraw.Draw(td_alpha)  # draws tick opacity into the overlay alpha
    tick_outer = dial_radius  # ticks run inward from the dial circle (px)

    # Shared jitter for all ticks (fixed-count draws first)
    angle_jitter = rng.uniform(*tcfg["angle_jitter_range"])  # float, deg
    grey = rng.randint(*tcfg["gray_range"])  # int gray level of all ticks
    ticks_per_digit = rng.randint(*tcfg["per_digit_range"])  # int

    sub_deg = DEG_PER_DIGIT / ticks_per_digit  # float, deg between ticks
    max_major_len = 0  # longest major tick (int, px) for digit placement

    for i in range(NUM_DIGITS):
        for s in range(ticks_per_digit):
            angle_rad = math.radians(i * DEG_PER_DIGIT + s * sub_deg
                                     + angle_jitter - 90)  # -90 so 0 = up
            if s == 0:  # major tick
                tick_len = rng.randint(*tcfg["len_range"])
                tick_w = rng.randint(*tcfg["width_range"])
                max_major_len = max(max_major_len, tick_len)
            else:  # minor tick
                tick_len = max(1, rng.randint(*tcfg["len_range"]) // 2)
                tick_w = max(1, rng.randint(*tcfg["width_range"]) - 1)
            tick_inner = tick_outer - tick_len
            seg = [(cx + tick_inner * math.cos(angle_rad), cy + tick_inner * math.sin(angle_rad)),
                   (cx + tick_outer * math.cos(angle_rad), cy + tick_outer * math.sin(angle_rad))]
            vdraw.line(seg, fill=grey, width=tick_w)
            adraw.line(seg, fill=255, width=tick_w)

    return tick_outer - max_major_len


def draw_scale_digits(td_value, td_alpha, cx, cy, dial_radius, tick_inner_radius,
                      dcfg, rng, font_size):
    """Draw digits 0-9 around the dial on the overlay layer (0 at top, CW).

    Digits sit inside or outside the ticks with a clearance gap.  The radial
    jitter is one-directional: inside digits move only inward (toward the
    centre, inside_jitter_range), outside digits only outward (away from the
    centre, outside_jitter_range).  dcfg = cfg["digits"]; rng = "digits" stream.
    """
    try:
        font = ImageFont.truetype("arial.ttf", font_size)  # FreeTypeFont, digit glyphs
    except OSError:
        # no Arial: Pillow's bundled font, still at font_size px (Pillow >= 10.1)
        font = ImageFont.load_default(size=font_size)

    ang_jitter = rng.uniform(*dcfg["angle_jitter_range"]) * rng.choice([-1, 1])  # float, deg
    grey = rng.randint(*dcfg["gray_range"])  # int gray level of all digits
    gap = rng.uniform(*dcfg["gap_range"])  # float, px clearance to ticks
    inside = rng.random() < dcfg["inside_prob"]  # bool
    # exactly one uniform draw in either branch, so the stream's draw count
    # does not depend on `inside`
    if inside:
        base_dist = tick_inner_radius - font_size / 2 - gap  # float, px
        dist = base_dist - rng.uniform(*dcfg["inside_jitter_range"])  # float px, toward centre
    else:
        base_dist = dial_radius + font_size / 2 + gap  # float, px
        dist = base_dist + rng.uniform(*dcfg["outside_jitter_range"])  # float px, away from centre

    # anti-aliased digit coverage 0-255, "L" (canvas, canvas).  Text goes to
    # coverage only: the value stays a flat `grey` (straight, not
    # premultiplied), as a text draw on a transparent colour layer stored it,
    # so edges are not attenuated twice when composited.
    cover = Image.new("L", td_alpha.size, 0)
    cdraw = ImageDraw.Draw(cover)  # draws into the coverage mask
    for i in range(NUM_DIGITS):
        angle_rad = math.radians(i * DEG_PER_DIGIT + ang_jitter - 90)
        text = str(i)
        bbox = font.getbbox(text)
        tx = cx + dist * math.cos(angle_rad) - (bbox[2] - bbox[0]) / 2
        ty = cy + dist * math.sin(angle_rad) - (bbox[3] - bbox[1]) / 2
        cdraw.text((tx, ty), text, fill=255, font=font)

    m = np.asarray(cover, dtype=np.int32)  # (C, C) int32 coverage
    a = np.asarray(td_alpha, dtype=np.int32)  # (C, C) int32 existing overlay alpha
    td_value.paste(grey, mask=Image.fromarray(((m > 0) * 255).astype(np.uint8)))
    # alpha "over": m + a * (255 - m) / 255, (C, C) uint8, written in place
    td_alpha.paste(Image.fromarray((m + (a * (255 - m) + 127) // 255).astype(np.uint8)))


# ── shadows ─────────────────────────────────────────────────────────────

def draw_shadow(mask, apex, base_left, base_right,
                cx, cy, hub_radius, dial_radius,
                offset_dx, offset_dy, gap_start_deg, gap_span_deg):
    """Draw the offset pointer shadow and the circle shadow shape into a mask.

    mask : "L" image (C, C), 0 = no shadow.  The triangle + hub are drawn at
    255 (shape 1.0), the offset dial-circle arc at SHADOW_ARC_LEVEL (~0.5).
    The arc goes first, so the pointer shape wins where they overlap.  The
    mask is blurred later, so the tip naturally loses its sharpness.
    """
    mdraw = ImageDraw.Draw(mask)  # draws the shadow shape into the mask
    tri = [(p[0] + offset_dx, p[1] + offset_dy) for p in (apex, base_left, base_right)]
    scx, scy = cx + offset_dx, cy + offset_dy  # float px, shadow hub centre
    # offset outer circle (with matching arc gap), half as dark as the pointer
    bbox = [scx - dial_radius, scy - dial_radius, scx + dial_radius, scy + dial_radius]
    _arc_or_circle(mdraw, bbox, gap_start_deg, gap_span_deg, SHADOW_ARC_LEVEL, 2)
    hub_box = [scx - hub_radius, scy - hub_radius, scx + hub_radius, scy + hub_radius]
    mdraw.polygon(tri, fill=255)
    mdraw.ellipse(hub_box, fill=255)


def draw_rim_shadows(cx, cy, dial_radius, canvas_size, light_angle,
                     offset_frac, cutout_frac, blur_r):
    """Blurred shape mask of the crescent shadow cast by the bowl rim.

    The crescent is the part of the dial disc outside a cutout circle that
    is shifted toward the light.  Returns float32 (C, C) in 0-1 (crescent
    = 1.0 before the blur).
    """
    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    dial_mask = (xs - cx) ** 2 + (ys - cy) ** 2 <= dial_radius ** 2  # (C, C) bool
    light_rad = math.radians(light_angle - 90)  # 0 = up, CW -> trig
    co_cx = cx + offset_frac * dial_radius * math.cos(light_rad)
    co_cy = cy + offset_frac * dial_radius * math.sin(light_rad)
    cutout = (xs - co_cx) ** 2 + (ys - co_cy) ** 2 <= (cutout_frac * dial_radius) ** 2
    shape = np.zeros((canvas_size, canvas_size), dtype=np.uint8)  # (C, C) uint8, 0/255
    shape[dial_mask & ~cutout] = 255
    blurred = Image.fromarray(shape).filter(ImageFilter.GaussianBlur(radius=blur_r))
    return np.asarray(blurred, dtype=np.float32) / 255.0  # (C, C) float32, 0-1


def shadow_darkening(m_p, m_r, s_p, s_r):
    """Per-pixel darkening fraction of the pointer + rim shadows.

    m_p, m_r : (H, W) float shape masks in 0-1 (pointer / rim shadow)
    s_p, s_r : float strengths in 0-1 of the two shadows
    Both shadows block the same light, so they do not compound: the result
    is max(s_p*m_p, s_r*m_r), not 1 - (1 - a)(1 - b).
    Returns (H, W) float32 d in 0-1; the shaded pixel is in * (1 - d).
    """
    return np.maximum(np.float32(s_p) * np.asarray(m_p, dtype=np.float32),
                      np.float32(s_r) * np.asarray(m_r, dtype=np.float32))


def _shadow_strengths(S, srng):
    """Draw the shared shadow strength and its two per-shadow jitters.

    S: shadows config section (dict); srng: the shadows random.Random,
    consumed for exactly 3 draws (s, jitter_p, jitter_r).
    Returns (s_p, s_r): pointer / rim strengths, floats clipped to 0-1.
    """
    s = srng.uniform(*S["strength_range"])  # float, shared strength
    j = S["strength_jitter"]  # float, jitter amplitude (+/-)
    jp = srng.uniform(-j, j)  # float, pointer-shadow jitter
    jr = srng.uniform(-j, j)  # float, rim-shadow jitter
    return min(max(s + jp, 0.0), 1.0), min(max(s + jr, 0.0), 1.0)


# ── pointer ─────────────────────────────────────────────────────────────

def _pointer_variation(img, mask, cx, cy, angle_deg, tip_dist, grad_strength,
                       noise_amp, nrng):
    """Lengthwise brightness gradient + per-pixel noise on the pointer body.

    img: "L" canvas; mask: "L" pointer mask; nrng: "pointer" np Generator.
    """
    arr = np.array(img).astype(np.int16)  # (C, C) int16
    ys, xs = np.where(np.array(mask) > 0)
    if len(ys) == 0:
        return img
    a = math.radians(angle_deg - 90)
    along = ((xs - cx) * math.cos(a) + (ys - cy) * math.sin(a)).astype(np.float32)
    along_norm = along / max(tip_dist, 1)  # (N,) float32, -1 hub .. +1 tip
    grad_shift = (along_norm * grad_strength).astype(np.int16)  # (N,) int16
    noise = nrng.integers(-noise_amp, noise_amp + 1, size=len(ys), dtype=np.int16)
    arr[ys, xs] = np.clip(arr[ys, xs] + grad_shift + noise, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _draw_pointer_edge(img, mask, edge_gray, bg_arr, thickness, noise_amp, nrng):
    """Variable-thickness edge band on the pointer mask boundary.

    The outermost erosion ring blends toward the pre-pointer background
    (bg_arr, (C, C) uint8), inner rings toward edge_gray; each pixel gets
    independent noise from nrng.
    """
    m = np.array(mask)  # (C, C) uint8
    arr = np.array(img)  # (C, C) uint8
    # dist[y, x] = erosion steps survived (0 = outermost ring), (C, C) float32
    dist = np.full(m.shape, thickness, dtype=np.float32)
    current = m.copy()
    for step in range(thickness):
        e1 = np.minimum(np.roll(current, 1, 0), np.roll(current, -1, 0))
        e2 = np.minimum(np.roll(current, 1, 1), np.roll(current, -1, 1))
        eroded = np.minimum(e1, e2) & current
        dist[(current > 0) & (eroded == 0)] = step
        current = eroded

    ys, xs = np.where((m > 0) & (dist < thickness))
    if len(ys) == 0:
        return img
    blend = dist[ys, xs] / max(thickness - 1, 1)  # (N,) 0 = bg .. 1 = edge
    noise = nrng.integers(-noise_amp, noise_amp + 1, size=len(ys), dtype=np.int16)
    bg = bg_arr[ys, xs].astype(np.int16)  # (N,) int16
    blended = (bg + blend * (edge_gray - bg) + noise).astype(np.int16)
    arr[ys, xs] = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _pick_pointer_gray(rng, lo, hi, bg_gray, min_contrast):
    """Draw the pointer body gray (int) with at least `min_contrast` levels of
    separation from the face background gray.

    rng          : the "pointer" random.Random; exactly ONE randint call is made
    lo, hi       : int, inclusive pointer.gray_range
    bg_gray      : int, palette bg_gray_range draw (face background level)
    min_contrast : int >= 0, pointer.min_bg_contrast; 0 = off

    The allowed set {g in [lo, hi] : |g - bg_gray| >= min_contrast} is at most
    two contiguous intervals, [lo, bg - c] and [bg + c, hi].  An index drawn
    with randint(0, n_allowed - 1) is mapped onto them in ascending order:

        lo ......... bg-c | forbidden | bg+c ......... hi
        [ index 0 .. n_low-1 ]          [ n_low .. n-1 ]

    With c = 0 the set is all of [lo, hi] and randint(0, n-1) + lo equals
    randint(lo, hi) bit for bit (both are lo + randbelow(n)).  If the set is
    empty, one randint(lo, hi) draw is still consumed and the endpoint of
    [lo, hi] farthest from bg_gray is returned (the lower one on a tie).
    """
    low_hi = min(hi, bg_gray - min_contrast)  # int, top of the lower interval
    high_lo = max(lo, bg_gray + min_contrast)  # int, bottom of the upper interval
    if min_contrast <= 0:  # whole range allowed (c = 0 must not split at bg)
        low_hi, high_lo = hi, hi + 1
    n_low = max(0, low_hi - lo + 1)  # int, size of [lo, low_hi]
    n_high = max(0, hi - high_lo + 1)  # int, size of [high_lo, hi]
    if n_low + n_high == 0:
        rng.randint(lo, hi)  # keep the stream's draw count fixed
        return hi if hi - bg_gray > bg_gray - lo else lo
    idx = rng.randint(0, n_low + n_high - 1)  # int, index into the allowed set
    return lo + idx if idx < n_low else high_lo + (idx - n_low)


def draw_pointer(img, cx, cy, angle_deg, hub_radius, tip_dist, chord_half,
                 chord_dist, ptr_gray, edge_gray, grad_strength, pcfg, nrng,
                 edge_thickness, cut_tip_radius):
    """Draw the pointer on the "L" canvas.

    Returns (img, apex, base_left, base_right, mask) where mask is the "L"
    pointer mask with the apex disc erased when cut_tip_radius is not None.
    """
    a = math.radians(angle_deg - 90)
    apex = (cx + tip_dist * math.cos(a), cy + tip_dist * math.sin(a))
    base_mx = cx - chord_dist * math.cos(a)
    base_my = cy - chord_dist * math.sin(a)
    perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
    base_left = (base_mx + chord_half * perp[0], base_my + chord_half * perp[1])
    base_right = (base_mx - chord_half * perp[0], base_my - chord_half * perp[1])
    hub_box = [cx - hub_radius, cy - hub_radius, cx + hub_radius, cy + hub_radius]

    # pointer shape mask, "L" (C, C)
    mask = Image.new("L", img.size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon([apex, base_left, base_right], fill=255)
    mdraw.ellipse(hub_box, fill=255)

    bg_arr = np.array(img)  # pre-pointer canvas, (C, C) uint8
    draw = ImageDraw.Draw(img)
    draw.polygon([apex, base_left, base_right], fill=ptr_gray, outline=ptr_gray)
    draw.ellipse(hub_box, fill=ptr_gray, outline=ptr_gray)

    img = _pointer_variation(img, mask, cx, cy, angle_deg, tip_dist,
                             grad_strength, pcfg["noise_amplitude"], nrng)
    img = _draw_pointer_edge(img, mask, edge_gray, bg_arr, edge_thickness,
                             pcfg["edge_noise_amplitude"], nrng)

    if cut_tip_radius is not None:
        mask_arr = np.array(mask)
        ys_t, xs_t = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1]]
        mask_arr[(xs_t - apex[0]) ** 2 + (ys_t - apex[1]) ** 2 <= cut_tip_radius ** 2] = 0
        mask = Image.fromarray(mask_arr)
    return img, apex, base_left, base_right, mask


# ── main render ────────────────────────────────────────────────────────

# ── reflections + glares (cover glass, shapes from Places365 patches) ────

def places_dir():
    """Resolved Places365 folder (str): V7_PLACES_DIR if set, else PLACES_DIR."""
    return os.path.abspath(os.path.expanduser(os.environ.get("V7_PLACES_DIR", PLACES_DIR)))


def places_files(folder=None):
    """Sorted list[str] of every *.jpg / *.jpeg / *.png under folder (recursive).

    folder: str or None (-> places_dir()).  Cached per resolved folder.
    Raises RuntimeError naming the folder when it holds no photos.
    """
    folder = places_dir() if folder is None else os.path.abspath(os.path.expanduser(folder))
    files = _PLACES_CACHE.get(folder)  # list[str] or None
    if files is None:
        files = sorted(os.path.join(root, f) for root, _, names in os.walk(folder)
                       for f in names if f.lower().endswith(_PLACES_EXTS))
        if files:
            _PLACES_CACHE[folder] = files
    if not files:
        raise RuntimeError(
            f"reflections/glares need Places365 photos, but none (*.jpg, *.jpeg, "
            f"*.png) were found in {folder}. Run `python download_places365.py` "
            f"(or point V7_PLACES_DIR at a folder of photos).")
    return files


def _patch_shape(rng, sec, files):
    """Shape mask of one effect from a random Places365 patch.

    rng: the section's random.Random; sec: reflections / glares config
    section (dict); files: sorted photo paths (list[str]).  Draws, in order:
    file index, patch side, x0, y0, flip, rot90 count, percentile.  Keeps the
    patch pixels at or above the p-th brightness percentile, then softens the
    edge with a Gaussian blur of radius edge_soft (the patch is zero-padded by
    3 * edge_soft first so the softened edge is not cut off).
    Returns (S', S') float32 in 0-1, S' = s + 2 * pad.
    """
    path = files[rng.randrange(len(files))]  # str, chosen photo
    with Image.open(path) as photo:
        gray = photo.convert("L")  # "L" photo, (h, w)
    s = rng.randint(*sec["patch_size_range"])  # int px, patch side
    s = max(1, min(s, gray.width, gray.height))
    x0 = rng.randint(0, gray.width - s)  # int px, patch left
    y0 = rng.randint(0, gray.height - s)  # int px, patch top
    patch = np.asarray(gray.crop((x0, y0, x0 + s, y0 + s)), dtype=np.float32)  # (s, s)
    if rng.random() < 0.5:
        patch = patch[:, ::-1]
    patch = np.rot90(patch, rng.randint(0, 3))
    thr = np.percentile(patch, rng.uniform(*sec["threshold_pct_range"]))  # float gray
    m = (patch >= thr).astype(np.float32)  # (s, s) float32, 1 = shape
    soft = float(sec["edge_soft"])  # float px, blur radius
    if soft > 0:
        pad = int(math.ceil(3 * soft))  # int px of zero border around the patch
        m_img = Image.fromarray(np.pad(m * 255, pad).astype(np.uint8))  # "L" (s+2p, s+2p)
        m = np.asarray(m_img.filter(ImageFilter.GaussianBlur(radius=soft)),
                       dtype=np.float32) / 255.0
    return np.clip(m, 0, 1)


def _place_shape(rng, m, canvas_size, bcx, bcy, b_radius, inside_outer):
    """Paste shape m at a random centre uniform in the disc of boundary B.

    m: (h, w) float32 mask; bcx, bcy: B centre (int px, canvas);
    b_radius: float px; inside_outer: (C, C) bool, inside B.  Draws radius
    then angle (polar, sqrt for uniform area).  Returns a (C, C) float32
    canvas mask, cropped at the canvas edges and zero outside B.
    """
    rr = b_radius * math.sqrt(rng.random())  # float px, centre distance from B centre
    th = rng.uniform(0, 2 * math.pi)  # float rad, centre direction
    h, w = m.shape
    x0 = int(round(bcx + rr * math.cos(th) - w / 2.0))  # int px, paste left (canvas)
    y0 = int(round(bcy + rr * math.sin(th) - h / 2.0))  # int px, paste top
    out = np.zeros((canvas_size, canvas_size), dtype=np.float32)  # (C, C) float32
    cx0, cy0 = max(x0, 0), max(y0, 0)  # int px, clipped paste start
    cx1, cy1 = min(x0 + w, canvas_size), min(y0 + h, canvas_size)  # int px, clipped end (excl.)
    if cx1 > cx0 and cy1 > cy0:
        out[cy0:cy1, cx0:cx1] = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
    out[~inside_outer] = 0
    return out


def _glass_effects(img, cfg, rngs, inside_outer, bcx, bcy, b_radius, tip, xs, ys):
    """Step 13c: reflections (out = in - sum darken*m), then glares
    (out = lerp(out, V, m), glare zeroed within clear_distance of the tip).

    img: canvas "L" (C, C); rngs: section_rngs dict; tip: visible tip (x, y)
    float px in canvas coords; xs (1, C) / ys (C, 1) int pixel grids.
    The photo folder is only read when an image actually gets an effect.
    Returns the canvas as an "L" image (C, C).
    """
    canvas_size = img.size[0]  # int px, C
    plan = []  # list of (name, section dict, random.Random, item count int)
    for name in ("reflections", "glares"):
        sec, rng = cfg[name], rngs[name][0]
        active = rng.random() < sec["prob"]  # bool, fixed-count draw 1
        n = rng.randint(*sec["count_range"])  # int, fixed-count draw 2
        plan.append((name, sec, rng, n if active else 0))
    if not any(n for *_, n in plan):
        return img
    files = places_files()  # list[str], sorted photo paths
    work = np.asarray(img, dtype=np.float32)  # (C, C) float32 working canvas
    (_, R, rrng, n_r), (_, Gl, grng, n_g) = plan
    dark = np.zeros_like(work)  # (C, C) float32, summed reflection darkening
    for _ in range(n_r):
        m = _patch_shape(rrng, R, files)
        delta = rrng.randint(*R["darken_range"])  # int gray levels subtracted
        dark += delta * _place_shape(rrng, m, canvas_size, bcx, bcy, b_radius, inside_outer)
    work = np.clip(work - dark, 0, 255)
    if n_g:
        clear = float(Gl["clear_distance"])  # float px
        # (C, C) bool, pixels a glare may never cover (around the visible tip)
        near_tip = (xs - tip[0]) ** 2 + (ys - tip[1]) ** 2 <= clear ** 2
        for _ in range(n_g):
            m = _patch_shape(grng, Gl, files)
            v = grng.randint(*Gl["value_range"])  # int gray level of the glare
            mc = _place_shape(grng, m, canvas_size, bcx, bcy, b_radius, inside_outer)
            mc[near_tip] = 0
            work = work * (1 - mc) + v * mc
    return Image.fromarray(np.clip(np.rint(work), 0, 255).astype(np.uint8))


def render_image(cfg, image_seed):
    """Render one dial from cfg and an image seed. Pure: no I/O, no global RNG.

    cfg: nested config dict (see v7_config); image_seed: int.
    Returns RenderResult (image (img_size, img_size) uint8, tip, angle).
    """
    rngs = section_rngs(image_seed)
    img_size = cfg["canvas"]["img_size"]  # output side (int, px)
    G, O = cfg["geometry"], cfg["outer"]  # geometry / outer sections, dict[str, value]

    # --- angle ---
    angle_deg = rngs["angle"][0].randint(*cfg["dataset"]["angle_deg_range"])  # int

    # --- geometry stream (all fixed-count; arc gap first) ---
    grng = rngs["geometry"][0]
    gap_start_deg = grng.uniform(0, 360)  # float, PIL arc degrees
    gap_span_deg = grng.uniform(*G["arc_gap_range"])  # float, deg
    scale = grng.uniform(*G["scale_range"])  # float
    dial_radius = int(img_size // 3 * scale)  # int, px
    # displacement: everything inside the circular boundary B shifts by
    # (disp_x, disp_y) in the image; B itself stays centred, outside it is 0
    disp_mag = grng.uniform(*G["displacement_range"])  # float px, shift length
    disp_ang = grng.uniform(0, 2 * math.pi)  # float rad, shift direction
    disp_x = int(round(disp_mag * math.cos(disp_ang)))  # int px, image shift x (lossless)
    disp_y = int(round(disp_mag * math.sin(disp_ang)))  # int px, image shift y
    hub_radius = dial_radius * grng.uniform(*G["hub_ratio_range"])  # float, px
    chord_half = grng.uniform(*G["chord_ratio_range"]) * hub_radius  # float, px
    max_d = math.sqrt(max(hub_radius ** 2 - chord_half ** 2, 0))
    chord_dist = grng.uniform(0, max_d)  # float, px: chord behind the centre
    tip_dist = (dial_radius - 2) * (1 + grng.uniform(-G["tip_variation"], G["tip_variation"]))
    ptr_disp_mag = grng.uniform(*G["ptr_displacement_range"])  # float px, pointer shift length
    ptr_disp_ang = grng.uniform(0, 2 * math.pi)  # float rad, pointer shift direction
    hole_r = hub_radius * grng.uniform(*G["hole_ratio_range"])  # float, px
    hd_mag = grng.uniform(*G["hole_disp_range"])
    hd_ang = grng.uniform(0, 2 * math.pi)
    stretch_h = grng.uniform(*G["stretch_h_range"])  # float
    stretch_v = grng.uniform(*G["stretch_v_range"])  # float
    # outer boundary radius / dial radius (float): the first and only draw of
    # the outer stream, drawn here so the canvas can be sized to cover B
    outer_scale = rngs["outer"][0].uniform(*O["scale_range"])  # float
    # largest possible |displacement| component (int px) for this config
    dm = int(math.ceil(max(abs(v) for v in G["displacement_range"])))
    # canvas half side (int px): dial + margin, or the stretched boundary B,
    # plus dm so B is fully covered wherever it lands in the canvas
    half = max(dial_radius + G["canvas_margin"],
               int(math.ceil(dial_radius * outer_scale * max(stretch_h, stretch_v))) + 1) + dm
    canvas_size = 2 * half  # int, even, px
    cx = cy = canvas_size // 2  # canvas centre (int, px); all content drawn about it
    # boundary B centre in canvas coords (int px): placing the canvas at the
    # centred offset + d lands it on the image centre
    bcx, bcy = cx - disp_x, cy - disp_y
    ptr_cx = cx + ptr_disp_mag * math.cos(ptr_disp_ang)  # float, pointer centre
    ptr_cy = cy + ptr_disp_mag * math.sin(ptr_disp_ang)
    hole_cx = ptr_cx + hd_mag * math.cos(hd_ang)  # float, hole centre
    hole_cy = ptr_cy + hd_mag * math.sin(hd_ang)

    # --- palette stream ---
    P, prng = cfg["palette"], rngs["palette"][0]
    bg_gray = prng.randint(*P["bg_gray_range"])  # int gray levels below
    circle_gray = prng.randint(*P["circle_gray_range"])
    outer_gray = prng.randint(*P["outer_gray_range"])
    rim_gray = prng.randint(*P["rim_gray_range"])

    # --- 1 non-uniform background (background stream, numpy) ---
    B, bnp = cfg["background"], rngs["background"][1]
    n_lf, var = B["lowfreq_size"], B["variation"]
    lf = np.clip(bg_gray + bnp.integers(-var, var + 1, (n_lf, n_lf), dtype=np.int16),
                 0, 255).astype(np.uint8)  # (n_lf, n_lf) uint8 tiles
    bg_arr = np.array(Image.fromarray(lf).resize((canvas_size, canvas_size), Image.BICUBIC),
                      dtype=np.int16)  # (C, C) int16
    ni = B["noise_intensity"]
    bg_arr += bnp.integers(-ni, ni + 1, (canvas_size, canvas_size), dtype=np.int16)
    img = Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8))  # canvas, "L" (C, C)

    # --- 2 faint dial circle with arc gap ---
    dial_box = [cx - dial_radius, cy - dial_radius, cx + dial_radius, cy + dial_radius]
    _arc_or_circle(ImageDraw.Draw(img), dial_box, gap_start_deg, gap_span_deg, circle_gray, 1)

    # --- 3 ticks on a separate overlay (digits stream decides visibility) ---
    D, drng = cfg["digits"], rngs["digits"][0]
    show_digits = drng.random() < D["show_prob"]  # bool, consumed by ticks too
    font_size = drng.randint(*D["font_size_range"])  # int px
    td_value, td_alpha = _new_layer(canvas_size)  # overlay value/alpha, "L" (C, C)
    tick_inner_r = None
    if show_digits:
        tick_inner_r = draw_ticks(td_value, td_alpha, cx, cy, dial_radius,
                                  cfg["ticks"], rngs["ticks"][0])

    # --- 4/5 shadows (shadows stream; all fixed-count draws up front) ---
    S, srng = cfg["shadows"], rngs["shadows"][0]
    has_shadow = srng.random() < S["prob"]  # bool
    light_angle = srng.uniform(0, 360)  # deg, 0 = up, CW
    s_p, s_r = _shadow_strengths(S, srng)  # float pointer / rim strengths, 0-1
    rim_offset = srng.uniform(*S["rim_offset_frac"])  # float, crescent offset / dial radius
    rim_cutout = srng.uniform(*S["rim_cutout_frac"])  # float, cutout radius / dial radius
    rim_blur = srng.uniform(*S["rim_blur_range"])  # float px, rim mask blur
    shd_dist = srng.uniform(*S["offset_range"])  # float px, pointer shadow offset
    # shadow darkening fraction on the canvas, (C, C) float32 in 0-1, or None
    # when the image has no shadow; nothing is composited onto img here, it
    # is applied once at step 13 as img * (1 - d)
    shade = None
    if has_shadow:
        m_r = draw_rim_shadows(cx, cy, dial_radius, canvas_size, light_angle,
                               rim_offset, rim_cutout, rim_blur)  # (C, C) float32 rim mask

        a = math.radians(angle_deg - 90)  # pointer direction (float, rad; 0 = +x, CW in image)
        s_apex = (ptr_cx + tip_dist * math.cos(a), ptr_cy + tip_dist * math.sin(a))
        bmx, bmy = ptr_cx - chord_dist * math.cos(a), ptr_cy - chord_dist * math.sin(a)
        # unit vector across the pointer (tuple of 2 float), along its base chord
        perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
        s_bl = (bmx + chord_half * perp[0], bmy + chord_half * perp[1])
        s_br = (bmx - chord_half * perp[0], bmy - chord_half * perp[1])
        light_rad = math.radians(light_angle - 90)  # shadow-cast direction (float, rad)
        p_mask = Image.new("L", (canvas_size, canvas_size), 0)  # pointer shadow shape, "L" (C, C)
        draw_shadow(p_mask, s_apex, s_bl, s_br, ptr_cx, ptr_cy,
                    hub_radius, dial_radius, shd_dist * math.cos(light_rad),
                    shd_dist * math.sin(light_rad), gap_start_deg, gap_span_deg)
        blur = ImageFilter.GaussianBlur(radius=S["blur_radius"])  # shadow blur filter (PIL)
        m_p = np.asarray(p_mask.filter(blur), dtype=np.float32) / 255.0  # (C, C) float32, 0-1
        shade = shadow_darkening(m_p, m_r, s_p, s_r)  # (C, C) float32

    # --- 6 digits on the overlay ---
    if show_digits:
        draw_scale_digits(td_value, td_alpha, cx, cy, dial_radius, tick_inner_r,
                          D, drng, font_size)

    # --- 7 mounting hole + pointer (pointer stream) ---
    PT = cfg["pointer"]
    ptr_rng, ptr_np = rngs["pointer"]
    ptr_gray = _pick_pointer_gray(ptr_rng, *PT["gray_range"], bg_gray,
                                  PT["min_bg_contrast"])  # int body gray
    hole_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    edge_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    grad_strength = ptr_rng.uniform(*PT["gradient_range"])  # float
    edge_thickness = ptr_rng.randint(*PT["edge_thickness_range"])  # int px

    # hole drawn BEFORE the pre-pointer snapshot so it survives the cut-tip restore
    ImageDraw.Draw(img).ellipse([hole_cx - hole_r, hole_cy - hole_r,
                                 hole_cx + hole_r, hole_cy + hole_r], fill=hole_gray)

    C = cfg["cut_tip"]
    cut_r = rngs["cut_tip"][0].randint(*C["radius_range"])  # int px
    bg_pre_ptr = img.copy() if C["enabled"] else None  # pre-pointer canvas, "L"
    img, apex, _, _, ptr_mask = draw_pointer(
        img, ptr_cx, ptr_cy, angle_deg, hub_radius, tip_dist, chord_half,
        chord_dist, ptr_gray, edge_gray, grad_strength, PT, ptr_np,
        edge_thickness, cut_r if C["enabled"] else None)
    if bg_pre_ptr is not None:
        img.paste(bg_pre_ptr, mask=Image.fromarray(255 - np.array(ptr_mask)))
        # label = visible tip: the centre of the cut arc on the pointer axis,
        # apex - cut_r * u with u = (cos a, sin a) the unit pointer direction
        a_ptr = math.radians(angle_deg - 90)  # float rad, canvas pointer angle
        apex = (apex[0] - cut_r * math.cos(a_ptr),
                apex[1] - cut_r * math.sin(a_ptr))  # (x, y) float px, visible tip

    # --- 8 horizontal / vertical stretch inside dial ---
    # inverse affine (output -> input) coefficients, 6-tuple of float: stretch about (cx, cy)
    affine = (1.0 / stretch_h, 0, cx * (1 - 1.0 / stretch_h),
              0, 1.0 / stretch_v, cy * (1 - 1.0 / stretch_v))
    size2 = (canvas_size, canvas_size)  # canvas (width, height) tuple of int
    img = img.transform(size2, Image.AFFINE, affine, resample=Image.BICUBIC)
    # overlay stretched as one "LA" image: Pillow resamples LA premultiplied
    # (La), so fringe pixels keep their own gray instead of fading toward 0
    td_la = Image.merge("LA", (td_value, td_alpha)).transform(
        size2, Image.AFFINE, affine, resample=Image.BICUBIC)  # "LA" (C, C)
    td_value, td_alpha = td_la.split()  # two "L" images (C, C)
    if shade is not None:
        # darkening follows the canvas through the same affine, (C, C) float32
        shade = np.asarray(Image.fromarray(shade).transform(
            size2, Image.AFFINE, affine, resample=Image.BILINEAR), dtype=np.float32)
    apex = (cx + (apex[0] - cx) * stretch_h, cy + (apex[1] - cy) * stretch_v)
    dial_rx, dial_ry = dial_radius * stretch_h, dial_radius * stretch_v  # float px

    # --- 9 exposure jitter ---
    J, jrng = cfg["jitter"], rngs["jitter"][0]
    img = ImageEnhance.Brightness(img).enhance(jrng.uniform(*J["brightness_range"]))
    img = ImageEnhance.Contrast(img).enhance(jrng.uniform(*J["contrast_range"]))
    img = ImageEnhance.Sharpness(img).enhance(jrng.uniform(*J["sharpness_range"]))

    # --- 10 outer annulus + rim band ---
    outer_rx, outer_ry = dial_rx * outer_scale, dial_ry * outer_scale  # float px
    arr = np.array(img)  # (C, C) uint8
    # open pixel grids: ys (C, 1) int, xs (1, C) int; they broadcast to (C, C)
    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    # squared normalised radius vs the dial ellipse (about (cx, cy)) / the
    # boundary B (about (bcx, bcy)), (C, C) float64; <= 1 inside
    inner_r2 = ((xs - cx) / max(dial_rx, 1)) ** 2 + ((ys - cy) / max(dial_ry, 1)) ** 2
    outer_r2 = ((xs - bcx) / max(outer_rx, 1)) ** 2 + ((ys - bcy) / max(outer_ry, 1)) ** 2
    inside_outer = outer_r2 <= 1  # (C, C) bool, inside B
    arr[(inner_r2 > 1) & inside_outer] = outer_gray
    r_norm = np.sqrt(inner_r2)  # (C, C) float64, radius / dial ellipse radius
    rim_half = O["rim_width"] / min(max(dial_rx, 1), max(dial_ry, 1))
    arr[(r_norm > 1 - rim_half) & (r_norm < 1 + rim_half)] = rim_gray

    # --- 11 hierarchical noise inside the outer ring ---
    H, hnp = cfg["hier_noise"], rngs["hier_noise"][1]
    nb, bv, pv = H["blocks"], H["block_variance"], H["pixel_variance"]
    block_u8 = np.clip(hnp.integers(-bv, bv + 1, (nb, nb), dtype=np.int16) + 128,
                       0, 255).astype(np.uint8)  # (nb, nb) uint8, offset by 128
    block_up = np.array(Image.fromarray(block_u8).resize(size2, Image.BICUBIC),
                        dtype=np.int16) - 128  # (C, C) int16
    combined = block_up + hnp.integers(-pv, pv + 1, size2, dtype=np.int16)  # (C, C) int16 block + pixel noise
    ch = arr.astype(np.int16)  # (C, C) int16 working copy of the canvas
    ch[inside_outer] += combined[inside_outer]
    arr = np.clip(ch, 0, 255).astype(np.uint8)

    # --- 12 adversarial triangle artifacts (avoid pointer + hole) ---
    ptr_mask_s = ptr_mask.transform(size2, Image.AFFINE, affine, resample=Image.NEAREST)
    ptr_blocked = np.array(ptr_mask_s) > 127  # (C, C) bool, stretched pointer
    s_hole_cx = cx + (hole_cx - cx) * stretch_h
    s_hole_cy = cy + (hole_cy - cy) * stretch_v
    ptr_blocked |= (((xs - s_hole_cx) / max(hole_r * stretch_h, 1)) ** 2
                    + ((ys - s_hole_cy) / max(hole_r * stretch_v, 1)) ** 2) <= 1

    T = cfg["tri_artifacts"]
    trng, tnp = rngs["tri_artifacts"]
    n_tris = trng.randint(*T["count_range"])  # int, first draw of the stream
    if not T["enabled"]:
        n_tris = 0
    tv = T["pixel_var"]
    for _ in range(n_tris):
        tri_gray = trng.randint(*T["gray_range"])  # int mean gray
        tcx = trng.randint(0, canvas_size - 1)
        tcy = trng.randint(0, canvas_size - 1)
        sz = trng.randint(*T["size_range"])  # int px
        verts = []  # list of 3 (x, y) float tuples, triangle vertices
        for _v in range(3):
            va = trng.uniform(0, 2 * math.pi)
            vr = trng.uniform(*T["vertex_dist_range"]) * sz
            verts.append((tcx + vr * math.cos(va), tcy + vr * math.sin(va)))
        tri_mask = Image.new("L", size2, 0)
        ImageDraw.Draw(tri_mask).polygon(verts, fill=255)
        valid = (np.array(tri_mask) > 127) & inside_outer & ~ptr_blocked  # (C, C) bool
        if not valid.any():
            continue
        noise = tnp.integers(-tv, tv + 1, size2, dtype=np.int16)  # (C, C) int16 per-pixel fill noise
        arr[valid] = np.clip(tri_gray + noise, 0, 255).astype(np.uint8)[valid]

    arr[~inside_outer] = 0
    img = Image.fromarray(arr)

    # --- 13 erase overlay over pointer/hole, composite ticks/digits ---
    td_a = np.array(td_alpha)
    td_a[ptr_blocked] = 0
    img = _composite_gray(img, td_value, Image.fromarray(td_a))

    # --- 13b physical shadow: out = round(in * (1 - d)) on the dial face ---
    if shade is not None:
        # where a shadow may fall, (C, C) bool: inside the dial ellipse, off
        # the rim band, not on the stretched pointer or hole (those sit above
        # the face, as the old overwrite order had it)
        face = (r_norm <= 1 - rim_half) & ~ptr_blocked
        d = np.where(face, shade, np.float32(0))  # (C, C) float32 darkening
        lit = np.asarray(img, dtype=np.float32)  # (C, C) float32 unshadowed canvas
        img = Image.fromarray(np.clip(np.rint(lit * (1 - d)), 0, 255).astype(np.uint8))

    # --- 13c cover-glass reflections, then glares (glare wins), inside B ---
    img = _glass_effects(img, cfg, rngs, inside_outer, bcx, bcy,
                         min(outer_rx, outer_ry), apex, xs, ys)

    # --- 14 Gaussian blur inside the outer ring ---
    if B["dial_circle_blur"]:
        blurred = img.filter(ImageFilter.GaussianBlur(radius=B["dial_circle_blur_radius"]))
        outer_mask_img = Image.new("L", size2, 0)
        ImageDraw.Draw(outer_mask_img).ellipse(
            [bcx - outer_rx, bcy - outer_ry, bcx + outer_rx, bcy + outer_ry], fill=255)
        img = Image.composite(blurred, img, outer_mask_img)
    arr = np.array(img)
    arr[~inside_outer] = 0
    img = Image.fromarray(arr)

    # --- 15 place the canvas at the centred offset + d onto a black image ---
    off_x = (img_size - canvas_size) // 2 + disp_x  # int px, canvas x0 in the image (may be < 0)
    off_y = (img_size - canvas_size) // 2 + disp_y  # int px, canvas y0 in the image
    out = np.zeros((img_size, img_size), dtype=np.uint8)  # (S, S) uint8 output image
    x0, y0 = max(off_x, 0), max(off_y, 0)  # int px, pasted region start (image)
    x1 = min(off_x + canvas_size, img_size)  # int px, pasted region end x (excl.)
    y1 = min(off_y + canvas_size, img_size)  # int px, pasted region end y (excl.)
    out[y0:y1, x0:x1] = arr[y0 - off_y:y1 - off_y, x0 - off_x:x1 - off_x]
    # safety net: zero everything outside B in image coords (B centre (int px)
    # = (bcx + off_x, bcy + off_y), same pixel convention as the canvas)
    iys, ixs = np.ogrid[:img_size, :img_size]  # (S, 1), (1, S) int grids
    out[((ixs - (bcx + off_x)) / max(outer_rx, 1)) ** 2
        + ((iys - (bcy + off_y)) / max(outer_ry, 1)) ** 2 > 1] = 0

    return RenderResult(image=out,
                        tip_x=apex[0] + off_x, tip_y=apex[1] + off_y,
                        angle_deg=angle_deg)


# ── Multiprocessing worker plumbing + dataset driver ──────────────────────
#
# Each image index gets its own seed (_seed_for_index), so output is
# bit-exact regardless of worker count.  Windows uses 'spawn': the cfg
# reaches workers through Pool initializer args, stored in _WORKER.

_WORKER = {}  # per-process state: {"output_dir": str, "seed_base": int, "cfg": dict}


def _seed_for_index(seed_base, idx):
    """Per-image seed, independent of worker count or task-scheduling
    order -- what makes --workers N bit-exact with --workers 1."""
    return (seed_base + idx * 2654435761) & 0xFFFFFFFF


def _worker_init(output_dir, seed_base, cfg):
    """Pool initializer: store output_dir (str), seed_base (int) and the
    validated cfg (dict section -> key -> value) in this process's _WORKER."""
    _WORKER["output_dir"] = output_dir
    _WORKER["seed_base"] = seed_base
    _WORKER["cfg"] = cfg


def _worker_task(idx):
    """Render image idx (int), save it as dial_<idx:04d>.png in the output dir;
    -> label row dict {filename: str, angle_deg: int, tip_x: float, tip_y: float}."""
    res = render_image(_WORKER["cfg"], _seed_for_index(_WORKER["seed_base"], idx))
    fname = f"dial_{idx:04d}.png"
    Image.fromarray(res.image).save(os.path.join(_WORKER["output_dir"], fname))
    return {"filename": fname, "angle_deg": res.angle_deg,
            "tip_x": round(res.tip_x, 2), "tip_y": round(res.tip_y, 2)}


LABEL_FIELDS = ["filename", "angle_deg", "tip_x", "tip_y"]  # labels.csv columns (list[str])
GENERATOR_NAME = "generate_synthetic_v7_mp.py"  # recorded in config.yaml run block (str)


class OutputDirNotEmpty(RuntimeError):
    """Output folder already holds dial_*.png / labels.csv / config.yaml and
    overwrite was not requested."""


def _dataset_files(output_dir):
    """Files generate_dataset owns in output_dir -> list[str] of absolute paths
    (dial_*.png, labels.csv, config.yaml); other files are never touched."""
    # escape: a '[...]' in output_dir is a folder name, not a pattern
    found = glob.glob(os.path.join(glob.escape(output_dir), "dial_*.png"))
    for name in ("labels.csv", "config.yaml"):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            found.append(path)
    return found


def _write_config(cfg, output_dir, run):
    """Write config.yaml = cfg + run block (run: dict[str, int|str|bool])."""
    with open(os.path.join(output_dir, "config.yaml"), "w", encoding="utf-8") as f:
        f.write(v7_config.dump_config(cfg, run=run))


def check_output_dir(output_dir, overwrite):
    """Raise OutputDirNotEmpty if output_dir (str; may not exist yet) holds dataset
    files and overwrite (bool) is False. Deletes nothing."""
    existing = _dataset_files(output_dir)
    if existing and not overwrite:
        raise OutputDirNotEmpty(
            f"'{output_dir}' already contains dataset files "
            f"({len(existing)} of dial_*.png/labels.csv/config.yaml); "
            f"pass --overwrite to replace them")


def generate_dataset(cfg, output_dir, num_images, seed=0, workers=None, chunksize=4,
                     overwrite=False, progress_cb=None, cancel_event=None):
    """Render num_images dials into output_dir with PNGs, labels.csv, config.yaml.

    cfg: validated config dict; progress_cb(done, total) is called after every
    finished image; cancel_event (threading.Event) stops the run, keeping only
    finished PNGs whose rows are in labels.csv.
    Returns {"completed": bool, "done": int, "output_dir": str}.
    Raises OutputDirNotEmpty if dataset files exist and overwrite is False.
    """
    os.makedirs(output_dir, exist_ok=True)
    check_output_dir(output_dir, overwrite)
    for path in _dataset_files(output_dir):
        os.remove(path)

    n_workers = max(1, min(workers or os.cpu_count() or 1, num_images))
    run = {"num_images": num_images, "seed": seed, "workers": n_workers,
           "chunksize": chunksize, "generator": GENERATOR_NAME,
           "completed": False, "finished_at": None}
    _write_config(cfg, output_dir, run)  # provisional, overwritten at the end

    written = []  # label rows already in labels.csv, list[dict]
    cancelled = False  # True once a cancel was seen or the run ended early (bool)
    pool = None  # mp.Pool when n_workers > 1, else None
    try:
        with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            writer.writeheader()

            def accept(label):
                """Record one finished label row; True if a cancel was requested."""
                writer.writerow(label)
                written.append(label)
                if progress_cb is not None:
                    progress_cb(len(written), num_images)
                return cancel_event is not None and cancel_event.is_set()

            indices = range(num_images)
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True  # cancel set before start: render nothing
            elif n_workers == 1:
                _worker_init(output_dir, seed, cfg)
                for idx in indices:
                    if accept(_worker_task(idx)):
                        cancelled = True
                        break
            else:
                pool = mp.Pool(processes=n_workers, initializer=_worker_init,
                               initargs=(output_dir, seed, cfg))
                for label in pool.imap(_worker_task, indices, chunksize=chunksize):
                    if accept(label):
                        cancelled = True
                        break
            f.flush()
    except BaseException:
        cancelled = True  # worker error, progress_cb error, Ctrl-C: finalize, re-raise
        raise
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        if len(written) < num_images:
            cancelled = True
        if cancelled:
            # in-flight workers may have written later PNGs; keep only labelled ones
            keep = {r["filename"] for r in written}
            for path in glob.glob(os.path.join(glob.escape(output_dir), "dial_*.png")):
                if os.path.basename(path) not in keep:
                    os.remove(path)
        run["completed"] = not cancelled
        run["finished_at"] = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
        _write_config(cfg, output_dir, run)
    return {"completed": not cancelled, "done": len(written), "output_dir": output_dir}


def main():
    """CLI entry: parse flags, load + validate the YAML config, run
    generate_dataset. Exit codes: 0 ok, 1 output dir not empty, 2 bad config."""
    parser = argparse.ArgumentParser(
        description="Generate native-grayscale synthetic dial images "
                    "(YAML-configured, bit-exact across worker counts)")
    parser.add_argument("--config", default=None,
                        help="YAML config merged over configs/v7_default.yaml")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"directory for PNGs, labels.csv, config.yaml (default: {OUTPUT_DIR})")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help=f"number of images (default: {NUM_IMAGES})")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default: 0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="worker processes (default: os.cpu_count())")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="Pool.imap chunksize (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete existing dial_*.png, labels.csv, config.yaml first")
    args = parser.parse_args()
    if args.num_images <= 0:
        parser.error("--num-images must be a positive integer")

    try:
        cfg = v7_config.load_config(args.config)
    except v7_config.ConfigError as e:
        print("Invalid config:", file=sys.stderr)
        for err in e.errors:
            print(f"  {err}", file=sys.stderr)
        sys.exit(2)
    try:
        res = generate_dataset(cfg, args.output_dir, args.num_images, seed=args.seed,
                               workers=args.workers, chunksize=args.chunksize,
                               overwrite=args.overwrite)
    except OutputDirNotEmpty as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if res["completed"]:
        print(f"Generated {res['done']} images in '{args.output_dir}/'")
    else:
        print(f"Cancelled after {res['done']} images in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
