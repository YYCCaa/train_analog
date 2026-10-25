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
  │ 13c cover glass, inside B: m_k = transformed font glyphs;        │
  │    reflections out = in - sum(dk * m_k),                         │
  │    then glares out = lerp(out, V, m) one by one (glare wins);    │
  │    glare m zeroed within clear_distance of the visible tip       │
  │ 14 Gaussian blur inside outer ring; black outside again          │
  │ 15 place canvas on a black img_size^2 at centred offset + d;     │
  │    boundary B stays centred (everything inside B shifts by d)    │
  └──────────────────────────────────────────────────────────────────┘
           v
    RenderResult(image (256, 256) uint8, tip_x, tip_y, angle_deg)

The pointer: a hub circle + an isosceles triangle whose base is a chord
of the hub, apex pointing outward.  Pixel coordinates: pixel i's centre at i.

Label axis = the triangle's height (apex -> base-chord midpoint) as drawn:
ImageDraw floors float polygon vertices, so the drawn apex is floor(apex) and
the base corners floor(base_left / _right).  The axis is mapped through the
same stretch as the image (Image.AFFINE works in pixel-corner coords, so the
stretch is about (cx - 0.5, cy - 0.5)); after a non-uniform stretch the
triangle is no longer isosceles and the label follows the stretched image of
the original height, not the stretched triangle's perpendicular altitude.
Then the integer canvas offset is added.  The label = drawn apex - t u, on that
line by construction.

Label rule (blur aware, see _TipField / _patch_tip):

    m    = visible pointer weight (edge ring, cut, rim / outer fill aware,
           through the same bicubic stretch as the image)
    Cmap = G_sigma_eff * m   per-pixel contrast fraction kept after the blur
           (sigma_eff = sqrt(dial blur^2 + tip_label.base_blur^2); a long bar
           of width w keeps erf(w / 2 sqrt 2 sigma) on its centre line, so a
           bar meets c exactly at w* = 2 sqrt 2 erfinv(c) sigma_eff)
    B    = Cmap >= c          pixels meeting the criterion (c = contrast_frac)

    label = drawn apex - t u, smallest t (0.05 px grid) with
            Cmap(p) >= c  and  share of B in the P x P patch on p >= min_ratio

         hub side                                       apex ->
          ##########################                    .
          ##############################   +---------+  .
     axis ===============================B=B=*=B=B==t==|  1-px tail: Cmap ok,
          ##############################   |B B B B .|     but B fills < 20 %
          ##########################       +---------+     of the patch -> no
                                       patch P x P on the label p: >= min_ratio
                                       of it meets Cmap >= c  -> label here

Fallbacks: contrast-only point (min_ratio ignored) -> visible end (farthest
pixel with weight > 1/2 within TIP_STRIP px of the axis) -> cut-radius guess;
the label is never ahead of the visible end ("clamped").  Raising sigma,
base_blur, c, patch_size or min_ratio never moves it forward (blur envelope +
nested patch squares, see _TipField).  params["derived"] / ["cut_tip"] record
ideal_apex_x/y (ideal apex), axis_apex_x/y and axis_base_mid_x/y (the drawn
height, derived only), tip_recession (px from the drawn apex, pre-stretch),
sigma_eff, w_star, contrast_at_tip (Cmap), patch_ratio_at_tip (plain P x P
share) and tip_rule ("patch" | "contrast" | "clamped" | "visible_end" |
"fallback").

Shadows are physical: a shadow blocks the direct light and keeps only the
ambient light, so it scales the surface by an albedo-independent factor,
out = in * (1 - d).  One shared strength s per image, jittered separately
for the pointer (s_p) and rim (s_r) shadows; overlapping shadows of the
one light do not compound (max, not product).  Ticks, digits, noise and
triangles under a shadow are darkened by the same factor.

Reflections and glares sit on the cover glass.  Each item takes a random
ASCII glyph (GLYPHS = A-Z a-z 0-9) in a random available font (GLYPH_FONTS,
rendered in memory and cached per (font, char); no data files), gives it a
continuous random transform with extreme ranges, softens its edge by
edge_soft into the shape m (0-1), and lands it at a centre uniform in the
disc of B:

    char c, font f ──render 320 px, crop, area-downscale──> sil (h, w) bool, longest 128
        │ resize: longest side = s ~ U(size_range)
        │ affine about the centre: rotate(θ) · shear(k) · aspect(√a, 1/√a)
        │ perspective: q = H q_a, H = I + [[0,0,0],[0,0,0],[px,py,0]] / s,
        │              P ~ U(perspective_range), (px, py) ~ U(-P, P) (per item),
        │              clamped so the denominator stays > 0.05
        │ elastic warp: g x g (dx, dy) grid ~ U(-A, A), A = strength · s,
        │              strength ~ U(warp_strength_range), g ~ randint(warp_grid_range)
        │              (per item),
        │              bicubic-upsampled, cv2.remap bilinear
        │ threshold 0.5, Gaussian blur edge_soft, crop to m > 0.01 (side <= 4·s)
        v
    m (h', w') float32 0-1 ──> placed m (C, C), zero outside B
    reflection: out = in - darken * m     (values, overlaps add)
    glare:      out = in*(1-m) + V*m      (after all reflections; not near tip)

Params: RenderResult.params records the values this image actually drew or
derived, dict[str, dict] grouped by section.  Recording adds no draws and
changes no pixel.  Values are JSON-plain (int, float rounded to 4 decimals,
bool, str, list); positions are image px unless named otherwise.

    angle          angle_deg
    geometry       arc_gap_start_deg, arc_gap_span_deg, scale, dial_radius,
                   displacement_mag / _angle_rad / _x / _y, hub_radius,
                   chord_half, chord_dist, tip_dist, ptr_displacement_mag /
                   _angle_rad, hole_radius, hole_displacement_mag / _angle_rad,
                   stretch_h, stretch_v, canvas_size
    outer          scale
    palette        bg_gray, circle_gray, outer_gray, rim_gray
    background     lowfreq_min, lowfreq_max, lowfreq_mean (of the tile grid)
    digits         show, font_size; when shown: angle_jitter, gray, gap,
                   inside, radial_jitter, dist
    ticks          drawn; when drawn: angle_jitter, gray, per_digit,
                   major_lens[10], major_widths[10], inner_radius
    shadows        active, light_angle, s_p, s_r, rim_offset_frac,
                   rim_cutout_frac, rim_blur, offset
    pointer        gray, hole_gray, edge_gray, gradient, edge_thickness
    cut_tip        enabled, cut_r, ideal_apex_x / _y, tip_recession,
                   sigma_eff, w_star, contrast_at_tip, patch_ratio_at_tip, tip_rule
    jitter         brightness, contrast, sharpness
    hier_noise     block_min, block_max, block_std (of the block offsets)
    tri_artifacts  enabled, count_drawn, n, items[n] {gray, center_x,
                   center_y, size, placed}
    reflections    active, count_drawn, n, items[n] {char, font, size,
                   rotation, aspect, shear, P, px, py, warp_strength, g,
                   center_x, center_y, darken}
    glares         same as reflections with value instead of darken
    blur           enabled, radius
    derived        img_size, canvas_offset_x / _y, dial_rx, dial_ry,
                   outer_rx, outer_ry, tip_x, tip_y, ideal_apex_x / _y,
                   axis_apex_x / _y, axis_base_mid_x / _y, tip_recession,
                   sigma_eff, w_star, contrast_at_tip, patch_ratio_at_tip, tip_rule
"""

import argparse
import csv
import datetime
import glob
import math
import multiprocessing as mp
import os
import random
import string
import sys
import zlib
from dataclasses import dataclass, field

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
    tip_x: float  # visible pointer end x on its axis, image px (float; edge ring + cut aware)
    tip_y: float  # visible pointer end y on its axis, image px (float; edge ring + cut aware)
    angle_deg: int  # pointer angle, whole degrees (0 = up, clockwise)
    # sampled per-image values, dict[str, dict]: section name -> key -> JSON-plain
    # value (int / float rounded to 4 decimals / bool / str / list); see "Params"
    # in the module docstring.  Recorded only, never drawn for.
    params: dict = field(default_factory=dict)


def _plain(v):
    """JSON-plain copy of a recorded value: bool / int / str kept, floats (and
    numpy scalars) rounded to 4 decimals, tuples -> lists, dicts recursed."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 4)
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    return v


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


def draw_ticks(td_value, td_alpha, cx, cy, dial_radius, tcfg, rng, rec=None):
    """Draw the 0-9 scale ticks (+ subdivisions) on the overlay layer.

    td_value / td_alpha : overlay "L" images (canvas, canvas)
    tcfg                : cfg["ticks"]; rng: the "ticks" random.Random
    rec                 : optional dict, filled with the drawn values (no draws added)
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
    major_lens, major_widths = [], []  # list[int] px per digit, recorded only

    for i in range(NUM_DIGITS):
        for s in range(ticks_per_digit):
            angle_rad = math.radians(i * DEG_PER_DIGIT + s * sub_deg
                                     + angle_jitter - 90)  # -90 so 0 = up
            if s == 0:  # major tick
                tick_len = rng.randint(*tcfg["len_range"])
                tick_w = rng.randint(*tcfg["width_range"])
                max_major_len = max(max_major_len, tick_len)
                major_lens.append(tick_len)
                major_widths.append(tick_w)
            else:  # minor tick
                tick_len = max(1, rng.randint(*tcfg["len_range"]) // 2)
                tick_w = max(1, rng.randint(*tcfg["width_range"]) - 1)
            tick_inner = tick_outer - tick_len
            seg = [(cx + tick_inner * math.cos(angle_rad), cy + tick_inner * math.sin(angle_rad)),
                   (cx + tick_outer * math.cos(angle_rad), cy + tick_outer * math.sin(angle_rad))]
            vdraw.line(seg, fill=grey, width=tick_w)
            adraw.line(seg, fill=255, width=tick_w)

    if rec is not None:
        rec.update(angle_jitter=angle_jitter, gray=grey, per_digit=ticks_per_digit,
                   major_lens=major_lens, major_widths=major_widths,
                   inner_radius=tick_outer - max_major_len)
    return tick_outer - max_major_len


def draw_scale_digits(td_value, td_alpha, cx, cy, dial_radius, tick_inner_radius,
                      dcfg, rng, font_size, rec=None):
    """Draw digits 0-9 around the dial on the overlay layer (0 at top, CW).

    Digits sit inside or outside the ticks with a clearance gap.  The radial
    jitter is one-directional: inside digits move only inward (toward the
    centre, inside_jitter_range), outside digits only outward (away from the
    centre, outside_jitter_range).  dcfg = cfg["digits"]; rng = "digits" stream;
    rec: optional dict, filled with the drawn values (no draws added).
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
    if rec is not None:
        rec.update(angle_jitter=ang_jitter, gray=grey, gap=gap, inside=inside,
                   radial_jitter=abs(dist - base_dist), dist=dist)

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


def _erosion_dist(m, thickness):
    """Erosion-ring index of every pixel of a pointer mask.

    m         : (C, C) uint8 pointer mask (0 / 255)
    thickness : int >= 0, number of 4-neighbour roll-erosion steps
    Returns dist, (C, C) float32: the step at which a mask pixel is eroded away
    (0 = outermost ring) or `thickness` for pixels deeper than the band and for
    every pixel outside the mask.  Shared by _draw_pointer_edge and the visible
    mask in draw_pointer so the two always agree.
    """
    dist = np.full(m.shape, thickness, dtype=np.float32)  # (C, C) float32
    current = m.copy()  # (C, C) uint8, mask still standing after `step` erosions
    for step in range(thickness):
        e1 = np.minimum(np.roll(current, 1, 0), np.roll(current, -1, 0))
        e2 = np.minimum(np.roll(current, 1, 1), np.roll(current, -1, 1))
        eroded = np.minimum(e1, e2) & current
        dist[(current > 0) & (eroded == 0)] = step
        current = eroded
    return dist


def _draw_pointer_edge(img, mask, edge_gray, bg_arr, thickness, noise_amp, nrng):
    """Variable-thickness edge band on the pointer mask boundary.

    The outermost erosion ring blends toward the pre-pointer background
    (bg_arr, (C, C) uint8), inner rings toward edge_gray; each pixel gets
    independent noise from nrng.  Because ring 0 is repainted with the
    background (blend = 0), the visible pointer is the mask minus ring 0.
    """
    m = np.array(mask)  # (C, C) uint8
    arr = np.array(img)  # (C, C) uint8
    # dist[y, x] = erosion steps survived (0 = outermost ring), (C, C) float32
    dist = _erosion_dist(m, thickness)

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

    Returns (img, apex, base_left, base_right, mask, vis) where mask is the "L"
    pointer mask with the apex disc erased when cut_tip_radius is not None, and
    vis, (C, C) float32 in [0, 1], is the VISIBLE pointer: that (cut) mask
    weighted by how far _draw_pointer_edge blends each edge ring away from the
    background (ring 0 = background = 0, same erosion of the uncut mask), plus,
    with the cut off, the outline-only pixels the img draw adds (weight 1).  apex is the ideal triangle apex, (x, y) float px.
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

    m_uncut = np.array(mask)  # (C, C) uint8, uncut pointer mask (before the cut)
    if cut_tip_radius is not None:
        mask_arr = m_uncut.copy()  # (C, C) uint8, mask to cut
        ys_t, xs_t = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1]]
        mask_arr[(xs_t - apex[0]) ** 2 + (ys_t - apex[1]) ** 2 <= cut_tip_radius ** 2] = 0
        mask = Image.fromarray(mask_arr)
    # visible pointer, (C, C) float32 in [0, 1]: how much each pixel shows the
    # pointer rather than the background.  0 outside the (cut) mask; ring k of
    # the edge band of the uncut mask blends bg -> edge gray by
    # k / max(t - 1, 1) (see _draw_pointer_edge), so it gets that weight (ring 0
    # = pure background = 0); deeper pixels 1.  Thickness 0 draws no band.
    dist = _erosion_dist(m_uncut, edge_thickness)  # (C, C) float32 ring index
    vis = np.where(dist < edge_thickness, dist / max(edge_thickness - 1, 1),
                   1.0).astype(np.float32)  # (C, C) float32, band weight
    vis[np.array(mask) == 0] = 0.0
    if cut_tip_radius is None:
        # the img polygon / ellipse are drawn WITH an outline, which can paint a
        # sliver of pixels the fill-only mask lacks (thin apex); they keep the
        # flat ptr_gray (weight 1).  With the cut on, render_image restores
        # everything outside the cut mask, so they vanish there.
        drawn = Image.new("L", img.size, 0)  # "L" (C, C), pixels the img draw painted
        ddraw = ImageDraw.Draw(drawn)
        ddraw.polygon([apex, base_left, base_right], fill=255, outline=255)
        ddraw.ellipse(hub_box, fill=255, outline=255)
        vis[(np.array(drawn) > 0) & (m_uncut == 0)] = 1.0
    return img, apex, base_left, base_right, mask, vis


# float px, half-width of the strip about the pointer axis in which visible
# pixels count toward the tip: catches a 1-px apex sliver drawn just off-axis
# and the two "horns" a cut arc leaves where it meets the triangle sides
TIP_STRIP = 6.0


def _visible_tip(vis, apex, u, max_back, fallback=None, cover=None):
    """Visible end of the pointer on its axis.

    vis      : (C, C) bool or float in [0, 1], visible-pointer weight; pixel
               centres at integer coords (ImageDraw convention)
    apex     : (x, y) float px, ideal triangle apex
    u        : (ux, uy) float, unit pointer direction (hub -> apex)
    max_back : float px, how far back from the apex to search
    fallback : (x, y) float px (or None), returned when no visible pixel is found
    cover    : None, or (C, C) bool, pixels painted over the pointer later
               (rim band / outer fill); they never count as visible
    A pixel is visible when its weight is > 0.5 (more pointer than background;
    an even 50/50 edge ring does not count) and it is not covered.  The
    tip is the point on the axis, apex - t * u, level with the farthest visible
    pixel (largest projection on u) inside the strip |lateral| <= TIP_STRIP and
    at most max_back behind the apex; t is rounded up to the 0.05 px walk grid
    (so the label never sits beyond that pixel).  Returns (x, y) float px.
    """
    reach = int(math.ceil(max_back + TIP_STRIP)) + 2  # int px, search window half-size
    h, w = vis.shape
    xa, ya = int(math.floor(apex[0])), int(math.floor(apex[1]))  # int px, window anchor
    x_lo, x_hi = max(xa - reach, 0), min(xa + reach + 1, w)  # int px column range
    y_lo, y_hi = max(ya - reach, 0), min(ya + reach + 1, h)  # int px row range
    if x_lo >= x_hi or y_lo >= y_hi:
        return fallback
    ys, xs = np.mgrid[y_lo:y_hi, x_lo:x_hi]  # (h', w') int pixel coords of the window
    ok = vis[y_lo:y_hi, x_lo:x_hi] > 0.5  # (h', w') bool, more pointer than background
    if cover is not None:
        ok &= ~cover[y_lo:y_hi, x_lo:x_hi]
    dx, dy = xs - apex[0], ys - apex[1]  # (h', w') float64 offsets from the apex
    back = -(dx * u[0] + dy * u[1])  # (h', w') float64, distance behind the apex
    lat = np.abs(-dx * u[1] + dy * u[0])  # (h', w') float64, distance off the axis
    sel = ok & (lat <= TIP_STRIP) & (back <= max_back)  # (h', w') bool
    if not sel.any():
        return fallback
    t = math.ceil(round(float(back[sel].min()) / 0.05, 6)) * 0.05  # float px on the grid
    return apex[0] - t * u[0], apex[1] - t * u[1]


# ── blur-aware tip label: per-pixel contrast + patch support (_patch_tip) ──
TIP_T_STEP = 0.05  # float px, along-axis grid of the label scan
TIP_CHUNK = 200  # int, grid points scanned per vectorised chunk (10 px)
TIP_PATCH_MIN = 3  # int px, smallest square of the nested patch support (_TipField.rule_support)
# blur ladder (tuple[float, ...] px, increasing, the same for every sigma) of
# the monotone contrast envelope in _TipField: 0.25 px steps up to 2, 0.5 up
# to 6, 1 up to 40 (fine where the raster jaggies matter, cheap at big blurs)
TIP_SIGMA_LADDER = tuple([0.25 * k for k in range(1, 8)] + [2.0 + 0.5 * k for k in range(8)]
                         + [6.0 + k for k in range(35)])


def _erfinv(c):
    """Inverse error function of c (float in (-1, 1)) -> float, Newton on math.erf."""
    x = math.sqrt(-math.log((1 - c) * (1 + c))) * math.copysign(0.8, c)  # float, start
    for _ in range(60):
        step = (math.erf(x) - c) / (2 / math.sqrt(math.pi) * math.exp(-x * x))
        x -= step
        if abs(step) < 1e-15:
            break
    return x


def tip_width_threshold(contrast_frac, sigma_eff):
    """w* (float px) = 2 sqrt(2) erfinv(c) sigma_eff: the narrowest bar that keeps
    contrast fraction c (float) after a Gaussian blur of std sigma_eff (float px).
    Recorded for reference: the contrast map reaches c on a long bar's centre
    line exactly when the bar is w* wide."""
    if sigma_eff <= 0:
        return 0.0
    return 2 * math.sqrt(2) * _erfinv(contrast_frac) * sigma_eff


class _TipField:
    """Per-pixel blurred pointer contrast and patch support near the label scan.

    Built on a window around the search segment apex - t u, 0 <= t <= t_max:

        Cmap = G_sigma * m       contrast fraction the pointer keeps per pixel
        B    = Cmap >= c         "meets the contrast criterion" (0 off-canvas)
        R_s(p) = mean of B over the s x s square on p's nearest pixel

    Monotone guard: Cmap is in fact min(G_s * m) over s in {0} + the rungs of
    TIP_SIGMA_LADDER below sigma + {sigma}.  On a convex end (and inside a
    region star-shaped about the point) G_s * m only falls as s grows, so this
    is G_sigma * m there; it differs only on raster jaggies / concave notches,
    where a larger blur could otherwise "fill in" a point and move the label
    forward.  The rungs are the same for every sigma, so raising sigma (dial
    blur or base_blur) cannot raise Cmap anywhere except between rungs.

    m      : (C, C) float, visible pointer weight in [0, 1], covered pixels zeroed;
             pixel centres at integer coords
    apex   : (x, y) float px, scan origin; u : (ux, uy) float unit axis direction
    t_max  : float px, scan length
    sigma  : float px, sigma_eff (0 = no blur: Cmap = m)
    c      : float, contrast_frac
    patch  : int px, patch side P (odd; an even value is rounded up)

    Attributes: x0, y0 int px, canvas coords of window pixel [0, 0]; cmap
    (h, w) float32 Cmap over the window, exact (the blur runs on a crop
    kr = ceil(4 sigma) px larger, zero beyond the canvas); integ (h + 1, w + 1)
    int64, integral image of B; patch int.
    """

    def __init__(self, m, apex, u, t_max, sigma, c, patch):
        self.patch = int(patch) + 1 - int(patch) % 2  # int px, odd patch side
        end = (apex[0] - t_max * u[0], apex[1] - t_max * u[1])  # (x, y) float px, scan end
        mar = self.patch // 2 + 2  # int px, window margin: patches + bilinear neighbours
        self.x0 = int(math.floor(min(apex[0], end[0]))) - mar  # int px, window origin x
        self.y0 = int(math.floor(min(apex[1], end[1]))) - mar  # int px, window origin y
        x1 = int(math.ceil(max(apex[0], end[0]))) + mar + 1  # int px, window end x (excl.)
        y1 = int(math.ceil(max(apex[1], end[1]))) + mar + 1  # int px, window end y (excl.)
        kr = int(math.ceil(4 * sigma)) if sigma > 0 else 0  # int px, blur kernel radius
        h, w = m.shape
        # crop of m, window + kr all round, (y1-y0+2kr, x1-x0+2kr) float32, 0 off-canvas
        crop = np.zeros((y1 - self.y0 + 2 * kr, x1 - self.x0 + 2 * kr), dtype=np.float32)
        ya, yb = max(self.y0 - kr, 0), min(y1 + kr, h)  # int px, canvas rows in the crop
        xa, xb = max(self.x0 - kr, 0), min(x1 + kr, w)  # int px, canvas cols in the crop
        if ya < yb and xa < xb:
            crop[ya - self.y0 + kr:yb - self.y0 + kr, xa - self.x0 + kr:xb - self.x0 + kr] = \
                np.asarray(m[ya:yb, xa:xb], dtype=np.float32)
        # monotone envelope over the blur ladder sigma_k (TIP_SIGMA_LADDER) < sigma
        # (k = 0, 1, ...) and sigma itself: Cmap = min of G_sigma' * m (G_0 * m = m).
        # Rung k is rung k-1 blurred by sqrt(sigma_k^2 - sigma_(k-1)^2) (Gaussians
        # add in quadrature; small kernels, cheap), so the rungs are the same for
        # every sigma.  Sampled small kernels blur a little less than their std,
        # so the rungs sit slightly above the true G_sigma_k * m: they only ever
        # guard, never set, the value.  G_sigma * m itself is blurred directly
        env = crop  # (h'+2kr, w'+2kr) float32, running minimum
        if kr:
            import cv2

            def gblur(a, sg):
                """a (H, W) float32 blurred by a Gaussian of std sg (float px), 0 outside."""
                kk = int(math.ceil(4 * sg))  # int px, kernel radius
                return cv2.GaussianBlur(a, (2 * kk + 1, 2 * kk + 1), sg, sigmaY=sg,
                                        borderType=cv2.BORDER_CONSTANT)
            rung, prev = crop, 0.0  # (h'+2kr, w'+2kr) float32 G_prev * m, float px std
            for sk in TIP_SIGMA_LADDER:
                if sk >= sigma - 1e-9:
                    break
                rung = gblur(rung, math.sqrt(sk * sk - prev * prev))
                env = np.minimum(env, rung)
                prev = sk
            env = np.minimum(env, gblur(crop, sigma))
        self.cmap = env[kr:env.shape[0] - kr, kr:env.shape[1] - kr]  # (h', w') float32
        gy, gx = np.ogrid[self.y0:y1, self.x0:x1]  # (h', 1), (1, w') int canvas coords
        meets = (self.cmap >= c) & (gy >= 0) & (gy < h) & (gx >= 0) & (gx < w)  # (h', w') bool
        self.integ = np.zeros((meets.shape[0] + 1, meets.shape[1] + 1), dtype=np.int64)
        self.integ[1:, 1:] = meets.cumsum(0).cumsum(1)

    def contrast(self, x, y):
        """Bilinear Cmap at canvas points (x, y) (float or arrays) -> float64 array."""
        hh, ww = self.cmap.shape
        fx = np.clip(np.asarray(x, dtype=np.float64) - self.x0, 0, ww - 1.001)  # window coords
        fy = np.clip(np.asarray(y, dtype=np.float64) - self.y0, 0, hh - 1.001)
        x0, y0 = fx.astype(np.intp), fy.astype(np.intp)  # top-left pixel (>= 0)
        ax, ay = fx - x0, fy - y0  # float64 fractions
        g = self.cmap
        return ((g[y0, x0] * (1 - ax) + g[y0, x0 + 1] * ax) * (1 - ay)
                + (g[y0 + 1, x0] * (1 - ax) + g[y0 + 1, x0 + 1] * ax) * ay)

    def ratio(self, x, y, size=None):
        """R_s: share of the s x s square (s = size or the patch, odd) centred on
        the nearest pixel of each (x, y) whose Cmap >= c -> float64 array."""
        s = self.patch if size is None else int(size)  # int px, square side
        r = s // 2  # int px, half side
        hh, ww = self.cmap.shape
        ix = np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(np.intp) - self.x0
        iy = np.floor(np.asarray(y, dtype=np.float64) + 0.5).astype(np.intp) - self.y0
        xa, xb = np.clip(ix - r, 0, ww), np.clip(ix + r + 1, 0, ww)  # window col bounds
        ya, yb = np.clip(iy - r, 0, hh), np.clip(iy + r + 1, 0, hh)  # window row bounds
        I = self.integ  # (h' + 1, w' + 1) int64
        return (I[yb, xb] - I[ya, xb] - I[yb, xa] + I[ya, xa]) / float(s * s)

    def rule_support(self, x, y):
        """Support the patch rule tests: min over odd s = TIP_PATCH_MIN..P of R_s,
        float64 array.  Nested squares make a larger patch never accept a point a
        smaller one rejects (a plain R_P can grow with P near a wedge apex)."""
        out = self.ratio(x, y)  # float64, R_P
        for s in range(TIP_PATCH_MIN, self.patch, 2):
            out = np.minimum(out, self.ratio(x, y, s))
        return out


def _patch_tip(field, apex, u, t_max, c, min_ratio, t_start=0.0):
    """Label: the smallest t (TIP_T_STEP grid, t_start..t_max) behind apex with
    Cmap(p) >= c (bilinear) and, when min_ratio > 0, rule_support(p) >= min_ratio,
    p = apex - t u.  field is the _TipField of that segment.

    Model: a Gaussian blur of std sigma keeps a fraction Cmap = G_sigma * m of
    the pointer's contrast at each pixel (a long bar of width w keeps
    erf(w / (2 sqrt 2 sigma)) on its centre line, the width rule); a point can
    pass that per-pixel test on a thin sliver yet stay invisible when its
    neighbourhood holds almost no other passing pixel, hence the patch ratio.
    The scan runs in chunks of TIP_CHUNK points and stops at the first hit
    (each point is 4 Cmap reads + a few integral-image boxes).
    Returns ((x, y) float px, t float px) or None.
    """
    n_t = int(math.floor(t_max / TIP_T_STEP + 1e-9)) + 1  # int, grid points 0..t_max
    i0 = int(math.ceil(t_start / TIP_T_STEP - 1e-9))  # int, first grid index scanned
    for lo in range(i0, n_t, TIP_CHUNK):
        idx = np.arange(lo, min(lo + TIP_CHUNK, n_t))  # (<= TIP_CHUNK,) int grid indices
        ts = idx * TIP_T_STEP  # (K,) float64 px behind the apex
        px, py = apex[0] - ts * u[0], apex[1] - ts * u[1]  # (K,) float64 points
        ok = field.contrast(px, py) >= c  # (K,) bool
        if min_ratio > 0 and ok.any():
            ok &= field.rule_support(px, py) >= min_ratio
        hit = np.flatnonzero(ok)  # (H,) int
        if hit.size:
            t = float(ts[hit[0]])
            return (apex[0] - t * u[0], apex[1] - t * u[1]), t
    return None


# ── main render ────────────────────────────────────────────────────────

# ── reflections + glares (cover glass, shapes from ASCII font glyphs) ─────

GLYPHS = string.ascii_letters + string.digits  # str (62 chars), the glyphs shapes are drawn from
# TrueType font names (tuple[str, ...]) tried with ImageFont.truetype, in this
# order; the ones that fail to load are skipped (see glyph_fonts)
GLYPH_FONTS = ("arial.ttf", "arialbd.ttf", "times.ttf", "timesbd.ttf", "cour.ttf",
               "courbd.ttf", "georgia.ttf", "verdana.ttf", "impact.ttf", "comic.ttf",
               "segoeui.ttf", "consola.ttf")
_GLYPH_DEFAULT = "<default>"  # str, font name standing for ImageFont.load_default (no TrueType found)
_GLYPH_RENDER_PX = 320  # int px font size glyphs are drawn at (then area-downscaled)
_GLYPH_SIDE = 128  # int px, longest side of every cached glyph silhouette
_SIL_SMOOTH = 1.0  # float px (at the 128-px glyph scale), pre-blur that rounds the stair steps
_GLYPH_FONT_NAMES = None  # tuple[str, ...] loadable font names, resolved once (None = not yet)
_GLYPH_FONT_OBJ = {}  # dict[str, ImageFont], loaded fonts by name
_GLYPH_CACHE = {}  # dict[tuple[int, str], (h, w) bool], silhouettes by (font index, char)
_PERSP_MIN_DEN = 0.05  # float, smallest projective denominator allowed over the source box
_PERSP_CLAMP_DEN = 0.1  # float, denominator a too-strong keystone is clamped to
_MAX_SIDE_FACTOR = 4  # int, output mask side is at most this times the glyph size s


def _load_glyph_font(name):
    """ImageFont for font name (str) at _GLYPH_RENDER_PX (raises OSError if missing)."""
    if name == _GLYPH_DEFAULT:
        return ImageFont.load_default(size=_GLYPH_RENDER_PX)
    return ImageFont.truetype(name, _GLYPH_RENDER_PX)


def glyph_fonts():
    """Loadable glyph font names (tuple[str, ...]), in GLYPH_FONTS order.

    Resolved once per process and cached; ("<default>",) when no TrueType
    font of GLYPH_FONTS loads (then PIL's built-in default font is used).
    """
    global _GLYPH_FONT_NAMES
    if _GLYPH_FONT_NAMES is None:
        names = []  # list[str], fonts that loaded
        for name in GLYPH_FONTS:
            try:
                _GLYPH_FONT_OBJ[name] = _load_glyph_font(name)
            except OSError:
                continue
            names.append(name)
        if not names:
            _GLYPH_FONT_OBJ[_GLYPH_DEFAULT] = _load_glyph_font(_GLYPH_DEFAULT)
            names = [_GLYPH_DEFAULT]
        _GLYPH_FONT_NAMES = tuple(names)
    return _GLYPH_FONT_NAMES


def glyph_silhouette(char, font_index=0):
    """Silhouette of one glyph -> (h, w) bool, tight bbox, longest side 128.

    char: str, one char of GLYPHS; font_index: int into glyph_fonts().
    Drawn white on black at _GLYPH_RENDER_PX with a margin (ascenders and
    descenders are never clipped), cropped to its tight bbox, area-
    downscaled to longest side 128 and re-thresholded at 0.5.  Cached per
    (font_index, char).
    """
    import cv2
    key = (font_index, char)  # tuple[int, str] cache key
    hit = _GLYPH_CACHE.get(key)
    if hit is not None:
        return hit
    font = _GLYPH_FONT_OBJ[glyph_fonts()[font_index]]  # ImageFont
    x0, y0, x1, y1 = font.getbbox(char)  # int px ink box relative to the draw origin
    pad = _GLYPH_RENDER_PX // 4  # int px margin on every side
    im = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), 0)  # "L" render canvas
    ImageDraw.Draw(im).text((pad - x0, pad - y0), char, fill=255, font=font)
    a = np.asarray(im) >= 128  # (H, W) bool, ink
    ys, xs = np.nonzero(a)
    a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # (h0, w0) bool, tight bbox
    f = _GLYPH_SIDE / float(max(a.shape))  # float, scale to longest side 128
    w, h = max(1, int(round(a.shape[1] * f))), max(1, int(round(a.shape[0] * f)))  # int px
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR  # int cv2 flag
    sil = cv2.resize(a.astype(np.float32), (w, h), interpolation=interp)  # (h, w) float32
    sil = sil.reshape(h, w) >= 0.5  # (h, w) bool
    ys, xs = np.nonzero(sil)
    sil = sil[ys.min():ys.max() + 1, xs.min():xs.max() + 1]  # (h, w) bool, tight again
    _GLYPH_CACHE[key] = sil
    return sil


def transform_shape(sil, size, rotation_deg, aspect, shear, warp, edge_soft,
                    perspective=(0.0, 0.0)):
    """Continuous random transform of one silhouette -> shape mask m.

    sil: (h, w) bool silhouette; size: int px s, longest side after scaling;
    rotation_deg: float, > 0 turns the shape counter-clockwise as displayed
    (image y down; +90 equals np.rot90(., 1)); aspect: float a, x *= sqrt(a),
    y /= sqrt(a); shear: float k, x += k * y; warp: (2, g, g) float px grid
    of (dx, dy) displacements, or None for rigid; edge_soft: float px blur;
    perspective: (px, py) float keystone, H = I + [[0,0,0],[0,0,0],[px,py,0]] / s.

        sil ─pre-blur─> float ─resize─> (s-scaled) ─remap(bilinear):
        q -> L^-1 H^-1 (q + d(q)) + c_src─> >= 0.5 ─blur─> crop m > 0.01

        source box ──L (affine)──> parallelogram ──H (keystone)──> quad
                                   q_a           q = q_a / (1 + (px, py)·q_a / s)

    The pre-blur (_SIL_SMOOTH at the 128 scale) turns stair-stepped edges
    into smooth curves before the re-binarisation.  L = rotate · shear ·
    aspect about the shape centre; H maps centred coords.  When the
    projective denominator would fall <= 0.05 anywhere over the box, (px, py)
    are scaled down so its minimum is 0.1.  The output canvas holds the
    projected corners + warp amplitude + blur, at most 4·s per side (pixels
    beyond it are lost).  Returns (h', w') float32 in 0-1 cropped to its
    tight bbox ((1, 1) zeros if nothing is left).
    """
    import cv2
    h, w = sil.shape
    f = size / float(max(h, w))  # float, scale factor to the target size
    sw, sh = max(1, int(round(w * f))), max(1, int(round(h * f)))  # int px, scaled w / h
    src = cv2.GaussianBlur(sil.astype(np.float32), (0, 0), _SIL_SMOOTH,
                           borderType=cv2.BORDER_CONSTANT)  # (h, w) float32 0-1, smoothed
    # area averaging when shrinking (anti-aliased), bilinear when enlarging (smooth)
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR  # int cv2 flag
    src = cv2.resize(src, (sw, sh), interpolation=interp)  # (sh, sw) float32
    if src.ndim == 1:
        src = src.reshape(sh, sw)
    th = math.radians(rotation_deg)  # float rad
    c, s_ = math.cos(th), math.sin(th)
    ra = math.sqrt(aspect)  # float, x stretch (y gets 1 / ra)
    # (2, 2) float64 linear map, centred source coords -> centred affine coords
    lin = (np.array([[c, s_], [-s_, c]]) @ np.array([[1.0, shear], [0.0, 1.0]])
           @ np.diag([ra, 1.0 / ra]))
    # corners of the source box (pixel edges at -0.5 .. n - 0.5), centred: (4, 2) float64
    corners = np.array([[-sw, -sh], [sw, -sh], [-sw, sh], [sw, sh]], dtype=np.float64) / 2.0
    ac = corners @ lin.T  # (4, 2) float64, affine-mapped corners
    # (2,) float64 bottom row of H (without the 1): [px, py] / s
    hb = np.array([float(perspective[0]), float(perspective[1])]) / float(size)
    den_min = 1.0 + float((ac @ hb).min())  # float, smallest denominator (linear -> at a corner)
    if den_min <= _PERSP_MIN_DEN:
        hb *= (1.0 - _PERSP_CLAMP_DEN) / (1.0 - den_min)  # clamp: new minimum = 0.1
    pc = ac / (1.0 + ac @ hb)[:, None]  # (4, 2) float64, projected corners
    lo, hi = pc.min(axis=0), pc.max(axis=0)  # (2,) float px bbox of the quad
    ctr, ext = (lo + hi) / 2.0, (hi - lo) / 2.0  # (2,) float px, bbox centre / half extents
    amp = 0.0 if warp is None else float(np.abs(warp).max())  # float px, max displacement
    margin = int(math.ceil(amp)) + int(math.ceil(3 * edge_soft)) + 1  # int px border
    cap = _MAX_SIDE_FACTOR * int(size)  # int px, max output side
    ow = min(cap, int(math.ceil(2 * ext[0] - 1e-6)) + 2 * margin)  # int px, output width
    oh = min(cap, int(math.ceil(2 * ext[1] - 1e-6)) + 2 * margin)  # int px, output height
    gy, gx = np.mgrid[:oh, :ow].astype(np.float32)  # (oh, ow) float32 output pixel grids
    if warp is not None and amp > 0:
        dx = np.clip(cv2.resize(np.asarray(warp[0], np.float32), (ow, oh),
                                interpolation=cv2.INTER_CUBIC), -amp, amp)  # (oh, ow) float32
        dy = np.clip(cv2.resize(np.asarray(warp[1], np.float32), (ow, oh),
                                interpolation=cv2.INTER_CUBIC), -amp, amp)  # (oh, ow) float32
        gx, gy = gx + dx.reshape(oh, ow), gy + dy.reshape(oh, ow)
    # (oh, ow) float32, centred projected coords of every (warped) output pixel
    qx = gx - np.float32((ow - 1) / 2.0 - ctr[0])
    qy = gy - np.float32((oh - 1) / 2.0 - ctr[1])
    persp_on = bool(hb.any())  # bool, a keystone is applied
    if persp_on:
        den = 1.0 - np.float32(hb[0]) * qx - np.float32(hb[1]) * qy  # (oh, ow) float32, H^-1 den.
        ok = den > 1e-6  # (oh, ow) bool, on the source side of the horizon
        den = np.where(ok, den, np.float32(1.0))
        qx, qy = qx / den, qy / den  # (oh, ow) float32, centred affine coords
    inv = np.linalg.inv(lin)  # (2, 2) float64, affine -> source
    lim = float(4 * (sw + sh) + 16)  # float px, far outside the source (keeps maps finite)
    map_x = np.clip(inv[0, 0] * qx + inv[0, 1] * qy + (sw - 1) / 2.0,
                    -lim, lim).astype(np.float32)  # (oh, ow) float32 source x
    map_y = np.clip(inv[1, 0] * qx + inv[1, 1] * qy + (sh - 1) / 2.0,
                    -lim, lim).astype(np.float32)  # (oh, ow) float32 source y
    if persp_on:
        map_x[~ok] = -lim  # beyond the horizon -> outside the source
    out = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0)  # (oh, ow) float32
    m = (out >= 0.5).astype(np.float32)  # (oh, ow) float32, binary shape
    if edge_soft > 0:
        m_img = Image.fromarray((m * 255).astype(np.uint8))  # "L" (oh, ow)
        m = np.asarray(m_img.filter(ImageFilter.GaussianBlur(radius=edge_soft)),
                       dtype=np.float32) / 255.0
    ys, xs = np.nonzero(m > 0.01)
    if len(ys) == 0:
        return np.zeros((1, 1), dtype=np.float32)
    return np.clip(m[ys.min():ys.max() + 1, xs.min():xs.max() + 1], 0, 1)


def _glyph_shape(rng, nrng, sec, rec=None):
    """Shape mask of one effect: a random glyph in a random font, randomly transformed.

    rng / nrng: the section's random.Random / np.random.Generator; sec:
    reflections / glares config section (dict); rec: optional dict, filled
    with this item's drawn values (char, font, size, rotation, aspect, shear,
    P, px, py, warp_strength, g) -- recording adds no draws.  Draws, in order, from rng:
    1 glyph index, 2 font index, 3 size s, 4 rotation, 5 aspect (log-uniform),
    6 shear, 7 P ~ U(perspective_range), 8 px ~ U(-P, P), 9 py ~ U(-P, P),
    10 strength ~ U(warp_strength_range), 11 g ~ randint(warp_grid_range);
    then from nrng always exactly 2*G*G values ~ U(-1, 1), G = warp_grid_range[1],
    scaled by A = strength * s (drawn even when A = 0), so the draw count per
    item is fixed for a given config.  The g x g grid takes the first 2*g*g
    values (dx then dy).  Returns (h', w') float32 in 0-1.
    """
    fonts = glyph_fonts()  # tuple[str, ...]
    ch = GLYPHS[rng.randrange(len(GLYPHS))]  # str, the glyph
    fi = rng.randrange(len(fonts))  # int, font index
    size = rng.randint(*sec["size_range"])  # int px, longest side s
    rot = rng.uniform(*sec["rotation_range"])  # float deg
    lo, hi = sec["aspect_range"]
    aspect = math.exp(rng.uniform(math.log(lo), math.log(hi)))  # float, log-uniform stretch
    shear = rng.uniform(*sec["shear_range"])  # float shear factor
    P = rng.uniform(*sec["perspective_range"])  # float, this item's max keystone strength
    persp = (rng.uniform(-P, P), rng.uniform(-P, P))  # (px, py) float
    strength = rng.uniform(*sec["warp_strength_range"])  # float, warp amplitude / size
    g = rng.randint(*sec["warp_grid_range"])  # int, this item's warp control grid side
    gmax = int(sec["warp_grid_range"][1])  # int, G: the upper end, fixes the nrng draw count
    amp = strength * size  # float px, warp amplitude A
    noise = nrng.uniform(-1.0, 1.0, 2 * gmax * gmax)  # (2*G*G,) float64, always fully drawn
    warp = (amp * noise[:2 * g * g].reshape(2, g, g)) if amp > 0 else None  # (2, g, g) float px
    if rec is not None:
        rec.update(char=ch, font=fonts[fi], size=size, rotation=rot, aspect=aspect,
                   shear=shear, P=P, px=persp[0], py=persp[1], warp_strength=strength, g=g)
    return transform_shape(glyph_silhouette(ch, fi), size, rot, aspect, shear, warp,
                           float(sec["edge_soft"]), persp)


def _place_shape(rng, m, canvas_size, bcx, bcy, b_radius, inside_outer, rec=None):
    """Paste shape m at a random centre uniform in the disc of boundary B.

    m: (h, w) float32 mask; bcx, bcy: B centre (int px, canvas);
    b_radius: float px; inside_outer: (C, C) bool, inside B; rec: optional
    dict, gets center_x / center_y (float px, canvas coords).  Draws radius
    then angle (polar, sqrt for uniform area).  Returns a (C, C) float32
    canvas mask, cropped at the canvas edges and zero outside B.
    """
    rr = b_radius * math.sqrt(rng.random())  # float px, centre distance from B centre
    th = rng.uniform(0, 2 * math.pi)  # float rad, centre direction
    if rec is not None:
        rec.update(center_x=bcx + rr * math.cos(th), center_y=bcy + rr * math.sin(th))
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


def _glass_effects(img, cfg, rngs, inside_outer, bcx, bcy, b_radius, tip, xs, ys,
                   rec=None):
    """Step 13c: reflections (out = in - sum darken*m), then glares
    (out = lerp(out, V, m), glare zeroed within clear_distance of the tip).

    img: canvas "L" (C, C); rngs: section_rngs dict; tip: visible tip (x, y)
    float px in canvas coords; xs (1, C) / ys (C, 1) int pixel grids.
    Shapes are transformed font glyphs (_glyph_shape).  rec: optional dict,
    gets rec["reflections"] / rec["glares"] = {active bool, count_drawn int,
    n int (items rendered), items list[dict] (one _glyph_shape record +
    center_x/center_y canvas px + darken / value int each)}.
    Returns the canvas as an "L" image (C, C).
    """
    canvas_size = img.size[0]  # int px, C
    plan = []  # list of (name, section dict, random.Random, item count int)
    # per-section records, dict[str, dict] (filled below; discarded when rec is None)
    recs = {}
    for name in ("reflections", "glares"):
        sec, rng = cfg[name], rngs[name][0]
        active = rng.random() < sec["prob"]  # bool, fixed-count draw 1
        n = rng.randint(*sec["count_range"])  # int, fixed-count draw 2
        plan.append((name, sec, rng, n if active else 0))
        recs[name] = {"active": active, "count_drawn": n, "n": n if active else 0, "items": []}
    if rec is not None:
        rec.update(recs)
    if not any(n for *_, n in plan):
        return img
    work = np.asarray(img, dtype=np.float32)  # (C, C) float32 working canvas
    (_, R, rrng, n_r), (_, Gl, grng, n_g) = plan
    rnrng, gnrng = rngs["reflections"][1], rngs["glares"][1]  # np.random.Generator per section
    dark = np.zeros_like(work)  # (C, C) float32, summed reflection darkening
    for _ in range(n_r):
        it = {}  # dict, this item's record
        m = _glyph_shape(rrng, rnrng, R, it)
        delta = rrng.randint(*R["darken_range"])  # int gray levels subtracted
        it["darken"] = delta
        dark += delta * _place_shape(rrng, m, canvas_size, bcx, bcy, b_radius, inside_outer, it)
        recs["reflections"]["items"].append(it)
    work = np.clip(work - dark, 0, 255)
    if n_g:
        clear = float(Gl["clear_distance"])  # float px
        # (C, C) bool, pixels a glare may never cover (around the visible tip)
        near_tip = (xs - tip[0]) ** 2 + (ys - tip[1]) ** 2 <= clear ** 2
        for _ in range(n_g):
            it = {}  # dict, this item's record
            m = _glyph_shape(grng, gnrng, Gl, it)
            v = grng.randint(*Gl["value_range"])  # int gray level of the glare
            it["value"] = v
            mc = _place_shape(grng, m, canvas_size, bcx, bcy, b_radius, inside_outer, it)
            recs["glares"]["items"].append(it)
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
    # sampled values of this image, dict[str, dict]: section -> key -> value
    # (raw; made JSON-plain by _plain at the end).  Recording only, no draws.
    rec = {
        "angle": {"angle_deg": angle_deg},
        "geometry": {"arc_gap_start_deg": gap_start_deg, "arc_gap_span_deg": gap_span_deg,
                     "scale": scale, "dial_radius": dial_radius,
                     "displacement_mag": disp_mag, "displacement_angle_rad": disp_ang,
                     "displacement_x": disp_x, "displacement_y": disp_y,
                     "hub_radius": hub_radius, "chord_half": chord_half,
                     "chord_dist": chord_dist, "tip_dist": tip_dist,
                     "ptr_displacement_mag": ptr_disp_mag,
                     "ptr_displacement_angle_rad": ptr_disp_ang,
                     "hole_radius": hole_r, "hole_displacement_mag": hd_mag,
                     "hole_displacement_angle_rad": hd_ang,
                     "stretch_h": stretch_h, "stretch_v": stretch_v},
        "outer": {"scale": outer_scale},
    }
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
    rec["palette"] = {"bg_gray": bg_gray, "circle_gray": circle_gray,
                      "outer_gray": outer_gray, "rim_gray": rim_gray}
    rec["geometry"].update(canvas_size=canvas_size)

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
    # background summary (the draws are arrays; their range / mean is what shows)
    rec["background"] = {"lowfreq_min": int(lf.min()), "lowfreq_max": int(lf.max()),
                         "lowfreq_mean": float(lf.mean())}

    # --- 2 faint dial circle with arc gap ---
    dial_box = [cx - dial_radius, cy - dial_radius, cx + dial_radius, cy + dial_radius]
    _arc_or_circle(ImageDraw.Draw(img), dial_box, gap_start_deg, gap_span_deg, circle_gray, 1)

    # --- 3 ticks on a separate overlay (digits stream decides visibility) ---
    D, drng = cfg["digits"], rngs["digits"][0]
    show_digits = drng.random() < D["show_prob"]  # bool, consumed by ticks too
    font_size = drng.randint(*D["font_size_range"])  # int px
    td_value, td_alpha = _new_layer(canvas_size)  # overlay value/alpha, "L" (C, C)
    tick_inner_r = None
    rec["digits"] = {"show": show_digits, "font_size": font_size}
    rec["ticks"] = {"drawn": show_digits}
    if show_digits:
        tick_inner_r = draw_ticks(td_value, td_alpha, cx, cy, dial_radius,
                                  cfg["ticks"], rngs["ticks"][0], rec["ticks"])

    # --- 4/5 shadows (shadows stream; all fixed-count draws up front) ---
    S, srng = cfg["shadows"], rngs["shadows"][0]
    has_shadow = srng.random() < S["prob"]  # bool
    light_angle = srng.uniform(0, 360)  # deg, 0 = up, CW
    s_p, s_r = _shadow_strengths(S, srng)  # float pointer / rim strengths, 0-1
    rim_offset = srng.uniform(*S["rim_offset_frac"])  # float, crescent offset / dial radius
    rim_cutout = srng.uniform(*S["rim_cutout_frac"])  # float, cutout radius / dial radius
    rim_blur = srng.uniform(*S["rim_blur_range"])  # float px, rim mask blur
    shd_dist = srng.uniform(*S["offset_range"])  # float px, pointer shadow offset
    rec["shadows"] = {"active": has_shadow, "light_angle": light_angle, "s_p": s_p, "s_r": s_r,
                      "rim_offset_frac": rim_offset, "rim_cutout_frac": rim_cutout,
                      "rim_blur": rim_blur, "offset": shd_dist}
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
                          D, drng, font_size, rec["digits"])

    # --- 7 mounting hole + pointer (pointer stream) ---
    PT = cfg["pointer"]
    ptr_rng, ptr_np = rngs["pointer"]
    ptr_gray = _pick_pointer_gray(ptr_rng, *PT["gray_range"], bg_gray,
                                  PT["min_bg_contrast"])  # int body gray
    hole_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    edge_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    grad_strength = ptr_rng.uniform(*PT["gradient_range"])  # float
    edge_thickness = ptr_rng.randint(*PT["edge_thickness_range"])  # int px
    rec["pointer"] = {"gray": ptr_gray, "hole_gray": hole_gray, "edge_gray": edge_gray,
                      "gradient": grad_strength, "edge_thickness": edge_thickness}

    # hole drawn BEFORE the pre-pointer snapshot so it survives the cut-tip restore
    ImageDraw.Draw(img).ellipse([hole_cx - hole_r, hole_cy - hole_r,
                                 hole_cx + hole_r, hole_cy + hole_r], fill=hole_gray)

    C = cfg["cut_tip"]
    cut_r = rngs["cut_tip"][0].randint(*C["radius_range"])  # int px
    rec["cut_tip"] = {"enabled": bool(C["enabled"]), "cut_r": cut_r}
    bg_pre_ptr = img.copy() if C["enabled"] else None  # pre-pointer canvas, "L"
    img, apex, base_l, base_r, ptr_mask, ptr_vis = draw_pointer(
        img, ptr_cx, ptr_cy, angle_deg, hub_radius, tip_dist, chord_half,
        chord_dist, ptr_gray, edge_gray, grad_strength, PT, ptr_np,
        edge_thickness, cut_r if C["enabled"] else None)
    if bg_pre_ptr is not None:
        img.paste(bg_pre_ptr, mask=Image.fromarray(255 - np.array(ptr_mask)))
    # label inputs; the label itself is found after step 10 (it must see the rim)
    ideal_apex = apex  # (x, y) float px, ideal triangle apex (canvas, pre-stretch)
    # the label axis is the pointer triangle's HEIGHT (apex -> base-chord
    # midpoint) as drawn: ImageDraw floors float polygon vertices, so the drawn
    # triangle's apex is floor(apex) and its base corners floor(base_l / _r);
    # (x, y) float px, canvas, pre-stretch
    apex_drawn = (float(math.floor(apex[0])), float(math.floor(apex[1])))
    base_mid_drawn = ((math.floor(base_l[0]) + math.floor(base_r[0])) / 2.0,
                      (math.floor(base_l[1]) + math.floor(base_r[1])) / 2.0)

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
    dial_rx, dial_ry = dial_radius * stretch_h, dial_radius * stretch_v  # float px

    # --- 9 exposure jitter ---
    J, jrng = cfg["jitter"], rngs["jitter"][0]
    # enhance factors (float), drawn in this order: brightness, contrast, sharpness
    j_bright = jrng.uniform(*J["brightness_range"])
    img = ImageEnhance.Brightness(img).enhance(j_bright)
    j_contrast = jrng.uniform(*J["contrast_range"])
    img = ImageEnhance.Contrast(img).enhance(j_contrast)
    j_sharp = jrng.uniform(*J["sharpness_range"])
    img = ImageEnhance.Sharpness(img).enhance(j_sharp)
    rec["jitter"] = {"brightness": j_bright, "contrast": j_contrast, "sharpness": j_sharp}

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

    # label = visible tip: the visible end of the pointer on its axis, after the
    # background-blended edge ring 0, the cut and the rim band / outer fill that
    # paint over it (raster-exact walk back from the drawn apex along -u,
    # u the unit stretched height direction), stretched like the image
    covered = r_norm > 1 - rim_half  # (C, C) bool, rim band + outer fill + outside
    # visible weight through the same bicubic stretch as the canvas: the image
    # is bg + w * (pointer - bg) with w resampled exactly like it, (C, C) float32
    vis_s = np.asarray(Image.fromarray(np.round(ptr_vis * 255).astype(np.uint8)).transform(
        size2, Image.AFFINE, affine, resample=Image.BICUBIC), dtype=np.float32) / 255.0
    # Image.AFFINE maps pixel-corner coords, so with pixel centres at integer
    # coords the stretch is about (cx - 0.5, cy - 0.5): point (x, y) -> sxy(x, y)
    def sxy(x, y):
        """Canvas point (x, y) float px -> where the stretch puts it, (x, y) float px."""
        return (cx - 0.5 + (x - cx + 0.5) * stretch_h, cy - 0.5 + (y - cy + 0.5) * stretch_v)
    ideal_apex = sxy(*ideal_apex)  # (x, y) float px, stretched ideal apex (reference)
    axis_apex = sxy(*apex_drawn)  # (x, y) float px, stretched drawn apex: the scan origin
    # stretched base-chord midpoint, (x, y) float px.  The stretch is affine, so
    # the label follows the stretched image of the original height (apex ->
    # base midpoint); after a non-uniform stretch the triangle is no longer
    # isosceles and that line differs slightly from its perpendicular altitude
    axis_base_mid = sxy(*base_mid_drawn)
    d_len = math.hypot(apex_drawn[0] - base_mid_drawn[0],
                       apex_drawn[1] - base_mid_drawn[1])  # float px, height length
    u_d = ((apex_drawn[0] - base_mid_drawn[0]) / d_len,
           (apex_drawn[1] - base_mid_drawn[1]) / d_len)  # (ux, uy) float, unit height, pre-stretch
    v_s = (u_d[0] * stretch_h, u_d[1] * stretch_v)  # stretched axis direction
    v_n = math.hypot(*v_s)  # float, its length (stretched px per pre-stretch px)
    u_s = (v_s[0] / v_n, v_s[1] / v_n)  # (ux, uy) float, unit stretched axis
    back0 = (cut_r if C["enabled"] else 0) * v_n  # float px, the old formula's recession
    vis_end = _visible_tip(vis_s, axis_apex, u_s, back0 + 20,
                           cover=covered)  # (x, y) float px visible end, or None
    apex = vis_end if vis_end is not None else (axis_apex[0] - back0 * u_s[0],
                                                axis_apex[1] - back0 * u_s[1])
    # blur-aware label (see _patch_tip): the farthest axis point whose blurred
    # contrast Cmap = G_sigma_eff * (visible weight minus cover) is >= c there
    # and on >= min_ratio of the patch around it; fallbacks contrast-only ->
    # visible end -> its fallback; never ahead of the visible end; no draws
    TL = cfg["tip_label"]
    sigma_blur = float(B["dial_circle_blur_radius"]) if B["dial_circle_blur"] else 0.0  # float px
    sigma_eff = math.hypot(sigma_blur, float(TL["base_blur"]))  # float px
    c_tip = float(TL["contrast_frac"])  # float, contrast fraction c
    w_star = tip_width_threshold(c_tip, sigma_eff)  # float px, reference bar width
    min_ratio = float(TL["min_ratio"])  # float, patch support threshold (0 = off)
    t_max = (tip_dist + chord_dist) * v_n  # float px, scan length (apex -> base chord)
    vis_cov = np.where(covered, np.float32(0), vis_s)  # (C, C) float32, weight minus cover
    field = _TipField(vis_cov, axis_apex, u_s, t_max, sigma_eff, c_tip, TL["patch_size"])
    hit_c = _patch_tip(field, axis_apex, u_s, t_max, c_tip, 0.0)  # contrast-only, or None
    hit = hit_c  # ((x, y), t) or None, the label candidate
    tip_rule = "contrast"  # str, which rule produced the label
    if hit_c is not None and min_ratio > 0:
        hit_p = _patch_tip(field, axis_apex, u_s, t_max, c_tip, min_ratio, hit_c[1])
        if hit_p is not None:
            hit, tip_rule = hit_p, "patch"
    if hit is None:
        tip_rule = "visible_end" if vis_end is not None else "fallback"
    else:
        t_end = -1.0 if vis_end is None else -(
            (vis_end[0] - axis_apex[0]) * u_s[0] + (vis_end[1] - axis_apex[1]) * u_s[1])
        if hit[1] >= t_end:
            apex = hit[0]
        else:  # ahead of the visible end: clamp onto it
            tip_rule = "clamped"
    contrast_at_tip = float(field.contrast(apex[0], apex[1]))  # float, Cmap at the label
    patch_ratio_at_tip = float(field.ratio(apex[0], apex[1]))  # float, R_P at the label
    # px between the drawn apex and the label, measured before the stretch (float)
    tip_recession = math.hypot((apex[0] - axis_apex[0]) / stretch_h,
                               (apex[1] - axis_apex[1]) / stretch_v)

    # --- 11 hierarchical noise inside the outer ring ---
    H, hnp = cfg["hier_noise"], rngs["hier_noise"][1]
    nb, bv, pv = H["blocks"], H["block_variance"], H["pixel_variance"]
    block_u8 = np.clip(hnp.integers(-bv, bv + 1, (nb, nb), dtype=np.int16) + 128,
                       0, 255).astype(np.uint8)  # (nb, nb) uint8, offset by 128
    block_up = np.array(Image.fromarray(block_u8).resize(size2, Image.BICUBIC),
                        dtype=np.int16) - 128  # (C, C) int16
    combined = block_up + hnp.integers(-pv, pv + 1, size2, dtype=np.int16)  # (C, C) int16 block + pixel noise
    blk = block_u8.astype(np.int16) - 128  # (nb, nb) int16, the drawn block offsets
    rec["hier_noise"] = {"block_min": int(blk.min()), "block_max": int(blk.max()),
                         "block_std": float(blk.std())}
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
    rec["tri_artifacts"] = {"enabled": bool(T["enabled"]), "count_drawn": n_tris}
    if not T["enabled"]:
        n_tris = 0
    tri_items = []  # list[dict], one record per triangle (gray, centre, size, placed)
    rec["tri_artifacts"].update(n=n_tris, items=tri_items)
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
        tri_items.append({"gray": tri_gray, "center_x": tcx, "center_y": tcy, "size": sz,
                          "placed": bool(valid.any())})
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
                         min(outer_rx, outer_ry), apex, xs, ys, rec)

    # --- 14 Gaussian blur inside the outer ring ---
    rec["blur"] = {"enabled": bool(B["dial_circle_blur"]),
                   "radius": B["dial_circle_blur_radius"]}
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

    # canvas -> image coords for recorded placement centres (float px)
    for name in ("reflections", "glares", "tri_artifacts"):
        for it in rec[name]["items"]:
            it["center_x"] += off_x
            it["center_y"] += off_y
    rec["derived"] = {"img_size": img_size, "canvas_offset_x": off_x, "canvas_offset_y": off_y,
                      "dial_rx": dial_rx, "dial_ry": dial_ry,
                      "outer_rx": outer_rx, "outer_ry": outer_ry,
                      "tip_x": apex[0] + off_x, "tip_y": apex[1] + off_y,
                      "ideal_apex_x": ideal_apex[0] + off_x,
                      "ideal_apex_y": ideal_apex[1] + off_y,
                      "axis_base_mid_x": axis_base_mid[0] + off_x,
                      "axis_base_mid_y": axis_base_mid[1] + off_y,
                      "axis_apex_x": axis_apex[0] + off_x, "axis_apex_y": axis_apex[1] + off_y,
                      "tip_recession": tip_recession, "sigma_eff": sigma_eff,
                      "w_star": w_star, "contrast_at_tip": contrast_at_tip,
                      "patch_ratio_at_tip": patch_ratio_at_tip, "tip_rule": tip_rule}
    # ideal apex in image px (float), label recession behind it (float px,
    # pre-stretch), blur model sigma_eff / w* (float px), Cmap and patch ratio
    # at the label (float) and the rule that produced it (str)
    rec["cut_tip"].update(ideal_apex_x=ideal_apex[0] + off_x,
                          ideal_apex_y=ideal_apex[1] + off_y, tip_recession=tip_recession,
                          sigma_eff=sigma_eff, w_star=w_star, contrast_at_tip=contrast_at_tip,
                          patch_ratio_at_tip=patch_ratio_at_tip, tip_rule=tip_rule)
    return RenderResult(image=out,
                        tip_x=apex[0] + off_x, tip_y=apex[1] + off_y,
                        angle_deg=angle_deg, params=_plain(rec))


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
