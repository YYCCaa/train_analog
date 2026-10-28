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
  │    (hub swept from the face + floating triangle, penumbra ∝ d)   │
  │    d = max(s_p*m_p, s_r*m_r)  float32 darkening  [d layer]       │
  │    (built at 7, s_p / s_r as drawn; the pointer gray avoids them)│
  │ 6  digits -> overlay                                  [td layer] │
  │ 7  mounting hole, pointer body (gradient + noise), pointer edge, │
  │    cut-tip (restore pre-pointer pixels near the apex)            │
  │ 8  stretch h/v (canvas + overlay + d + apex)                     │
  │ 9  brightness -> contrast -> sharpness jitter, each factor       │
  │    capped per image so no surviving pixel saturates              │
  │ 10 outer annulus fill, rim band                                  │
  │ 11 hierarchical (block + pixel) noise inside outer ring          │
  │ 12 triangle artifacts, every pixel > d = clear_distance from the │
  │    pointer + hub + hole: centre from B minus that zone Z_d; one  │
  │    reaching Z_d is re-centred whole (<= 64 tries, then trimmed)  │
  │ 13 black outside outer ring; erase overlay over pointer;         │
  │    composite overlay; apply shadow out = round(in * (1 - d))     │
  │    (d zeroed outside the dial face, on the rim, pointer, hole)   │
  │ 13c cover glass, inside B: m_k = transformed font glyphs;        │
  │    reflections out = in - sum(dk * m_k),                         │
  │    then glares out = lerp(out, V, m) one by one (glare wins);    │
  │    glare m zeroed within clear_distance of the tip label         │
  │ 14 Gaussian blur inside outer ring; black outside again          │
  │ 15 place canvas on a black img_size^2 at centred offset + d;     │
  │    boundary B stays centred (everything inside B shifts by d)    │
  └──────────────────────────────────────────────────────────────────┘
           v
    RenderResult(image (256, 256) uint8, tip_x, tip_y, angle_deg)

The pointer: a hub circle + an isosceles triangle whose base is a chord
of the hub, apex pointing outward.  Pixel coordinates: pixel i's centre at i.
The triangle is our own hard raster (raster_pointer_triangle), not Pillow's
polygon, so a thin tip never forms a "ball":

          A ●            per scan line (rows, or columns for a near-
           ╱·╲             horizontal pointer): w = exact width (float),
          ╱ ▓ ╲  w 1.2      n = floor(w + 0.5) px, run centred on the
         ╱ ▓▓  ╲ w 1.9      true midline (start = floor(c - (n-1)/2 + 0.5))
        ╱  ▓▓   ╲ w 2.3   -> n never shrinks going away from A, |n - w| <= 0.5,
       ╱  ▓▓▓    ╲ w 2.7     no pixel where w < 0.5

Tip label (tip_count_label): a pixel count on the triangle actually DRAWN.

    1  triangle: the EXACT float apex A and base corners B_l, B_r (the
       pointer raster, raster_pointer_triangle, draws from them unrounded:
       per scan line round(width) px centred on the true midline);
       M = (B_l + B_r) / 2 is the base midpoint, A -> M the isosceles height
    2  stretch: the same affine as the image.  Image.AFFINE works in pixel-corner
       coords, so with pixel centres at integer coords the stretch is about
       (cx - 0.5, cy - 0.5): A, B_l, B_r, M -> A', B_l', B_r', M'.  Lines parallel
       to the base stay parallel; after a non-uniform stretch A'M' is the image of
       the height, not the stretched triangle's perpendicular altitude
    3  pointer pixels, on the stretched canvas, geometry only (before noise,
       blur and effects): the pre-stretch set (cut mask, minus the edge-ring
       pixels painted pure background; the body is painted through the mask,
       so nothing else) stretched BICUBIC like the image and thresholded >= 128, minus the
       rim band / outer fill (`covered`).  Hub-disc pixels count too.
    4  scan t on a 0.05 px grid in [0, |M' - A'|]:

            A'  ●            height' = A' -> M'
               ╱ ╲
          t ─ ●───●─ ─ ─  L(t) through P(t) = A' + t (M' - A') / |M' - A'|, parallel to B_l'B_r'
             ╱  ●  ╲       count(t) = # pointer pixels whose centre lies in the 1-px band
            ╱       ╲                 -0.5 < signed distance to L(t) <= 0.5
     B_l' ●─────M'────● B_r'

       label = P(t*), t* = the smallest t with count(t) > n = tip_label.min_pixels
       (no such t: M', tip_fallback = True)

The label lies on A'M' exactly; a larger n never moves it toward the apex.
Example, n = 2, cut tip on (o = pointer pixel, . = erased / background):

          . . . . .          t = 0 .. 7: count 0 (cut disc)
        o . . . . . o        t ~ 7.4: count 2 (the cut arc's two horns) -> not > 2
       o o o o o o o o   <-  t ~ 7.6: count 8 > 2 -> label on the height here

Then the integer canvas offset is added.  params["derived"] records
axis_apex_x/y (A'), axis_base_mid_x/y (M'), axis_base_dir_x/y (unit B_l' -> B_r'),
tip_t (t*, px along A'M'), tip_count (count(t*)), tip_fallback, tip_recession
(= t* measured before the stretch: t* |M - A| / |M' - A'|) and ideal_apex_x/y
(the ideal float apex, for reference); RenderResult.tip_axis holds A', M' and
the base direction unrounded.

Edge clamp (pointer.edge_max_frac = f, edge_thickness_map): the edge band of
thickness t (ring 0 blends to the background, inner rings to edge gray) is
narrowed on the triangle so a thin end is not eaten by its own band:

          A ●   W(h) = |B_r - B_l| h / |M - A|, h = projection on the height
           ╱█╲       t_eff = min(t, floor(f W(h)))
          ╱███╲      ring k painted iff k < t_eff, blend k / max(t_eff - 1, 1)
         ╱░███░╲     (█ body, ░ band; near A, f W < 1: no band, body to the edge)
        ╱░░███░░╲    hub-disc pixels keep t; f >= 1 skips the clamp (old pixels)
    B_l ●───M────● B_r

The edge noise is drawn for the unclamped band, so f never shifts a stream.

Shadows are physical: a shadow blocks the direct light and keeps only the
ambient light, so it scales the surface by an albedo-independent factor,
out = in * (1 - d).  One shared strength s per image, jittered separately
for the pointer (s_p) and rim (s_r) shadows; overlapping shadows of the
one light do not compound (max, not product).  Ticks, digits, noise and
triangles under a shadow are darkened by the same factor; the pointer, hub
and hole sit above the face and are never darkened.

The pointer shadow is modelled (pointer_shadow_shapes / pointer_shadow_mask):
the hub is a cylinder standing ON the face, the triangle plate sits on its
top, and the light casts every point at height h by h cot(elev) along u (the
unit vector away from the light, from light_angle).  One draw
distance_frac ~ U(shadows.distance_frac_range) gives d = distance_frac *
dial_radius, the plate's cast distance:

     side view                              top view (light from upper-left)
       light                                   ╭───╮
         ╲   plate (height z)                   │hub│▒▒╮   hub: disc swept along u from
          ╲ ▬▬▬▬▬▬▬▬▬▬▬▬▬                       ╰───╯▒▒│   0 to d (a stadium), attached
           ╲  ┃hub┃                               ╰▒▒▒▒╯   to the hub; the part under the
      ══════╲═┻━━━┻═══ face                                hub is hidden -> far-side crescent
             ●──d──►    d = z cot(elev)            triangle: shifted by d u (floating plate):
                                                   small d -> a sliver beside the pointer,
                                                   large d -> the whole triangle, separated

Penumbra: sigma = max(blur_min, blur_per_px * cast distance).  Triangle and
the shifted dial-circle arc (SHADOW_ARC_LEVEL): blur_per_px * d.  Hub: K =
SHADOW_HUB_SLICES slices of the sweep, slice k blurred with t_k d, t_k =
(k + 0.5) / K, so the edge is sharp at the hub and soft at the far end.
m_p = max of the blurred components.  Shadows keep their
drawn strengths; instead pointer.min_bg_contrast (c) restrains the POINTER gray,
which is drawn after the bg gray and the shadow strengths (separate streams,
drawn earlier, so no draw order changes):

    the two face levels the pointer sits on: the bare face bg and the darkest
    shadowed face sh = bg (1 - s_max),  s_max = max(s_p, s_r)  (0 without a
    shadow: sh == bg, the windows coincide)

    0 ...[sh-c ██ sh+c]..... allowed .....[bg-c ██ bg+c]... 255   total <= 4c
     darker than both      between the faces             lighter than both
    allowed: |g - sh| >= c and |g - bg| >= c (window ends allowed; overlapping
    windows merge into one)
    ptr_gray = one randint over [lo, hi] minus both windows  (_pick_pointer_gray)

(bg_gray_range draw; jitter and noise make this approximate, as for the bare
rule).  params["pointer"]["forbidden"] records the merged windows.

Pointer palette: no value is clamped at 0 / 255.  The gray draw keeps headroom
for the lengthwise gradient and the noise, and the variation is scaled down
continuously when even that is not enough (gray_range near 0 / 255):

    H   = max|gradient_range| * along_max + noise_amplitude   (pointer_headroom;
          along_max = (max(tip, hub) + 2) / tip, this image's geometry)
    I   = [lo, hi] ∩ [ceil H, floor(255 - H)]   (empty: I = [lo, hi])
    g   = one randint over I minus the forbidden windows       (_pick_pointer_gray)

    0 ─── H ───[════════ I: g lands here ════════]─── H ─── 255
                 g - H_img ◄── body ──► g + H_img

    body px = g + rint(k (grad_shift + noise)),  k = min(1, min(g, 255 - g) / H_img)
    edge px = v + k_e noise,  v = bg + blend (edge - bg),  k_e = min(1, min(v, 255 - v) / amp)

H_img is this image's max |grad_shift| + noise_amplitude.  k = k_e = 1 leaves
the old sums bit for bit; the noise arrays are drawn as before (same size,
dtype) and only multiplied.  params["pointer"] records headroom (H),
headroom_ok (I is the restricted interval) and amp_scale (k).

Exposure jitter (step 9): PIL's enhance is out = trunc(D + f (img - D)),
saturated at 0 / 255, with D = 0 (brightness), the int-rounded mean of the
whole canvas (contrast) or img.filter(SMOOTH) (sharpness).  Before each
enhance, on the CURRENT canvas, over the pixels that survive to the output
(inside the dial ellipse, off the rim band):

    f_max = the largest f with D + f (img - D) in [-0.5, 255.5]  (> 1 always;
            f <= 1 blends between img and D, so only the upper side can saturate)

         lo ═══════════════ hi                 f = U(lo, hi)            f_max >= hi
         lo ══════ f_max ┄┄ hi                 f = U(lo, f_max)         lo <= f_max < hi
    f_max lo/hi ══ f_max    lo ┄┄┄┄┄ hi        f = U(f_max lo/hi, f_max) f_max < lo

Each is one random() (random.uniform), so the jitter stream's draws are
unchanged.  params["jitter"] records the factors, <kind>_max (None when
unbounded) and capped (dict of bool per kind: the range was shortened).

Triangle artifacts (step 12), d = tri_artifacts.clear_distance.  P = the
stretched (cut) pointer mask incl. the hub disc | the stretched hole ellipse;
Z_d = the pixels whose centre lies within d (Euclidean, pixel centres) of a P
pixel (clear_zone; Z_0 = P); V = inside B & ~Z_d, the centre domain and the
paint clip.  Per triangle the draws are, in order: gray, centre (ONE draw from
V, row-major flat index randrange(|V|)), size, 3 vertices (angle, distance);
the triangle is translated so its centroid sits on the centre, which is always
inside it.  With d > 0 the WHOLE raster must clear Z_d, else only the centre
is redrawn (one more randrange, same shape) and the triangle moved there:

        try 1: reaches Z_d                 try k: clears Z_d, painted whole
       ┌──────────────────────────┐        ┌──────────────────────────┐
       │    ░░░░░░░░░░            │        │    ░░░░░░░░░░            │
       │  ░░░░  ▲  ░░░░           │        │  ░░░░  ▲  ░░░░      ▲    │
       │  ░░░  ╱█╲  ░░░ ▲         │        │  ░░░  ╱█╲  ░░░     ╱c╲   │
       │  ░░  ╱███╲  ░░╱c╲        │   ──►  │  ░░  ╱███╲  ░░    ╱───╲  │
       │  ░░░ (hub)  ░╱───╲       │        │  ░░░ (hub)  ░░░          │
       │    ░░░░░░░░░░            │        │    ░░░░░░░░░░            │
       └──────────────────────────┘        └──────────────────────────┘
        █ P   ░ Z_d minus P (d px wide)   c ∈ V = B minus Z_d, centroid G := c

    try 1 .. 64: raster ∩ Z_d = ∅ -> painted = raster ∩ B (whole, trimmed False)
    all 64 reach Z_d             -> painted = 64th raster ∩ V (trimmed True)
    (painted empty, sub-pixel: the centre pixel alone, which is in V)

d = 0 skips the fit check: no redraws, the triangles are clipped at P, the old
placement bit for bit.  The redraws come after the vertex draws, so only the
tri stream changes (later triangles' draws shift); every other section is
untouched.  Every triangle is placed (items[i].placed True, pixels >= 1),
except when V is empty (Z_d covers B): then no centre is drawn, nothing is
painted (placed False, pixels 0, tries 0) and its gray / size / vertex draws
are still made.  V depends on where B sits, so triangles do not follow the
displacement as a pure translation.  The per-triangle numpy noise is drawn for
every triangle.

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
                   rim_cutout_frac, rim_blur, distance_frac, distance (px,
                   = distance_frac * dial_radius), blur_triangle (sigma px),
                   blur_hub_far (sigma px of the last hub slice)
    pointer        gray, hole_gray, edge_gray, gradient, edge_thickness,
                   forbidden (list of [lo, hi] float pairs: the merged open
                   gray windows sh +- c / bg +- c it avoided; [] when
                   min_bg_contrast = 0), gray_fallback (bool:
                   the allowed set was empty, farthest endpoint used), headroom
                   (H), headroom_ok (bool), amp_scale (k)
    cut_tip        enabled, cut_r, ideal_apex_x / _y, tip_recession
    jitter         brightness, contrast, sharpness, brightness_max,
                   contrast_max, sharpness_max (cap, None = unbounded),
                   capped {brightness, contrast, sharpness} (bool)
    hier_noise     block_min, block_max, block_std (of the block offsets)
    tri_artifacts  enabled, count_drawn, n, items[n] {gray, center_x,
                   center_y, size, placed (False only when V is empty),
                   pixels (painted), tries (centre draws: 1 at d = 0, <= 64,
                   0 when V is empty), trimmed (bool: all 64 tries reached
                   Z_d, the last one painted trimmed), verts (3 [x, y], the
                   triangle's vertices at its final centre)}
    reflections    active, count_drawn, n, items[n] {char, font, size,
                   rotation, aspect, shear, P, px, py, warp_strength, g,
                   center_x, center_y, darken}
    glares         same as reflections with value instead of darken
    blur           enabled, radius
    derived        img_size, canvas_offset_x / _y, dial_rx, dial_ry,
                   outer_rx, outer_ry, tip_x, tip_y, ideal_apex_x / _y,
                   axis_apex_x / _y, axis_base_mid_x / _y, axis_base_dir_x / _y,
                   tip_t, tip_count, tip_fallback, tip_recession
"""

import argparse
import contextlib
import csv
import datetime
import glob
import json
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
SHADOW_HUB_SLICES = 8  # K (int): slices of the hub-shadow sweep, each blurred with its own sigma
# centre draws (int) a triangle artifact gets to clear the zone Z_d (first one
# included); when all fail, the last try is drawn trimmed at Z_d
TRI_MAX_TRIES = 64

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
    tip_x: float  # tip label x, image px (float): on the triangle's height (tip_count_label)
    tip_y: float  # tip label y, image px (float)
    angle_deg: int  # pointer angle, whole degrees (0 = up, clockwise)
    # sampled per-image values, dict[str, dict]: section name -> key -> JSON-plain
    # value (int / float rounded to 4 decimals / bool / str / list); see "Params"
    # in the module docstring.  Recorded only, never drawn for.
    params: dict = field(default_factory=dict)
    # label geometry, unrounded floats in image px: (A' (x, y), M' (x, y), unit
    # base direction (x, y)) of the stretched triangle; the label is
    # A' + tip_t (M' - A') / |M' - A'|.  Recorded only.
    tip_axis: tuple = None


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

def _stadium(canvas_size, x0, y0, x1, y1, radius):
    """Pixels whose centre lies within `radius` of the segment (x0, y0) -> (x1, y1).

    All args float px (canvas_size int); a zero-length segment gives a disc.
    Returns (C, C) bool.
    """
    ys, xs = np.ogrid[:canvas_size, :canvas_size]  # (C, 1), (1, C) int grids
    vx, vy = x1 - x0, y1 - y0  # float px, segment vector
    len2 = vx * vx + vy * vy  # float px^2, squared segment length
    if len2 <= 1e-12:
        t = 0.0  # float: degenerate segment, nearest point is the start
    else:
        # (C, C) float64 in 0-1: projection of each pixel centre onto the segment
        t = np.clip(((xs - x0) * vx + (ys - y0) * vy) / len2, 0.0, 1.0)
    dx, dy = xs - (x0 + t * vx), ys - (y0 + t * vy)  # (C, C) float64, offset to the nearest point
    return dx * dx + dy * dy <= radius * radius


def pointer_shadow_shapes(apex, base_left, base_right, cx, cy, hub_radius, dial_radius,
                          dist, light_rad, gap_start_deg, gap_span_deg, canvas_size,
                          slices=SHADOW_HUB_SLICES):
    """Pre-blur shape masks of the physically modelled pointer shadow.

    The hub is a cylinder standing ON the face; the triangle plate sits on its
    top.  Light from -u casts every point at height h by h cot(elev) along u;
    the plate is at the top, so it casts by the full distance d (dist):

         side view                          top view (u = away from the light)
           light                                ╭───╮
             ╲   plate (height z)               │hub│▒▒╮   hub: the disc swept along u
              ╲ ▬▬▬▬▬▬▬▬▬▬▬▬▬                   ╰───╯▒▒│   from 0 to d (a stadium); the
               ╲  ┃hub┃                           ╰▒▒▒▒╯   part under the hub is hidden
          ══════╲═┻━━━┻═══ face                           by the hub -> a crescent on
                 ●──d──►    d = z cot(elev)               the far side only
                                                triangle: shifted by d u (floating plate)

    apex, base_left, base_right : (x, y) float px, the pointer's exact vertices
    cx, cy, hub_radius          : float px, hub centre and radius
    dial_radius                 : int px; the printed dial circle is shifted too
    dist                        : float px, cast distance d (>= 0)
    light_rad                   : float rad, direction u the shadow is cast in
                                  (image coords, 0 = +x, clockwise)
    gap_start_deg, gap_span_deg : float deg, the dial circle's arc gap (PIL arc)
    canvas_size                 : int C
    slices                      : int K, hub-sweep slices

    Returns dict:
      "hub"        (C, C) bool, the whole stadium (union of the slices)
      "hub_slices" list of K (C, C) bool, slice k = the stadium segment for
                   t in [k/K, (k+1)/K] (hub disc centred at c + t d u)
      "triangle"   (C, C) bool, raster_pointer_triangle of the vertices + d u
      "arc"        (C, C) uint8, the dial-circle outline (width 2, same arc gap)
                   shifted by d u, at SHADOW_ARC_LEVEL (0 elsewhere)
    """
    ux, uy = math.cos(light_rad), math.sin(light_rad)  # float, unit cast direction u
    ox, oy = dist * ux, dist * uy  # float px, full cast offset d u
    hub_slices = []  # list of K (C, C) bool, stadium segments
    for k in range(slices):
        t0, t1 = k / slices, (k + 1) / slices  # float, slice's sweep interval
        hub_slices.append(_stadium(canvas_size, cx + t0 * ox, cy + t0 * oy,
                                   cx + t1 * ox, cy + t1 * oy, hub_radius))
    hub = np.logical_or.reduce(hub_slices)  # (C, C) bool, union stadium
    tri = [(p[0] + ox, p[1] + oy) for p in (apex, base_left, base_right)]  # 3 (x, y) float
    triangle = raster_pointer_triangle(tri[0], tri[1], tri[2], canvas_size)  # (C, C) bool
    arc_img = Image.new("L", (canvas_size, canvas_size), 0)  # "L" (C, C), shifted circle
    scx, scy = cx + ox, cy + oy  # float px, shifted hub centre (the circle's centre)
    _arc_or_circle(ImageDraw.Draw(arc_img),
                   [scx - dial_radius, scy - dial_radius, scx + dial_radius, scy + dial_radius],
                   gap_start_deg, gap_span_deg, SHADOW_ARC_LEVEL, 2)
    return {"hub": hub, "hub_slices": hub_slices, "triangle": triangle,
            "arc": np.array(arc_img, dtype=np.uint8)}


def shadow_sigmas(dist, blur_min, blur_per_px, slices=SHADOW_HUB_SLICES):
    """Penumbra widths of the pointer shadow (Gaussian sigma, float px).

    An edge at height h casts by h cot(elev) and its penumbra grows with that
    distance: sigma = max(blur_min, blur_per_px * cast distance).
      triangle / arc (all at the plate height): max(blur_min, blur_per_px d)
      hub slice k (height rises 0 -> top, t_k = (k + 0.5) / K): max(blur_min,
      blur_per_px t_k d)
    Returns (sigma_tri float, sigma_hub list of K float).
    """
    sig_tri = max(blur_min, blur_per_px * dist)  # float px
    sig_hub = [max(blur_min, blur_per_px * (k + 0.5) / slices * dist)
               for k in range(slices)]  # list of K float px
    return sig_tri, sig_hub


def _blur_mask(mask_u8, sigma):
    """Gaussian blur (PIL, radius = sigma px, float) of a (C, C) uint8 mask;
    returns (C, C) float32 in 0-1.  sigma <= 0 leaves it sharp."""
    im = Image.fromarray(mask_u8)
    if sigma > 0:
        im = im.filter(ImageFilter.GaussianBlur(radius=sigma))
    return np.asarray(im, dtype=np.float32) / 255.0


def pointer_shadow_mask(shapes, dist, blur_min, blur_per_px):
    """Blurred pointer-shadow shape m_p (0-1) from pointer_shadow_shapes.

    Every hub slice is blurred with its own sigma (sharp where the hub meets
    the face, soft at the far end); the triangle and the shifted arc together
    with the triangle's sigma (the triangle at 255 wins over the arc).  The
    components are combined with max.
    Returns (m_p (C, C) float32 0-1, sigma_tri float px, sigma_hub list of K float px).
    """
    sig_tri, sig_hub = shadow_sigmas(dist, blur_min, blur_per_px, len(shapes["hub_slices"]))
    ta = shapes["arc"].copy()  # (C, C) uint8, arc + triangle at 255
    ta[shapes["triangle"]] = 255
    m_p = _blur_mask(ta, sig_tri)  # (C, C) float32
    for sl, sg in zip(shapes["hub_slices"], sig_hub):
        np.maximum(m_p, _blur_mask(sl.astype(np.uint8) * 255, sg), out=m_p)
    return m_p, sig_tri, sig_hub


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

def _round_half_up(x):
    """Nearest int to float x, ties (x.5) rounded UP: floor(x + 0.5) (int).
    The one deterministic tie rule of the pointer raster."""
    return int(math.floor(x + 0.5))


def raster_pointer_triangle(apex, base_left, base_right, size):
    """Hard raster of the pointer triangle from its EXACT float vertices.

    apex, base_left, base_right : (x, y) float px each (pixel i's centre at i)
    size : int C (square canvas) or (width, height) tuple of int
    Returns (H, W) bool, True = triangle pixel.

    ImageDraw.polygon rounds every edge separately and fills boundary pixels,
    so on a slender triangle the per-row width wobbles (3 3 2 3 ... on a
    1.5 .. 2.7 px wide tip): a "ball" on a neck.  Here every scan line gets
    round(width) pixels centred on the true midline, so the width grows
    steadily away from the apex:

        scan lines: pixel rows when the height A -> M is closer to vertical
        (|dy| >= |dx|), else pixel columns (the same rule, transposed)

                 A                 line s (integer pixel-centre coordinate):
                 ●                   [lo, hi] = exact line ∩ triangle (float)
        s=1 ── ·[▓]· ──  w = 0.4     w = hi - lo,  c = (lo + hi) / 2
        s=2 ── ·[▓]· ──  w = 0.8 ->  n = round_half_up(w)   pixels to fill
        s=3 ── [▓▓]·  ──  w = 1.6    start = round_half_up(c - (n - 1) / 2)
        s=4 ── [▓▓]·  ──  w = 2.0    fill start .. start + n - 1
        s=5 ─ [▓▓▓]   ──  w = 2.4
               ╱  M  ╲             -> |n - w| <= 0.5, run centre within 0.5 px
          B_l ●───────● B_r           of c, n non-decreasing wherever w is,
                                      and no pixel where w < 0.5 (the tip end)

    (the s=1 row above has w < 0.5, so it is empty; drawn for the rule only)

    Ties go up (round_half_up = floor(x + 0.5)), so the raster is
    deterministic.  A plain loop over the ~100 scan lines (well under 1 ms).
    """
    if isinstance(size, int):
        width, height = size, size  # int px, canvas width / height
    else:
        width, height = int(size[0]), int(size[1])
    out = np.zeros((height, width), dtype=bool)  # (H, W) bool, the raster
    pts = [(float(apex[0]), float(apex[1])), (float(base_left[0]), float(base_left[1])),
           (float(base_right[0]), float(base_right[1]))]  # 3 (x, y) float vertices
    mx = (pts[1][0] + pts[2][0]) / 2.0  # float px, base midpoint M x
    my = (pts[1][1] + pts[2][1]) / 2.0  # float px, base midpoint M y
    rows = abs(my - pts[0][1]) >= abs(mx - pts[0][0])  # bool, scan pixel rows (else columns)
    # (s, u) per vertex: s = scan coordinate (row y or column x), u = position
    # along the scan line; 3 (float, float) tuples
    su = [(p[1], p[0]) if rows else (p[0], p[1]) for p in pts]
    n_s, n_u = (height, width) if rows else (width, height)  # int, scan-line count / line length
    s_lo = max(int(math.ceil(min(p[0] for p in su) - 1e-9)), 0)  # int, first scan line
    s_hi = min(int(math.floor(max(p[0] for p in su) + 1e-9)), n_s - 1)  # int, last scan line
    edges = [(su[0], su[1]), (su[1], su[2]), (su[2], su[0])]  # 3 vertex pairs
    for s in range(s_lo, s_hi + 1):
        us = []  # list[float], where the line meets the triangle's edges
        for (s0, u0), (s1, u1) in edges:
            if min(s0, s1) - 1e-9 <= s <= max(s0, s1) + 1e-9:
                if abs(s1 - s0) < 1e-12:  # edge along the line: both ends
                    us.extend((u0, u1))
                else:
                    us.append(u0 + (u1 - u0) * (s - s0) / (s1 - s0))
        if not us:
            continue
        lo, hi = min(us), max(us)  # float px, exact intersection [lo, hi]
        n = _round_half_up(hi - lo)  # int, pixels on this line
        if n <= 0:
            continue
        start = _round_half_up((lo + hi) / 2.0 - (n - 1) / 2.0)  # int, first pixel
        a, b = max(start, 0), min(start + n, n_u)  # int, clipped run [a, b)
        if a >= b:
            continue
        if rows:
            out[s, a:b] = True
        else:
            out[a:b, s] = True
    return out


def pointer_along_max(hub_radius, tip_dist):
    """Largest |along_norm| (float) a pointer body pixel can reach in this image.

    along_norm = along / max(tip_dist, 1) is +1 at the ideal apex and
    -hub_radius / tip_dist at the back of the hub.  Rasterised pixels can sit
    up to ~1 px beyond the ideal outline (raster_pointer_triangle rounds each
    scan line's run, PIL fills the hub ellipse), so 2 px of slack are added:
    (max(tip, hub) + 2) / max(tip, 1).
    Uses only the geometry stream's draws (hub_radius, tip_dist; float px).
    """
    return (max(tip_dist, hub_radius) + 2.0) / max(tip_dist, 1.0)


def pointer_headroom(gradient_range, along_max, noise_amp):
    """Config-level headroom H (float, gray levels) the body gray keeps from 0 / 255:

        H = max(|gradient_range[0]|, |gradient_range[1]|) * along_max + noise_amp

    It bounds |grad_shift| + |noise| for ANY gradient the pointer stream may
    still draw (grad_strength is drawn after the gray, so it is not used).
    """
    g_abs = max(abs(float(gradient_range[0])), abs(float(gradient_range[1])))  # float
    return g_abs * float(along_max) + float(noise_amp)


def _pointer_variation(img, mask, cx, cy, angle_deg, tip_dist, grad_strength,
                       noise_amp, nrng, info=None):
    """Lengthwise brightness gradient + per-pixel noise on the pointer body.

    img: "L" canvas, the body drawn flat at the pointer gray g; mask: "L"
    pointer mask; nrng: "pointer" np Generator; info: optional dict, filled
    with info["amp_scale"] = k (float).

    Continuous amplitude scale, so the body never leaves 0..255:

        H_img = max|grad_shift| + noise_amp          (this image's actual gradient)
        k     = min(1, min(g, 255 - g) / H_img)      (1 when H_img = 0)
        px    = g + rint(k * (grad_shift + noise))   (k = 1: bit-identical to the old sum)

           0 ─────── g-kH ═════ g ═════ g+kH ─────── 255
                      └── body spread, never touches 0 / 255 ──┘

    The noise array is drawn exactly as before (same size and dtype) and only
    multiplied, so the pointer stream's draw count is unchanged.  The np.clip
    stays as a safety net; it never changes a value.
    """
    arr = np.array(img).astype(np.int16)  # (C, C) int16
    ys, xs = np.where(np.array(mask) > 0)
    if len(ys) == 0:
        if info is not None:
            info["amp_scale"] = 1.0
        return img
    a = math.radians(angle_deg - 90)
    along = ((xs - cx) * math.cos(a) + (ys - cy) * math.sin(a)).astype(np.float32)
    along_norm = along / max(tip_dist, 1)  # (N,) float32, -hub/tip at the hub back .. +1 tip
    grad_shift = (along_norm * grad_strength).astype(np.int16)  # (N,) int16
    noise = nrng.integers(-noise_amp, noise_amp + 1, size=len(ys), dtype=np.int16)  # (N,) int16
    base = arr[ys, xs]  # (N,) int16, the flat body gray before variation
    # int, levels from the body gray to the nearer of 0 / 255 (the body is flat
    # g, so this is min(g, 255 - g))
    room = int(min(base.min(), 255 - base.max()))
    h_img = int(np.abs(grad_shift).max()) + int(noise_amp)  # int, this image's max |shift + noise|
    k = 1.0 if h_img <= room else room / h_img  # float in [0, 1], amplitude scale
    var = grad_shift + noise  # (N,) int16, unscaled variation
    if k < 1.0:
        var = np.rint(var.astype(np.float64) * k).astype(np.int16)  # (N,) int16, |var| <= room
    if info is not None:
        info["amp_scale"] = k
    arr[ys, xs] = np.clip(base + var, 0, 255)  # safety net only: never changes a value
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


def edge_thickness_map(ys, xs, thickness, tri, edge_max_frac):
    """Per-pixel effective edge thickness t_eff of the band pixels (ys, xs).

    ys, xs        : (N,) int, band pixel coords (pre-stretch canvas)
    thickness     : int >= 0, t (pointer edge_thickness draw)
    tri           : None, or a 4-tuple (A, B_l, B_r, in_tri): the triangle's
                    exact float vertices, (x, y) float px each (the ones
                    raster_pointer_triangle draws from), and in_tri, (C, C)
                    bool, the triangle's raster pixels
    edge_max_frac : float, pointer.edge_max_frac; >= 1 = no clamp

        A ●            h  = (p - A) . (M - A) / |M - A|   (projection on the height)
         ╱ ╲           W(h) = |B_r - B_l| h / |M - A|     (cross-section width)
      h ╱ ▒ ╲  W(h)    t_eff = min(t, floor(frac W(h)))   in the triangle, 0 <= h <= |M - A|
       ╱ ▒▒▒ ╲         t_eff = t                          hub disc (elsewhere)
    B_l ●──M──● B_r    ring k painted iff k < t_eff, blend k / max(t_eff - 1, 1)

    Returns (N,) int64 t_eff, or None when no clamp applies (frac >= 1 or no
    triangle given): then every band pixel uses t, exactly as before the clamp.
    """
    if tri is None or edge_max_frac >= 1 or len(ys) == 0:
        return None
    (ax, ay), bl, br, in_tri = tri
    mx, my = (bl[0] + br[0]) / 2.0, (bl[1] + br[1]) / 2.0  # float px, base midpoint M
    h_len = math.hypot(mx - ax, my - ay)  # float px, |M - A|
    t_eff = np.full(len(ys), int(thickness), dtype=np.int64)  # (N,) int64, hub default t
    if h_len <= 0:
        return t_eff
    hx, hy = (mx - ax) / h_len, (my - ay) / h_len  # (float, float) unit height
    h = (xs - ax) * hx + (ys - ay) * hy  # (N,) float64, projection on the height
    w = math.hypot(br[0] - bl[0], br[1] - bl[1]) * h / h_len  # (N,) float64, W(h)
    on_tri = in_tri[ys, xs] & (h >= 0) & (h <= h_len)  # (N,) bool, triangle part
    cap = np.floor(edge_max_frac * w + 1e-9).astype(np.int64)  # (N,) int64, floor(frac W)
    t_eff[on_tri] = np.minimum(int(thickness), np.maximum(cap[on_tri], 0))
    return t_eff


def _draw_pointer_edge(img, mask, edge_gray, bg_arr, thickness, noise_amp, nrng,
                       tri=None, edge_max_frac=1.0):
    """Variable-thickness edge band on the pointer mask boundary.

    The outermost erosion ring blends toward the pre-pointer background
    (bg_arr, (C, C) uint8), inner rings toward edge_gray; each pixel gets
    independent noise from nrng.  Because ring 0 is repainted with the
    background (blend = 0), the visible pointer is the mask minus ring 0.
    Thickness 0 draws no band (and no noise).

    Edge clamp (tri, edge_max_frac, see edge_thickness_map): on the triangle
    part each pixel uses t_eff = min(t, floor(frac W(h))); ring k is painted iff
    k < t_eff, with blend k / max(t_eff - 1, 1); unpainted band pixels keep the
    body.  The noise is drawn for the UNCLAMPED band (fixed draw count) and
    indexed, so the clamp never shifts the stream.  frac >= 1 skips the clamp:
    the pixels are bit-identical to the unclamped band.

    Returns (img "L" (C, C), bg_painted (C, C) bool): bg_painted marks the
    band pixels painted pure background (blend 0), which the tip label does
    not count as pointer pixels.
    """
    m = np.array(mask)  # (C, C) uint8
    arr = np.array(img)  # (C, C) uint8
    # dist[y, x] = erosion steps survived (0 = outermost ring), (C, C) float32
    dist = _erosion_dist(m, thickness)
    bg_painted = np.zeros(m.shape, dtype=bool)  # (C, C) bool, ring pixels with blend 0

    ys, xs = np.where((m > 0) & (dist < thickness))  # (N,) int, unclamped band pixels
    if len(ys) == 0:
        return img, bg_painted
    noise = nrng.integers(-noise_amp, noise_amp + 1, size=len(ys), dtype=np.int16)  # (N,) int16
    t_eff = edge_thickness_map(ys, xs, thickness, tri, edge_max_frac)  # (N,) int64 or None
    ring = dist[ys, xs]  # (N,) float32, ring index k
    if t_eff is None:
        blend = ring / max(thickness - 1, 1)  # (N,) float32, 0 = bg .. 1 = edge
    else:
        keep = ring < t_eff  # (N,) bool, painted under the clamp
        ys, xs, noise, ring, t_eff = ys[keep], xs[keep], noise[keep], ring[keep], t_eff[keep]
        blend = (ring / np.maximum(t_eff - 1, 1)).astype(np.float32)  # (N',) float32
    bg = bg_arr[ys, xs].astype(np.int16)  # (N,) int16
    v = bg + blend * (edge_gray - bg)  # (N,) float, noiseless band value in [0, 255]
    if noise_amp > 0:
        # per-pixel noise scale k_e = min(1, min(v, 255 - v) / amp), (N,) float:
        # v + k_e * noise stays in [0, 255]; k_e = 1 leaves the old sum bit for bit
        k_e = np.minimum(1.0, np.minimum(v, 255.0 - v) / float(noise_amp))
        blended = (v + np.where(k_e < 1.0, k_e * noise, noise)).astype(np.int16)  # (N,) int16
    else:
        blended = (v + noise).astype(np.int16)  # (N,) int16, noise is all 0
    arr[ys, xs] = np.clip(blended, 0, 255).astype(np.uint8)  # safety net: never changes a value
    bg_painted[ys[blend == 0], xs[blend == 0]] = True
    return Image.fromarray(arr), bg_painted


def pointer_forbidden_windows(bg_gray, min_contrast, s_max=0.0):
    """Merged gray windows (list of [lo, hi] float pairs, ascending) a pointer
    must stay out of: the bare face bg and the darkest shadowed face
    sh = bg (1 - s_max), each +- c, open intervals ([] when c <= 0):

        ───[sh-c ██ sh+c]──── allowed ────[bg-c ██ bg+c]───   total width <= 4c

    bg_gray      : int, palette bg_gray_range draw (bare face level)
    min_contrast : int >= 0, c = pointer.min_bg_contrast
    s_max        : float 0-1, max(s_p, s_r) when the image has a shadow, else 0
                   (sh == bg: the two windows coincide, the bare |g - bg| >= c rule)
    A gray g is allowed iff |g - sh| >= c and |g - bg| >= c (window ends
    allowed).  Overlapping windows (sh + c > bg - c) merge into one; windows
    that only touch stay two (the shared end is allowed).  The overlap test
    carries the 2e-9 slack of _pointer_allowed_pieces' two 1e-9 tolerances, so
    a float-rounded touch (sh + c = bg - c + 1e-14) still leaves the shared
    int allowed and the merged and the two-window int sets never differ.  A
    gray between the faces (sh + c <= g <= bg - c) contrasts with both.
    """
    c = float(min_contrast)  # float, contrast c
    if c <= 0:
        return []
    sh = bg_gray * (1.0 - s_max)  # float, darkest shadowed face level
    bg = float(bg_gray)  # float, bare face level
    if sh + c > bg - c + 2e-9:  # overlap (always when s_max = 0, c >= 1): one window
        return [[sh - c, bg + c]]
    return [[sh - c, sh + c], [bg - c, bg + c]]


def _pointer_allowed_pieces(i_lo, i_hi, windows):
    """Allowed int pieces (list of (a, b) int pairs, inclusive, ascending, each
    a <= b) of I = [i_lo, i_hi] minus the open float windows (list of [lo, hi]
    floats, ascending by lo; pointer_forbidden_windows; windows that overlap
    by float rounding are handled too).  A window
    (w_lo, w_hi) forbids the ints floor(w_lo) + 1 .. ceil(w_hi) - 1, so g with
    |g - level| >= c is allowed exactly (1e-9 tolerance for float rounding):

        i_lo ── piece ──]( w0 )[── piece ──]( w1 )[── piece ── i_hi   (<= 3 pieces)
    """
    pieces = []  # list of (int, int), allowed pieces so far
    a = i_lo  # int, bottom of the next candidate piece
    for w_lo, w_hi in windows:
        top = min(i_hi, int(math.floor(w_lo + 1e-9)))  # int, last allowed int below the window
        if a <= top:
            pieces.append((a, top))
        a = max(a, int(math.ceil(w_hi - 1e-9)))  # int, first allowed int above the window
    if a <= i_hi:
        pieces.append((a, i_hi))
    return pieces


def _pick_pointer_gray(rng, lo, hi, bg_gray, min_contrast, s_max=0.0, info=None,
                       headroom=0.0):
    """Draw the pointer body gray (int) with at least `min_contrast` levels of
    separation from the bare face gray AND from the darkest shadowed face, and
    `headroom` levels of room from 0 / 255 for the gradient + noise.

    rng          : the "pointer" random.Random; exactly ONE randint call is made
    lo, hi       : int, inclusive pointer.gray_range
    bg_gray      : int, palette bg_gray_range draw (face background level)
    min_contrast : int >= 0, pointer.min_bg_contrast; 0 = off
    s_max        : float 0-1, max(s_p, s_r) with a shadow, 0 without one
    info         : optional dict; info["fallback"] (bool) is set True when the
                   allowed set was empty (farthest-endpoint fallback), else
                   False; info["headroom_ok"] (bool) True when I is restricted
    headroom     : float >= 0, H (pointer_headroom); 0 = no restriction

    Base interval (headroom-restricted draw, nothing piles up at 0 / 255):

        0 ── H ──[ I = [lo, hi] ∩ [ceil(H), floor(255 - H)] ]── H ── 255
        (I empty, e.g. gray_range [255, 255]: I = [lo, hi]; the amplitude
         scale in _pointer_variation then keeps the body in range)

    Everything below (forbidden windows, index map, fallback) works on I.

    The pointer gray is drawn after the bg and the shadow strengths (they live
    on the palette / shadows streams, drawn earlier).  It must contrast with
    the two face levels it can sit on, the darkest shadowed face
    sh = bg (1 - s_max) and the bare face bg: two open windows of +- c
    (pointer_forbidden_windows; merged when they overlap).  The allowed set
    is I minus the windows, up to three contiguous int pieces
    (_pointer_allowed_pieces).  ONE index drawn with randint(0, n_allowed - 1)
    is mapped onto them in ascending order:

        I_lo ... sh-c ](  sh +- c  )[ sh+c ... bg-c ](  bg +- c  )[ bg+c ... I_hi
        [ idx 0 .. n0-1 ]            [ n0 .. n0+n1-1 ]            [ .. n - 1   ]
          darker than both            between the faces            lighter than both

    With c = 0 the set is all of I = [I_lo, I_hi] (whatever s_max) and
    randint(0, n-1) + I_lo equals randint(I_lo, I_hi) bit for bit (both are
    I_lo + randbelow(n)).  With s_max = 0 the windows coincide: the plain
    |g - bg| >= c rule.  If the set is empty, one randint(I_lo, I_hi) draw is
    still consumed and the endpoint of I farthest from its NEAREST face level
    (min(|e - sh|, |e - bg|)) is returned; on a tie the lower endpoint.
    """
    # base interval I = [lo, hi] ∩ [ceil(H), floor(255 - H)] when non-empty
    # (headroom for the gradient + noise), else [lo, hi]; ints
    i_lo = max(lo, int(math.ceil(headroom - 1e-9)))  # int, bottom of I
    i_hi = min(hi, int(math.floor(255 - headroom + 1e-9)))  # int, top of I
    headroom_ok = i_lo <= i_hi  # bool, I is the restricted interval
    if not headroom_ok:
        i_lo, i_hi = lo, hi
    if info is not None:
        info["headroom_ok"] = headroom_ok
    lo, hi = i_lo, i_hi  # everything below works on I
    windows = pointer_forbidden_windows(bg_gray, min_contrast, s_max)  # list of [lo, hi] floats
    pieces = _pointer_allowed_pieces(lo, hi, windows)  # list of (a, b) ints, allowed
    n = sum(b - a + 1 for a, b in pieces)  # int, allowed set size
    if info is not None:
        info["fallback"] = n == 0
    if n == 0:
        rng.randint(lo, hi)  # keep the stream's draw count fixed
        sh = bg_gray * (1.0 - s_max)  # float, darkest shadowed face level
        d_lo = min(abs(lo - sh), abs(lo - bg_gray))  # float, lo to its nearest face
        d_hi = min(abs(hi - sh), abs(hi - bg_gray))  # float, hi to its nearest face
        return hi if d_hi > d_lo else lo
    idx = rng.randint(0, n - 1)  # int, index into the allowed set
    for a, b in pieces:  # ascending: map idx onto the pieces in order
        if idx <= b - a:
            return a + idx
        idx -= b - a + 1
    raise AssertionError("unreachable: idx < n")


# ── config-level pointer gray warnings (studio + CLI; no rendering) ─────

GRAY_RANGE_MIN_LEVELS = 32  # int, fewer usable pointer grays than this -> warning (b)
FALLBACK_WARN_SHARE = 0.005  # float, contrast-fallback share above this -> warning (c)


def pointer_headroom_bound(cfg):
    """Worst-case (largest) headroom H (float, gray levels) any image of cfg can get.

    Mirrors render_image: dial_radius R = int(img_size // 3 * scale),
    hub_radius = R * h, tip_dist = (R - 2) (1 + u), u in [-tip_variation,
    +tip_variation].  pointer_along_max = (max(tip, hub) + 2) / max(tip, 1)
    grows as tip shrinks and as hub grows (both branches), so for each
    reachable int R the maximum sits at tip = (R - 2)(1 - tip_variation),
    hub = R * hub_ratio_range[1]; every int R the scale_range allows is tried.
    """
    G, PT = cfg["geometry"], cfg["pointer"]  # dict sections
    base = cfg["canvas"]["img_size"] // 3  # int, dial radius at scale 1
    s_lo, s_hi = G["scale_range"]  # float, dial radius multiplier bounds
    tv = float(G["tip_variation"])  # float, fractional tip jitter
    h_hi = float(G["hub_ratio_range"][1])  # float, largest hub / dial radius
    along = 0.0  # float, largest along_max over the reachable radii
    for r in range(int(base * s_lo), int(base * s_hi) + 1):  # int R, every reachable dial radius
        along = max(along, pointer_along_max(r * h_hi, (r - 2) * (1 - tv)))
    return pointer_headroom(PT["gradient_range"], along, PT["noise_amplitude"])


def _max_strength_cdf(t, s_lo, s_hi, j):
    """P(s_max <= t) (float) for s_max = max(clip(s + jp), clip(s + jr)) as
    _shadow_strengths draws it: s ~ U(s_lo, s_hi) SHARED, jp, jr ~ U(-j, j)
    independent, clip to [0, 1].  clip is monotone, so s_max = clip(s + M),
    M = max(jp, jr) with CDF F_M(m) = ((m + j) / 2j)^2 on [-j, j].  Then

        P(s + M <= t) = (1 / (s_hi - s_lo)) * integral_{s_lo}^{s_hi} F_M(t - s) ds
                      = (G(t - s_lo) - G(t - s_hi)) / (s_hi - s_lo)

    with G the antiderivative of F_M: 0 below -j, (m + j)^3 / 12 j^2 on
    [-j, j], 2j/3 + (m - j) above j (exact, closed form; j = 0 or
    s_lo = s_hi degenerate to a step / a point).  The clip makes
    P = 1 for t >= 1 and 0 for t < 0.
    """
    if t >= 1.0:
        return 1.0
    if t < 0.0:
        return 0.0

    def f_m(m):  # float CDF of M at m
        if j <= 0:
            return 1.0 if m >= 0 else 0.0
        return min(max((m + j) / (2 * j), 0.0), 1.0) ** 2

    def g_m(m):  # float antiderivative of f_m (0 at -inf)
        if j <= 0:
            return max(m, 0.0)
        if m <= -j:
            return 0.0
        if m <= j:
            return (m + j) ** 3 / (12 * j * j)
        return 2 * j / 3 + (m - j)

    if s_hi <= s_lo:
        return f_m(t - s_lo)
    return (g_m(t - s_lo) - g_m(t - s_hi)) / (s_hi - s_lo)


def pointer_fallback_share(cfg, headroom=None):
    """Exact share (float 0-1) of images whose allowed pointer gray set is
    empty, i.e. _pick_pointer_gray takes the endpoint fallback (gray_fallback).

    Enumerated, not sampled: bg uniform over the ints of palette.bg_gray_range;
    no shadow with probability 1 - shadows.prob (s_max = 0); with a shadow
    s_max follows _max_strength_cdf.  I is built from `headroom` (float;
    default pointer_headroom_bound, the worst case; an image with a smaller H
    can have an I one level wider, see pointer_gray_warnings).

        I_lo ......... A ......... a_hi ](  bg +- c  )[ bg+c .. I_hi
               ](  sh +- c  )[   sh = bg (1 - s) slides left as s grows

    The upper part [bg + c, I_hi] does not depend on s; when it is non-empty
    the set is never empty.  Else the allowed grays all lie in
    A = [I_lo, a_hi], a_hi = min(I_hi, floor(bg - c)).  A empty: always empty.
    A non-empty: empty iff the shadowed window covers all of A, i.e.
    sh - c < I_lo AND sh + c > a_hi, i.e. s* < s_max < s2 with
    s* = 1 - (I_lo + c) / bg and s2 = 1 - (a_hi - c) / bg (s = 0 never: A's top
    is >= c below bg), so P(empty | bg, shadow) = P(s_max < s2) - P(s_max <= s*).
    """
    PT, S = cfg["pointer"], cfg["shadows"]  # dict sections
    c = int(PT["min_bg_contrast"])  # int, contrast c
    if c <= 0:
        return 0.0
    H = pointer_headroom_bound(cfg) if headroom is None else headroom  # float
    lo, hi = PT["gray_range"]  # int, pointer.gray_range
    i_lo = max(lo, int(math.ceil(H - 1e-9)))  # int, bottom of I (as _pick_pointer_gray)
    i_hi = min(hi, int(math.floor(255 - H + 1e-9)))  # int, top of I
    if i_lo > i_hi:
        i_lo, i_hi = lo, hi
    b_lo, b_hi = cfg["palette"]["bg_gray_range"]  # int, bg draw bounds
    p = float(S["prob"])  # float, shadow probability
    s_lo, s_hi = (float(v) for v in S["strength_range"])  # float, shared strength bounds
    j = float(S["strength_jitter"])  # float, jitter amplitude
    total = 0.0  # float, sum of P(empty | bg) over the bg ints
    for bg in range(b_lo, b_hi + 1):  # int, every bg gray
        if max(i_lo, bg + c) <= i_hi:  # upper part non-empty: never empty
            continue
        a_hi = min(i_hi, int(math.floor(bg - c + 1e-9)))  # int, top of A (below the bare window)
        if a_hi < i_lo:  # A empty: no allowed gray whatever the shadow
            total += 1.0
            continue
        # A non-empty (so bg >= I_lo + c > 0): without a shadow never empty; with
        # one, empty for s* < s_max < s2 (same 1e-9 tolerance as the pieces)
        s_star = 1.0 - (i_lo + c - 1e-9) / bg  # float, largest s keeping I_lo allowed
        s_two = 1.0 - (a_hi - c + 1e-9) / bg  # float, smallest s freeing a_hi
        if s_two > s_star:
            e1 = (_max_strength_cdf(s_two - 1e-12, s_lo, s_hi, j)
                  - _max_strength_cdf(s_star, s_lo, s_hi, j))  # float, P(empty | shadow)
            total += p * e1
    return total / (b_hi - b_lo + 1)


def pointer_gray_warnings(cfg):
    """Config-level warnings (list[str], each "section.key: message") when the
    pointer gray ranges are too tight; [] when fine.  Deterministic, no rendering.

    Picture (gray axis, H = pointer_headroom_bound, the worst case over the
    geometry ranges; c = min_bg_contrast; sh = bg (1 - s_max) the darkest
    shadowed face of one image, both windows open, pointer_forbidden_windows):

        0 ── H ──[════════ I = [lo, hi] ∩ [ceil H, floor(255 - H)] ════════]── H ── 255
                 [ I_lo .. ](░ sh +- c ░)[ .. between .. ](░ bg +- c ░)[ .. I_hi ]
                  darker        window       the faces       window      lighter

      (a) I empty               -> grays from all of [lo, hi], texture scaled down
      (b) 0 < |I| < 32 levels   -> narrow, few distinct pointer grays
      (c) allowed set empty     -> endpoint fallback (point mass); share by
          exact enumeration (pointer_fallback_share), warned above 0.5 %

    (a) uses the worst-case H: all-or-nothing for the config.  An image with
    a smaller H (larger dial / tip) may still have room, so near the edge
    (a) can fire when only part of the images lose the headroom.
    """
    PT = cfg["pointer"]  # dict section
    lo, hi = PT["gray_range"]  # int, pointer.gray_range
    H = pointer_headroom_bound(cfg)  # float, worst-case headroom
    h_lo = int(math.ceil(H - 1e-9))  # int, lowest gray with headroom
    h_hi = int(math.floor(255 - H + 1e-9))  # int, highest gray with headroom
    i_lo, i_hi = max(lo, h_lo), min(hi, h_hi)  # int, I bounds
    warns = []  # list[str]
    if i_lo > i_hi:
        warns.append(
            f"pointer.gray_range: [{lo}, {hi}] leaves no room for gradient+noise headroom "
            f"H≈{H:.0f} (needs overlap with [{h_lo}, {h_hi}]); pointer grays are drawn from "
            f"the full range and the texture is scaled down near 0/255 — lower "
            f"pointer.gradient_range / noise_amplitude or widen gray_range")
    elif i_hi - i_lo + 1 < GRAY_RANGE_MIN_LEVELS:
        n = i_hi - i_lo + 1  # int, usable levels
        warns.append(
            f"pointer.gray_range: only {n} levels ([{i_lo}, {i_hi}]) remain after the "
            f"gradient+noise headroom H≈{H:.0f} (band [{h_lo}, {h_hi}]); fewer than "
            f"{GRAY_RANGE_MIN_LEVELS} distinct pointer grays — widen gray_range or lower "
            f"pointer.gradient_range / noise_amplitude")
    share = pointer_fallback_share(cfg, H)  # float 0-1
    if share > FALLBACK_WARN_SHARE:
        b_lo, b_hi = cfg["palette"]["bg_gray_range"]  # int, bg bounds
        warns.append(
            f"pointer.min_bg_contrast: {PT['min_bg_contrast']} leaves no allowed pointer "
            f"gray for {share * 100:.1f}% of images (bg {b_lo}-{b_hi}, shadows.prob "
            f"{cfg['shadows']['prob']}); those images get an endpoint gray of the range "
            f"(a point mass) — lower min_bg_contrast or widen pointer.gray_range / "
            f"palette.bg_gray_range")
    return warns


def draw_pointer(img, cx, cy, angle_deg, hub_radius, tip_dist, chord_half,
                 chord_dist, ptr_gray, edge_gray, grad_strength, pcfg, nrng,
                 edge_thickness, cut_tip_radius, info=None):
    """Draw the pointer on the "L" canvas.  pcfg = cfg["pointer"] (its
    edge_max_frac clamps the edge band on the triangle, see _draw_pointer_edge).
    info: optional dict, filled with info["amp_scale"] (float, the body's
    gradient + noise scale k of _pointer_variation).

    Returns (img, apex, base_left, base_right, mask, ptr_pix): apex / base_left
    / base_right are the ideal (float) triangle vertices, (x, y) float px; mask
    is the "L" pointer mask with the apex disc erased when cut_tip_radius is
    not None; ptr_pix, (C, C) bool, is the set of POINTER pixels the tip label
    counts (pre-stretch): inside that (cut) mask and not an edge pixel painted
    pure background (bg_painted of _draw_pointer_edge, rings of the uncut
    mask).

    The triangle is raster_pointer_triangle's hard raster of the exact float
    vertices (per scan line round(width) px, centred: the width grows steadily
    from the apex, no "ball" at a thin tip); the hub disc is Pillow's filled
    ellipse (no outline).  The body gray is painted through the mask, so the
    painted pointer pixels are exactly the mask pixels.
    """
    a = math.radians(angle_deg - 90)
    apex = (cx + tip_dist * math.cos(a), cy + tip_dist * math.sin(a))
    base_mx = cx - chord_dist * math.cos(a)
    base_my = cy - chord_dist * math.sin(a)
    perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
    base_left = (base_mx + chord_half * perp[0], base_my + chord_half * perp[1])
    base_right = (base_mx - chord_half * perp[0], base_my - chord_half * perp[1])
    hub_box = [cx - hub_radius, cy - hub_radius, cx + hub_radius, cy + hub_radius]

    # triangle pixels, (C, C) bool: our own hard raster of the exact float
    # vertices (per scan line round(width) px, centred; raster_pointer_triangle),
    # never Pillow's polygon (its per-edge rounding makes a ball at a thin tip)
    in_tri = raster_pointer_triangle(apex, base_left, base_right, img.size)
    # pointer shape mask, "L" (C, C): triangle raster + the hub disc (Pillow's
    # filled ellipse, no outline)
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).ellipse(hub_box, fill=255)
    m0 = np.array(mask)  # (C, C) uint8, hub disc
    m0[in_tri] = 255
    mask = Image.fromarray(m0)

    bg_arr = np.array(img)  # pre-pointer canvas, (C, C) uint8
    # the body is painted THROUGH the mask: painted pointer pixels == mask pixels
    body = bg_arr.copy()  # (C, C) uint8, canvas with the flat body
    body[m0 > 0] = ptr_gray
    img = Image.fromarray(body)

    img = _pointer_variation(img, mask, cx, cy, angle_deg, tip_dist,
                             grad_strength, pcfg["noise_amplitude"], nrng, info)
    # triangle for the edge clamp: the exact float vertices + their raster
    tri = (apex, base_left, base_right, in_tri)
    # bg_painted: (C, C) bool, edge pixels repainted pure background (blend 0)
    img, bg_painted = _draw_pointer_edge(img, mask, edge_gray, bg_arr, edge_thickness,
                                         pcfg["edge_noise_amplitude"], nrng,
                                         tri, float(pcfg["edge_max_frac"]))

    m_uncut = np.array(mask)  # (C, C) uint8, uncut pointer mask (before the cut)
    if cut_tip_radius is not None:
        mask_arr = m_uncut.copy()  # (C, C) uint8, mask to cut
        ys_t, xs_t = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1]]
        mask_arr[(xs_t - apex[0]) ** 2 + (ys_t - apex[1]) ** 2 <= cut_tip_radius ** 2] = 0
        mask = Image.fromarray(mask_arr)
    # (C, C) bool, pointer pixels: the body is painted through the mask, so
    # nothing outside it is pointer (no outline sliver to add)
    ptr_pix = (np.array(mask) > 0) & ~bg_painted
    return img, apex, base_left, base_right, mask, ptr_pix


# ── exposure jitter: per-image capped factors (no saturation) ─────────────
# float levels: PIL's Image.blend computes deg + f (img - deg) as a float and
# truncates it to uint8, saturating only at >= 256 (-> 255) or <= 0 (-> 0); a
# value in [-0.5, 255.5] truncates to what the unclipped value reads (1-level
# rounding tolerance), so the factor caps keep every surviving pixel inside it
JITTER_TOP = 255.5  # float, highest unclipped blend value allowed
JITTER_BOTTOM = -0.5  # float, lowest unclipped blend value allowed
JITTER_KINDS = ("brightness", "contrast", "sharpness")  # str names, the draw order


def enhance_degenerate(img, kind):
    """PIL ImageEnhance's degenerate image for `kind`, (C, C) float64.

    img: "L" image; kind: "brightness" | "contrast" | "sharpness".
        brightness  D = 0                                (ImageEnhance.Brightness)
        contrast    D = int(mean(whole image) + 0.5)     (ImageEnhance.Contrast:
                        ImageStat mean over EVERY pixel, int-rounded, not the region)
        sharpness   D = img.filter(ImageFilter.SMOOTH)   (ImageEnhance.Sharpness)
    enhance(f) = Image.blend(D, img, f) = trunc(D + f (img - D)), saturated.
    """
    if kind == "brightness":
        return np.zeros(img.size[::-1], dtype=np.float64)
    if kind == "contrast":
        from PIL import ImageStat
        mean = int(ImageStat.Stat(img).mean[0] + 0.5)  # int, PIL's own rounding
        return np.full(img.size[::-1], float(mean), dtype=np.float64)
    return np.asarray(img.filter(ImageFilter.SMOOTH), dtype=np.float64)


def enhance_factor_max(img, deg, region):
    """Largest factor f (float, inf if unbounded) with D + f (img - D) inside
    [JITTER_BOTTOM, JITTER_TOP] on every `region` pixel.

    img: "L" image (C, C); deg: (C, C) float64 degenerate D; region: (C, C) bool.

        img > D:  f <= (255.5 - D) / (img - D)       img < D:  f <= (D + 0.5) / (D - img)

    f = 1 gives img itself, so f_max > 1 always, and any f in [0, 1] blends
    between img and D (both in 0..255): only the upper bound can saturate.
    """
    a = np.asarray(img, dtype=np.float64)[region]  # (N,) float64, region pixels
    d = deg[region]  # (N,) float64, their degenerate values
    diff = a - d  # (N,) float64
    up, dn = diff > 0, diff < 0  # (N,) bool, pixels f pushes up / down
    f_max = math.inf  # float
    if up.any():
        f_max = min(f_max, float(((JITTER_TOP - d[up]) / diff[up]).min()))
    if dn.any():
        f_max = min(f_max, float(((d[dn] - JITTER_BOTTOM) / -diff[dn]).min()))
    return f_max


def capped_uniform(rng, lo, hi, f_max):
    """One uniform draw (exactly one rng.random()) of a factor capped at f_max.

    rng: random.Random; lo, hi: float, the config range; f_max: float, the cap.
        f_max >= lo:  f = U(lo, min(hi, f_max))            (restricted draw)
        f_max <  lo:  f = U(f_max lo / hi, f_max)          (range shrunk in proportion)

          lo ════════════ hi               lo ══════ f_max ┄┄┄┄ hi
          └ f_max >= hi: the old draw ┘    └ capped ─┘
    Returns (f float, capped bool: the range was shortened).
    random.uniform(a, b) = a + (b - a) random(), one draw whatever the bounds.
    """
    if f_max >= hi:
        return rng.uniform(lo, hi), False
    if f_max >= lo:
        return rng.uniform(lo, f_max), True
    return rng.uniform(f_max * lo / hi, f_max), True


def _jitter_enhance(img, kind, rng, lo, hi, region):
    """Apply one capped exposure-jitter enhance to the "L" canvas.

    kind: one of JITTER_KINDS; rng: the "jitter" random.Random (one draw);
    lo, hi: float, cfg["jitter"][kind + "_range"]; region: (C, C) bool, the
    pixels that survive to the output (inside the dial ellipse, off the rim).
    The cap is computed on the CURRENT image, just before this enhance.
    Returns (img "L", f float, f_max float, capped bool).
    """
    deg = enhance_degenerate(img, kind)  # (C, C) float64
    f_max = enhance_factor_max(img, deg, region)  # float
    f, capped = capped_uniform(rng, lo, hi, f_max)
    enh = {"brightness": ImageEnhance.Brightness, "contrast": ImageEnhance.Contrast,
           "sharpness": ImageEnhance.Sharpness}[kind]  # ImageEnhance class
    return enh(img).enhance(f), f, f_max, capped


# ── tip label: first base-parallel line with > n pointer pixels ──────────
TIP_T_STEP = 0.05  # float px, grid of the scan along the triangle height


def tip_count_label(pix, apex, base_left, base_right, n):
    """Tip label on the (stretched) pointer triangle by pixel count.

    pix        : (H, W) bool, pointer pixels; pixel centres at integer coords
    apex       : (x, y) float px, A' (exact apex, stretched)
    base_left, base_right : (x, y) float px, B_l' and B_r' (exact base corners, stretched)
    n          : int >= 0, tip_label.min_pixels

        A' ●   P(t) = A' + t h,  h = (M' - A') / |M' - A'|,  M' = (B_l' + B_r') / 2
          ╱ ╲
     ── ●─●─● ── L(t): the line through P(t) parallel to B_l' B_r'
        ╱     ╲   count(t) = # pointer pixels whose unit SQUARE L(t) passes
    B_l' ●──M'──● B_r'   through: -r <= d(p) - t k < r, d(p) = (p - A') . nb,
                         nb the unit base normal (toward the base), k = h . nb,
                         r = (|nb_x| + |nb_y|) / 2 = half the square's extent
                         along nb (0.5 for an axis-aligned base, up to 0.707)

        tilted L(t):  ┌──┬──┐        a fixed 1-px band (r = 0.5) skips pixels
                      │ ╱│  │        a tilted line clearly crosses (count 0
                      │╱ │  │        inside a continuous 1-px sliver); r makes
                     ╱┼──┤  │        the count = pixels the line touches

    t* = the smallest t on the TIP_T_STEP grid in [0, |M' - A'|] with
    count(t) > n; fallback (no such t): M'.  Vectorised: each pixel projects
    once onto nb, covers a grid-index interval of t, and a difference array +
    cumsum gives count on the whole grid.  A larger n can only move t* toward
    the base (monotonic by construction).

    Returns (label (x, y) float px, t* float px from A', count(t*) int,
    fallback bool).  The label is A' + t* h: on the line A'M' exactly.
    """
    ax, ay = float(apex[0]), float(apex[1])
    mx = (base_left[0] + base_right[0]) / 2.0  # float px, M' x
    my = (base_left[1] + base_right[1]) / 2.0  # float px, M' y
    h_len = math.hypot(mx - ax, my - ay)  # float px, |M' - A'|
    if h_len <= 0:  # degenerate triangle: no height to scan
        return (ax, ay), 0.0, 0, True
    hx, hy = (mx - ax) / h_len, (my - ay) / h_len  # (float, float) unit height
    bx, by = base_right[0] - base_left[0], base_right[1] - base_left[1]  # float, base vector
    b_len = math.hypot(bx, by)  # float px, base length
    if b_len > 0:
        nbx, nby = -by / b_len, bx / b_len  # (float, float) unit base normal
    else:
        nbx, nby = hx, hy
    k = hx * nbx + hy * nby  # float, h . nb: rate of the band along t
    if k < 0:
        nbx, nby, k = -nbx, -nby, -k  # orient nb toward the base
    k = max(k, 1e-6)
    r = (abs(nbx) + abs(nby)) / 2.0  # float px, pixel square's half extent along nb
    n_grid = int(math.floor(h_len / TIP_T_STEP + 1e-9))  # int, last grid index
    ys, xs = np.nonzero(pix)  # (N,) int64 pointer pixel coords
    d = (xs - ax) * nbx + (ys - ay) * nby  # (N,) float64 signed distance from A' along nb
    # pixel counted at grid t_i = i * step iff d - r <= t_i k < d + r, i.e.
    # i in [ceil((d - r) / (k step)), ceil((d + r) / (k step)) - 1]
    i_lo = np.ceil(np.round((d - r) / (k * TIP_T_STEP), 6)).astype(np.int64)  # (N,) int64
    i_hi = np.ceil(np.round((d + r) / (k * TIP_T_STEP), 6)).astype(np.int64) - 1  # (N,) int64
    i_lo = np.maximum(i_lo, 0)
    i_hi = np.minimum(i_hi, n_grid)
    ok = i_lo <= i_hi  # (N,) bool, pixel's band meets the scan
    diff = np.zeros(n_grid + 2, dtype=np.int64)  # (n_grid + 2,) int64 difference array
    np.add.at(diff, i_lo[ok], 1)
    np.add.at(diff, i_hi[ok] + 1, -1)
    count = np.cumsum(diff[:n_grid + 1])  # (n_grid + 1,) int64, count(t_i)
    hit = np.flatnonzero(count > n)  # (H,) int64 grid indices passing
    if hit.size:
        i = int(hit[0])
        t = i * TIP_T_STEP  # float px
        return (ax + t * hx, ay + t * hy), t, int(count[i]), False
    return (mx, my), h_len, int(count[-1]) if count.size else 0, True


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

    img: canvas "L" (C, C); rngs: section_rngs dict; tip: tip label (x, y)
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
        # (C, C) bool, pixels a glare may never cover (around the tip label)
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


# ── triangle artifacts (step 12) ──────────────────────────────────────────

def clear_zone(blocked, d):
    """Z_d: the pixels whose centre lies within Euclidean distance d of the
    centre of a blocked pixel.

    blocked: (H, W) bool, the set P; d: float px >= 0.  Returns a new (H, W)
    bool, Z_d = {q : min over p in P of |q - p| <= d}; Z_d == P for d < 1.
    Exact disc dilation in numpy: the disc of radius d is a stack of
    horizontal runs, row offset dy in [-r, r] (r = floor(d)) with half-width
    k(dy) = floor(sqrt(d^2 - dy^2)):

        dy -2   · · ▓ ▓ ▓ · ·   k = 1        (d = 2.5: every (dx, dy) with
        dy -1   · ▓ ▓ ▓ ▓ ▓ ·   k = 2         dx^2 + dy^2 <= 6.25)
        dy  0   · ▓ ▓ ● ▓ ▓ ·   k = 2
        dy +1   · ▓ ▓ ▓ ▓ ▓ ·   k = 2
        dy +2   · · ▓ ▓ ▓ · ·   k = 1

    so Z_d = OR over dy of (P grown along x by k(dy), one row-cumsum
    difference per distinct k) shifted by dy rows.  The work is confined to
    P's bounding box grown by r.
    """
    zone = blocked.copy()  # (H, W) bool, Z_d; starts as P
    r = int(math.floor(d))  # int px, largest row / column offset inside the disc
    if r < 1 or not blocked.any():
        return zone
    H, W = blocked.shape
    rows = np.flatnonzero(blocked.any(axis=1))  # (n,) int64, rows holding P
    cols = np.flatnonzero(blocked.any(axis=0))  # (m,) int64, columns holding P
    y0, y1 = max(int(rows[0]) - r, 0), min(int(rows[-1]) + r + 1, H)  # int, crop rows [y0, y1)
    x0, x1 = max(int(cols[0]) - r, 0), min(int(cols[-1]) + r + 1, W)  # int, crop cols [x0, x1)
    sub = blocked[y0:y1, x0:x1]  # (h, w) bool, P inside the crop
    h, w = sub.shape
    # row prefix counts of sub padded with r empty columns on each side,
    # (h, w + 2r + 1) int32: cs[:, j] = # P pixels in padded columns [0, j)
    cs = np.zeros((h, w + 2 * r + 1), dtype=np.int32)
    cs[:, r + 1:r + 1 + w] = np.cumsum(sub, axis=1, dtype=np.int32)
    cs[:, r + 1 + w:] = cs[:, r + w:r + w + 1]
    grown = np.zeros((h, w), dtype=bool)  # (h, w) bool, Z_d inside the crop
    runs = {}  # dict[int, (h, w) bool]: P grown along x by half-width k
    d2 = float(d) * float(d)  # float px^2
    for dy in range(-r, r + 1):
        if abs(dy) >= h:  # row offset past the crop (a mask shorter than r rows)
            continue
        k = math.isqrt(int(math.floor(d2 - dy * dy)))  # int px, run half-width at this row offset
        if k not in runs:
            # a P pixel within [x - k, x + k] of column x, on the same row
            runs[k] = (cs[:, r + k + 1:r + k + 1 + w] - cs[:, r - k:r - k + w]) > 0
        run = runs[k]
        if dy >= 0:  # Z row y gets P row y - dy
            grown[dy:] |= run[:h - dy]
        else:
            grown[:h + dy] |= run[-dy:]
    zone[y0:y1, x0:x1] = grown
    return zone


def _tri_raster(offs, gx, gy, tcx, tcy, size2):
    """One artifact triangle with its centroid on the centre (tcx, tcy).

    offs: list of 3 (dx, dy) float vertex offsets from the draw point; (gx, gy):
    float, their centroid; (tcx, tcy): int canvas px; size2: canvas (C, C).
    Returns (verts, raster): verts, list of 3 (x, y) float canvas px; raster,
    (C, C) bool, Pillow's polygon fill of verts (no clipping).
    """
    verts = [(tcx + dx - gx, tcy + dy - gy) for dx, dy in offs]  # 3 (x, y) float vertices
    tri_mask = Image.new("L", size2, 0)  # "L" (C, C) polygon mask
    ImageDraw.Draw(tri_mask).polygon(verts, fill=255)
    return verts, np.array(tri_mask) > 127


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
    # cast distance of the pointer shadow: ONE uniform draw at the position of
    # the old px offset draw, so the stream's draw count is unchanged
    dist_frac = srng.uniform(*S["distance_frac_range"])  # float, d / dial_radius
    shd_dist = dist_frac * dial_radius  # float px, cast distance d
    sig_tri, sig_hub = shadow_sigmas(shd_dist, S["blur_min"], S["blur_per_px"])  # float px, list
    rec["shadows"] = {"active": has_shadow, "light_angle": light_angle, "s_p": s_p, "s_r": s_r,
                      "rim_offset_frac": rim_offset, "rim_cutout_frac": rim_cutout,
                      "rim_blur": rim_blur, "distance_frac": dist_frac, "distance": shd_dist,
                      "blur_triangle": sig_tri, "blur_hub_far": sig_hub[-1]}
    # shadow darkening fraction on the canvas, (C, C) float32 in 0-1, or None
    # when the image has no shadow; nothing is composited onto img here, it
    # is applied once at step 13 as img * (1 - d)
    shade = None
    m_p = m_r = None  # (C, C) float32 pointer / rim masks; shade is built at step 7
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
        # pre-blur component masks (dict, see pointer_shadow_shapes): hub swept
        # from the face, triangle plate shifted by d u, shifted dial-circle arc
        shapes = pointer_shadow_shapes(s_apex, s_bl, s_br, ptr_cx, ptr_cy, hub_radius,
                                       dial_radius, shd_dist, light_rad,
                                       gap_start_deg, gap_span_deg, canvas_size)
        m_p = pointer_shadow_mask(shapes, shd_dist, S["blur_min"],
                                  S["blur_per_px"])[0]  # (C, C) float32, 0-1

    # --- 6 digits on the overlay ---
    if show_digits:
        draw_scale_digits(td_value, td_alpha, cx, cy, dial_radius, tick_inner_r,
                          D, drng, font_size, rec["digits"])

    # --- 7 mounting hole + pointer (pointer stream) ---
    PT = cfg["pointer"]
    ptr_rng, ptr_np = rngs["pointer"]
    # the faces the pointer may sit on: bare bg_gray and, with a shadow, the
    # darkest shadowed face bg_gray (1 - max(s_p, s_r)); both drawn already
    # (palette / shadows streams), so the pointer stream's order is unchanged
    s_max = max(s_p, s_r) if has_shadow else 0.0  # float 0-1, strongest shadow
    forbidden = pointer_forbidden_windows(bg_gray, PT["min_bg_contrast"], s_max)  # list of [lo, hi] floats
    # headroom H (float, gray levels) the body gray keeps from 0 / 255: the
    # largest |gradient| the config allows times this image's largest
    # |along_norm| (geometry, drawn already) + the noise amplitude
    headroom = pointer_headroom(PT["gradient_range"], pointer_along_max(hub_radius, tip_dist),
                                PT["noise_amplitude"])
    gray_info = {}  # dict, {"fallback": bool, "headroom_ok": bool} from _pick_pointer_gray
    ptr_gray = _pick_pointer_gray(ptr_rng, *PT["gray_range"], bg_gray,
                                  PT["min_bg_contrast"], s_max, gray_info,
                                  headroom)  # int body gray
    hole_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    edge_gray = int(ptr_gray * ptr_rng.uniform(*PT["darker_frac_range"]))  # int
    grad_strength = ptr_rng.uniform(*PT["gradient_range"])  # float
    edge_thickness = ptr_rng.randint(*PT["edge_thickness_range"])  # int px (0 = no edge band)
    if has_shadow:
        shade = shadow_darkening(m_p, m_r, s_p, s_r)  # (C, C) float32
    rec["pointer"] = {"gray": ptr_gray, "hole_gray": hole_gray, "edge_gray": edge_gray,
                      "gradient": grad_strength, "edge_thickness": edge_thickness,
                      "forbidden": forbidden,
                      "gray_fallback": gray_info["fallback"],
                      "headroom": headroom, "headroom_ok": gray_info["headroom_ok"]}

    # hole drawn BEFORE the pre-pointer snapshot so it survives the cut-tip restore
    ImageDraw.Draw(img).ellipse([hole_cx - hole_r, hole_cy - hole_r,
                                 hole_cx + hole_r, hole_cy + hole_r], fill=hole_gray)

    C = cfg["cut_tip"]
    cut_r = rngs["cut_tip"][0].randint(*C["radius_range"])  # int px
    rec["cut_tip"] = {"enabled": bool(C["enabled"]), "cut_r": cut_r}
    bg_pre_ptr = img.copy() if C["enabled"] else None  # pre-pointer canvas, "L"
    img, apex, base_l, base_r, ptr_mask, ptr_pix = draw_pointer(
        img, ptr_cx, ptr_cy, angle_deg, hub_radius, tip_dist, chord_half,
        chord_dist, ptr_gray, edge_gray, grad_strength, PT, ptr_np,
        edge_thickness, cut_r if C["enabled"] else None, rec["pointer"])
    if bg_pre_ptr is not None:
        img.paste(bg_pre_ptr, mask=Image.fromarray(255 - np.array(ptr_mask)))
    # label inputs; the label itself is found after step 10 (it must see the rim)
    ideal_apex = apex  # (x, y) float px, ideal triangle apex (canvas, pre-stretch)
    # the label axis is the pointer triangle's HEIGHT (apex -> base-chord
    # midpoint) of the EXACT float vertices: raster_pointer_triangle draws from
    # them unrounded and centres every scan line's run on the true midline;
    # (x, y) float px, canvas, pre-stretch
    tri_apex = apex  # (x, y) float px, exact apex A
    tri_bl, tri_br = base_l, base_r  # (x, y) float px each, exact B_l / B_r
    tri_base_mid = ((tri_bl[0] + tri_br[0]) / 2.0, (tri_bl[1] + tri_br[1]) / 2.0)  # M

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

    # dial-ellipse / boundary geometry (no draws), needed from step 9 on
    outer_rx, outer_ry = dial_rx * outer_scale, dial_ry * outer_scale  # float px
    # open pixel grids: ys (C, 1) int, xs (1, C) int; they broadcast to (C, C)
    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    # squared normalised radius vs the dial ellipse (about (cx, cy)) / the
    # boundary B (about (bcx, bcy)), (C, C) float64; <= 1 inside
    inner_r2 = ((xs - cx) / max(dial_rx, 1)) ** 2 + ((ys - cy) / max(dial_ry, 1)) ** 2
    outer_r2 = ((xs - bcx) / max(outer_rx, 1)) ** 2 + ((ys - bcy) / max(outer_ry, 1)) ** 2
    inside_outer = outer_r2 <= 1  # (C, C) bool, inside B
    r_norm = np.sqrt(inner_r2)  # (C, C) float64, radius / dial ellipse radius
    rim_half = O["rim_width"] / min(max(dial_rx, 1), max(dial_ry, 1))  # float, rim half width / radius
    # pixels of the jittered canvas that survive to the output, (C, C) bool:
    # inside the dial ellipse and off the rim band (step 10 repaints the rest)
    survive = r_norm <= 1 - rim_half

    # --- 9 exposure jitter (factors capped per image: nothing saturates) ---
    J, jrng = cfg["jitter"], rngs["jitter"][0]
    rec["jitter"] = {}  # dict: factor, its cap (None = unbounded) and capped flag per kind
    capped = {}  # dict[str, bool], per kind: the range was shortened
    # enhance factors (float), drawn in this order: brightness, contrast, sharpness;
    # each cap is computed on the current image, just before its enhance
    for kind in JITTER_KINDS:
        img, f, f_max, capped[kind] = _jitter_enhance(img, kind, jrng,
                                                      *J[kind + "_range"], survive)
        rec["jitter"][kind] = f
        rec["jitter"][kind + "_max"] = f_max if math.isfinite(f_max) else None
    rec["jitter"]["capped"] = capped

    # --- 10 outer annulus + rim band ---
    arr = np.array(img)  # (C, C) uint8
    arr[(inner_r2 > 1) & inside_outer] = outer_gray
    arr[(r_norm > 1 - rim_half) & (r_norm < 1 + rim_half)] = rim_gray

    # --- tip label (geometry only: before noise, blur and effects) ---
    # pointer pixels on the stretched canvas: the pre-stretch pointer pixels
    # (cut mask minus the edge ring painted pure background; see
    # draw_pointer) stretched with the image's affine and BICUBIC resample, thresholded at >= 128 (more pointer than background, the
    # same resample as the image, so a black-on-face pointer pixel counts iff it
    # renders darker than the face/pointer midpoint), minus `covered` (rim band
    # and outer fill paint over them)
    covered = r_norm > 1 - rim_half  # (C, C) bool, rim band + outer fill + outside
    pix_s = np.asarray(Image.fromarray(ptr_pix.astype(np.uint8) * 255).transform(
        size2, Image.AFFINE, affine, resample=Image.BICUBIC)) >= 128  # (C, C) bool
    pix_s &= ~covered
    # Image.AFFINE maps pixel-corner coords, so with pixel centres at integer
    # coords the stretch is about (cx - 0.5, cy - 0.5): point (x, y) -> sxy(x, y)
    def sxy(x, y):
        """Canvas point (x, y) float px -> where the stretch puts it, (x, y) float px."""
        return (cx - 0.5 + (x - cx + 0.5) * stretch_h, cy - 0.5 + (y - cy + 0.5) * stretch_v)
    ideal_apex = sxy(*ideal_apex)  # (x, y) float px, stretched ideal apex (reference)
    axis_apex = sxy(*tri_apex)  # (x, y) float px, A': stretched exact apex
    axis_bl, axis_br = sxy(*tri_bl), sxy(*tri_br)  # (x, y) float px, B_l', B_r'
    # stretched base-chord midpoint M', (x, y) float px.  The stretch is affine,
    # so A'M' is the image of the height and lines parallel to the base
    # stay parallel; after a non-uniform stretch the triangle is no longer
    # isosceles and A'M' differs slightly from its perpendicular altitude
    axis_base_mid = sxy(*tri_base_mid)
    TL = cfg["tip_label"]
    # label (x, y) float px, t* float px along A'M', count(t*) int, fallback bool
    apex, tip_t, tip_count, tip_fallback = tip_count_label(
        pix_s, axis_apex, axis_bl, axis_br, int(TL["min_pixels"]))
    bdx, bdy = axis_br[0] - axis_bl[0], axis_br[1] - axis_bl[1]  # float, stretched base vector
    b_n = math.hypot(bdx, bdy) or 1.0  # float px, its length
    base_dir = (bdx / b_n, bdy / b_n)  # (x, y) float, unit B_l' -> B_r'
    h_pre = math.hypot(tri_apex[0] - tri_base_mid[0],
                       tri_apex[1] - tri_base_mid[1])  # float px, height pre-stretch
    h_post = math.hypot(axis_apex[0] - axis_base_mid[0],
                        axis_apex[1] - axis_base_mid[1])  # float px, |M' - A'|
    # tip_t measured before the stretch (float px): the affine scales the height
    # line uniformly, by h_post / h_pre
    tip_recession = tip_t * h_pre / h_post if h_post > 0 else 0.0

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

    # --- 12 adversarial triangle artifacts (kept clear of pointer + hub + hole) ---
    ptr_mask_s = ptr_mask.transform(size2, Image.AFFINE, affine, resample=Image.NEAREST)
    # P, (C, C) bool: stretched (cut) pointer incl. the hub disc, | the hole
    ptr_blocked = np.array(ptr_mask_s) > 127
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
    # list[dict], one record per triangle (gray, centre, size, placed, pixels,
    # tries, trimmed, verts)
    tri_items = []
    rec["tri_artifacts"].update(n=n_tris, items=tri_items)
    tv = T["pixel_var"]
    clear = float(T["clear_distance"])  # float px, d: no triangle pixel within d of P
    # Z_d, (C, C) bool: pixels within d (pixel-centre Euclidean) of P; d = 0 is
    # P itself (the old placement, no fit check)
    zone = clear_zone(ptr_blocked, clear) if clear > 0 else ptr_blocked
    # V, (C, C) bool: inside B and off Z_d, the centre domain and the paint
    # clip.  B sits where the displacement puts it, so the centres depend on
    # the displacement: the triangles are the one content layer that is not a
    # pure translation
    allowed = inside_outer & ~zone
    v_idx = np.flatnonzero(allowed)  # (|V|,) int64, row-major flat indices of V
    for _ in range(n_tris):
        tri_gray = trng.randint(*T["gray_range"])  # int mean gray
        # centre drawn from V (ONE draw); row-major flat index -> (row, col).
        # V empty (Z_d covers B): no draw, the canvas centre is recorded and
        # nothing is painted (allowed is empty)
        flat = int(v_idx[trng.randrange(len(v_idx))]) if len(v_idx) else \
            cy * canvas_size + cx  # int, flat canvas index of the centre
        tries = 1 if len(v_idx) else 0  # int, centre draws used
        tcy, tcx = divmod(flat, canvas_size)  # int px, centre row / column
        sz = trng.randint(*T["size_range"])  # int px
        offs = []  # list of 3 (dx, dy) float tuples, vertex offsets from the draw point
        for _v in range(3):
            va = trng.uniform(0, 2 * math.pi)
            vr = trng.uniform(*T["vertex_dist_range"]) * sz
            offs.append((vr * math.cos(va), vr * math.sin(va)))
        # translate so the triangle's CENTROID sits on the centre (the centroid
        # is always inside the triangle, so the centre pixel is a valid spot)
        gx, gy = sum(o[0] for o in offs) / 3.0, sum(o[1] for o in offs) / 3.0  # float, centroid offset
        # 3 (x, y) float vertices, (C, C) bool unclipped raster
        verts, tri_px = _tri_raster(offs, gx, gy, tcx, tcy, size2)
        trimmed = False  # bool, all TRI_MAX_TRIES centres reached Z_d: last one trimmed
        if clear > 0 and len(v_idx):
            # the WHOLE raster must clear Z_d; otherwise redraw only the centre
            # (one randrange per retry, after the vertex draws) and move the
            # same shape there
            hit = bool((tri_px & zone).any())  # bool, the raster reaches Z_d
            while hit and tries < TRI_MAX_TRIES:
                flat = int(v_idx[trng.randrange(len(v_idx))])
                tries += 1
                tcy, tcx = divmod(flat, canvas_size)
                verts, tri_px = _tri_raster(offs, gx, gy, tcx, tcy, size2)
                hit = bool((tri_px & zone).any())
            trimmed = hit
        valid = tri_px & allowed  # (C, C) bool, painted pixels (= tri_px & B when it fits)
        if not valid.any():  # sub-pixel triangle: paint the centre pixel alone
            valid[tcy, tcx] = bool(allowed[tcy, tcx])
        tri_items.append({"gray": tri_gray, "center_x": tcx, "center_y": tcy, "size": sz,
                          # False only when V is empty (Z_d covers B)
                          "placed": bool(valid.any()), "pixels": int(valid.sum()),
                          "tries": tries, "trimmed": trimmed,
                          "verts": [list(v) for v in verts]})
        # drawn for every triangle, so the numpy stream's draw count is fixed
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
    for it in rec["tri_artifacts"]["items"]:
        it["verts"] = [[x + off_x, y + off_y] for x, y in it["verts"]]  # 3 [x, y] float image px
    rec["derived"] = {"img_size": img_size, "canvas_offset_x": off_x, "canvas_offset_y": off_y,
                      "dial_rx": dial_rx, "dial_ry": dial_ry,
                      "outer_rx": outer_rx, "outer_ry": outer_ry,
                      "tip_x": apex[0] + off_x, "tip_y": apex[1] + off_y,
                      "ideal_apex_x": ideal_apex[0] + off_x,
                      "ideal_apex_y": ideal_apex[1] + off_y,
                      "axis_base_mid_x": axis_base_mid[0] + off_x,
                      "axis_base_mid_y": axis_base_mid[1] + off_y,
                      "axis_apex_x": axis_apex[0] + off_x, "axis_apex_y": axis_apex[1] + off_y,
                      "axis_base_dir_x": base_dir[0], "axis_base_dir_y": base_dir[1],
                      "tip_t": tip_t, "tip_count": tip_count, "tip_fallback": tip_fallback,
                      "tip_recession": tip_recession}
    # ideal apex in image px (float) and the label recession behind the drawn
    # apex (float px, pre-stretch), also under cut_tip
    rec["cut_tip"].update(ideal_apex_x=ideal_apex[0] + off_x,
                          ideal_apex_y=ideal_apex[1] + off_y, tip_recession=tip_recession)
    return RenderResult(image=out,
                        tip_x=apex[0] + off_x, tip_y=apex[1] + off_y,
                        angle_deg=angle_deg, params=_plain(rec),
                        tip_axis=((axis_apex[0] + off_x, axis_apex[1] + off_y),
                                  (axis_base_mid[0] + off_x, axis_base_mid[1] + off_y),
                                  base_dir))


# ── Multiprocessing worker plumbing + dataset driver ──────────────────────
#
# Each image index gets its own seed (_seed_for_index), so output is
# bit-exact regardless of worker count.  Windows uses 'spawn': the cfg
# reaches workers through Pool initializer args, stored in _WORKER.

_WORKER = {}  # per-process state: {"output_dir": str, "seed_base": int, "cfg": dict, "save_params": bool}

PARAMS_FILE = "params.jsonl"  # per-image sampled parameters, one JSON line per labels.csv row (str)


def _seed_for_index(seed_base, idx):
    """Per-image seed, independent of worker count or task-scheduling
    order -- what makes --workers N bit-exact with --workers 1."""
    return (seed_base + idx * 2654435761) & 0xFFFFFFFF


def _worker_init(output_dir, seed_base, cfg, save_params=False):
    """Pool initializer: store output_dir (str), seed_base (int), the
    validated cfg (dict section -> key -> value) and save_params (bool: also
    build the params.jsonl line) in this process's _WORKER."""
    _WORKER["output_dir"] = output_dir
    _WORKER["seed_base"] = seed_base
    _WORKER["cfg"] = cfg
    _WORKER["save_params"] = save_params


def _strict(v):
    """Copy of the JSON-able value v with every non-finite float replaced by
    the string "NaN" / "Infinity" / "-Infinity" (dicts and lists recursed)."""
    if isinstance(v, float) and not math.isfinite(v):
        return "NaN" if math.isnan(v) else ("Infinity" if v > 0 else "-Infinity")
    if isinstance(v, dict):
        return {k: _strict(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_strict(x) for x in v]
    return v


def params_line(record):
    """One strict-JSON line (str, no trailing newline) for record (dict);
    non-finite floats anywhere in it become the strings "NaN" / "Infinity" /
    "-Infinity", so json.loads works on every reader."""
    return json.dumps(_strict(record), allow_nan=False)


def _worker_task(idx):
    """Render image idx (int), save it as dial_<idx:04d>.png in the output dir;
    -> (label, line): label = row dict {filename: str, angle_deg: int,
    tip_x: float, tip_y: float}; line = params.jsonl line (str) when the pool
    was initialised with save_params, else None."""
    img_seed = _seed_for_index(_WORKER["seed_base"], idx)  # seed render_image gets (int)
    res = render_image(_WORKER["cfg"], img_seed)
    fname = f"dial_{idx:04d}.png"
    Image.fromarray(res.image).save(os.path.join(_WORKER["output_dir"], fname))
    label = {"filename": fname, "angle_deg": res.angle_deg,
             "tip_x": round(res.tip_x, 2), "tip_y": round(res.tip_y, 2)}
    line = None  # params.jsonl line (str) or None
    if _WORKER.get("save_params"):
        line = params_line({"filename": fname, "seed": img_seed,
                            "angle_deg": label["angle_deg"], "tip_x": label["tip_x"],
                            "tip_y": label["tip_y"], "sampled": res.params})
    return label, line


LABEL_FIELDS = ["filename", "angle_deg", "tip_x", "tip_y"]  # labels.csv columns (list[str])
GENERATOR_NAME = "generate_synthetic_v7_mp.py"  # recorded in config.yaml run block (str)


class OutputDirNotEmpty(RuntimeError):
    """Output folder already holds dial_*.png / labels.csv / params.jsonl / config.yaml and
    overwrite was not requested."""


def _dataset_files(output_dir):
    """Files generate_dataset owns in output_dir -> list[str] of absolute paths
    (dial_*.png, labels.csv, params.jsonl, config.yaml); other files are never touched."""
    # escape: a '[...]' in output_dir is a folder name, not a pattern
    found = glob.glob(os.path.join(glob.escape(output_dir), "dial_*.png"))
    for name in ("labels.csv", PARAMS_FILE, "config.yaml"):
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
            f"({len(existing)} of dial_*.png/labels.csv/params.jsonl/config.yaml); "
            f"pass --overwrite to replace them")


def generate_dataset(cfg, output_dir, num_images, seed=0, workers=None, chunksize=4,
                     overwrite=False, progress_cb=None, cancel_event=None,
                     save_params=False):
    """Render num_images dials into output_dir with PNGs, labels.csv, config.yaml.

    cfg: validated config dict; progress_cb(done, total) is called after every
    finished image; cancel_event (threading.Event) stops the run, keeping only
    finished PNGs whose rows are in labels.csv.  save_params (bool) also writes
    params.jsonl: one record per labels.csv row, written in the same step.
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
           "save_params": bool(save_params), "completed": False, "finished_at": None}
    _write_config(cfg, output_dir, run)  # provisional, overwritten at the end

    written = []  # label rows already in labels.csv, list[dict]
    cancelled = False  # True once a cancel was seen or the run ended early (bool)
    pool = None  # mp.Pool when n_workers > 1, else None
    try:
        with open(os.path.join(output_dir, "labels.csv"), "w", newline="") as f, \
                contextlib.ExitStack() as stack:
            # params.jsonl handle (text file) when save_params, else None
            pf = (stack.enter_context(open(os.path.join(output_dir, PARAMS_FILE), "w",
                                           encoding="utf-8", newline="\n"))
                  if save_params else None)
            writer = csv.DictWriter(f, fieldnames=LABEL_FIELDS)
            writer.writeheader()

            def accept(result):
                """Record one finished image: its label row and, when saving,
                its params line (result: (label dict, line str|None)); True if
                a cancel was requested."""
                label, line = result
                writer.writerow(label)
                if pf is not None:
                    pf.write(line + "\n")
                    pf.flush()
                f.flush()
                written.append(label)
                if progress_cb is not None:
                    progress_cb(len(written), num_images)
                return cancel_event is not None and cancel_event.is_set()

            indices = range(num_images)
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True  # cancel set before start: render nothing
            elif n_workers == 1:
                _worker_init(output_dir, seed, cfg, bool(save_params))
                for idx in indices:
                    if accept(_worker_task(idx)):
                        cancelled = True
                        break
            else:
                pool = mp.Pool(processes=n_workers, initializer=_worker_init,
                               initargs=(output_dir, seed, cfg, bool(save_params)))
                for result in pool.imap(_worker_task, indices, chunksize=chunksize):
                    if accept(result):
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
                        help=f"directory for PNGs, labels.csv, params.jsonl (with --save-params), config.yaml (default: {OUTPUT_DIR})")
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES,
                        help=f"number of images (default: {NUM_IMAGES})")
    parser.add_argument("--seed", type=int, default=0, help="base RNG seed (default: 0)")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1,
                        help="worker processes (default: os.cpu_count())")
    parser.add_argument("--chunksize", type=int, default=4,
                        help="Pool.imap chunksize (default: 4)")
    parser.add_argument("--overwrite", action="store_true",
                        help="delete existing dial_*.png, labels.csv, params.jsonl, config.yaml first")
    parser.add_argument("--save-params", action="store_true",
                        help="also write params.jsonl: every image's sampled parameters "
                             "(~3 KB per image)")
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
    # too-tight pointer gray ranges (list[str]); printed once, generation goes on
    enc = getattr(sys.stderr, "encoding", None) or "utf-8"  # str, console codec
    for w in pointer_gray_warnings(cfg):
        line = f"[v7] warning: {w}"  # str, one stderr line
        print(line.encode(enc, "replace").decode(enc), file=sys.stderr)
    try:
        res = generate_dataset(cfg, args.output_dir, args.num_images, seed=args.seed,
                               workers=args.workers, chunksize=args.chunksize,
                               overwrite=args.overwrite, save_params=args.save_params)
    except OutputDirNotEmpty as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)
    if res["completed"]:
        print(f"Generated {res['done']} images in '{args.output_dir}/'")
    else:
        print(f"Cancelled after {res['done']} images in '{args.output_dir}/'")


if __name__ == "__main__":
    main()
