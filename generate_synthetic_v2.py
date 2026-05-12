"""
Generate synthetic training images of an analog dial meter (128x128).

The pointer is: a central hub circle (radius = 0.3-0.5 of dial radius)
+ an isosceles triangle whose base is an internal chord of the hub circle
(length = 0.4-1.0 of hub diameter), with the apex pointing outward.

A very faint outer circle marks the dial boundary.  No scales or digits.

Label = (tip_x, tip_y) pixel coordinates of the pointer apex.
"""

import argparse
import colorsys
import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont

OUTPUT_DIR = "sv2"
IMG_SIZE = 256
CENTER = IMG_SIZE // 2  # 128
NUM_IMAGES = 15000

# Base dial face parameters (before scaling)
DIAL_RADIUS_BASE = IMG_SIZE // 3

# Scale: 0-9 clockwise, 36 deg per division, 0 at top
NUM_DIGITS = 10
DEG_PER_DIGIT = 360.0 / NUM_DIGITS
VALUE_MIN = 0.0
VALUE_MAX = 9.0

# ── geometry ──────────────────────────────────────────────────────────
SCALE_RANGE = (0.9, 1.1)            # dial radius multiplier (fraction of DIAL_RADIUS_BASE)
HUB_RATIO_RANGE = (0.4, 0.45)       # hub circle radius as fraction of dial radius
CHORD_RATIO_RANGE = (0.99, 1.0)     # pointer base chord length / hub diameter
HOLE_RATIO_RANGE = (1.05, 1.08)     # mounting hole radius / hub radius
HOLE_DISP_RANGE = (-8, 8)           # mounting hole centre offset from pointer centre (px)
TIP_VARIATION = 0.02                 # fractional jitter on pointer tip distance from centre
DISPLACEMENT_RANGE = (-3, 3)         # whole-dial displacement from canvas centre (px)
PTR_DISPLACEMENT_RANGE = (-5, 5)     # pointer+hole offset from dial centre (px)
STRETCH_H_RANGE = (0.98, 1.02)      # horizontal stretch factor applied inside dial
STRETCH_V_RANGE = (0.98, 1.02)      # vertical stretch factor applied inside dial
CANVAS_MARGIN = 30                   # px beyond dial_radius in tight canvas

# ── outer canvas (housing/bezel beyond dial face) ────────────────────
OUTER_SCALE_RANGE = (1.25, 1.32)     # outer boundary radius / dial radius
RIM_WIDTH = 3                        # rim band thickness (px)

# ── scale ticks ──────────────────────────────────────────────────────
TICKS_PER_DIGIT_RANGE = (1, 5)      # number of subdivision ticks between digits
TICK_ANGLE_JITTER = (-4, 4)         # shared angular offset for all ticks (degrees)
TICK_LEN_RANGE = (6, 9)             # major tick length (px); minor = half of this
TICK_WIDTH_RANGE = (2, 4)           # major tick line thickness (px); minor = max(1, this-1)
TICK_GREY_RANGE = (0, 30)           # grey value for tick colour (0=black, 30=dark grey)

# ── scale digits ─────────────────────────────────────────────────────
DIGIT_DIST = IMG_SIZE // 3 - 5      # fallback distance from centre (px, used when no ticks)
DIGIT_DIST_JITTER = (0, 4)         # bidirectional radial jitter from base position (px)
DIGIT_ANGLE_JITTER = (0, 2)         # shared angular jitter for all digits (degrees)
DIGIT_FONT_SIZE_RANGE = (16, 26)    # font size range for rendered digits
DIGIT_SHOW_PROB = 0.5               # probability of showing digits per image
DIGIT_GREY_RANGE = (0, 30)          # grey value for digit colour
DIGIT_GAP_RANGE = (0, 1)          # clearance gap between digit edge and tick edge (px)
DIGIT_INSIDE_PROB = 0.5             # probability digits are inside ticks (vs outside)

# ── HSV colour palette (shared hue/sat per image, varying value) ─────
DIAL_S_RANGE = (0, 0.15)            # dial palette saturation (low = near-grey)
BG_V_RANGE = (0.67, 0.71)           # background HSV value (darkest surface)
CIRCLE_V_RANGE = (0.84, 0.88)       # faint outer dial circle HSV value
OUTER_V_RANGE = (0.76, 0.86)        # annular region (between dial and bezel) HSV value
RIM_V_RANGE = (0.71, 0.76)          # rim band HSV value

# ── pointer colour ───────────────────────────────────────────────────
PTR_S_RANGE = (0.5, 1.0)            # pointer HSV value (bright half of the spectrum)
PTR_DARKER_V_FRAC = (0.4, 0.6)      # mounting hole / edge colour: fraction of pointer's V
PTR_GRADIENT_RANGE = (-25, 25)       # lengthwise brightness shift hub-to-tip
PTR_NOISE_RANGE = 25                  # fine per-pixel noise on pointer body
PTR_EDGE_THICKNESS_RANGE = (1, 3)    # edge band width (px)
PTR_EDGE_NOISE_RANGE = 5            # per-pixel random variation on pointer edge

# ── background noise / blur ──────────────────────────────────────────
BG_LOWFREQ_SIZE = 8                  # low-frequency gradient tile resolution
BG_VARIATION = 8                     # max per-channel deviation in low-freq background
NOISE_INTENSITY = 3                  # per-pixel RGB noise amplitude on background
BLUR_RANGE = (1, 3)                  # Gaussian blur radius range (unused / final pass)
DIAL_CIRCLE_BLUR = True              # enable Gaussian blur inside dial circle
DIAL_CIRCLE_BLUR_RADIUS = 3         # blur radius inside dial circle (px)
HIER_NOISE_BLOCKS = 5                # coarse noise grid size (NxN blocks)
HIER_NOISE_BLOCK_VARIANCE = 15       # per-block noise amplitude
HIER_NOISE_PIXEL_VARIANCE = 3        # fine per-pixel noise amplitude inside outer ring

# ── pointer tip erasure ──────────────────────────────────────────────
CUT_TIP = True                       # erase pointer pixels near apex (forces extrapolation)
CUT_TIP_RADIUS_RANGE = (7, 10)      # radius of erased circle around apex (px)

# ── shadows ──────────────────────────────────────────────────────────
SHADOW_PROB = 0.5                    # probability of rendering pointer + rim shadows
SHADOW_OFFSET_RANGE = (15, 25)       # pointer shadow displacement (px)
SHADOW_OPACITY_RANGE = (160, 180)    # pointer shadow alpha (0-255)
SHADOW_BLUR_RADIUS = 2.5             # Gaussian blur on pointer shadow layer (px)
SHADOW_S_RANGE = (0.05, 0.2)           # shadow saturation (low = near-grey, matches dial palette)
SHADOW_V_RANGE = (0.05, 0.25)       # shadow HSV value range (many shades of dark grey)
ARC_GAP_RANGE = (50, 120)            # arc gap width on dial circle (degrees)
RIM_SHADOW_OFFSET_FRAC = (0.15, 0.35)       # crescent offset as fraction of dial radius
RIM_SHADOW_CUTOUT_FRAC = (0.80, 0.95)       # crescent cutout radius / dial radius
RIM_SHADOW_PRIMARY_OPACITY = (200, 220)      # alpha for key-light rim shadow
RIM_SHADOW_SECONDARY_OPACITY = (200, 220)    # alpha for fill-light rim shadow
RIM_SHADOW_BLUR_RANGE = (4.0, 8.0)          # blur radius for rim shadow falloff (px)

# ── colour jitter (ImageEnhance, 1.0 = original) ────────────────────
BRIGHTNESS_RANGE = (0.95, 1.35)
CONTRAST_RANGE = (0.80, 1.20)
SATURATION_RANGE = (0.75, 1.25)
SHARPNESS_RANGE = (0.7, 1.5)

# ── adversarial triangular artifacts ─────────────────────────────────
TRI_ARTIFACT_ENABLED = True          # master toggle for triangle artifacts
TRI_ARTIFACT_COUNT = (0, 6)          # number of triangles per image
TRI_ARTIFACT_SIZE = (15, 40)         # triangle base length (px)
TRI_ARTIFACT_MEAN_RANGE = (50, 200)  # per-triangle per-channel colour mean
TRI_ARTIFACT_PIXEL_VAR = 5           # per-pixel noise around the channel mean
TRI_ARTIFACT_VERTEX_DIST = (0.3, 1.0)  # vertex distance multiplier (fraction of size)


def _hsv_rgb(h, s, v):
    """HSV (all 0-1) to (R, G, B) tuple in 0-255."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


# ── dial face ───────────────────────────────────────────────────────────

def draw_dial_face(draw, cx, cy, dial_radius, gap_start_deg=None, gap_span_deg=0,
                   augment=True, show_ticks=True, show_circle=True,
                   circle_color=None):
    """Draw a very faint outer circle with tick marks and one arc gap.

    gap_start_deg : where the gap begins (PIL arc degrees: 0 = 3-o'clock, CW)
    gap_span_deg  : angular width of the gap in degrees
    """
    faint = (circle_color + (255,)) if circle_color else (220, 220, 218, 255)

    if show_circle:
        if gap_span_deg <= 0 or gap_start_deg is None:
            draw.ellipse(
                [cx - dial_radius, cy - dial_radius,
                 cx + dial_radius, cy + dial_radius],
                outline=faint, width=1,
            )
        else:
            arc_start = (gap_start_deg + gap_span_deg) % 360
            arc_end = gap_start_deg % 360
            bbox = [cx - dial_radius, cy - dial_radius,
                    cx + dial_radius, cy + dial_radius]
            draw.arc(bbox, start=arc_start, end=arc_end, fill=faint, width=1)

    # 0-9 scale ticks drawn inward from the outer circle, with subdivision
    # ticks evenly spaced between each pair of consecutive digits.
    if not show_ticks:
        return dial_radius  # no ticks drawn, inner edge = outer edge
    tick_outer = dial_radius

    # Shared jitter for all ticks (same values applied uniformly)
    if augment:
        angle_jitter = random.uniform(*TICK_ANGLE_JITTER)
        grey = random.randint(*TICK_GREY_RANGE)
        tick_color = (grey, grey, grey, 255)
        ticks_per_digit = random.randint(*TICKS_PER_DIGIT_RANGE)
    else:
        angle_jitter = 0
        tick_color = (0, 0, 0, 255)
        ticks_per_digit = 1

    sub_deg = DEG_PER_DIGIT / ticks_per_digit
    max_major_len = 0  # track longest major tick for digit placement

    for i in range(NUM_DIGITS):
        for s in range(ticks_per_digit):
            nominal_deg = i * DEG_PER_DIGIT + s * sub_deg
            angle_deg = nominal_deg + angle_jitter
            angle_rad = math.radians(angle_deg - 90)  # -90 so 0° = up

            # Per-tick randomised length and width
            if augment:
                if s == 0:  # major tick
                    tick_len = random.randint(*TICK_LEN_RANGE)
                    tick_w = random.randint(*TICK_WIDTH_RANGE)
                    max_major_len = max(max_major_len, tick_len)
                else:  # minor tick
                    tick_len = max(1, random.randint(*TICK_LEN_RANGE) // 2)
                    tick_w = max(1, random.randint(*TICK_WIDTH_RANGE) - 1)
            else:
                tick_len = 5 if s == 0 else 2
                tick_w = 1

            tick_inner = tick_outer - tick_len
            x0 = cx + tick_inner * math.cos(angle_rad)
            y0 = cy + tick_inner * math.sin(angle_rad)
            x1 = cx + tick_outer * math.cos(angle_rad)
            y1 = cy + tick_outer * math.sin(angle_rad)
            draw.line([(x0, y0), (x1, y1)], fill=tick_color, width=tick_w)

    if max_major_len == 0:
        max_major_len = TICK_LEN_RANGE[1]  # fallback for non-augment
    return tick_outer - max_major_len  # inner edge of longest major tick


def draw_scale_digits(draw, cx, cy, dial_radius, augment=True,
                      tick_inner_radius=None):
    """Draw digits 0-9 around the dial. 0 at top (positive y), clockwise.

    If tick_inner_radius is given, digits are placed either inside or
    outside the ticks (50/50) with a clearance gap so they never overlap.
    dist_jitter applies bidirectionally from the base position.
    """
    font_size = random.randint(*DIGIT_FONT_SIZE_RANGE) if augment else DIGIT_FONT_SIZE_RANGE[0]
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()

    # Shared jitter for all digits (same values applied uniformly)
    if augment:
        ang_jitter = random.uniform(*DIGIT_ANGLE_JITTER) * random.choice([-1, 1])
        grey = random.randint(*DIGIT_GREY_RANGE)
        color = (grey, grey, grey, 255)
    else:
        ang_jitter = 0
        color = (0, 0, 0, 255)

    # Place digits inside or outside ticks, with a clearance gap
    if tick_inner_radius is not None and augment:
        gap = random.uniform(*DIGIT_GAP_RANGE)
        inside = random.random() < DIGIT_INSIDE_PROB
        if inside:
            base_dist = tick_inner_radius - font_size / 2 - gap
        else:
            tick_outer = dial_radius
            base_dist = tick_outer + font_size / 2 + gap
    else:
        base_dist = DIGIT_DIST
    dist_jitter = random.uniform(*DIGIT_DIST_JITTER) * random.choice([-1, 1]) if augment else 0
    dist = base_dist + dist_jitter  # same for all digits in this image

    for i in range(NUM_DIGITS):
        nominal_deg = i * DEG_PER_DIGIT
        angle_deg = nominal_deg + ang_jitter
        angle_rad = math.radians(angle_deg - 90)  # -90 so 0° = up

        dx = dist * math.cos(angle_rad)
        dy = dist * math.sin(angle_rad)

        text = str(i)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = cx + dx - tw / 2
        ty = cy + dy - th / 2
        draw.text((tx, ty), text, fill=color, font=font)


# ── shadow ──────────────────────────────────────────────────────────────

def draw_shadow(shadow_draw, apex, base_left, base_right,
                cx, cy, hub_radius, dial_radius,
                offset_dx, offset_dy, opacity,
                gap_start_deg=None, gap_span_deg=0,
                base_rgb=(40, 40, 40)):
    """Draw a soft, offset shadow of the pointer shape and the outer circle.

    The shadow is blurred later, so the tip naturally loses its sharpness.
    """
    shadow_color = (*base_rgb, opacity)

    # Offset triangle
    s_apex = (apex[0] + offset_dx, apex[1] + offset_dy)
    s_bl = (base_left[0] + offset_dx, base_left[1] + offset_dy)
    s_br = (base_right[0] + offset_dx, base_right[1] + offset_dy)
    shadow_draw.polygon([s_apex, s_bl, s_br], fill=shadow_color)

    # Offset hub
    scx = cx + offset_dx
    scy = cy + offset_dy
    shadow_draw.ellipse(
        [scx - hub_radius, scy - hub_radius,
         scx + hub_radius, scy + hub_radius],
        fill=shadow_color,
    )

    # Offset outer circle (with matching arc gap)
    bbox = [scx - dial_radius, scy - dial_radius,
            scx + dial_radius, scy + dial_radius]
    circle_shadow = (40, 40, 40, opacity // 2)  # lighter than pointer shadow
    if gap_span_deg > 0 and gap_start_deg is not None:
        arc_start = (gap_start_deg + gap_span_deg) % 360
        arc_end = gap_start_deg % 360
        shadow_draw.arc(bbox, start=arc_start, end=arc_end,
                        fill=circle_shadow, width=2)
    else:
        shadow_draw.ellipse(bbox, outline=circle_shadow, width=2)


# ── pointer colour variation ───────────────────────────────────────────

def _pointer_color_variation(img, mask, base_rgb, cx, cy, angle_deg,
                             hub_radius, tip_dist):
    """Apply realistic structured colour variation to the pointer body.

    Layers (applied additively to brightness):
    1. Lengthwise gradient — smooth brightness ramp from hub to tip.
    2. Fine per-pixel noise — subtle texture.
    """
    arr = np.array(img).astype(np.int16)
    m = np.array(mask)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return img

    a = math.radians(angle_deg - 90)
    axis_dx = math.cos(a)   # unit vector along pointer (hub → tip)
    axis_dy = math.sin(a)
    # Signed distance along axis from centre, normalised to [-1, 1]
    rel_x = (xs - cx).astype(np.float32)
    rel_y = (ys - cy).astype(np.float32)
    along = rel_x * axis_dx + rel_y * axis_dy          # projection on axis
    along_norm = along / max(tip_dist, 1)               # -1 .. +1 (hub .. tip)

    # ── 1. Lengthwise gradient ──
    grad_strength = random.uniform(*PTR_GRADIENT_RANGE)
    grad_shift = (along_norm * grad_strength).astype(np.int16)

    # ── 2. Fine per-pixel noise ──
    noise = np.random.randint(-PTR_NOISE_RANGE, PTR_NOISE_RANGE + 1,
                              size=len(ys), dtype=np.int16)

    # Combine all shifts
    total_shift = grad_shift + noise

    for c in range(3):
        arr[ys, xs, c] = np.clip(arr[ys, xs, c] + total_shift, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), img.mode)


def _draw_pointer_edge(img, mask, edge_color_rgb, bg_arr):
    """Draw a variable-thickness edge on the boundary of the pointer mask.

    Edge thickness is randomised per image.  Pixels in the outermost
    erosion band are blended mostly toward the background (the actual
    dial face colour from *bg_arr*, captured before the pointer was
    drawn), while inner bands are mostly the edge colour.
    Each pixel also gets independent random noise.
    """
    thickness = random.randint(*PTR_EDGE_THICKNESS_RANGE)
    m = np.array(mask)
    arr = np.array(img)

    # Build per-pixel distance from the mask boundary (0 = outermost)
    # by successive erosions.  distance[y,x] = how many erosion steps
    # the pixel survives (0 for the outermost ring, 1 for the next, …).
    dist = np.full(m.shape, thickness, dtype=np.float32)
    current = m.copy()
    for step in range(thickness):
        e1 = np.minimum(np.roll(current, 1, 0), np.roll(current, -1, 0))
        e2 = np.minimum(np.roll(current, 1, 1), np.roll(current, -1, 1))
        eroded = np.minimum(e1, e2) & current
        # Pixels that just disappeared are at this erosion depth
        ring = (current > 0) & (eroded == 0)
        dist[ring] = step
        current = eroded

    # Edge band = mask pixels within the thickness
    edge_band = (m > 0) & (dist < thickness)
    ys, xs = np.where(edge_band)
    if len(ys) == 0:
        return img

    # Blend factor: 0.0 at outermost ring (= background), 1.0 at
    # innermost ring (= full edge colour).  This makes the edge
    # gradually assimilate into the background.
    blend = dist[ys, xs] / max(thickness - 1, 1)

    # Per-pixel random noise on each channel
    noise = np.random.randint(-PTR_EDGE_NOISE_RANGE, PTR_EDGE_NOISE_RANGE + 1,
                              size=(len(ys), 3), dtype=np.int16)

    bg = bg_arr[ys, xs, :3].astype(np.int16)
    ec = np.array(edge_color_rgb, dtype=np.int16)
    blended = (bg + blend[:, None] * (ec - bg) + noise).astype(np.int16)
    arr[ys, xs, :3] = np.clip(blended, 0, 255).astype(np.uint8)
    if arr.shape[2] == 4:
        arr[ys, xs, 3] = 255
    return Image.fromarray(arr, img.mode)


# ── pointer ─────────────────────────────────────────────────────────────

def draw_pointer(img, cx, cy, angle_deg, hub_radius, tip_dist,
                 chord_half, chord_dist, pointer_base_rgb, edge_color_rgb=None,
                 cut_tip_radius=2):
    """Draw the pointer with realistic colour variation.  Return (apex, bl, br)."""
    a = math.radians(angle_deg - 90)

    apex = (cx + tip_dist * math.cos(a),
            cy + tip_dist * math.sin(a))

    base_mx = cx - chord_dist * math.cos(a)
    base_my = cy - chord_dist * math.sin(a)

    perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
    base_left = (base_mx + chord_half * perp[0],
                 base_my + chord_half * perp[1])
    base_right = (base_mx - chord_half * perp[0],
                  base_my - chord_half * perp[1])

    draw = ImageDraw.Draw(img)

    # Build a mask of the pointer shape
    mask = Image.new("L", img.size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon([apex, base_left, base_right], fill=255)
    mdraw.ellipse([cx - hub_radius, cy - hub_radius,
                   cx + hub_radius, cy + hub_radius], fill=255)

    # Snapshot background before painting (for edge blending outward)
    bg_arr = np.array(img)

    # Fill pointer region with base colour
    base_rgba = pointer_base_rgb + (255,)
    draw.polygon([apex, base_left, base_right], fill=base_rgba, outline=base_rgba)
    draw.ellipse([cx - hub_radius, cy - hub_radius,
                  cx + hub_radius, cy + hub_radius],
                 fill=base_rgba, outline=base_rgba)

    # Apply structured colour variation
    img = _pointer_color_variation(img, mask, pointer_base_rgb, cx, cy,
                                   angle_deg, hub_radius, tip_dist)

    # Draw edge in a colour from the luminance strip of the pointer colour
    if edge_color_rgb is not None:
        img = _draw_pointer_edge(img, mask, edge_color_rgb, bg_arr)

    # Optionally erase the tip region (pointer drawn without its apex)
    if CUT_TIP:
        r = cut_tip_radius
        mask_arr = np.array(mask)
        ys_t, xs_t = np.ogrid[:mask_arr.shape[0], :mask_arr.shape[1]]
        tip_circle = (xs_t - apex[0]) ** 2 + (ys_t - apex[1]) ** 2 <= r ** 2
        mask_arr[tip_circle] = 0
        mask = Image.fromarray(mask_arr, "L")

    return img, apex, base_left, base_right, mask


# ── glare ───────────────────────────────────────────────────────────────

# ── rim shadow (bowl cavity) ───────────────────────────────────────────

def draw_rim_shadows(cx, cy, dial_radius, canvas_size, base_rgb=(30, 30, 30)):
    """Create a blurred RGBA layer with crescent shadows cast by the bowl rim.

    Physical model: the dial is a recessed bowl.  A light source illuminates
    one side; the opposite rim arc blocks it and casts a crescent-shaped
    shadow inward onto the dial face.

    Geometry: the crescent (lune) is the area inside the dial circle but
    *outside* a cutout circle that is shifted toward the light source.  The
    cutout "bites" from the light side, leaving a crescent that hugs the rim
    on the shadow side.

    Returns an RGBA Image ready to alpha-composite onto the main canvas.
    """
    layer_a = np.zeros((canvas_size, canvas_size), dtype=np.float32)

    # Pixel coordinate grids
    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    dial_dist2 = (xs - cx) ** 2 + (ys - cy) ** 2
    dial_mask = dial_dist2 <= dial_radius ** 2

    light_angle = random.uniform(0, 360)

    # Single light source
    light_rad = math.radians(light_angle - 90)  # convert: 0=up,CW → trig

    offset_frac = random.uniform(*RIM_SHADOW_OFFSET_FRAC)
    cutout_frac = random.uniform(*RIM_SHADOW_CUTOUT_FRAC)
    co_offset = offset_frac * dial_radius
    co_radius = cutout_frac * dial_radius

    # Cutout centre shifted toward the light
    co_cx = cx + co_offset * math.cos(light_rad)
    co_cy = cy + co_offset * math.sin(light_rad)

    cutout_mask = ((xs - co_cx) ** 2 + (ys - co_cy) ** 2) <= co_radius ** 2

    # Crescent: inside dial AND outside cutout
    crescent = dial_mask & (~cutout_mask)

    opacity = random.randint(*RIM_SHADOW_PRIMARY_OPACITY)
    layer_a[crescent] += opacity

    layer_a = np.clip(layer_a, 0, 255)

    rgba = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    rgba[:, :, 0] = base_rgb[0]
    rgba[:, :, 1] = base_rgb[1]
    rgba[:, :, 2] = base_rgb[2]
    rgba[:, :, 3] = layer_a.astype(np.uint8)

    layer = Image.fromarray(rgba, "RGBA")
    blur_r = random.uniform(*RIM_SHADOW_BLUR_RANGE)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return layer, light_angle


# ── edge-mean padding ─────────────────────────────────────────────────

def pad_to_target(img, target_size, pad_left, pad_top):
    """Pad an RGB image to *target_size* x *target_size* with edge-mean fill.

    Uses the same logic as ``infer_tflite.preprocess``: the fill colour is
    the mean of the two edge rows / columns so the transition is smooth.
    """
    arr = np.array(img)
    h, w = arr.shape[:2]
    ch = arr.shape[2]
    pad_right = target_size - w - pad_left
    pad_bottom = target_size - h - pad_top

    # --- horizontal padding ---
    if pad_left > 0 or pad_right > 0:
        edge_mean = arr[:, [0, -1], :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((h, target_size, ch), dtype=np.uint8)
        for c in range(ch):
            if pad_left > 0:
                new[:, :pad_left, c] = edge_mean[c]
            new[:, pad_left:pad_left + w, c] = arr[:, :, c]
            if pad_right > 0:
                new[:, pad_left + w:, c] = edge_mean[c]
        arr = new

    # --- vertical padding ---
    cur_h, cur_w = arr.shape[:2]
    if pad_top > 0 or pad_bottom > 0:
        edge_mean = arr[[0, -1], :, :].mean(axis=(0, 1)).astype(np.uint8)
        new = np.empty((target_size, cur_w, ch), dtype=np.uint8)
        for c in range(ch):
            if pad_top > 0:
                new[:pad_top, :, c] = edge_mean[c]
            new[pad_top:pad_top + cur_h, :, c] = arr[:, :, c]
            if pad_bottom > 0:
                new[pad_top + cur_h:, :, c] = edge_mean[c]
        arr = new

    return Image.fromarray(arr, img.mode)


# ── helpers ─────────────────────────────────────────────────────────────

def value_to_angle(value):
    return value * DEG_PER_DIGIT


def _random_pointer_rgb():
    """Return a random (R, G, B) from the top half of HSV value axis."""
    h = random.random()
    v = random.random()
    s = random.uniform(*PTR_S_RANGE)
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (int(r * 255), int(g * 255), int(b * 255))


def _luminance_strip_color(base_rgb, darker_only=False):
    """Pick a random colour from the lighter/darker strip of *base_rgb*.

    Same hue and saturation, random brightness -- like the vertical
    luminance strip in the Windows colour-picker dialog.

    If *darker_only* is True, the brightness is capped at the base
    colour's own value so the result is never brighter than the input.
    """
    r, g, b = [c / 255.0 for c in base_rgb]
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    new_v = random.uniform(PTR_DARKER_V_FRAC[0]*v, PTR_DARKER_V_FRAC[1]*v) if darker_only else random.random()
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, new_v)
    return (int(nr * 255), int(ng * 255), int(nb * 255))


# ── main generation ────────────────────────────────────────────────────

def generate_image(angle_deg, path, augment=True):
    """Generate one synthetic dial image and return (tip_x, tip_y).

    The dial is rendered on a tight canvas sized to the scaled dial body,
    then edge-mean-padded to IMG_SIZE x IMG_SIZE (matching the padding
    logic in ``infer_tflite.preprocess``).
    """

    # --- scaling ---
    scale = random.uniform(*SCALE_RANGE) if augment else 1.0
    dial_radius = int(DIAL_RADIUS_BASE * scale)

    # --- canvas size (tight around dial + margin for shadows/ticks) ---
    canvas_size = 2 * (dial_radius + CANVAS_MARGIN)
    if canvas_size % 2 != 0:
        canvas_size += 1
    cx = canvas_size // 2
    cy = canvas_size // 2

    # --- displacement (applied during padding, not drawing) ---
    if augment:
        disp_mag = random.uniform(*DISPLACEMENT_RANGE)
        disp_ang = random.uniform(0, 2 * math.pi)
        disp_x = int(round(disp_mag * math.cos(disp_ang)))
        disp_y = int(round(disp_mag * math.sin(disp_ang)))
    else:
        disp_x, disp_y = 0, 0

    # --- hub ---
    hub_ratio = random.uniform(*HUB_RATIO_RANGE) if augment else 0.4
    hub_radius = dial_radius * hub_ratio

    # --- triangle base chord ---
    chord_ratio = random.uniform(*CHORD_RATIO_RANGE) if augment else 1.0
    chord_length = chord_ratio * hub_radius * 2
    chord_half = chord_length / 2.0
    max_d = math.sqrt(max(hub_radius ** 2 - chord_half ** 2, 0))
    chord_dist = random.uniform(0, max_d) if (augment and max_d > 0) else 0.0

    # --- tip distance ---
    tick_inner = int(dial_radius)
    base_tip_dist = tick_inner - 2
    tip_var = random.uniform(-TIP_VARIATION, TIP_VARIATION) if augment else 0.0
    tip_dist = base_tip_dist * (1 + tip_var)

    # --- arc gap ---
    if augment:
        gap_start_deg = random.uniform(0, 360)
        gap_span_deg = random.uniform(*ARC_GAP_RANGE)
    else:
        gap_start_deg = None
        gap_span_deg = 0

    # --- pointer / hole displacement from dial centre ---
    if augment:
        ptr_disp_mag = random.uniform(*PTR_DISPLACEMENT_RANGE)
        ptr_disp_ang = random.uniform(0, 2 * math.pi)
        ptr_dx = ptr_disp_mag * math.cos(ptr_disp_ang)
        ptr_dy = ptr_disp_mag * math.sin(ptr_disp_ang)
    else:
        ptr_dx, ptr_dy = 0, 0
    ptr_cx = cx + ptr_dx
    ptr_cy = cy + ptr_dy

    # --- shadow / rim base colours (HSV grey shades) ---
    def _sample_shadow_rgb():
        h = random.random()
        s = random.uniform(*SHADOW_S_RANGE)
        v = random.uniform(*SHADOW_V_RANGE)
        return _hsv_rgb(h, s, v)

    if augment:
        pointer_shadow_rgb = _sample_shadow_rgb()
        rim_shadow_rgb = _sample_shadow_rgb()
    else:
        pointer_shadow_rgb = (30, 30, 30)
        rim_shadow_rgb = (30, 30, 30)

    # --- shared HSV palette for background / rim / outer dial ---
    if augment:
        dial_h = random.random()           # shared hue
        dial_s = random.uniform(*DIAL_S_RANGE)
        bg_v = random.uniform(*BG_V_RANGE)
        bg_rgb = _hsv_rgb(dial_h, dial_s, bg_v)
        circle_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*CIRCLE_V_RANGE))
    else:
        bg_rgb = (175, 175, 175)
        circle_rgb = (220, 220, 218)

    # --- non-uniform background (small canvas) ---
    if augment:
        bg_base = np.array(bg_rgb, dtype=np.int16)

        lf_var = np.random.randint(-BG_VARIATION, BG_VARIATION + 1,
                                   (BG_LOWFREQ_SIZE, BG_LOWFREQ_SIZE, 3),
                                   dtype=np.int16)
        lf_img = Image.fromarray(
            np.clip(bg_base + lf_var, 0, 255).astype(np.uint8), "RGB"
        )
        lf_img = lf_img.resize((canvas_size, canvas_size), Image.BICUBIC)
        bg_arr = np.array(lf_img, dtype=np.int16)

        noise = np.random.randint(-NOISE_INTENSITY, NOISE_INTENSITY + 1,
                                  (canvas_size, canvas_size, 3), dtype=np.int16)
        bg_arr = np.clip(bg_arr + noise, 0, 255).astype(np.uint8)

        alpha = np.full((canvas_size, canvas_size, 1), 255, dtype=np.uint8)
        bg_arr = np.concatenate([bg_arr, alpha], axis=2)
        img = Image.fromarray(bg_arr, "RGBA")
    else:
        img = Image.new("RGBA", (canvas_size, canvas_size), "white")

    draw = ImageDraw.Draw(img)

    # --- faint dial circle (with gap) + scale marks/digits ---
    show_digits = random.random() < DIGIT_SHOW_PROB if augment else True
    draw_dial_face(draw, cx, cy, dial_radius, gap_start_deg, gap_span_deg,
                   augment=augment, show_ticks=False, circle_color=circle_rgb)

    # Ticks & digits on a separate RGBA overlay so the outer-dial annular
    # fill and rim band (drawn later) cannot occlude them.
    td_layer = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    td_draw = ImageDraw.Draw(td_layer)
    tick_inner_r = None
    if show_digits:
        tick_inner_r = draw_dial_face(td_draw, cx, cy, dial_radius,
                                      gap_start_deg, gap_span_deg,
                                      augment=augment, show_ticks=True,
                                      show_circle=False)

    # --- shadows (rim + pointer, controlled by one random boolean) ---
    has_shadow = random.random() < SHADOW_PROB if augment else False
    light_angle = random.uniform(0, 360)

    if augment and has_shadow:
        rim_layer, light_angle = draw_rim_shadows(cx, cy, dial_radius,
                                                  canvas_size,
                                                  base_rgb=rim_shadow_rgb)
        img = Image.alpha_composite(img, rim_layer)

    if augment and has_shadow:
        shadow_layer = Image.new("RGBA", (canvas_size, canvas_size),
                                 (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_layer)

        a = math.radians(angle_deg - 90)
        s_apex = (ptr_cx + tip_dist * math.cos(a), ptr_cy + tip_dist * math.sin(a))
        bmx = ptr_cx - chord_dist * math.cos(a)
        bmy = ptr_cy - chord_dist * math.sin(a)
        perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
        s_bl = (bmx + chord_half * perp[0], bmy + chord_half * perp[1])
        s_br = (bmx - chord_half * perp[0], bmy - chord_half * perp[1])

        # Shadow falls on the side away from the light.
        # Rim cutout is shifted toward light at light_rad; crescent is opposite.
        # Pointer shadow offset must point the same way as the crescent.
        light_rad = math.radians(light_angle - 90)  # same conv as rim
        shd_dist = random.uniform(*SHADOW_OFFSET_RANGE)
        # Offset AWAY from light (negate the light direction)
        shd_dx = shd_dist * math.cos(light_rad)
        shd_dy = shd_dist * math.sin(light_rad)
        shd_opacity = random.randint(*SHADOW_OPACITY_RANGE)

        draw_shadow(sdraw, s_apex, s_bl, s_br,
                    ptr_cx, ptr_cy, hub_radius, dial_radius,
                    shd_dx, shd_dy, shd_opacity,
                    gap_start_deg, gap_span_deg,
                    base_rgb=pointer_shadow_rgb)

        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=SHADOW_BLUR_RADIUS))
        img = Image.alpha_composite(img, shadow_layer)

    # --- scale digits on overlay ---
    if show_digits:
        draw_scale_digits(td_draw, cx, cy, dial_radius, augment=augment,
                          tick_inner_radius=tick_inner_r)

    # --- pointer with non-uniform colour + feathered edges ---
    hole_ratio = random.uniform(*HOLE_RATIO_RANGE) if augment else 1.25
    hole_r = hub_radius * hole_ratio
    ptr_rgb = _random_pointer_rgb() if augment else (210, 30, 30)

    # Draw mounting hole BEFORE saving pre-pointer background so it
    # survives the CUT_TIP restoration (fixes hole-invisible bug).
    hole_color = _luminance_strip_color(ptr_rgb, darker_only=True) + (255,)
    if augment:
        hd_mag = random.uniform(*HOLE_DISP_RANGE)
        hd_ang = random.uniform(0, 2 * math.pi)
        hole_cx = ptr_cx + hd_mag * math.cos(hd_ang)
        hole_cy = ptr_cy + hd_mag * math.sin(hd_ang)
    else:
        hole_cx, hole_cy = ptr_cx, ptr_cy
    hole_draw = ImageDraw.Draw(img)
    hole_draw.ellipse([hole_cx - hole_r, hole_cy - hole_r,
                       hole_cx + hole_r, hole_cy + hole_r],
                      fill=hole_color)

    bg_pre_ptr = img.copy() if CUT_TIP else None
    cut_r = random.randint(*CUT_TIP_RADIUS_RANGE) if augment else 2
    edge_color = _luminance_strip_color(ptr_rgb, darker_only=True)
    img, apex, bl, br, ptr_mask = draw_pointer(
        img, ptr_cx, ptr_cy, angle_deg,
        hub_radius=hub_radius, tip_dist=tip_dist,
        chord_half=chord_half, chord_dist=chord_dist,
        pointer_base_rgb=ptr_rgb, edge_color_rgb=edge_color,
        cut_tip_radius=cut_r,
    )
    if CUT_TIP and bg_pre_ptr is not None:
        inv_mask = Image.fromarray(255 - np.array(ptr_mask), "L")
        img.paste(bg_pre_ptr, mask=inv_mask)

    # --- random horizontal / vertical stretch inside dial ---
    if augment:
        stretch_h = random.uniform(*STRETCH_H_RANGE)
        stretch_v = random.uniform(*STRETCH_V_RANGE)
        affine_coeffs = (1.0 / stretch_h, 0, cx * (1 - 1.0 / stretch_h),
                         0, 1.0 / stretch_v, cy * (1 - 1.0 / stretch_v))
        img = img.transform(
            (canvas_size, canvas_size),
            Image.AFFINE,
            affine_coeffs,
            resample=Image.BICUBIC,
        )
        td_layer = td_layer.transform(
            (canvas_size, canvas_size),
            Image.AFFINE,
            affine_coeffs,
            resample=Image.BICUBIC,
        )
        apex = (cx + (apex[0] - cx) * stretch_h,
                cy + (apex[1] - cy) * stretch_v)
        dial_rx = dial_radius * stretch_h
        dial_ry = dial_radius * stretch_v
    else:
        dial_rx = dial_radius
        dial_ry = dial_radius

    # --- convert to RGB ---
    img = img.convert("RGB")

    # --- colour / exposure jitter ---
    if augment:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
        img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # --- outer canvas (housing/bezel around dial) ---
    outer_scale = random.uniform(*OUTER_SCALE_RANGE) if augment else 1.08
    outer_rx = dial_rx * outer_scale
    outer_ry = dial_ry * outer_scale

    arr = np.array(img)
    ys, xs = np.ogrid[:canvas_size, :canvas_size]
    inner_r2 = ((xs - cx) / max(dial_rx, 1)) ** 2 + ((ys - cy) / max(dial_ry, 1)) ** 2
    outer_r2 = ((xs - cx) / max(outer_rx, 1)) ** 2 + ((ys - cy) / max(outer_ry, 1)) ** 2
    inside_outer = outer_r2 <= 1
    annular = (inner_r2 > 1) & inside_outer

    # Fill annular region (between dial and outer boundary)
    if augment:
        outer_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*OUTER_V_RANGE))
    else:
        outer_rgb = (70, 70, 70)
    arr[annular] = outer_rgb

    # Rim at dial boundary (before noise & blur so edges blend naturally)
    r_norm = np.sqrt(inner_r2)
    rim_half = RIM_WIDTH / min(max(dial_rx, 1), max(dial_ry, 1))
    rim_band = (r_norm > 1 - rim_half) & (r_norm < 1 + rim_half)
    if augment:
        rim_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*RIM_V_RANGE))
    else:
        rim_rgb = (190, 190, 190)
    arr[rim_band] = rim_rgb

    # Hierarchical noise inside outer canvas
    if augment:
        # Block-level noise (coarse 5x5 grid, higher variance)
        block_vals = np.random.randint(
            -HIER_NOISE_BLOCK_VARIANCE, HIER_NOISE_BLOCK_VARIANCE + 1,
            (HIER_NOISE_BLOCKS, HIER_NOISE_BLOCKS), dtype=np.int16)
        block_u8 = np.clip(block_vals + 128, 0, 255).astype(np.uint8)
        block_up = np.array(
            Image.fromarray(block_u8, "L").resize(
                (canvas_size, canvas_size), Image.BICUBIC),
            dtype=np.int16) - 128
        # Per-pixel noise (fine grain, lower variance)
        pixel_noise = np.random.randint(
            -HIER_NOISE_PIXEL_VARIANCE, HIER_NOISE_PIXEL_VARIANCE + 1,
            (canvas_size, canvas_size), dtype=np.int16)
        combined = block_up + pixel_noise
        for c in range(3):
            ch = arr[:, :, c].astype(np.int16)
            ch[inside_outer] += combined[inside_outer]
            arr[:, :, c] = np.clip(ch, 0, 255).astype(np.uint8)

    # Adversarial triangular artifacts (before blur so they get softened)
    if augment:
        # Stretch ptr_mask to match current (post-stretch) coordinate space
        ptr_mask_s = ptr_mask.transform(
            (canvas_size, canvas_size),
            Image.AFFINE,
            (1.0 / stretch_h, 0, cx * (1 - 1.0 / stretch_h),
             0, 1.0 / stretch_v, cy * (1 - 1.0 / stretch_v)),
            resample=Image.NEAREST,
        )
        ptr_blocked = np.array(ptr_mask_s) > 127
        # Also block mounting hole (stretched coordinates)
        s_hole_cx = cx + (hole_cx - cx) * stretch_h
        s_hole_cy = cy + (hole_cy - cy) * stretch_v
        s_hole_rx = hole_r * stretch_h
        s_hole_ry = hole_r * stretch_v
        hole_dist2 = ((xs - s_hole_cx) / max(s_hole_rx, 1)) ** 2 + \
                     ((ys - s_hole_cy) / max(s_hole_ry, 1)) ** 2
        ptr_blocked |= (hole_dist2 <= 1)

        n_tris = random.randint(*TRI_ARTIFACT_COUNT) if TRI_ARTIFACT_ENABLED else 0

        for _ in range(n_tris):
            base_rgb = tuple(random.randint(*TRI_ARTIFACT_MEAN_RANGE) for _ in range(3))
            # Random centre anywhere on canvas
            tcx = random.randint(0, canvas_size - 1)
            tcy = random.randint(0, canvas_size - 1)
            sz = random.randint(*TRI_ARTIFACT_SIZE)

            # 3 random vertices around centre
            verts = []
            for _v in range(3):
                va = random.uniform(0, 2 * math.pi)
                vr = random.uniform(*TRI_ARTIFACT_VERTEX_DIST) * sz
                verts.append((tcx + vr * math.cos(va),
                              tcy + vr * math.sin(va)))

            # Rasterize triangle
            tri_mask = Image.new("L", (canvas_size, canvas_size), 0)
            ImageDraw.Draw(tri_mask).polygon(verts, fill=255)
            tri_pixels = np.array(tri_mask) > 127

            # Only place where inside outer canvas AND not on pointer
            valid = tri_pixels & inside_outer & (~ptr_blocked)
            if not valid.any():
                continue

            # Noisy fill: per-channel mean + per-pixel variance
            for c in range(3):
                noise = np.random.randint(
                    -TRI_ARTIFACT_PIXEL_VAR, TRI_ARTIFACT_PIXEL_VAR + 1,
                    (canvas_size, canvas_size), dtype=np.int16)
                fill = np.clip(base_rgb[c] + noise, 0, 255).astype(np.uint8)
                arr[:, :, c][valid] = fill[valid]

    arr[~inside_outer] = 0
    img = Image.fromarray(arr, img.mode)

    # Erase ticks/digits that overlap pointer body or mounting hole
    if augment:
        td_arr = np.array(td_layer)
        td_arr[ptr_blocked, 3] = 0
        td_layer = Image.fromarray(td_arr, "RGBA")

    # Composite tick/digit overlay so scales aren't occluded by outer dial
    img = Image.alpha_composite(img.convert("RGBA"), td_layer).convert("RGB")

    # Gaussian blur inside outer canvas
    if augment and DIAL_CIRCLE_BLUR:
        dcb_r = DIAL_CIRCLE_BLUR_RADIUS
        if dcb_r > 0.1:
            blurred = img.filter(ImageFilter.GaussianBlur(radius=dcb_r))
            outer_mask_img = Image.new("L", (canvas_size, canvas_size), 0)
            ImageDraw.Draw(outer_mask_img).ellipse(
                [cx - outer_rx, cy - outer_ry,
                 cx + outer_rx, cy + outer_ry], fill=255)
            img = Image.composite(blurred, img, outer_mask_img)

    # Mask everything outside outer canvas to black
    arr = np.array(img)
    arr[~inside_outer] = 0
    img = Image.fromarray(arr, img.mode)

    # --- pad to IMG_SIZE x IMG_SIZE (edge-mean, centre-aligned + displacement) ---
    pad_left = (IMG_SIZE - canvas_size) // 2 + disp_x
    pad_top = (IMG_SIZE - canvas_size) // 2 + disp_y
    pad_left = max(0, min(pad_left, IMG_SIZE - canvas_size))
    pad_top = max(0, min(pad_top, IMG_SIZE - canvas_size))

    img = pad_to_target(img, IMG_SIZE, pad_left, pad_top)
    img.save(path)

    # Tip coordinates in final (padded) image space
    tip_x = apex[0] + pad_left
    tip_y = apex[1] + pad_top
    return tip_x, tip_y


def main():
    global DIAL_CIRCLE_BLUR
    parser = argparse.ArgumentParser(description="Generate synthetic dial images")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help=f"Output directory (default: {OUTPUT_DIR})")
    parser.add_argument("--dial-circle-blur", action="store_true",
                        default=DIAL_CIRCLE_BLUR,
                        help="Enable gaussian blur inside dial circle")
    args = parser.parse_args()

    DIAL_CIRCLE_BLUR = args.dial_circle_blur
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, "labels.csv")
    rows = []

    for i in range(NUM_IMAGES):
        angle_deg = random.randint(0, 359)
        fname = f"dial_{i:04d}.png"
        fpath = os.path.join(output_dir, fname)

        tip_x, tip_y = generate_image(angle_deg, fpath)
        rows.append({"filename": fname,
                      "angle_deg": angle_deg,
                      "tip_x": round(tip_x, 2),
                      "tip_y": round(tip_y, 2)})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "angle_deg", "tip_x", "tip_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_IMAGES} images in '{output_dir}/'")
    print(f"Labels written to '{csv_path}'")


if __name__ == "__main__":
    main()
