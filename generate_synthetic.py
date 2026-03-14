"""
Generate synthetic training images of an analog dial meter (128x128).

The pointer is: a central hub circle (radius = 0.3-0.5 of dial radius)
+ an isosceles triangle whose base is an internal chord of the hub circle
(length = 0.4-1.0 of hub diameter), with the apex pointing outward.

A very faint outer circle marks the dial boundary.  No scales or digits.

Label = (tip_x, tip_y) pixel coordinates of the pointer apex.
"""

import csv
import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

OUTPUT_DIR = "synthetic_data"
IMG_SIZE = 192
CENTER = IMG_SIZE // 2  # 128
NUM_IMAGES = 20

# Base dial face parameters (before scaling)
DIAL_RADIUS_BASE = 64

# Scale: 0-9 clockwise, 36 deg per division, 0 at top
NUM_DIGITS = 10
DEG_PER_DIGIT = 360.0 / NUM_DIGITS
VALUE_MIN = 0.0
VALUE_MAX = 9.0

# Augmentation ranges
SCALE_RANGE = (0.75, 1.0)
HUB_RATIO_RANGE = (0.32, 0.38)
CHORD_RATIO_RANGE = (0.8, 1.0)     # chord_length / hub_diameter
TIP_VARIATION = 0.10
DISPLACEMENT_RANGE = (3, 5)
NOISE_INTENSITY = 12
BLUR_RANGE = (0.3, 0.7)

# Shadow parameters (pointer cast shadow)
SHADOW_OFFSET_RANGE = (10, 15)        # pixels
SHADOW_OPACITY_RANGE = (160, 180)    # alpha 0-255  (≈30-70%)
ARC_GAP_RANGE = (50, 120)           # degrees of arc to delete

# Rim shadow (bowl cavity — crescent/lune model)
# Cutout circle offset toward the light, as fraction of dial_radius
# (larger offset → thicker crescent on the shadow side)
RIM_SHADOW_OFFSET_FRAC = (0.15, 0.35)
# Cutout circle radius relative to dial_radius
# (smaller → thicker crescent, larger → thinner crescent)
RIM_SHADOW_CUTOUT_FRAC = (0.80, 0.95)
RIM_SHADOW_PRIMARY_OPACITY = (200, 200)      # alpha for key light shadow
RIM_SHADOW_SECONDARY_OPACITY = (200, 200)     # alpha for fill light shadow
RIM_SHADOW_BLUR_RANGE = (4.0, 8.0)          # blur radius for soft falloff

# Glare parameters
GLARE_RADIUS_RANGE = (15, 40)       # semi-axis range for the ellipse
GLARE_OPACITY_RANGE = (40, 110)     # alpha (≈15-43%)
GLARE_BLUR = 6.0                    # heavy blur for soft falloff

# Color jitter ranges (ImageEnhance factor: 1.0 = original)
BRIGHTNESS_RANGE = (0.85, 1.15)
CONTRAST_RANGE = (0.80, 1.20)
SATURATION_RANGE = (0.75, 1.25)
SHARPNESS_RANGE = (0.7, 1.5)

# Non-uniform background: low-frequency smooth variation
BG_LOWFREQ_SIZE = 8                 # generate gradient at this resolution
BG_VARIATION = 8                   # max pixel deviation from base colour

# Canvas: dial is drawn on a tight canvas, then edge-mean-padded to IMG_SIZE
CANVAS_MARGIN = 10                  # px beyond dial_radius in small canvas


# ── dial face ───────────────────────────────────────────────────────────

def draw_dial_face(draw, cx, cy, dial_radius, gap_start_deg=None, gap_span_deg=0,
                   augment=True):
    """Draw a very faint outer circle with tick marks and one arc gap.

    gap_start_deg : where the gap begins (PIL arc degrees: 0 = 3-o'clock, CW)
    gap_span_deg  : angular width of the gap in degrees
    """
    faint = (220, 220, 218, 255)

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

    # 0-9 scale ticks (no digits), drawn inward from the outer circle
    tick_outer = dial_radius
    for i in range(NUM_DIGITS):
        nominal_deg = i * DEG_PER_DIGIT  # 0° = top, clockwise

        # Per-tick randomisation
        if augment:
            angle_jitter = random.uniform(-4, 4)       # ±1-4 deg displacement
            tick_len = random.randint(4, 6)             # 4-6 px length
            grey = random.randint(0, 30)                # near-black colour
            tick_color = (grey, grey, grey, 255)
        else:
            angle_jitter = 0
            tick_len = 5
            tick_color = (0, 0, 0, 255)

        angle_deg = nominal_deg + angle_jitter
        angle_rad = math.radians(angle_deg - 90)  # -90 so 0° = up

        tick_inner = tick_outer - tick_len
        x0 = cx + tick_inner * math.cos(angle_rad)
        y0 = cy + tick_inner * math.sin(angle_rad)
        x1 = cx + tick_outer * math.cos(angle_rad)
        y1 = cy + tick_outer * math.sin(angle_rad)
        draw.line([(x0, y0), (x1, y1)], fill=tick_color, width=1)


# ── shadow ──────────────────────────────────────────────────────────────

def draw_shadow(shadow_draw, apex, base_left, base_right,
                cx, cy, hub_radius, dial_radius,
                offset_dx, offset_dy, opacity,
                gap_start_deg=None, gap_span_deg=0):
    """Draw a soft, offset shadow of the pointer shape and the outer circle.

    The shadow is blurred later, so the tip naturally loses its sharpness.
    """
    shadow_color = (40, 40, 40, opacity)

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

# Lengthwise gradient: brightness shift from hub to tip (positive = tip lighter)
PTR_GRADIENT_RANGE = (-25, 25)
# Edge darkening (bevel): max darkening at the edge of the pointer body
PTR_EDGE_DARKEN_RANGE = (10, 35)
# Specular highlight: bright streak along one side
PTR_SPECULAR_STRENGTH_RANGE = (15, 50)
PTR_SPECULAR_WIDTH_FRAC = (0.2, 0.5)   # fraction of pointer width
# Fine per-pixel noise (on top of structured variation)
PTR_NOISE_RANGE = 10


def _pointer_color_variation(img, mask, base_rgb, cx, cy, angle_deg,
                             hub_radius, tip_dist):
    """Apply realistic structured colour variation to the pointer body.

    Layers (applied additively to brightness):
    1. Lengthwise gradient — smooth brightness ramp from hub to tip.
    2. Edge darkening — darker near the silhouette edges (bevel effect).
    3. Specular highlight — bright streak offset to one side of the
       pointer axis, simulating a light reflection along the body.
    4. Fine per-pixel noise — subtle texture.
    """
    arr = np.array(img).astype(np.int16)
    m = np.array(mask)
    ys, xs = np.where(m > 0)
    if len(ys) == 0:
        return img

    a = math.radians(angle_deg - 90)
    axis_dx = math.cos(a)   # unit vector along pointer (hub → tip)
    axis_dy = math.sin(a)
    perp_dx = -axis_dy       # perpendicular (left of axis)
    perp_dy = axis_dx

    # Signed distance along axis from centre, normalised to [-1, 1]
    rel_x = (xs - cx).astype(np.float32)
    rel_y = (ys - cy).astype(np.float32)
    along = rel_x * axis_dx + rel_y * axis_dy          # projection on axis
    across = rel_x * perp_dx + rel_y * perp_dy         # projection on perp
    along_norm = along / max(tip_dist, 1)               # -1 .. +1 (hub .. tip)

    # ── 1. Lengthwise gradient ──
    grad_strength = random.uniform(*PTR_GRADIENT_RANGE)
    grad_shift = (along_norm * grad_strength).astype(np.int16)

    # ── 2. Edge darkening (bevel) ──
    # Approximate edge distance via successive erosions of the mask
    eroded = m.copy()
    edge_dist = np.zeros_like(m, dtype=np.float32)
    for _ in range(int(hub_radius * 2)):
        kernel_row = np.array([[0], [1], [0]], dtype=np.uint8)
        kernel_col = np.array([[0, 1, 0]], dtype=np.uint8)
        # Simple 4-connected erosion via min-filter
        e1 = np.minimum(np.roll(eroded, 1, 0), np.roll(eroded, -1, 0))
        e2 = np.minimum(np.roll(eroded, 1, 1), np.roll(eroded, -1, 1))
        eroded = np.minimum(e1, e2) & eroded
        edge_dist += (eroded > 0).astype(np.float32)
        if eroded.sum() == 0:
            break
    max_edge = edge_dist.max() if edge_dist.max() > 0 else 1
    edge_factor = edge_dist[ys, xs] / max_edge          # 0 at edge, 1 at centre
    darken_strength = random.uniform(*PTR_EDGE_DARKEN_RANGE)
    edge_shift = ((edge_factor - 1.0) * darken_strength).astype(np.int16)  # negative at edges

    # ── 3. Specular highlight streak ──
    spec_strength = random.uniform(*PTR_SPECULAR_STRENGTH_RANGE)
    spec_width = random.uniform(*PTR_SPECULAR_WIDTH_FRAC) * hub_radius
    # Offset the highlight to one side randomly
    spec_side = random.choice([-1, 1])
    spec_offset = random.uniform(0.1, 0.4) * hub_radius * spec_side
    spec_dist = across - spec_offset
    spec_profile = np.exp(-0.5 * (spec_dist / max(spec_width, 1)) ** 2)
    # Taper specular toward the tip (strongest in middle section)
    spec_taper = np.clip(1.0 - 0.6 * np.abs(along_norm), 0, 1)
    spec_shift = (spec_profile * spec_taper * spec_strength).astype(np.int16)

    # ── 4. Fine per-pixel noise ──
    noise = np.random.randint(-PTR_NOISE_RANGE, PTR_NOISE_RANGE + 1,
                              size=len(ys), dtype=np.int16)

    # Combine all shifts
    total_shift = grad_shift + edge_shift + spec_shift + noise

    for c in range(3):
        arr[ys, xs, c] = np.clip(arr[ys, xs, c] + total_shift, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), img.mode)


# ── pointer ─────────────────────────────────────────────────────────────

def draw_pointer(img, cx, cy, angle_deg, hub_radius, tip_dist,
                 chord_half, chord_dist, pointer_base_rgb):
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

    # Build a mask of the pointer shape
    mask = Image.new("L", img.size, 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.polygon([apex, base_left, base_right], fill=255)
    mdraw.ellipse([cx - hub_radius, cy - hub_radius,
                   cx + hub_radius, cy + hub_radius], fill=255)

    # Fill pointer region with base colour
    draw = ImageDraw.Draw(img)
    base_rgba = pointer_base_rgb + (255,)
    draw.polygon([apex, base_left, base_right], fill=base_rgba, outline=base_rgba)
    draw.ellipse([cx - hub_radius, cy - hub_radius,
                  cx + hub_radius, cy + hub_radius],
                 fill=base_rgba, outline=base_rgba)

    # Apply structured colour variation
    img = _pointer_color_variation(img, mask, pointer_base_rgb, cx, cy,
                                   angle_deg, hub_radius, tip_dist)

    return img, apex, base_left, base_right, mask


# ── glare ───────────────────────────────────────────────────────────────

def draw_glare(img, pointer_mask):
    """Add a soft elliptical glare highlight, masked to avoid the pointer.

    The glare simulates light reflecting off the glass cover of the dial.
    A random ellipse is drawn on a separate layer, heavily blurred for a
    natural falloff, then zeroed wherever the pointer body is.
    """
    cs = img.size[0]  # canvas is square

    # Random ellipse centre (biased toward image interior)
    gx = random.randint(cs // 4, 3 * cs // 4)
    gy = random.randint(cs // 4, 3 * cs // 4)

    # Random semi-axes and rotation
    ra = random.randint(*GLARE_RADIUS_RANGE)
    rb = random.randint(*GLARE_RADIUS_RANGE)
    opacity = random.randint(*GLARE_OPACITY_RANGE)

    # Draw a white ellipse on an oversized layer (to survive blur at edges)
    pad = int(GLARE_BLUR * 3)
    layer_size = cs + 2 * pad
    glare_big = Image.new("L", (layer_size, layer_size), 0)
    gdraw = ImageDraw.Draw(glare_big)
    gcx, gcy = gx + pad, gy + pad
    gdraw.ellipse([gcx - ra, gcy - rb, gcx + ra, gcy + rb], fill=255)

    # Blur then crop back to image size
    glare_big = glare_big.filter(ImageFilter.GaussianBlur(radius=GLARE_BLUR))
    glare_mask = glare_big.crop((pad, pad, pad + cs, pad + cs))

    # Zero out pointer region so glare doesn't overlap the pointer body
    glare_arr = np.array(glare_mask, dtype=np.int16)
    ptr_arr = np.array(pointer_mask)
    glare_arr[ptr_arr > 0] = 0
    glare_arr = np.clip(glare_arr, 0, 255).astype(np.uint8)

    # Build RGBA glare layer
    glare_rgba = np.zeros((cs, cs, 4), dtype=np.uint8)
    glare_rgba[:, :, 0] = 255  # white
    glare_rgba[:, :, 1] = 255
    glare_rgba[:, :, 2] = 255
    glare_rgba[:, :, 3] = (glare_arr.astype(np.float32) * opacity / 255).astype(np.uint8)

    glare_layer = Image.fromarray(glare_rgba, "RGBA")
    return Image.alpha_composite(img, glare_layer)


# ── rim shadow (bowl cavity) ───────────────────────────────────────────

def draw_rim_shadows(cx, cy, dial_radius, canvas_size):
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

    num_lights = random.choice([1, 2])

    light_angles = [random.uniform(0, 360)]
    if num_lights == 2:
        offset_ang = random.uniform(60, 180)
        light_angles.append((light_angles[0] + offset_ang) % 360)

    for i, light_deg in enumerate(light_angles):
        # Cutout circle is shifted TOWARD the light (away from shadow side)
        light_rad = math.radians(light_deg - 90)  # convert: 0=up,CW → trig

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

        if i == 0:
            opacity = random.randint(*RIM_SHADOW_PRIMARY_OPACITY)
        else:
            opacity = random.randint(*RIM_SHADOW_SECONDARY_OPACITY)

        layer_a[crescent] += opacity

    layer_a = np.clip(layer_a, 0, 255)

    rgba = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    rgba[:, :, 0] = 30
    rgba[:, :, 1] = 30
    rgba[:, :, 2] = 30
    rgba[:, :, 3] = layer_a.astype(np.uint8)

    layer = Image.fromarray(rgba, "RGBA")
    blur_r = random.uniform(*RIM_SHADOW_BLUR_RANGE)
    layer = layer.filter(ImageFilter.GaussianBlur(radius=blur_r))
    return layer, light_angles


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


def _random_red_rgb():
    """Return a random reddish (R, G, B) tuple as the pointer base."""
    r = random.randint(180, 230)
    g = random.randint(15, 55)
    b = random.randint(10, 50)
    return (r, g, b)


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
    tick_inner = int(dial_radius * 0.85)
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

    # --- non-uniform background (small canvas) ---
    if augment:
        bg_base = random.randint(210, 250)

        lf_lum = np.random.randint(-BG_VARIATION, BG_VARIATION + 1,
                                   (BG_LOWFREQ_SIZE, BG_LOWFREQ_SIZE),
                                   dtype=np.int16)
        lf_gray = Image.fromarray(
            np.clip(bg_base + lf_lum, 0, 255).astype(np.uint8), "L"
        )
        lf_gray = lf_gray.resize((canvas_size, canvas_size), Image.BICUBIC)
        lum = np.array(lf_gray, dtype=np.int16)

        tint = np.array([random.randint(-6, 6),
                         random.randint(-6, 6),
                         random.randint(-6, 6)], dtype=np.int16)
        bg_arr = np.stack([lum, lum, lum], axis=-1) + tint[None, None, :]
        bg_arr = np.clip(bg_arr, 0, 255).astype(np.int16)

        noise = np.random.randint(-NOISE_INTENSITY, NOISE_INTENSITY + 1,
                                  (canvas_size, canvas_size, 3), dtype=np.int16)
        bg_arr = np.clip(bg_arr + noise, 0, 255).astype(np.uint8)

        alpha = np.full((canvas_size, canvas_size, 1), 255, dtype=np.uint8)
        bg_arr = np.concatenate([bg_arr, alpha], axis=2)
        img = Image.fromarray(bg_arr, "RGBA")
    else:
        img = Image.new("RGBA", (canvas_size, canvas_size), "white")

    draw = ImageDraw.Draw(img)

    # --- faint dial circle (with gap) ---
    draw_dial_face(draw, cx, cy, dial_radius, gap_start_deg, gap_span_deg,
                   augment=augment)

    # --- rim shadow (bowl cavity — 1-2 light sources) ---
    # light_angles is reused for the pointer cast shadow direction
    if augment:
        rim_layer, light_angles = draw_rim_shadows(cx, cy, dial_radius,
                                                   canvas_size)
        img = Image.alpha_composite(img, rim_layer)
    else:
        light_angles = [random.uniform(0, 360)]

    # --- pointer cast shadow (same light source as rim shadow) ---
    if augment:
        shadow_layer = Image.new("RGBA", (canvas_size, canvas_size),
                                 (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow_layer)

        a = math.radians(angle_deg - 90)
        s_apex = (cx + tip_dist * math.cos(a), cy + tip_dist * math.sin(a))
        bmx = cx - chord_dist * math.cos(a)
        bmy = cy - chord_dist * math.sin(a)
        perp = (math.cos(a + math.pi / 2), math.sin(a + math.pi / 2))
        s_bl = (bmx + chord_half * perp[0], bmy + chord_half * perp[1])
        s_br = (bmx - chord_half * perp[0], bmy - chord_half * perp[1])

        # Shadow falls on the side away from the light.
        # Rim cutout is shifted toward light at light_rad; crescent is opposite.
        # Pointer shadow offset must point the same way as the crescent.
        primary_light_deg = light_angles[0]
        light_rad = math.radians(primary_light_deg - 90)  # same conv as rim
        shd_dist = random.uniform(*SHADOW_OFFSET_RANGE)
        # Offset AWAY from light (negate the light direction)
        shd_dx = shd_dist * math.cos(light_rad)
        shd_dy = shd_dist * math.sin(light_rad)
        shd_opacity = random.randint(*SHADOW_OPACITY_RANGE)

        draw_shadow(sdraw, s_apex, s_bl, s_br,
                    cx, cy, hub_radius, dial_radius,
                    shd_dx, shd_dy, shd_opacity,
                    gap_start_deg, gap_span_deg)

        shadow_layer = shadow_layer.filter(ImageFilter.GaussianBlur(radius=2.5))
        img = Image.alpha_composite(img, shadow_layer)

    # --- pointer with non-uniform colour ---
    ptr_rgb = _random_red_rgb() if augment else (210, 30, 30)
    img, apex, bl, br, ptr_mask = draw_pointer(
        img, cx, cy, angle_deg,
        hub_radius=hub_radius,
        tip_dist=tip_dist,
        chord_half=chord_half,
        chord_dist=chord_dist,
        pointer_base_rgb=ptr_rgb,
    )

    # --- glare (after pointer, before final blur) ---
    if augment:
        img = draw_glare(img, ptr_mask)

    # --- convert to RGB ---
    img = img.convert("RGB")

    # --- colour / exposure jitter ---
    if augment:
        img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
        img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
        img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
        img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # --- final gaussian blur ---
    if augment:
        blur_r = random.uniform(*BLUR_RANGE)
        img = img.filter(ImageFilter.GaussianBlur(radius=blur_r))

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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = os.path.join(OUTPUT_DIR, "labels.csv")
    rows = []

    for i in range(NUM_IMAGES):
        angle_deg = random.randint(0, 359)
        fname = f"dial_{i:04d}.png"
        fpath = os.path.join(OUTPUT_DIR, fname)

        tip_x, tip_y = generate_image(angle_deg, fpath)
        rows.append({"filename": fname,
                      "angle_deg": angle_deg,
                      "tip_x": round(tip_x, 2),
                      "tip_y": round(tip_y, 2)})

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "angle_deg", "tip_x", "tip_y"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {NUM_IMAGES} images in '{OUTPUT_DIR}/'")
    print(f"Labels written to '{csv_path}'")


if __name__ == "__main__":
    main()
