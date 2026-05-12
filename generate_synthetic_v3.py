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
DIAL_POLY_SEGS = 180     # number of segments for the dial-boundary polyline

# ── Scale ticks ──────────────────────────────────────────────────────────
TICKS_PER_DIGIT_RANGE = (1, 5)
TICK_LEN_FRAC_RANGE = (0.06, 0.10)   # fraction of a (semi-major) for tick length
TICK_WIDTH_RANGE = (2, 4)
TICK_GREY_RANGE = (0, 30)

# ── Digits ───────────────────────────────────────────────────────────────
DIGIT_FONT_SIZE_RANGE = (12, 24)
DIGIT_GREY_RANGE = (0, 30)
DIGIT_INSIDE_PROB = 0.5              # probability digits are inside ticks (vs outside)
DIGIT_RC_INSIDE_OFFSET = (0.18, 0.28)   # r_c inset when digits inside ticks
DIGIT_RC_OUTSIDE_OFFSET = (0.08, 0.18)  # r_c outset when digits outside ticks

# ── Pointer ──────────────────────────────────────────────────────────────
HUB_RATIO_RANGE = (0.35, 0.45)       # hub_radius / a
PTR_S_RANGE = (0.5, 1.0)             # pointer HSV saturation range
PTR_V_RANGE = (0.40, 0.95)           # pointer HSV value range
HOLE_RATIO_RANGE = (1.05, 1.10)
PTR_APEX_RC = 0.92                   # apex radius (fraction of semi-axis)
PTR_BASE_W_FRAC = 0.6                # pointer triangle base half-width / hub_r
PTR_HOLE_DARKEN = 60                 # per-channel offset to darken hub hole

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


# ── Outer canvas (bezel) and dial face ───────────────────────────────────

def draw_outer_canvas(img, params, outer_scale, dial_h, dial_s):
    """Render the bezel (annular fill + rim band) around the dial.

    The bezel is an ellipse with the same orientation as the dial, scaled by
    `outer_scale`. The annular region between the dial face and the bezel
    boundary is filled with `outer_rgb`; the rim band sits on the dial edge.
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
    annular = (inner_r2 > 1.0) & inside_outer

    outer_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*OUTER_V_RANGE))
    rim_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*RIM_V_RANGE))

    arr = np.array(img)
    arr[annular] = outer_rgb

    # Rim band: thin shell straddling inner_r2 == 1
    rim_half = RIM_WIDTH / min(max(a, 1), max(b, 1))
    r_norm = np.sqrt(inner_r2)
    rim_band = (r_norm > 1.0 - rim_half) & (r_norm < 1.0 + rim_half)
    arr[rim_band] = rim_rgb

    return Image.fromarray(arr, img.mode), inside_outer


def draw_dial_boundary(img, params, dial_h, dial_s):
    """Draw the faint dial circle as a polyline mapped through canonical→image."""
    cx, cy, a, b, theta = params["cx"], params["cy"], params["a"], params["b"], params["theta"]
    circle_rgb = _hsv_rgb(dial_h, dial_s, random.uniform(*CIRCLE_V_RANGE))

    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    pts = []
    n_seg = DIAL_POLY_SEGS
    for k in range(n_seg + 1):  # +1 to close the polyline (k=0 and k=n_seg map to same point)
        t = 2 * math.pi * k / n_seg
        cos_t = math.cos(t)
        sin_t = math.sin(t)
        x = cx + (a * cos_t * cos_o - b * sin_t * sin_o)
        y = cy + (a * cos_t * sin_o + b * sin_t * cos_o)
        pts.append((x, y))

    draw = ImageDraw.Draw(img)
    draw.line(pts, fill=circle_rgb, width=1)
    return img


# ── Scale ticks ──────────────────────────────────────────────────────────

def _canonical_to_image(t, r_c, cx, cy, a, b, theta):
    """Inline copy of the canonical→image map for use in the generator.

    Mirror of `autocal_utils.canonical_to_image_np` — keep in sync.
    """
    cos_t = math.cos(t)
    sin_t = math.sin(t)
    cos_o = math.cos(theta)
    sin_o = math.sin(theta)
    x = cx + r_c * (a * cos_t * cos_o - b * sin_t * sin_o)
    y = cy + r_c * (a * cos_t * sin_o + b * sin_t * cos_o)
    return x, y


def draw_scale_ticks(img, params):
    """Render scale ticks at canonical angles t_i = phi_0 + i·36°.

    Sub-ticks evenly spaced between consecutive main ticks. Tick lengths and
    widths are jittered per image (shared values across all ticks in the image).
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]

    ticks_per_digit = random.randint(*TICKS_PER_DIGIT_RANGE)
    main_len_frac = random.uniform(*TICK_LEN_FRAC_RANGE)
    main_width = random.randint(*TICK_WIDTH_RANGE)
    minor_width = max(1, main_width - 1)
    grey = random.randint(*TICK_GREY_RANGE)
    color = (grey, grey, grey)

    draw = ImageDraw.Draw(img)
    step_t = 2 * math.pi / N_MARKS / ticks_per_digit

    for i in range(N_MARKS):
        t_main = phi_0 + i * (2 * math.pi / N_MARKS)
        for s in range(ticks_per_digit):
            t = t_main + s * step_t
            if s == 0:
                len_frac = main_len_frac
                width = main_width
            else:
                len_frac = main_len_frac * 0.5
                width = minor_width
            r_inner = 1.0 - len_frac
            x0, y0 = _canonical_to_image(t, r_inner, cx, cy, a, b, theta)
            x1, y1 = _canonical_to_image(t, 1.0, cx, cy, a, b, theta)
            draw.line([(x0, y0), (x1, y1)], fill=color, width=width)
    return img


# ── Digits ───────────────────────────────────────────────────────────────

def draw_scale_digits(img, params):
    """Render digits 0..9 at canonical t_i = phi_0 + i·36°.

    Random font size shared across the 10 digits. Half the time inside the
    ticks, half outside.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]

    font_size = random.randint(*DIGIT_FONT_SIZE_RANGE)
    font = _load_font("arial.ttf", font_size)
    grey = random.randint(*DIGIT_GREY_RANGE)
    color = (grey, grey, grey)

    inside = random.random() < DIGIT_INSIDE_PROB
    if inside:
        r_c = 1.0 - random.uniform(*DIGIT_RC_INSIDE_OFFSET)
    else:
        r_c = 1.0 + random.uniform(*DIGIT_RC_OUTSIDE_OFFSET)

    draw = ImageDraw.Draw(img)
    for i in range(N_MARKS):
        t = phi_0 + i * (2 * math.pi / N_MARKS)
        x, y = _canonical_to_image(t, r_c, cx, cy, a, b, theta)
        text = str(i)
        bbox = font.getbbox(text)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw / 2, y - th / 2), text, fill=color, font=font)
    return img


# ── Pointer ──────────────────────────────────────────────────────────────

def _random_pointer_rgb():
    h = random.random()
    s = random.uniform(*PTR_S_RANGE)
    v = random.uniform(*PTR_V_RANGE)
    return _hsv_rgb(h, s, v)


def draw_pointer(img, params):
    """Draw the pointer (hub circle + isosceles triangle) at canonical angle.

    Apex at canonical angle phi_0 + pointer_value · 36°, r_c = 1.0 - small margin.
    """
    cx, cy = params["cx"], params["cy"]
    a, b, theta = params["a"], params["b"], params["theta"]
    phi_0 = params["phi_0"]
    pv = params["pointer_value"]

    hub_ratio = random.uniform(*HUB_RATIO_RANGE)
    hub_r = a * hub_ratio
    hole_r = hub_r * random.uniform(*HOLE_RATIO_RANGE)
    ptr_rgb = _random_pointer_rgb()
    hole_rgb = tuple(max(0, c - PTR_HOLE_DARKEN) for c in ptr_rgb)

    t_apex = phi_0 + pv * (2 * math.pi / N_MARKS)
    apex_x, apex_y = _canonical_to_image(t_apex, PTR_APEX_RC, cx, cy, a, b, theta)

    # Base of triangle: a chord at the hub edge.
    # We render the pointer in image space as a triangle from hub to apex.
    dx = apex_x - cx
    dy = apex_y - cy
    norm = math.hypot(dx, dy) or 1.0
    ux = dx / norm
    uy = dy / norm
    perp_x = -uy
    perp_y = ux
    half_w = hub_r * PTR_BASE_W_FRAC
    base_mx = cx
    base_my = cy
    base_left = (base_mx + half_w * perp_x, base_my + half_w * perp_y)
    base_right = (base_mx - half_w * perp_x, base_my - half_w * perp_y)

    draw = ImageDraw.Draw(img)
    draw.polygon([(apex_x, apex_y), base_left, base_right],
                 fill=ptr_rgb)
    draw.ellipse([cx - hub_r, cy - hub_r, cx + hub_r, cy + hub_r],
                 fill=ptr_rgb)
    draw.ellipse([cx - hole_r, cy - hole_r, cx + hole_r, cy + hole_r],
                 fill=hole_rgb)
    return img


# ── Adversarial triangle artifacts ───────────────────────────────────────

def add_triangle_artifacts(img):
    """Opaque random-colour triangles scattered on the canvas."""
    arr = np.array(img)
    canvas_size = arr.shape[0]
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
        if not m.any():
            continue
        for c in range(3):
            noise = np.random.randint(-TRI_PIXEL_VAR, TRI_PIXEL_VAR + 1,
                                      size=m.sum(), dtype=np.int16)
            arr[:, :, c][m] = np.clip(base[c] + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, img.mode)


# ── Top-level image generation ───────────────────────────────────────────

def generate_image(path, img_size=IMG_SIZE):
    """Generate one synthetic dial image. Returns dict of label fields."""
    params = sample_params(img_size=img_size)

    # 1. Procedural background fill (full canvas).
    img = make_procedural_background(img_size)

    # Shared HSV dial palette for bezel/circle/rim.
    dial_h = random.random()  # full hue wheel
    dial_s = random.uniform(*DIAL_S_RANGE)

    outer_scale = random.uniform(*OUTER_SCALE_RANGE)

    # 2. Distractor text (outside bezel disc only).
    img = draw_distractor_text(img, params, outer_scale)

    # 3. Outer canvas (bezel + rim).
    img, _inside_outer = draw_outer_canvas(img, params, outer_scale,
                                           dial_h, dial_s)  # mask unused: v3 fills full canvas

    # 4. Dial boundary, ticks, digits.
    img = draw_dial_boundary(img, params, dial_h, dial_s)
    img = draw_scale_ticks(img, params)
    img = draw_scale_digits(img, params)

    # 5. Pointer.
    img = draw_pointer(img, params)

    # 6. Triangle artifacts (across whole canvas).
    img = add_triangle_artifacts(img)

    # 7. Colour/exposure jitter.
    img = ImageEnhance.Brightness(img).enhance(random.uniform(*BRIGHTNESS_RANGE))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(*CONTRAST_RANGE))
    img = ImageEnhance.Color(img).enhance(random.uniform(*SATURATION_RANGE))
    img = ImageEnhance.Sharpness(img).enhance(random.uniform(*SHARPNESS_RANGE))

    # 8. Save.
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


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic autocal images")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-images", type=int, default=NUM_IMAGES)
    args = parser.parse_args()

    if args.num_images <= 0:
        parser.error("--num-images must be a positive integer")

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "labels.csv")

    fieldnames = ["filename", "cx", "cy", "a", "b", "theta", "phi_0"]
    for i in range(N_MARKS):
        fieldnames += [f"mark_{i}_x", f"mark_{i}_y"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(args.num_images):
            fname = f"dial_{i:05d}.png"
            fpath = os.path.join(args.output_dir, fname)
            writer.writerow(generate_image(fpath))

    print(f"Generated {args.num_images} images in '{args.output_dir}/'")
    print(f"Labels written to '{csv_path}'")


if __name__ == "__main__":
    main()
