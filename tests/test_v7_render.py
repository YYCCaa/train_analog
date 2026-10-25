"""Tests for the native-gray v7 renderer (generate_synthetic_v7_mp.render_image).

These call render_image directly (the pure render core); the CLI-level
generator tests live in tests/test_v7_gray_generator.py.
"""

import copy
import math
import os

import numpy as np
import pytest

import generate_synthetic_v7_mp as g7
import v7_config

# side of every rendered image (int), fixed by canvas.img_size
IMG = 256
# nominal dial radius in px (float) = img_size / 3, used to size test regions
DIAL = IMG / 3.0
# path (str) of the module under test, for the source-text check
SRC_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "generate_synthetic_v7_mp.py")


def _cfg(**overrides):
    """Default cfg with "section.key" -> value overrides applied (deep copy)."""
    cfg = v7_config.default_config()
    for dotted, value in overrides.items():
        sec, key = dotted.split(".")
        cfg[sec][key] = copy.deepcopy(value)
    return cfg


def _radius_grid():
    """Distance of every pixel from the image centre, (256, 256) float64."""
    ys, xs = np.mgrid[:IMG, :IMG]
    return np.hypot(xs - IMG / 2.0, ys - IMG / 2.0)


def _same_labels(a, b):
    """True when two RenderResults have the same angle and tip."""
    return a.angle_deg == b.angle_deg and a.tip_x == b.tip_x and a.tip_y == b.tip_y


def test_render_shape_and_dtype():
    res = g7.render_image(v7_config.default_config(), 1234)
    assert res.image.shape == (IMG, IMG)
    assert res.image.dtype == np.uint8
    assert 0 <= res.tip_x < IMG and 0 <= res.tip_y < IMG
    assert 0 <= res.angle_deg <= 359


def test_render_deterministic():
    cfg = v7_config.default_config()
    a = g7.render_image(cfg, 42)
    b = g7.render_image(cfg, 42)
    assert np.array_equal(a.image, b.image)
    assert _same_labels(a, b)
    c = g7.render_image(cfg, 43)
    assert not np.array_equal(a.image, c.image)


def test_no_colour_code_in_v7():
    with open(SRC_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    for banned in ("colorsys", "hsv"):  # any casing
        assert banned not in src.lower(), f"found {banned!r} in generate_synthetic_v7_mp.py"
    for banned in ('"RGB"', '"RGBA"'):  # quoted image-mode literals, exact case
        assert banned not in src, f"found {banned!r} in generate_synthetic_v7_mp.py"


def test_section_rngs_are_independent_streams():
    a = g7.section_rngs(7)
    b = g7.section_rngs(7)
    assert set(a) == set(g7.SECTIONS)
    assert "angle" in g7.SECTIONS and "canvas" not in g7.SECTIONS
    # same seed -> same draws; different sections -> different draws
    assert a["ticks"][0].random() == b["ticks"][0].random()
    assert a["ticks"][1].integers(0, 1 << 30) == b["ticks"][1].integers(0, 1 << 30)
    assert g7.section_rngs(7)["ticks"][0].random() != g7.section_rngs(7)["digits"][0].random()


def _overlay_footprint(cfg, seed):
    """Pixels the tick/digit overlay reaches (incl. blur), (256, 256) bool.

    Renders cfg twice with tick and digit gray ranges shifted by 225 levels.
    The range width is kept, so every draw is identical and only overlay
    pixels (and anything they bleed into) change.
    """
    lo, hi = copy.deepcopy(cfg), copy.deepcopy(cfg)
    for sec in ("ticks", "digits"):
        a, b = cfg[sec]["gray_range"]
        assert b - a <= 30, "probe needs room for a +225 shift"
        lo[sec]["gray_range"] = [a, b]
        hi[sec]["gray_range"] = [a + 225, b + 225]
    return g7.render_image(lo, seed).image != g7.render_image(hi, seed).image


def test_tick_change_isolated():
    # contrast pin is defensive here: jitter runs before the overlay composite
    base = _cfg(**{"jitter.contrast_range": [1.0, 1.0]})
    changed = _cfg(**{"jitter.contrast_range": [1.0, 1.0],
                      "ticks.len_range": [12, 14],
                      "ticks.per_digit_range": [5, 5]})
    # brief: central disc r < 0.45 * dial.  Inside-placed digits follow the
    # tick inner radius (passed as data), so with large fonts they can bleed
    # into that disc; the spec's actual criterion is "pixels outside the
    # tick/digit overlay stay identical", checked over the whole image.
    any_diff = False
    for seed in range(8):
        a = g7.render_image(base, seed)
        b = g7.render_image(changed, seed)
        assert _same_labels(a, b), f"seed {seed}: label moved"
        # +4 px margin: far blur-tail pixels can flip by 1 level on rounding
        overlay = _dilate(_overlay_footprint(base, seed) | _overlay_footprint(changed, seed), 4)
        keep = ~overlay  # (256, 256) bool, pixels no tick/digit reaches
        assert keep.sum() > 0.5 * IMG * IMG
        assert np.array_equal(a.image[keep], b.image[keep]), f"seed {seed}: non-overlay pixels changed"
        any_diff |= not np.array_equal(a.image, b.image)
    assert any_diff, "tick change had no visible effect"


def test_light_overlay_has_no_dark_halo():
    # white ticks/digits must only brighten pixels: a non-premultiplied
    # stretch of the overlay would pull fringe values toward 0 (dark halo)
    pins = {"ticks.gray_range": [255, 255], "digits.gray_range": [255, 255],
            "background.dial_circle_blur": False,
            "jitter.brightness_range": [1.0, 1.0],
            "jitter.contrast_range": [1.0, 1.0],
            "jitter.sharpness_range": [1.0, 1.0]}
    shown = _cfg(**pins, **{"digits.show_prob": 1.0})
    hidden = _cfg(**pins, **{"digits.show_prob": 0.0})  # same draws, no overlay
    for seed in range(6):
        a = g7.render_image(shown, seed).image.astype(np.int16)  # (256, 256)
        b = g7.render_image(hidden, seed).image.astype(np.int16)  # (256, 256)
        footprint = a != b  # (256, 256) bool, pixels the overlay touched
        assert footprint.any(), f"seed {seed}: overlay invisible"
        darker = footprint & (a < b - 1)  # overlay made a pixel darker
        assert not darker.any(), (
            f"seed {seed}: {int(darker.sum())} overlay pixels darkened, "
            f"worst {int((b - a)[darker].max())} levels")


def _dilate(mask, steps):
    """Binary 4-neighbour dilation of a (H, W) bool mask, `steps` times."""
    out = mask.copy()
    for _ in range(steps):
        grown = out.copy()
        grown[1:, :] |= out[:-1, :]
        grown[:-1, :] |= out[1:, :]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def test_tri_change_isolated():
    # contrast pin is defensive here: jitter runs before the triangles are drawn
    flat = {"jitter.contrast_range": [1.0, 1.0]}
    base = _cfg(**flat)
    off = _cfg(**flat, **{"tri_artifacts.enabled": False})
    # triangle footprint probes: same draws, extreme fills -> every pixel a
    # triangle touches (incl. blur spread) differs between the two
    probe_lo = _cfg(**flat, **{"tri_artifacts.gray_range": [0, 0]})
    probe_hi = _cfg(**flat, **{"tri_artifacts.gray_range": [255, 255]})
    centre = _radius_grid() < 0.3 * DIAL  # (256, 256) bool, ~pointer hub region
    lo, hi = base["tri_artifacts"]["count_range"]
    # seeds whose tri_artifacts stream draws >= 1 triangle (count is its first draw)
    seeds = [s for s in range(40)
             if g7.section_rngs(s)["tri_artifacts"][0].randint(lo, hi) >= 1][:6]
    assert len(seeds) == 6
    any_diff = False
    for seed in seeds:
        a = g7.render_image(base, seed)
        b = g7.render_image(off, seed)
        assert _same_labels(a, b), f"seed {seed}: label moved"
        foot = (g7.render_image(probe_lo, seed).image
                != g7.render_image(probe_hi, seed).image)
        keep = centre & ~_dilate(foot, 2)  # disc pixels no triangle reached
        assert np.array_equal(a.image[keep], b.image[keep]), f"seed {seed}: pointer region changed"
        any_diff |= not np.array_equal(a.image, b.image)
    assert any_diff, "disabling triangles had no visible effect"


def test_pointer_gray_change_moves_only_pointer():
    flat = {"jitter.contrast_range": [1.0, 1.0]}
    white = _cfg(**flat, **{"pointer.gray_range": [255, 255]})
    black = _cfg(**flat, **{"pointer.gray_range": [0, 0]})
    r = _radius_grid()
    ys, xs = np.mgrid[:IMG, :IMG]
    # pixel bearing (deg, 0 = up, clockwise), (256, 256) float64
    bearing = np.degrees(np.arctan2(xs - IMG / 2.0, -(ys - IMG / 2.0))) % 360
    ring = (r >= 0.75 * DIAL) & (r <= 0.95 * DIAL)
    for seed in range(6):
        a = g7.render_image(white, seed)
        b = g7.render_image(black, seed)
        assert _same_labels(a, b), f"seed {seed}: label moved"
        diff_ang = np.abs((bearing - a.angle_deg + 180) % 360 - 180)
        far = ring & (diff_ang > 60)  # ring pixels away from the pointer
        da = abs(float(a.image[far].mean()) - float(b.image[far].mean()))
        assert da < 1.0, f"seed {seed}: far ring mean moved by {da:.3f}"
        assert not np.array_equal(a.image, b.image)


# every probability / bool gate forced ON: (section, key, value) triples
_GATES_ON = [("digits", "show_prob", 1.0), ("shadows", "prob", 1.0),
             ("cut_tip", "enabled", True), ("background", "dial_circle_blur", True),
             ("tri_artifacts", "enabled", True)]


def _extreme_cfg(which):
    """Every leaf at its schema min ("min"), max ("max"), min with all gates on
    ("min_gates_on"), or max with the smallest canvas_margin ("max_tight_canvas")."""
    schema = v7_config.load_schema()
    cfg = v7_config.default_config()
    at_max = which == "max"
    for sec, keys in schema.items():
        for key, leaf in keys.items():
            v = leaf["max"] if at_max else leaf["min"]
            t = leaf["type"]
            if t == "bool":
                cfg[sec][key] = at_max
            elif t in ("int_range", "float_range"):
                cfg[sec][key] = [v, v]
            else:
                cfg[sec][key] = v
    if which == "min_gates_on":
        for sec, key, value in _GATES_ON:
            cfg[sec][key] = value
    if which == "max_tight_canvas":
        # smallest canvas -> widest padding, so the max displacement moves the
        # dial furthest toward the image edge: the worst case for the tip label
        cfg = _extreme_cfg("max")
        cfg["geometry"]["canvas_margin"] = schema["geometry"]["canvas_margin"]["min"]
    v7_config.validate_config(cfg, schema)
    return cfg


@pytest.mark.parametrize("which", ["min", "max", "min_gates_on", "max_tight_canvas"])
def test_render_extremes_do_not_crash(which):
    cfg = _extreme_cfg(which)
    for seed in range(5):
        res = g7.render_image(cfg, seed)
        assert res.image.shape == (IMG, IMG)
        assert res.image.dtype == np.uint8
        # the tip label must stay on the image even at the schema extremes
        assert 0 <= res.tip_x <= IMG - 1, (seed, res.tip_x)
        assert 0 <= res.tip_y <= IMG - 1, (seed, res.tip_y)


# ── pointer min-bg-contrast + one-directional digit jitter ──────────────

def test_pick_pointer_gray():
    import random
    # c = 0: identical to the old randint(lo, hi) draw, for many seeds/ranges
    for lo, hi in ((0, 255), (40, 41), (7, 7), (100, 230)):
        for s in range(300):
            want = random.Random(s).randint(lo, hi)  # int, old draw
            got = g7._pick_pointer_gray(random.Random(s), lo, hi, 160, 0)  # int, new draw
            assert got == want, (lo, hi, s)
    # c > 0: every draw is allowed and both sides of bg appear
    rng = random.Random(1)  # shared stream (random.Random)
    vals = [g7._pick_pointer_gray(rng, 0, 255, 160, 60) for _ in range(2000)]  # list[int]
    assert all(abs(v - 160) >= 60 for v in vals)
    assert any(v < 160 for v in vals) and any(v > 160 for v in vals)
    assert set(vals) == set(range(0, 101)) | set(range(220, 256))
    # one side only: bg near the top of the range
    vals = {g7._pick_pointer_gray(rng, 100, 200, 190, 50) for _ in range(500)}  # set[int]
    assert vals == set(range(100, 141))
    # infeasible: farthest endpoint, lower one on a tie
    assert g7._pick_pointer_gray(random.Random(0), 100, 200, 120, 200) == 200
    assert g7._pick_pointer_gray(random.Random(0), 100, 200, 180, 200) == 100
    assert g7._pick_pointer_gray(random.Random(0), 100, 200, 150, 200) == 100
    # infeasible still consumes exactly one draw (the same one as randint(lo, hi))
    c1, c2 = random.Random(9), random.Random(9)  # two identical streams
    g7._pick_pointer_gray(c1, 100, 200, 150, 200)
    c2.randint(100, 200)
    assert c1.random() == c2.random(), "infeasible case must consume one draw"


# sha256 hex (str) of default-config renders with digits hidden (so the
# digit-jitter change cannot touch them), seeds 0-2, recorded BEFORE any code
# change for min_bg_contrast; re-pinned when displacement moved from the
# canvas pad to the content inside the centred circle canvas, and again when
# displacement became a shift of everything inside a fixed circular boundary
# (canvas grows by 2 * max|displacement|; expected change), and again for
# the physical multiplicative shadow model (only seed 1 has a shadow; seeds
# 0 and 2 are unchanged, so no other stream moved). Only the image is hashed,
# so moving the tip label to the visible (cut) tip left these unchanged.
_PRE_CONTRAST_HASHES = {
    0: "e0c38ede1599cf8d16dcaef4347406454bd2163df2c4866ffa86c1d96a1f6ad6",
    1: "e3545616c3cfe7f4dc5305faabbef780e3ea52d2cdd9cb068c6e684ae080b3a8",
    2: "6da293f03654acd47597967cfd86bd8683143d6add586da6678e09cf2e401717",
}


def test_default_render_unchanged_by_contrast_key():
    import hashlib
    cfg = _cfg(**{"digits.show_prob": 0.0})
    assert cfg["pointer"]["min_bg_contrast"] == 0
    for seed, want in _PRE_CONTRAST_HASHES.items():
        got = hashlib.sha256(g7.render_image(cfg, seed).image.tobytes()).hexdigest()
        assert got == want, f"seed {seed}: default render changed"


def _pointer_footprint(cfg, seed):
    """Pixels the pointer body/edge/hole reach, (256, 256) bool (c forced to 0)."""
    lo, hi = copy.deepcopy(cfg), copy.deepcopy(cfg)
    for c, g in ((lo, 0), (hi, 255)):
        c["pointer"]["gray_range"] = [g, g]
        c["pointer"]["min_bg_contrast"] = 0
    return g7.render_image(lo, seed).image != g7.render_image(hi, seed).image


def test_min_bg_contrast_isolated():
    flat = {"jitter.contrast_range": [1.0, 1.0]}
    base = _cfg(**flat)
    strong = _cfg(**flat, **{"pointer.min_bg_contrast": 90})
    any_diff = False
    for seed in range(8):
        a = g7.render_image(base, seed)
        b = g7.render_image(strong, seed)
        assert _same_labels(a, b), f"seed {seed}: label moved"
        keep = ~_dilate(_pointer_footprint(base, seed), 4)  # (256, 256) bool, non-pointer
        assert keep.sum() > 0.5 * IMG * IMG
        assert np.array_equal(a.image[keep], b.image[keep]), f"seed {seed}: non-pointer changed"
        any_diff |= not np.array_equal(a.image, b.image)
    assert any_diff, "min_bg_contrast had no visible effect"


# "section.key" -> value pins (dict) for the digit-jitter tests: digits always
# on, no global or pixel-smearing post effects
_DIGIT_PINS = {"digits.show_prob": 1.0, "jitter.contrast_range": [1.0, 1.0],
               "jitter.brightness_range": [1.0, 1.0], "jitter.sharpness_range": [1.0, 1.0],
               "background.dial_circle_blur": False}


def _digit_mean_radius(cfg, seed):
    """Mean distance (float, px) of the digit footprint from the image centre.

    Footprint = pixels that change when only digits.gray_range is shifted by
    +225 (same draws), (256, 256) bool.
    """
    hi = copy.deepcopy(cfg)
    a, b = cfg["digits"]["gray_range"]
    hi["digits"]["gray_range"] = [a + 225, b + 225]
    foot = g7.render_image(cfg, seed).image != g7.render_image(hi, seed).image
    assert foot.sum() > 50, f"seed {seed}: digits invisible"
    return float(_radius_grid()[foot].mean())


def test_digit_jitter_direction():
    zero = {"digits.inside_jitter_range": [0.0, 0.0],
            "digits.outside_jitter_range": [0.0, 0.0]}  # both jitters off
    for inside_prob, key, sign in ((1.0, "digits.inside_jitter_range", -1),
                                   (0.0, "digits.outside_jitter_range", +1)):
        off = _cfg(**_DIGIT_PINS, **zero, **{"digits.inside_prob": inside_prob})
        on = _cfg(**_DIGIT_PINS, **{**zero, "digits.inside_prob": inside_prob,
                                    key: [5.0, 5.0]})
        for seed in range(6):
            r_off = _digit_mean_radius(off, seed)  # float px
            r_on = _digit_mean_radius(on, seed)  # float px
            assert sign * (r_on - r_off) > 2.5, (key, seed, r_off, r_on)
    # the other branch's range changes nothing (byte-identical)
    for inside_prob, other in ((1.0, "digits.outside_jitter_range"),
                               (0.0, "digits.inside_jitter_range")):
        a = _cfg(**_DIGIT_PINS, **{"digits.inside_prob": inside_prob})
        b = _cfg(**_DIGIT_PINS, **{"digits.inside_prob": inside_prob, other: [9.0, 9.0]})
        for seed in range(4):
            assert np.array_equal(g7.render_image(a, seed).image,
                                  g7.render_image(b, seed).image), (other, seed)



# ── displacement translates everything inside a fixed circular boundary ─

# "section.key" -> value pins (dict): fixed dial size (radius 85, outer
# boundary radius 85 * 1.2 = 102), no stretch, shadows always on (they must
# translate too).  outer 1.2 keeps ceil(102) + 1 = 103 below radius + margin,
# so the canvas half is radius + margin + max|displacement|.
_DISP_PINS = {"geometry.scale_range": [1.0, 1.0], "outer.scale_range": [1.2, 1.2],
              "geometry.stretch_h_range": [1.0, 1.0], "geometry.stretch_v_range": [1.0, 1.0],
              "shadows.prob": 1.0}
_DISP_RB = 85 * 1.2  # float px, outer boundary radius under _DISP_PINS
_DISP_M = 20  # int px, canvas_margin of the displaced config


def _disp_cfg(d, margin, **extra):
    """_DISP_PINS cfg with displacement_range [d, d] (float px) and
    canvas_margin (int px); extra "section.key" overrides on top."""
    return _cfg(**_DISP_PINS, **{"geometry.displacement_range": [d, d],
                                 "geometry.canvas_margin": margin}, **extra)


def _dist_grid(cx, cy):
    """Distance of every pixel from (cx, cy) (float px), (256, 256) float64."""
    ys, xs = np.mgrid[:IMG, :IMG]
    return np.hypot(xs - cx, ys - cy)


def _check_translation(extra, erode):
    """A (d = 0, margin M+10) and B (|d| = 10, margin M) share one canvas
    size, so B must equal A shifted by the integer tip difference at every
    pixel deeper than `erode` px (float) inside BOTH boundaries (A coords).
    extra: dict of "section.key" overrides for both configs."""
    for seed in range(5):
        a = g7.render_image(_disp_cfg(0.0, _DISP_M + 10, **extra), seed)
        b = g7.render_image(_disp_cfg(10.0, _DISP_M, **extra), seed)
        fdx, fdy = b.tip_x - a.tip_x, b.tip_y - a.tip_y  # float px, tip shift
        dx, dy = int(round(fdx)), int(round(fdy))  # int px, displacement
        assert abs(fdx - dx) < 1e-6 and abs(fdy - dy) < 1e-6, (seed, fdx, fdy)
        assert abs(math.hypot(dx, dy) - 10.0) <= 1.0, (seed, dx, dy)
        c = IMG / 2.0  # float px, boundary centre in the image
        # (256, 256) bool: A pixels deep inside A's boundary that also lie
        # inside B's boundary once shifted (B's boundary sits at c - d in A)
        region = ((_dist_grid(c, c) <= _DISP_RB - erode)
                  & (_dist_grid(c - dx, c - dy) <= _DISP_RB - erode))
        ys, xs = np.nonzero(region)  # (N,) int each, A pixel coords
        assert len(ys) > 0.5 * math.pi * _DISP_RB ** 2, seed
        assert np.array_equal(b.image[ys + dy, xs + dx], a.image[ys, xs]), \
            f"seed {seed}: content inside the boundary is not a pure translation"


def test_inside_boundary_is_pure_translation():
    _check_translation({"background.dial_circle_blur": False}, 2.0)


def test_inside_boundary_translation_with_blur():
    # blur radius 3 mixes the black outside the boundary into its edge
    _check_translation({"background.dial_circle_blur": True}, 14.0)


def test_boundary_fixed_and_black_outside():
    c = IMG / 2.0  # float px, boundary centre in the image
    outside = _dist_grid(c, c) > _DISP_RB  # (256, 256) bool, outside B
    for seed in range(4):
        for d in (0.0, 10.0):
            img = g7.render_image(_disp_cfg(d, _DISP_M), seed).image  # (256, 256) uint8
            assert not img[outside].any(), (seed, d)
            ys, xs = np.nonzero(img)  # (N,) int each, nonzero pixel coords
            bx = (xs.min() + xs.max()) / 2.0  # float px, bbox centre x
            by = (ys.min() + ys.max()) / 2.0  # float px, bbox centre y
            assert abs(bx - c) <= 1.0 and abs(by - c) <= 1.0, (seed, d, bx, by)
            assert abs(xs.mean() - c) <= 1.0 and abs(ys.mean() - c) <= 1.0, (seed, d)


# ── physical multiplicative shadows ─────────────────────────────────────

# "section.key" -> value pins (dict) for the shadow tests: shadow always on
# (unless overridden), shared strength 0.5 with no jitter, no blur anywhere
# the config allows it (rim blur has a schema min of 0.5)
_SHD_PINS = {"shadows.prob": 1.0, "shadows.strength_range": [0.5, 0.5],
             "shadows.strength_jitter": 0.0, "shadows.blur_radius": 0.0,
             "shadows.rim_blur_range": [0.5, 0.5],
             "background.dial_circle_blur": False}
_SHD_SEEDS = range(6)  # seeds (range of int) every shadow render test uses


def _erode(mask, steps):
    """Binary 4-neighbour erosion of a (H, W) bool mask, `steps` times."""
    return ~_dilate(~mask, steps)


def _umbra(extra, seed):
    """Shadow umbra of one render plus the matching no-shadow image.

    Renders the pinned cfg (+ extra overrides, dict) at strength 1 and with
    the shadow off.  Pixels the full-strength shadow turns black while the
    unshadowed pixel is >= 60 lie where the shape mask is ~1; eroded by 2 px
    so the bilinear-stretched mask edge cannot reach them.
    Returns (umbra (256, 256) bool, no_shadow (256, 256) int32).
    """
    full = _cfg(**{**_SHD_PINS, **extra, "shadows.strength_range": [1.0, 1.0]})
    none = _cfg(**{**_SHD_PINS, **extra, "shadows.prob": 0.0})
    f = g7.render_image(full, seed).image.astype(np.int32)  # (256, 256) int32
    b = g7.render_image(none, seed).image.astype(np.int32)  # (256, 256) int32
    return _erode((f == 0) & (b >= 60), 2), b


def test_shadow_darkening_is_max_not_compound():
    m_p = np.array([[1.0, 0.0, 0.5, 1.0]], dtype=np.float32)  # (1, 4) pointer mask
    m_r = np.array([[1.0, 1.0, 0.5, 0.0]], dtype=np.float32)  # (1, 4) rim mask
    d = g7.shadow_darkening(m_p, m_r, 0.5, 0.4)  # (1, 4) float32
    assert d.dtype == np.float32 and d.shape == (1, 4)
    # overlap of two full umbras: max(0.5, 0.4) = 0.5, not 1 - 0.5 * 0.6 = 0.7
    np.testing.assert_allclose(d, [[0.5, 0.4, 0.25, 0.5]], atol=1e-6)


def test_shadow_never_brightens():
    for extra in ({}, {"palette.bg_gray_range": [0, 30]}):  # default and very dark face
        for seed in _SHD_SEEDS:
            on = _cfg(**{**extra, "shadows.prob": 1.0})
            off = _cfg(**{**extra, "shadows.prob": 0.0})
            a = g7.render_image(on, seed).image.astype(np.int16)  # (256, 256)
            b = g7.render_image(off, seed).image.astype(np.int16)  # (256, 256)
            assert (a <= b + 1).all(), (extra, seed, int((a - b).max()))
            assert (a < b - 5).any(), (extra, seed, "shadow invisible")


def test_shadow_is_multiplicative():
    for seed in _SHD_SEEDS:
        umbra, b = _umbra({}, seed)
        assert umbra.sum() > 200, (seed, int(umbra.sum()))
        a = g7.render_image(_cfg(**_SHD_PINS), seed).image.astype(np.int32)
        # out = round(b * (1 - 0.5)) inside the umbra: within 1 level
        assert np.abs(a[umbra] - 0.5 * b[umbra]).max() <= 1.0, seed
        # texture kept: shadowed std ~ 0.5 x unshadowed std, not ~ 0
        sb, sa = float(b[umbra].std()), float(a[umbra].std())  # float, gray levels
        assert sb > 2.0, (seed, sb)
        assert 0.4 * sb <= sa <= 0.6 * sb, (seed, sa, sb)


def test_shadow_darkens_ticks():
    bright = {"digits.show_prob": 1.0, "ticks.gray_range": [220, 230],
              "digits.gray_range": [220, 230]}  # light overlay, same draw widths
    dark = {"digits.show_prob": 1.0, "ticks.gray_range": [0, 10],
            "digits.gray_range": [0, 10]}
    hit = 0  # int, overlay pixels checked inside the umbra over all seeds
    for seed in _SHD_SEEDS:
        umbra, b = _umbra(bright, seed)
        a = g7.render_image(_cfg(**{**_SHD_PINS, **bright}), seed).image.astype(np.int32)
        # overlay footprint: same draws, only the tick/digit gray differs
        foot = (g7.render_image(_cfg(**{**_SHD_PINS, **dark}), seed).image
                != g7.render_image(_cfg(**{**_SHD_PINS, **bright}), seed).image)
        sel = foot & umbra & (b >= 200)  # (256, 256) bool, bright overlay in shadow
        hit += int(sel.sum())
        if sel.any():
            assert np.abs(a[sel] - 0.5 * b[sel]).max() <= 1.0, seed
    assert hit > 20, hit


def test_strength_jitter_individual():
    S = _cfg(**{"shadows.strength_range": [0.5, 0.5],
                "shadows.strength_jitter": 0.2})["shadows"]
    pairs = [g7._shadow_strengths(S, g7.section_rngs(s)["shadows"][0])
             for s in range(10)]  # list of (s_p, s_r) float pairs
    assert all(0.3 <= p <= 0.7 and 0.3 <= r <= 0.7 for p, r in pairs)
    assert any(abs(p - r) > 0.01 for p, r in pairs), pairs
    S0 = dict(S, strength_jitter=0.0)
    assert all(g7._shadow_strengths(S0, g7.section_rngs(s)["shadows"][0]) == (0.5, 0.5)
               for s in range(5))


# ── tip label = visible tip (cut_tip aware) ─────────────────────────────

# "section.key" -> value pins (dict): no stretch, so the label shift is exactly r
_NO_STRETCH = {"geometry.stretch_h_range": [1.0, 1.0], "geometry.stretch_v_range": [1.0, 1.0]}
_CUT_R = 8  # int px, pinned cut_tip radius for the label tests


def _cut_pair(seed, **extra):
    """Render seed with cut_tip off (A) and on with radius _CUT_R (B), all else
    equal; extra: "section.key" overrides for both. -> (A, B) RenderResults."""
    off = _cfg(**extra, **{"cut_tip.enabled": False, "cut_tip.radius_range": [_CUT_R, _CUT_R]})
    on = _cfg(**extra, **{"cut_tip.enabled": True, "cut_tip.radius_range": [_CUT_R, _CUT_R]})
    return g7.render_image(off, seed), g7.render_image(on, seed)


def _unit_dir(angle_deg):
    """Unit pointer direction u in image coords (0 = up, clockwise) -> (ux, uy) floats."""
    a = math.radians(angle_deg - 90)  # float rad, canvas angle of the pointer
    return math.cos(a), math.sin(a)


def _ideal(res):
    """Ideal triangle apex recorded with a render, image px -> (x, y) floats."""
    d = res.params["derived"]
    return d["ideal_apex_x"], d["ideal_apex_y"]


def test_tip_label_is_visible_tip():
    for seed in range(6):
        a, b = _cut_pair(seed, **_NO_STRETCH)
        assert a.angle_deg == b.angle_deg
        assert _ideal(a) == _ideal(b)  # the cut does not move the triangle
        ux, uy = _unit_dir(a.angle_deg)  # unit pointer direction (floats)
        for res in (a, b):
            ix, iy = _ideal(res)  # ideal apex (floats)
            dx, dy = res.tip_x - ix, res.tip_y - iy  # float px, label - ideal apex
            # the label sits ON the pointer axis, behind the ideal apex (along -u)
            # (params are rounded to 4 decimals, hence the 1e-3 tolerances)
            assert abs(-dx * uy + dy * ux) < 1e-3, (seed, dx, dy)
            back = -(dx * ux + dy * uy)  # float px, recession along the axis
            assert abs(back - res.params["derived"]["tip_recession"]) < 1e-3, seed
            assert res.params["cut_tip"]["tip_recession"] == res.params["derived"]["tip_recession"]
        # cut on: the visible end is about cut_r behind the apex (arc + edge ring)
        rb = b.params["derived"]["tip_recession"]  # float px
        assert _CUT_R - 1.5 <= rb <= _CUT_R + 3, (seed, rb)
        assert rb > a.params["derived"]["tip_recession"], seed


def test_tip_label_stretched():
    sh, sv = 1.05, 0.95  # float, pinned horizontal / vertical stretch
    stretch = {"geometry.stretch_h_range": [sh, sh], "geometry.stretch_v_range": [sv, sv]}
    for seed in range(4):
        a, b = _cut_pair(seed, **stretch)
        ux, uy = _unit_dir(a.angle_deg)  # unit pointer direction (floats)
        for res in (a, b):
            ix, iy = _ideal(res)  # ideal apex, stretched (floats)
            dx, dy = res.tip_x - ix, res.tip_y - iy  # float px, label - ideal apex
            # on the stretched axis (direction (ux*sh, uy*sv)), behind the apex
            assert abs(-dx * uy * sv + dy * ux * sh) < 1e-3, seed
            assert dx * ux * sh + dy * uy * sv <= 1e-3, seed
            # tip_recession is that offset measured before the stretch
            rec = math.hypot(dx / sh, dy / sv)  # float px
            assert abs(rec - res.params["derived"]["tip_recession"]) < 1e-3, seed


def test_tip_label_on_pointer_pixels():
    # black pointer on a bright, flat, clean face so "pointer-dark" is obvious
    clean = {**_NO_STRETCH,
             "pointer.gray_range": [0, 0], "pointer.gradient_range": [0.0, 0.0],
             "palette.bg_gray_range": [230, 230], "palette.circle_gray_range": [230, 230],
             "shadows.prob": 0.0, "digits.show_prob": 0.0, "tri_artifacts.enabled": False,
             "background.dial_circle_blur": False,
             "jitter.brightness_range": [1.0, 1.0], "jitter.contrast_range": [1.0, 1.0],
             "jitter.sharpness_range": [1.0, 1.0]}
    dark = 100  # int gray, below this a pixel counts as pointer-dark
    for seed in range(6):
        a, b = _cut_pair(seed, **clean)
        img = b.image  # (256, 256) uint8, cut-on render
        ux, uy = _unit_dir(b.angle_deg)  # unit pointer direction (floats)

        def px(x, y):
            """Gray (int) of the pixel containing float point (x, y)."""
            return int(img[int(math.floor(y + 0.5)), int(math.floor(x + 0.5))])

        assert px(b.tip_x - 2 * ux, b.tip_y - 2 * uy) < dark, (seed, "behind label")
        assert px(b.tip_x + 2 * ux, b.tip_y + 2 * uy) >= dark, (seed, "beyond label")
        assert px(*_ideal(b)) >= dark, (seed, "ideal apex is cut away")


# ── reflections + glares from ASCII glyph silhouettes ─────────────────────

_GLYPH_SIDE = 128  # int px, longest side of every cached glyph silhouette


def _sil(char):
    """Glyph silhouette of char in the first available font: (h, w) bool, longest side 128."""
    return g7.glyph_silhouette(char, 0)


# "section.key" -> value pins (dict) for the effect tests: no blur, so every
# changed pixel shows the raw effect value
_FX_QUIET = {"background.dial_circle_blur": False}
_FX_SEEDS = range(5)  # seeds (range of int) every effect test uses


def _refl(**extra):
    """Quiet cfg with reflections always on (count 1, darken 30, hard edge)."""
    return _cfg(**_FX_QUIET, **{"reflections.prob": 1.0, "reflections.count_range": [1, 1],
                                "reflections.darken_range": [30, 30],
                                "reflections.edge_soft": 0.0, **extra})


def _glare(**extra):
    """Quiet cfg with glares always on (value 240, hard edge)."""
    return _cfg(**_FX_QUIET, **{"glares.prob": 1.0, "glares.value_range": [240, 240],
                                "glares.edge_soft": 0.0, **extra})


def test_effects_off_by_default_hash_unchanged():
    cfg = v7_config.default_config()
    assert cfg["reflections"]["prob"] == 0.0 and cfg["glares"]["prob"] == 0.0
    test_default_render_unchanged_by_contrast_key()


def test_reflection_darkens_by_value():
    any_change = False
    for seed in _FX_SEEDS:
        base = g7.render_image(_cfg(**_FX_QUIET), seed).image.astype(np.int16)  # (256, 256)
        got = g7.render_image(_refl(), seed).image.astype(np.int16)  # (256, 256)
        changed = got != base  # (256, 256) bool
        assert not (got > base).any(), seed
        drop = (base - got)[changed]  # (N,) int16, darkening of changed pixels
        clipped = (got[changed] == 0) & (base[changed] < 30)  # (N,) bool
        assert ((drop == 30) | clipped).all(), (seed, np.unique(drop))
        any_change |= changed.any()
    assert any_change, "no reflection visible"


def test_glare_overwrites_with_value():
    any_change = False
    for seed in _FX_SEEDS:
        base = g7.render_image(_cfg(**_FX_QUIET), seed).image  # (256, 256) uint8
        got = g7.render_image(_glare(), seed).image  # (256, 256) uint8
        changed = got != base  # (256, 256) bool
        assert (got[changed] == 240).all(), (seed, np.unique(got[changed]))
        any_change |= changed.any()
    assert any_change, "no glare visible"


# "section.key" -> value pins (dict): the biggest rigid-ish glyphs, so effects cover a lot
_BIG = {"size_range": [256, 256], "perspective_range": [0.0, 0.0],
        "warp_strength_range": [0.0, 0.0],
        "aspect_range": [1.0, 1.0], "shear_range": [0.0, 0.0]}


def _big(section, count):
    """Dotted overrides making `section` items huge rigid glyphs, `count` per image."""
    out = {f"{section}.{k}": v for k, v in _BIG.items()}  # dict[str, value]
    out[f"{section}.count_range"] = [count, count]
    return out


def test_glare_clears_pointer_tip():
    clear = 15.0  # float px, pinned clear distance
    extra = {**_big("glares", 8), "glares.clear_distance": clear}
    ys, xs = np.mgrid[:IMG, :IMG]  # (256, 256) int pixel grids
    for seed in range(6):
        base = g7.render_image(_cfg(**_FX_QUIET), seed)
        got = g7.render_image(_glare(**extra), seed)
        assert _same_labels(base, got)
        changed = got.image != base.image  # (256, 256) bool
        near = np.hypot(xs - got.tip_x, ys - got.tip_y) <= clear  # (256, 256) bool
        assert not (changed & near).any(), seed
        assert changed.sum() > 100, seed


def test_glare_wins_over_reflection():
    big = {**_big("reflections", 4), **_big("glares", 4)}
    glare_on = {"glares.prob": 1.0, "glares.value_range": [240, 240], "glares.edge_soft": 0.0}
    both_seen = 0
    for seed in _FX_SEEDS:
        base = g7.render_image(_cfg(**_FX_QUIET, **big), seed).image  # (256, 256) uint8
        r = g7.render_image(_refl(**big), seed).image  # reflections only
        gl = g7.render_image(_glare(**big), seed).image  # glares only
        both = g7.render_image(_refl(**big, **glare_on), seed).image  # both kinds
        overlap = (r != base) & (gl != base)  # (256, 256) bool, covered by both
        assert (both[overlap] == 240).all(), seed
        both_seen += int(overlap.sum())
    assert both_seen > 100, "no overlap forced"


def test_effects_stay_inside_boundary():
    c = IMG / 2.0  # float px, boundary centre in the image
    outside = _dist_grid(c, c) > _DISP_RB  # (256, 256) bool, outside B
    fx = {"reflections.prob": 1.0, **_big("reflections", 8), "glares.prob": 1.0,
          **_big("glares", 8), "glares.value_range": [255, 255]}
    for seed in range(4):
        for d in (0.0, 10.0):
            img = g7.render_image(_disp_cfg(d, _DISP_M, **fx), seed).image  # (256, 256) uint8
            assert not img[outside].any(), (seed, d)
            assert (img == 255).sum() > 100, (seed, d)


def test_glyph_silhouette():
    from scipy import ndimage
    assert g7.GLYPHS == "".join(sorted(set(g7.GLYPHS), key=g7.GLYPHS.index))
    assert len(g7.GLYPHS) == 62 and len(g7.glyph_fonts()) >= 1
    for ch in g7.GLYPHS:
        sil = _sil(ch)  # (h, w) bool
        assert sil.dtype == bool and sil.any(), ch
        assert abs(max(sil.shape) - _GLYPH_SIDE) <= 2, (ch, sil.shape)
        # tight bbox: every border row / column has a pixel
        assert sil[0].any() and sil[-1].any() and sil[:, 0].any() and sil[:, -1].any(), ch
    # "O" has a hole: its background (padded so the outside is one piece) splits in two+
    bg = ~np.pad(_sil("O"), 1)  # (h + 2, w + 2) bool, background of "O"
    assert ndimage.label(bg)[1] > 1
    h, w = _sil("l").shape
    assert h == max(h, w) and h > 3 * w, (h, w)  # "l" is tall and thin


def _iou(a, b):
    """IoU (float) of two bool masks, zero-padded top-left to a common shape."""
    h, w = max(a.shape[0], b.shape[0]), max(a.shape[1], b.shape[1])
    pa = np.zeros((h, w), dtype=bool)  # (h, w) bool
    pb = np.zeros((h, w), dtype=bool)  # (h, w) bool
    pa[:a.shape[0], :a.shape[1]] = a
    pb[:b.shape[0], :b.shape[1]] = b
    return (pa & pb).sum() / max(1, (pa | pb).sum())


# rigid shape-transform kwargs (dict): no rotation / stretch / shear / perspective / warp / blur
_RIGID = dict(rotation_deg=0.0, aspect=1.0, shear=0.0, warp=None, edge_soft=0.0,
              perspective=(0.0, 0.0))


def _tight(a):
    """Crop bool mask a to its tight bbox -> (h', w') bool."""
    ys, xs = np.nonzero(a)
    return a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def test_shape_transform_rigid_identity():
    import cv2
    s = 50  # int px, target longest side
    for ch in "OLE":
        sil = _sil(ch)  # (h, w) bool
        m = g7.transform_shape(sil, s, **_RIGID)  # (h', w') float32
        assert m.dtype == np.float32 and s - 4 <= max(m.shape) <= s + 2
        h, w = sil.shape
        f = s / max(h, w)  # float scale
        # reference: plain resize of the raw silhouette, cropped like the output
        ref = _tight(cv2.resize(sil.astype(np.float32), (max(1, round(w * f)), max(1, round(h * f))),
                                interpolation=cv2.INTER_AREA) >= 0.5)  # (h', w') bool
        assert _iou(m > 0.5, ref) > 0.93, ch


def test_shape_edges_smoothed_not_stair_stepped():
    # a NEAREST-upscaled (stair-stepped) silhouette: the pre-blur must
    # round a coarse staircase into a smooth edge at 2x scale
    coarse = np.zeros((64, 64), dtype=bool)  # (64, 64) bool, 45-deg staircase triangle
    coarse[np.tril_indices(64)] = True
    sil = np.kron(coarse, np.ones((2, 2), dtype=bool))  # (128, 128) bool, 2-px steps (2x NEAREST)
    m = g7.transform_shape(sil, 256, **_RIGID) > 0.5  # (h', w') bool at 2x scale
    rows = m.sum(axis=1)[20:-20].astype(int)  # (n,) int filled width per row, away from corners
    # a raw 2-px staircase at 2x widens by 4 px every 4 rows (jumps of 4);
    # a smoothed edge widens by about 1 px per row
    assert np.abs(np.diff(rows)).max() <= 2


def test_shape_transform_rotation():
    # convention: rotation_deg > 0 turns the shape counter-clockwise as
    # displayed (image y down), so +90 deg equals np.rot90(rigid, k=1)
    sil = _sil("L")
    rigid = g7.transform_shape(sil, 60, **_RIGID) > 0.5  # (h, w) bool
    rot = g7.transform_shape(sil, 60, **{**_RIGID, "rotation_deg": 90.0}) > 0.5
    assert _iou(rot, np.rot90(rigid, 1)) > 0.9
    assert _iou(rot, np.rot90(rigid, -1)) < 0.9  # the direction is really pinned


def test_warp_is_smooth_and_bounded():
    from scipy import ndimage
    A = 6.0  # float px, warp amplitude
    sil = _sil("O")
    flat = g7.transform_shape(sil, 60, **_RIGID) > 0.5  # (h, w) bool, unwarped
    _, ncomp_flat = ndimage.label(flat)
    nrng = np.random.default_rng(3)
    for g in (4, 7):
        for _ in range(4):
            warp = nrng.uniform(-A, A, size=(2, g, g))  # (2, g, g) float: dx grid, dy grid (px)
            m = g7.transform_shape(sil, 60, **{**_RIGID, "warp": warp}) > 0.5  # (h', w') bool
            assert m.shape != flat.shape or (m != flat).any(), "warp changed nothing"
            _, ncomp = ndimage.label(m)
            assert ncomp == ncomp_flat
            assert abs(int(m.sum()) - int(flat.sum())) <= 0.35 * flat.sum()
            assert m.shape[0] - flat.shape[0] <= 2 * A + 2
            assert m.shape[1] - flat.shape[1] <= 2 * A + 2


def test_perspective_changes_shape():
    s = 80  # int px, target longest side
    sil = _sil("E")
    flat = g7.transform_shape(sil, s, **_RIGID) > 0.5  # (h, w) bool, affine only
    for p in ((0.6, 0.0), (0.0, 0.6), (-0.6, 0.6)):
        m = g7.transform_shape(sil, s, **{**_RIGID, "perspective": p})  # (h', w') float32
        assert np.isfinite(m).all() and (m > 0.5).any(), p
        assert max(m.shape) <= 4 * s, p
        assert _iou(m > 0.5, flat) < 0.95, p


# "section.key" -> value pins (dict): every shape key at its schema extreme
def _extreme(section):
    """Dotted overrides putting `section`'s shape keys at the schema maxima."""
    return {f"{section}.aspect_range": [0.1, 10.0], f"{section}.shear_range": [-3.0, 3.0],
            f"{section}.perspective_range": [0.9, 0.9],
            f"{section}.warp_strength_range": [1.0, 1.0],
            f"{section}.warp_grid_range": [12, 12], f"{section}.size_range": [8, 256]}


def test_extreme_ranges_safe():
    import random
    cfg = _cfg(**_extreme("reflections"))
    sec = cfg["reflections"]  # dict, extreme reflections section
    seen = 0  # int, non-empty masks
    for seed in range(200):
        r, n = random.Random(seed), np.random.default_rng(seed)
        m = g7._glyph_shape(r, n, sec)  # (h', w') float32
        # replay the documented draw order (glyph, font, size) to get this item's size
        rep = random.Random(seed)  # random.Random, same stream from the start
        rep.randrange(len(g7.GLYPHS)), rep.randrange(len(g7.glyph_fonts()))
        size = rep.randint(*sec["size_range"])  # int px, the item's longest side s
        assert m.dtype == np.float32 and np.isfinite(m).all(), seed
        assert max(m.shape) <= 4 * size, (seed, m.shape, size)
        seen += int((m > 0.5).any())
    assert seen > 150
    # and the full render path at the extremes, both kinds
    full = _cfg(**_extreme("reflections"), **_extreme("glares"),
                **{"reflections.prob": 1.0, "glares.prob": 1.0})
    for seed in range(5):
        img = g7.render_image(full, seed).image  # (256, 256) uint8
        assert img.shape == (IMG, IMG)


def test_glyph_shape_draw_count_fixed():
    # the per-item draws are fixed-count: after one item the streams' state does
    # not depend on the perspective / warp-strength values, nor on the g drawn
    # (nrng always advances by exactly 2*G*G, G = warp_grid_range[1])
    import random
    cfg = v7_config.default_config()
    states = []  # list[tuple[float, float]], next draw of each stream
    for persp, strength in (([0.0, 0.0], [0.0, 0.0]), ([0.7, 0.7], [0.5, 0.5])):
        sec = dict(cfg["reflections"], perspective_range=persp, warp_strength_range=strength)
        r, n = random.Random(1), np.random.default_rng(1)
        g7._glyph_shape(r, n, sec)
        states.append((r.random(), float(n.random())))
    assert states[0] == states[1]
    # g varies per item (range 2..12) yet nrng consumption is the same each time
    sec = dict(cfg["reflections"], warp_grid_range=[2, 12])
    want = float(np.random.default_rng(5).random(2 * 12 * 12 + 1)[-1])  # float, next draw after 2*G*G
    for seed in range(8):
        r, n = random.Random(seed), np.random.default_rng(5)
        g7._glyph_shape(r, n, sec)
        assert float(n.random()) == want, seed


def test_items_get_different_grid_and_strength():
    # two items drawn from one section's streams get their own g and strength
    import random
    cfg = v7_config.default_config()
    sec = dict(cfg["reflections"], warp_grid_range=[2, 12], warp_strength_range=[0.0, 1.0],
               size_range=[64, 64])
    r, n = random.Random(3), np.random.default_rng(3)
    seen = []  # list[tuple[int, float]], (g, max |warp| px) per item given to transform_shape
    orig = g7.transform_shape

    def spy(sil, size, rot, aspect, shear, warp, edge_soft, persp):
        seen.append(None if warp is None else (warp.shape[1], float(np.abs(warp).max())))
        return orig(sil, size, rot, aspect, shear, warp, edge_soft, persp)

    g7.transform_shape = spy
    try:
        for _ in range(2):
            g7._glyph_shape(r, n, sec)
    finally:
        g7.transform_shape = orig
    assert len(seen) == 2 and None not in seen
    assert seen[0][0] != seen[1][0], seen  # different g
    assert seen[0][1] != seen[1][1], seen  # different strength-scaled amplitude


def test_effects_deterministic():
    cfg = _refl(**{"glares.prob": 1.0, "reflections.count_range": [2, 3]})
    for seed in range(3):
        a = g7.render_image(cfg, seed)
        b = g7.render_image(cfg, seed)
        assert np.array_equal(a.image, b.image) and _same_labels(a, b), seed


# ── sampled per-image params (RenderResult.params) ────────────────────────

# params groups (tuple[str, ...]) every render must record, one per section
_PARAM_GROUPS = ("angle", "geometry", "outer", "palette", "background", "digits", "ticks",
                 "shadows", "pointer", "cut_tip", "jitter", "hier_noise", "tri_artifacts",
                 "reflections", "glares", "blur", "derived")


def _json_plain(v):
    """True if v is built only from dict/list/str/int/float/bool/None (JSON types)."""
    if isinstance(v, dict):
        return all(isinstance(k, str) and _json_plain(x) for k, x in v.items())
    if isinstance(v, list):
        return all(_json_plain(x) for x in v)
    return v is None or type(v) in (str, int, float, bool)


def test_params_json_serialisable_and_grouped():
    import json
    for seed in range(4):
        res = g7.render_image(v7_config.default_config(), seed)
        p = res.params  # dict[str, dict], sampled values by section
        assert _json_plain(p), seed
        json.dumps(p)
        for grp in _PARAM_GROUPS:
            assert isinstance(p.get(grp), dict), (seed, grp)
        assert p["angle"]["angle_deg"] == res.angle_deg
        assert p["geometry"]["dial_radius"] > 0
        assert p["derived"]["tip_x"] == round(res.tip_x, 4)


def test_params_effect_items_match_count():
    cfg = _cfg(**{"reflections.prob": 1.0, "reflections.count_range": [2, 4],
                  "glares.prob": 1.0, "glares.count_range": [1, 3]})
    keys = {"char", "font", "size", "rotation", "aspect", "shear", "P", "px", "py",
            "warp_strength", "g", "center_x", "center_y"}  # set[str], per-item record keys
    for seed in range(3):
        p = g7.render_image(cfg, seed).params
        for name, val in (("reflections", "darken"), ("glares", "value")):
            sec = p[name]  # dict, one effect section
            assert sec["active"] is True
            assert len(sec["items"]) == sec["n"] >= 1, (seed, name)
            for it in sec["items"]:
                assert keys | {val} <= set(it), (seed, name, set(it))
                assert it["char"] in g7.GLYPHS


def test_params_effects_off_has_no_items():
    p = g7.render_image(v7_config.default_config(), 0).params
    assert p["reflections"]["active"] is False and p["reflections"]["items"] == []
    assert p["glares"]["items"] == []


def test_params_recording_keeps_render_identical():
    # rendering twice gives identical params, and the pinned hashes still hold
    cfg = _cfg(**{"reflections.prob": 1.0, "glares.prob": 1.0})
    a, b = g7.render_image(cfg, 7), g7.render_image(cfg, 7)
    assert a.params == b.params and np.array_equal(a.image, b.image)
    test_default_render_unchanged_by_contrast_key()


# ── tip label = VISIBLE pointer end (edge ring + cut aware) ───────────────

# "section.key" -> value pins (dict): black pointer on a bright, flat, clean face
# with no noise / blur / shadows / jitter / artifacts, so "pointer-dark" is unambiguous
_CLEAN_TIP = {
    "pointer.gray_range": [0, 0], "pointer.noise_amplitude": 0,
    "pointer.edge_noise_amplitude": 0, "pointer.gradient_range": [0.0, 0.0],
    "palette.bg_gray_range": [200, 200], "palette.circle_gray_range": [220, 220],
    "background.noise_intensity": 0, "background.variation": 0,
    "background.dial_circle_blur": False,
    "hier_noise.block_variance": 0, "hier_noise.pixel_variance": 0,
    "jitter.brightness_range": [1.0, 1.0], "jitter.contrast_range": [1.0, 1.0],
    "jitter.sharpness_range": [1.0, 1.0],
    "shadows.prob": 0.0, "tri_artifacts.enabled": False,
    "digits.show_prob": 0.0, "ticks.gray_range": [215, 215]}

# case name (str) -> (cut_tip.enabled bool, cut_tip.radius_range [lo, hi] int)
_CUT_CASES = {"off": (False, [7, 10]), "r1": (True, [1, 1]), "r2": (True, [2, 2]),
              "r4": (True, [4, 4]), "r7-10": (True, [7, 10])}


def _tip_overshoot(res):
    """How far (float px) the label sits beyond the last dark pointer pixel,
    measured along the (stretched) pointer axis in the rendered image, the way a
    person would judge the tip by eye.  Positive = label floats past the pointer.

    Dark pixels (< 100) within |lateral| < 6 px of the axis and along-axis in
    (-30, 12) px of the label are the pointer end; overshoot = -max(along).
    Returns None when no such pixel exists.
    """
    g = res.params["geometry"]
    ux, uy = _unit_dir(res.angle_deg)  # unit pointer direction, unstretched (floats)
    vx, vy = ux * g["stretch_h"], uy * g["stretch_v"]  # stretched axis direction (floats)
    n = math.hypot(vx, vy)
    vx, vy = vx / n, vy / n  # unit stretched axis (floats)
    ys, xs = np.nonzero(res.image < 100)  # (N,) int64 dark pixel coords
    dx, dy = xs - res.tip_x, ys - res.tip_y  # (N,) float64, offsets from the label
    along = dx * vx + dy * vy  # (N,) float64, + = beyond the label
    lat = -dx * vy + dy * vx  # (N,) float64, perpendicular distance (signed)
    sel = (np.abs(lat) < 6) & (along > -30) & (along < 12)  # (N,) bool
    if not sel.any():
        return None
    return -float(along[sel].max())


@pytest.mark.parametrize("case", list(_CUT_CASES))
def test_tip_label_on_visible_end(case):
    # blur off and no observer floor (tip_label.base_blur 0): sigma_eff = 0, w* = 0,
    # so the label is the raw visible end; the default floor (sigma_0 = 1) only
    # ever moves it back from there (width-based label, w* ~ 1.35 px)
    enabled, rr = _CUT_CASES[case]
    pins = {**_CLEAN_TIP, "cut_tip.enabled": enabled, "cut_tip.radius_range": rr}  # dict
    cfg = _cfg(**pins, **{"tip_label.base_blur": 0.0})
    cfg_floor = _cfg(**pins)  # default base_blur 1.0
    over = []  # list[float] px, label overshoot per seed
    for seed in range(60):
        res = g7.render_image(cfg, seed)
        assert res.params["derived"]["w_star"] == 0.0
        o = _tip_overshoot(res)
        assert o is not None, (case, seed)
        over.append(o)
        o_floor = _tip_overshoot(g7.render_image(cfg_floor, seed))  # float px
        assert o_floor <= o + 1e-6, (case, seed, o, o_floor)
    over = np.array(over)  # (60,) float64
    assert abs(over.mean()) <= 0.35, (case, over.mean(), over.max())
    assert over.max() <= 1.0, (case, over.mean(), over.max())


# ── _visible_tip unit tests (pure helper) ─────────────────────────────────

def _wedge(apex, half_deg, size=64):
    """Bool mask (size, size) of a wedge opening left from apex (x, y), half-angle half_deg."""
    ys, xs = np.mgrid[:size, :size]  # (size, size) int pixel grids
    dx, dy = apex[0] - xs, ys - apex[1]  # (size, size) float, + dx = behind the apex
    return (dx >= 0) & (np.abs(dy) <= dx * math.tan(math.radians(half_deg)))


def test_visible_tip_wedge_recedes():
    apex = (40.3, 32.0)  # (x, y) float px, ideal apex; the wedge points to +x
    full = _wedge(apex, 20)  # (64, 64) bool
    tx, ty = g7._visible_tip(full, apex, (1.0, 0.0), 20)
    assert ty == apex[1] and abs(tx - 40.0) < 0.05 + 1e-9  # last pixel centre x = 40
    # erase the outer 3 px of the wedge: the visible end recedes by 3 px
    trimmed = full & (np.mgrid[:64, :64][1] <= 37)  # (64, 64) bool
    tx2, ty2 = g7._visible_tip(trimmed, apex, (1.0, 0.0), 20)
    assert ty2 == apex[1] and abs(tx2 - 37.0) < 0.05 + 1e-9


def test_visible_tip_empty_falls_back():
    vis = np.zeros((32, 32), dtype=bool)  # (32, 32) bool, nothing visible
    assert g7._visible_tip(vis, (16.0, 16.0), (0.0, -1.0), 20, fallback=(1.5, 2.5)) == (1.5, 2.5)
    assert g7._visible_tip(vis, (16.0, 16.0), (0.0, -1.0), 20) is None
    # a covered pointer is not visible either
    vis[10:20, 14:19] = True
    assert g7._visible_tip(vis, (16.0, 9.0), (0.0, -1.0), 20, fallback=(0.0, 0.0),
                           cover=np.ones((32, 32), dtype=bool)) == (0.0, 0.0)


def test_visible_tip_square_on_axis():
    vis = np.zeros((32, 32), dtype=float)  # (32, 32) float weights
    vis[12:30, 10:21] = 1.0  # flat-ended bar, top row y = 12, pointing up (-y)
    tx, ty = g7._visible_tip(vis, (15.0, 4.0), (0.0, -1.0), 20)
    assert tx == 15.0 and abs(ty - 12.0) < 0.05 + 1e-9  # on the axis, at the top row
    vis[12, :] = 0.5  # an even 50/50 ring does not count
    tx, ty = g7._visible_tip(vis, (15.0, 4.0), (0.0, -1.0), 20)
    assert tx == 15.0 and abs(ty - 13.0) < 0.05 + 1e-9


# ── width-based tip label (blur aware) ────────────────────────────────────

def _aa_wedge(apex, half_deg, size=96, ss=8):
    """Area-coverage wedge (size, size) float32 in [0, 1] opening left (-x) from
    apex (x, y), half-angle half_deg: each pixel (centre at integer coords) is
    the fraction of its ss x ss sub-samples inside the ideal wedge (no edge ring)."""
    off = (np.arange(ss) + 0.5) / ss - 0.5  # (ss,) float64, sub-sample offsets in a pixel
    ys, xs = np.mgrid[:size, :size]  # (size, size) int pixel grids
    sx = xs[..., None, None] + off[None, None, None, :]  # (size, size, 1, ss) float64
    sy = ys[..., None, None] + off[None, None, :, None]  # (size, size, ss, 1) float64
    dx, dy = apex[0] - sx, sy - apex[1]  # broadcast (size, size, ss, ss); + dx = behind the apex
    inside = (dx >= 0) & (np.abs(dy) <= dx * math.tan(math.radians(half_deg)))
    return inside.mean(axis=(2, 3)).astype(np.float32)


@pytest.mark.parametrize("half_deg", [4.0, 6.0, 10.0])
@pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 3.0])
def test_width_tip_wedge_closed_form(half_deg, sigma):
    apex = (80.3, 48.0)  # (x, y) float px, clean wedge apex; the wedge points to +x
    vis = _aa_wedge(apex, half_deg)  # (96, 96) float32
    w_star = g7.tip_width_threshold(0.5, sigma)  # float px
    assert abs(w_star - 2 * math.sqrt(2) * 0.4769362762044699 * sigma) < 1e-9
    hit = g7._width_tip(vis, apex, (1.0, 0.0), 75.0, 20.0, w_star)
    assert hit is not None
    (tx, ty), t, w = hit  # label (x, y), recession t, width at the label (floats)
    want = w_star / (2 * math.tan(math.radians(half_deg)))  # float px, closed form
    assert abs(t - want) <= 0.5, (half_deg, sigma, t, want)
    assert ty == apex[1] and abs(tx - (apex[0] - t)) < 1e-9
    assert w >= w_star and abs(w - w_star) < 0.2, (w, w_star)


def test_width_tip_none_and_cover():
    vis = np.zeros((32, 32), dtype=np.float32)  # (32, 32) nothing visible
    assert g7._width_tip(vis, (16.0, 16.0), (0.0, -1.0), 20.0, 8.0, 1.0) is None
    vis[10:20, 14:19] = 1.0  # 5-px wide bar, top row y = 10
    hit = g7._width_tip(vis, (16.0, 5.0), (0.0, -1.0), 20.0, 8.0, 4.0)
    # bilinear: the width ramps 0 -> 5 between y = 9 and y = 10, reaching 4 at y = 9.8
    assert hit is not None and abs(hit[1] - 4.8) <= 0.06, hit
    cov = np.ones((32, 32), dtype=bool)  # everything covered
    assert g7._width_tip(vis, (16.0, 5.0), (0.0, -1.0), 20.0, 8.0, 4.0, cover=cov) is None
    # a bar thinner than w* never qualifies
    assert g7._width_tip(vis, (16.0, 5.0), (0.0, -1.0), 20.0, 8.0, 6.0) is None


def test_tip_threshold_math():
    from scipy.special import erfinv
    for c in (0.05, 0.3, 0.5, 0.77, 0.95):
        assert abs(g7._erfinv(c) - float(erfinv(c))) < 1e-10, c
    assert g7.tip_width_threshold(0.5, 0.0) == 0.0
    assert abs(g7.tip_width_threshold(0.5, 1.0) - 1.3490) < 1e-3


def _slender(res):
    """Pointer wedge half-angle (float deg) recorded with a render."""
    g = res.params["geometry"]
    return math.degrees(math.atan2(g["chord_half"], g["tip_dist"] + g["chord_dist"]))


# "section.key" -> value pins (dict): a slender pointer, chord_half = 0.1 x
# dial radius, so the wedge half-angle is ~5-6 deg (the user's thin-tip case)
_SLENDER = {"geometry.hub_ratio_range": [0.2, 0.2], "geometry.chord_ratio_range": [0.5, 0.5]}


@pytest.mark.parametrize("pins", [{}, _SLENDER], ids=["default", "slender"])
def test_tip_label_recedes_with_blur(pins):
    radii = (1, 3, 6)  # int px, dial blur radii in increasing order
    for seed in range(8):
        recs = []  # list[float], tip_recession per radius
        for r in radii:
            res = g7.render_image(_cfg(**{**pins, "background.dial_circle_blur": True,
                                          "background.dial_circle_blur_radius": r}), seed)
            d = res.params["derived"]
            assert abs(d["sigma_eff"] - math.hypot(r, 1.0)) < 1e-3
            assert d["w_star"] == res.params["cut_tip"]["w_star"]
            assert d["tip_width"] == res.params["cut_tip"]["tip_width"]
            assert d["tip_width"] >= d["w_star"]
            recs.append(d["tip_recession"])
        # blurrier never moves the label forward along the axis
        assert recs[0] <= recs[1] + 1e-6 and recs[1] <= recs[2] + 1e-6, (seed, recs)
        if pins:  # a slender pointer moves strictly back, by ~ dw* / (2 tan a)
            assert _slender(res) < 7.0, _slender(res)
            assert recs[0] < recs[1] < recs[2], (seed, recs)
            assert recs[2] - recs[0] > 15.0, (seed, recs)


def _bilinear(img, x, y):
    """Bilinear gray of img (H, W) at float arrays x, y (same shape) -> float64 array."""
    h, w = img.shape
    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)  # int arrays, top-left pixel
    fx, fy = x - x0, y - y0  # float arrays, fractional offsets
    f = img.astype(np.float64)  # (H, W) float64 copy of the image
    return ((f[y0, x0] * (1 - fx) + f[y0, x0 + 1] * fx) * (1 - fy)
            + (f[y0 + 1, x0] * (1 - fx) + f[y0 + 1, x0 + 1] * fx) * fy)


def _axis_contrast(res, ahead):
    """Blurred pointer contrast (float) on the final image, judged the way a
    person reads a tip: the darkest point of the perpendicular profile through
    the point `ahead` px beyond the label along the (stretched) axis (within 3 px
    of the axis), against the face beside it (median of the profile at 9..14 px
    off the axis): (bg - min) / (bg - pointer gray)."""
    g = res.params["geometry"]
    ux, uy = _unit_dir(res.angle_deg)  # unit pointer direction, unstretched (floats)
    vx, vy = ux * g["stretch_h"], uy * g["stretch_v"]
    n = math.hypot(vx, vy)
    vx, vy = vx / n, vy / n  # unit stretched axis (floats)
    px, py = res.tip_x + ahead * vx, res.tip_y + ahead * vy  # profile centre (floats)
    s = np.arange(-14.0, 14.0 + 1e-9, 0.25)  # (113,) float64 lateral offsets
    prof = _bilinear(res.image, px - s * vy, py + s * vx)  # (113,) float64 gray profile
    bg = float(np.median(prof[np.abs(s) >= 9]))  # float, face gray beside the pointer
    return (bg - float(prof[np.abs(s) <= 3].min())) / (bg - res.params["pointer"]["gray"])


# contrast tolerances (float) picked from the measured distributions, see
# test_tip_label_visible_after_blur: at the label >= c - _AT_LABEL_TOL, and
# 3 sigma_eff beyond it < c + _AHEAD_TOL
# (40 seeds x radius 2 / 4, measured): slender at-label min 0.42 mean 0.48-0.50,
# 3 sigma ahead max 0.40.  The default (wide, a ~ 22 deg) pointer is cut off
# abruptly, wider than w*, so the label sits on the cut arc and the blurred
# contrast there is the half-max of a sharp end, ~ 0.5 x erf(w / 2 sqrt(2)
# sigma): min 0.35 mean 0.38-0.43, ahead max 0.03; it gets the wider tolerance
_AT_LABEL_TOL = {"slender": 0.12, "default": 0.17}  # case -> float
_AHEAD_TOL = 0.10


@pytest.mark.parametrize("case", ["slender", "default"])
def test_tip_label_visible_after_blur(case):
    c = 0.5  # float, default tip_label.contrast_frac
    at, ahead = [], []  # list[float], contrast at / 3 sigma_eff beyond the label
    pins = _SLENDER if case == "slender" else {}  # dict, geometry pins
    for radius in (2, 4):
        cfg = _cfg(**{**_CLEAN_TIP, **pins, "background.dial_circle_blur": True,
                      "background.dial_circle_blur_radius": radius,
                      "cut_tip.enabled": True, "cut_tip.radius_range": [7, 10]})
        for seed in range(40):
            res = g7.render_image(cfg, seed)
            se = res.params["derived"]["sigma_eff"]  # float px
            a, b = _axis_contrast(res, 0.0), _axis_contrast(res, 3 * se)
            at.append(a)
            ahead.append(b)
            assert a >= c - _AT_LABEL_TOL[case], (radius, seed, a)
            assert b < c + _AHEAD_TOL, (radius, seed, b)
    print("contrast at label: min %.3f mean %.3f max %.3f" % (min(at), np.mean(at), max(at)))
    print("contrast 3 sigma ahead: min %.3f mean %.3f max %.3f"
          % (min(ahead), np.mean(ahead), max(ahead)))
