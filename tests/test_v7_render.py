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
# (canvas grows by 2 * max|displacement|; expected change)
_PRE_CONTRAST_HASHES = {
    0: "e0c38ede1599cf8d16dcaef4347406454bd2163df2c4866ffa86c1d96a1f6ad6",
    1: "a8223fad801dc72b92bf99fdd84a912dc18a72b33f9658b325ac4b5c4a17048b",
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
