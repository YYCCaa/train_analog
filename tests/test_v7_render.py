"""Tests for the native-gray v7 renderer (generate_synthetic_v7_mp.render_image).

These call render_image directly (the pure render core); the CLI-level
generator tests live in tests/test_v7_gray_generator.py.
"""

import copy
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
    for banned in ("colorsys", '"RGB"', '"RGBA"', "hsv"):
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


def _extreme_cfg(which):
    """Every leaf at its schema min (which=0) or max (which=1)."""
    schema = v7_config.load_schema()
    cfg = v7_config.default_config()
    for sec, keys in schema.items():
        for key, leaf in keys.items():
            v = leaf["max"] if which else leaf["min"]
            t = leaf["type"]
            if t == "bool":
                cfg[sec][key] = bool(which)
            elif t in ("int_range", "float_range"):
                cfg[sec][key] = [v, v]
            else:
                cfg[sec][key] = v
    v7_config.validate_config(cfg, schema)
    return cfg


@pytest.mark.parametrize("which", [0, 1], ids=["min", "max"])
def test_render_extremes_do_not_crash(which):
    cfg = _extreme_cfg(which)
    for seed in range(5):
        res = g7.render_image(cfg, seed)
        assert res.image.shape == (IMG, IMG)
        assert res.image.dtype == np.uint8
