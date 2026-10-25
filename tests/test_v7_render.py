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
    # same range width as the default [65, 184] (120 levels), shifted to the
    # extremes: the tri stream makes the same draws (randint consumes a
    # width-dependent number of random bits), so the probes and `base` place
    # the same triangles
    w = base["tri_artifacts"]["gray_range"][1] - base["tri_artifacts"]["gray_range"][0]  # int
    probe_lo = _cfg(**flat, **{"tri_artifacts.gray_range": [0, w]})
    probe_hi = _cfg(**flat, **{"tri_artifacts.gray_range": [255 - w, 255]})
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


# "section.key" -> value pins (dict): exposure jitter off.  The jitter factors
# are capped per image from the canvas the pointer is on (no saturation), so a
# brighter / darker pointer can lower a cap and change the whole face; with
# every range [1, 1] the factor is 1 whatever the cap (the cap is always > 1)
_NO_JITTER = {"jitter.brightness_range": [1.0, 1.0], "jitter.contrast_range": [1.0, 1.0],
              "jitter.sharpness_range": [1.0, 1.0]}


def test_pointer_gray_change_moves_only_pointer():
    flat = _NO_JITTER
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
    # headroom H: c = 0 is randint(I_lo, I_hi) bit for bit, I = [lo, hi] ∩ [ceil H, floor(255 - H)]
    # (lo, hi, H) -> (I_lo, I_hi, headroom_ok): I empty falls back to [lo, hi]
    cases = {(0, 255, 50.56): (51, 204, True), (100, 230, 50.56): (100, 204, True),
             (0, 255, 0.0): (0, 255, True), (40, 41, 60.0): (40, 41, False),
             (250, 255, 50.0): (250, 255, False)}  # dict[tuple, tuple]
    for (lo, hi, h), (i_lo, i_hi, ok) in cases.items():
        for s in range(300):
            info = {}  # dict, fallback / headroom_ok flags out
            want = random.Random(s).randint(i_lo, i_hi)  # int, randint over I
            got = g7._pick_pointer_gray(random.Random(s), lo, hi, 160, 0, 0.0, info, h)
            assert got == want, (lo, hi, h, s)
            assert info == {"fallback": False, "headroom_ok": ok}, (lo, hi, h)
    # c > 0 on I: the forbidden interval is removed from I, the fallback endpoint is I's
    vals = {g7._pick_pointer_gray(rng, 0, 255, 160, 60, 0.0, None, 50.56) for _ in range(3000)}
    assert vals == set(range(51, 101))  # [51, 100]; the upper part [220, 204] is empty
    assert g7._pick_pointer_gray(random.Random(0), 0, 255, 150, 120, 0.0, None, 50.56) == 51


# ── C: triangle artifacts are always placed ─────────────────────────────────

def test_triangles_always_placed():
    cfg = _cfg(**{"tri_artifacts.count_range": [6, 6]})
    cxs, cys, rs = [], [], []  # list[float]: centre offsets from the B centre, / B radius
    for seed in range(500):
        p = g7.render_image(cfg, seed).params
        t = p["tri_artifacts"]
        assert t["n"] == 6 and len(t["items"]) == 6
        bx = p["derived"]["canvas_offset_x"] + p["geometry"]["canvas_size"] // 2 \
            - p["geometry"]["displacement_x"]  # int px, B centre x (image)
        by = p["derived"]["canvas_offset_y"] + p["geometry"]["canvas_size"] // 2 \
            - p["geometry"]["displacement_y"]  # int px, B centre y (image)
        orx, ory = p["derived"]["outer_rx"], p["derived"]["outer_ry"]  # float px, B radii
        for it in t["items"]:
            assert it["placed"] is True and it["pixels"] >= 1, (seed, it)
            dx, dy = (it["center_x"] - bx) / orx, (it["center_y"] - by) / ory
            assert dx * dx + dy * dy <= 1.0 + 1e-9, (seed, it)  # centre inside B
            cxs.append(dx)
            cys.append(dy)
            rs.append(math.hypot(dx, dy))
    cxs, cys, rs = np.array(cxs), np.array(cys), np.array(rs)  # (3000,) float64 each
    # quadrant balance (loose): each quadrant of B gets 15-35% of the centres
    quads = [((cxs >= 0) & (cys >= 0)).mean(), ((cxs < 0) & (cys >= 0)).mean(),
             ((cxs >= 0) & (cys < 0)).mean(), ((cxs < 0) & (cys < 0)).mean()]
    assert all(0.15 <= q <= 0.35 for q in quads), quads
    # radial coverage: the outer annulus r > 0.7 holds 51% of B's area, the
    # pointer only blocks part of the inner disc -> clearly more than a third
    assert (rs > 0.7).mean() > 0.4, float((rs > 0.7).mean())


def test_triangles_paint_off_pointer_inside_boundary():
    # footprint = pixels that differ between tri gray 0 and 255.  Both ranges
    # have width 1, so the tri stream makes the same draws in both (a range of
    # another width consumes a different number of random bits per randint)
    base = {"tri_artifacts.count_range": [6, 6], **_NO_JITTER,
            "background.dial_circle_blur": False}
    r = _radius_grid()
    for seed in range(10):
        lo_cfg = _cfg(**base, **{"tri_artifacts.gray_range": [0, 0]})  # dict
        lo = g7.render_image(lo_cfg, seed)
        hi = g7.render_image(_cfg(**base, **{"tri_artifacts.gray_range": [255, 255]}), seed)
        foot = lo.image != hi.image  # (256, 256) bool, triangle pixels
        assert foot.sum() >= 6, seed
        assert not (foot & (r > IMG / 2)).any()  # inside B (well inside the image)
        ptr = _pointer_footprint(lo_cfg, seed)  # (256, 256) bool, same triangles
        assert not (foot & _erode(ptr, 2)).any(), seed  # never on the pointer body (2 px: bicubic stretch spread)


# ── B: exposure jitter never saturates (capped factors) ────────────────────

# "section.key" -> value pins (dict): the brief's stressed jitter config
_JIT_STRESS = {"jitter.brightness_range": [1.5, 2.0], "jitter.contrast_range": [1.3, 1.5],
               "jitter.sharpness_range": [0.0, 3.0]}


@pytest.mark.parametrize("pins", [{}, _JIT_STRESS], ids=["default", "stressed"])
def test_jitter_never_saturates(monkeypatch, pins):
    from PIL import ImageFilter, ImageStat
    orig = g7._jitter_enhance
    # one record per enhance: (kind str, input float64 (C, C), region (C, C) bool,
    # factor float, output uint8 (C, C)); reset per render
    calls = []

    def wrapped(img, kind, rng, lo, hi, region):
        out, f, f_max, capped = orig(img, kind, rng, lo, hi, region)
        calls.append((kind, img, region, f, np.asarray(out)))
        assert f <= f_max + 1e-12 and (capped or lo <= f <= hi)
        return out, f, f_max, capped
    monkeypatch.setattr(g7, "_jitter_enhance", wrapped)
    cfg = _cfg(**pins)
    n_capped = 0  # int, renders with at least one capped factor
    for seed in range(300):
        calls.clear()
        p = g7.render_image(cfg, seed).params["jitter"]
        assert [c[0] for c in calls] == list(g7.JITTER_KINDS)
        n_capped += any(p["capped"].values())
        for kind, img, region, f, out in calls:
            # PIL's exact degenerate, recomputed independently of the module
            if kind == "brightness":
                d = np.zeros(img.size[::-1])
            elif kind == "contrast":
                d = np.full(img.size[::-1], int(ImageStat.Stat(img).mean[0] + 0.5), float)
            else:
                d = np.asarray(img.filter(ImageFilter.SMOOTH), dtype=np.float64)
            a = np.asarray(img, dtype=np.float64)  # (C, C) float64 input
            v = (d + f * (a - d))[region]  # (N,) float64, unclipped enhance
            assert v.max() <= 255 + 1 and v.min() >= 0 - 1, (seed, kind, f, v.min(), v.max())
            # the real PIL output equals the unclipped value truncated: no saturation
            assert np.array_equal(out[region].astype(np.int64),
                                  np.clip(np.trunc(v), 0, 255).astype(np.int64)) or                 np.abs(out[region] - np.clip(np.trunc(v), 0, 255)).max() <= 1, (seed, kind)
    if pins:
        assert n_capped == 300, "the stressed config must cap every image"


def test_jitter_stream_draw_count_unchanged(monkeypatch):
    # the jitter stream state after step 9 equals 3 plain uniform draws (old code)
    got = {}  # dict, the jitter random.Random of the last render
    orig = g7.section_rngs

    def capture(seed):
        r = orig(seed)
        got["jitter"] = r["jitter"][0]
        return r
    monkeypatch.setattr(g7, "section_rngs", capture)
    for pins in ({}, _JIT_STRESS):
        cfg = _cfg(**pins)
        for seed in range(20):
            p = g7.render_image(cfg, seed).params["jitter"]
            old = orig(seed)["jitter"][0]  # random.Random, a fresh jitter stream
            for kind in g7.JITTER_KINDS:
                f_old = old.uniform(*cfg["jitter"][kind + "_range"])  # float, the old draw
                if not p["capped"][kind]:
                    assert p[kind] == pytest.approx(f_old, abs=1e-4)
            assert got["jitter"].getstate() == old.getstate()


def test_capped_uniform_unit():
    import random
    for lo, hi, f_max, want_lo, want_hi, capped in (
            (0.95, 1.35, 9.0, 0.95, 1.35, False), (0.95, 1.35, 1.1, 0.95, 1.1, True),
            (1.5, 2.0, 1.2, 1.2 * 1.5 / 2.0, 1.2, True), (0.0, 3.0, math.inf, 0.0, 3.0, False)):
        for s in range(200):
            r1, r2 = random.Random(s), random.Random(s)
            f, c = g7.capped_uniform(r1, lo, hi, f_max)
            assert c is capped and want_lo <= f <= want_hi
            u = r2.random()  # float, the single draw it consumed
            assert f == pytest.approx(want_lo + (want_hi - want_lo) * u)
            assert r1.random() == r2.random()


# ── A: pointer body / edge never clip (headroom gray + amplitude scale) ────

# "section.key" -> value pins (dict): the brief's stressed pointer config
_PTR_STRESS = {"pointer.gray_range": [0, 255], "pointer.gradient_range": [-100.0, 100.0],
               "pointer.noise_amplitude": 64, "pointer.edge_noise_amplitude": 32}


def _clip_counter(monkeypatch):
    """Wrap np.clip so calls from the two pointer functions count changed values.

    Returns a dict[str, int] {"_pointer_variation": n, "_draw_pointer_edge": n}
    of elements the clip changed, updated live while renders run.
    """
    import sys
    counts = {"_pointer_variation": 0, "_draw_pointer_edge": 0}  # dict[str, int]
    orig = np.clip  # the real numpy clip

    def clip(a, lo, hi, *args, **kw):
        out = orig(a, lo, hi, *args, **kw)
        name = sys._getframe(1).f_code.co_name  # str, the calling function
        if name in counts:
            counts[name] += int(np.count_nonzero(np.asarray(out) != np.asarray(a)))
        return out
    monkeypatch.setattr(np, "clip", clip)
    return counts


@pytest.mark.parametrize("pins", [{}, _PTR_STRESS], ids=["default", "stressed"])
def test_pointer_clip_never_changes_a_value(monkeypatch, pins):
    counts = _clip_counter(monkeypatch)  # dict[str, int], live
    cfg = _cfg(**pins)
    scaled = 0  # int, renders whose amplitude scale k < 1
    for seed in range(300):
        p = g7.render_image(cfg, seed).params["pointer"]
        assert 0.0 <= p["amp_scale"] <= 1.0
        scaled += p["amp_scale"] < 1.0
    assert counts == {"_pointer_variation": 0, "_draw_pointer_edge": 0}, counts
    if pins:
        assert scaled > 0, "the stressed config never exercised the amplitude scale"


def test_pointer_gray_and_body_have_no_point_mass(monkeypatch):
    # body pixel values after gradient + noise, pooled over renders (list of (N,) uint8)
    body = []
    orig = g7._pointer_variation

    def wrapped(img, mask, *args, **kw):
        out = orig(img, mask, *args, **kw)
        body.append(np.asarray(out)[np.asarray(mask) > 0])
        return out
    monkeypatch.setattr(g7, "_pointer_variation", wrapped)
    cfg = v7_config.default_config()
    grays = []  # list[int], drawn body grays
    for seed in range(300):
        p = g7.render_image(cfg, seed).params["pointer"]
        grays.append(p["gray"])
        assert p["headroom_ok"] is True and p["headroom"] == pytest.approx(51.0, abs=1.0)
    lo_g, hi_g = min(grays), max(grays)  # int, the gray draw stays inside I
    assert lo_g >= 50 and hi_g <= 205, (lo_g, hi_g)
    px = np.concatenate(body)  # (M,) uint8, every body pixel
    freq = np.bincount(px, minlength=256) / px.size  # (256,) float64, value frequencies
    assert freq.max() <= 3.0 / 256, (int(freq.argmax()), float(freq.max()) * 256)
    # the gray draw itself: one randint over I (c = 0), so uniform; check on many draws
    import random
    rng = random.Random(5)  # random.Random, a stand-in pointer stream
    vals = np.array([g7._pick_pointer_gray(rng, 0, 255, 160, 0, 0.0, None, 50.56)
                     for _ in range(20000)])  # (20000,) int
    counts = np.bincount(vals, minlength=256)  # (256,) int
    n_support = 204 - 51 + 1  # int, |I|
    assert counts[:51].sum() == 0 and counts[205:].sum() == 0
    assert counts.max() <= 3 * 20000 / n_support


def test_edge_noise_scaled_near_extremes():
    # unit: band pixels whose noiseless value sits at 0 / 255 get no noise
    import random
    m = np.zeros((32, 32), np.uint8)  # (32, 32) uint8 mask
    m[8:24, 8:24] = 255
    from PIL import Image
    for edge_gray, bgv in ((0, 0), (255, 255), (3, 250)):
        img = Image.fromarray(np.full((32, 32), 128, np.uint8))  # "L" canvas
        bg = np.full((32, 32), bgv, np.uint8)  # (32, 32) uint8 pre-pointer canvas
        out, _ = g7._draw_pointer_edge(img, Image.fromarray(m), edge_gray, bg, 3, 32,
                                       np.random.default_rng(0))
        a = np.asarray(out)
        band = (m > 0) & (a != 128)  # (32, 32) bool, painted band pixels
        assert band.any()
        if edge_gray == bgv:
            assert (a[band] == bgv).all(), np.unique(a[band])  # k_e = 0: exact value
    # draw count unchanged: same stream state after, with and without noise scaling
    r1, r2 = np.random.default_rng(1), np.random.default_rng(1)
    img = Image.fromarray(np.full((32, 32), 128, np.uint8))
    g7._draw_pointer_edge(img, Image.fromarray(m), 0, np.zeros((32, 32), np.uint8), 3, 32, r1)
    g7._draw_pointer_edge(img, Image.fromarray(m), 128, np.full((32, 32), 128, np.uint8), 3, 32, r2)
    assert r1.integers(0, 1 << 30) == r2.integers(0, 1 << 30)


# sha256 hex (str) of default-config renders with digits hidden (so the
# digit-jitter change cannot touch them), seeds 0-2, recorded BEFORE any code
# change for min_bg_contrast; re-pinned when displacement moved from the
# canvas pad to the content inside the centred circle canvas, and again when
# displacement became a shift of everything inside a fixed circular boundary
# (canvas grows by 2 * max|displacement|; expected change), and again for
# the physical multiplicative shadow model (only seed 1 has a shadow; seeds
# 0 and 2 are unchanged, so no other stream moved). Only the image is hashed,
# so moving the tip label to the visible (cut) tip left these unchanged.
# Since the edge clamp they are the pointer.edge_max_frac = 1 renders.
# Re-pinned for the no-clamping change (headroom pointer gray, amplitude
# scale, capped exposure jitter, triangle centres drawn from the valid region:
# the gray draw, the jitter factors and the tri stream all moved; expected).
# Re-pinned for the pointer triangle's own hard raster (per-line round(width),
# centred, replacing Pillow's polygon in the pointer, its mask and its shadow;
# expected).  Previous values: 0 31d62628..., 1 e8ea7675..., 2 4ecdc0d1...
_PRE_CONTRAST_HASHES = {
    0: "296bcacea0929c338125d89bc0c198064d771054954f0786aca56d525a4b0709",
    1: "03b29b4969a846e369f1b01069184b7c7f927300c1976298432784ef810cd0ed",
    2: "2b07c73df1ccd6ab3dcadc64a07e6dead8df1971e46758bbaf9bc42608f4b725",
}


# sha256 hex (str) of the same renders with the DEFAULT pointer.edge_max_frac
# 0.25 (re-pinned when the edge clamp arrived: seeds 0 and 2 change near the
# apex; seed 1's apex sits under the rim band, so it is unchanged); re-pinned
# for the no-clamping change and the pointer raster, like _PRE_CONTRAST_HASHES
# (previous values: 0 64d4f5a3..., 1 e8ea7675..., 2 ab349759...)
_DEFAULT_HASHES = {
    0: "0842334bc121d696838fda4fe8c50245a999645ce6fb98fa051a04c373c67cc1",
    1: "03b29b4969a846e369f1b01069184b7c7f927300c1976298432784ef810cd0ed",
    2: "6325e541743e2848c4948f6dc70f9115ad00c6f5d904a3d7ce4d79587fcf98c3",
}


def test_default_render_unchanged_by_contrast_key():
    import hashlib
    cfg = _cfg(**{"digits.show_prob": 0.0})
    assert cfg["pointer"]["min_bg_contrast"] == 0
    for seed, want in _DEFAULT_HASHES.items():
        got = hashlib.sha256(g7.render_image(cfg, seed).image.tobytes()).hexdigest()
        assert got == want, f"seed {seed}: default render changed"


def test_edge_max_frac_one_reproduces_old_pixels():
    # edge_max_frac = 1 skips the clamp: the pre-clamp pins hold bit for bit
    import hashlib
    cfg = _cfg(**{"digits.show_prob": 0.0, "pointer.edge_max_frac": 1.0})
    for seed, want in _PRE_CONTRAST_HASHES.items():
        got = hashlib.sha256(g7.render_image(cfg, seed).image.tobytes()).hexdigest()
        assert got == want, f"seed {seed}: frac-1 render changed"


def _pointer_footprint(cfg, seed):
    """Pixels the pointer body/edge/hole reach, (256, 256) bool (c forced to 0)."""
    lo, hi = copy.deepcopy(cfg), copy.deepcopy(cfg)
    for c, g in ((lo, 0), (hi, 255)):
        c["pointer"]["gray_range"] = [g, g]
        c["pointer"]["min_bg_contrast"] = 0
    return g7.render_image(lo, seed).image != g7.render_image(hi, seed).image


def test_min_bg_contrast_isolated():
    # shadows off: with a shadow the forbidden interval also covers the shadowed face
    # (jitter off: its per-image caps see the pointer gray, see _NO_JITTER)
    flat = {**_NO_JITTER, "shadows.prob": 0.0}
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
# so the canvas half is radius + margin + max|displacement|.  Triangles off:
# their centres are drawn from the part of the content inside B (every
# triangle is placed), so they follow where B sits, not the content.
_DISP_PINS = {"geometry.scale_range": [1.0, 1.0], "outer.scale_range": [1.2, 1.2],
              "geometry.stretch_h_range": [1.0, 1.0], "geometry.stretch_v_range": [1.0, 1.0],
              "shadows.prob": 1.0, "tri_artifacts.enabled": False}
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


def _axis(res):
    """Label axis recorded with a render (the triangle's exact height, base-chord
    midpoint -> apex, stretched, image px) -> ((ax, ay) apex floats, (ux, uy)
    unit direction floats)."""
    d = res.params["derived"]
    ax, ay = d["axis_apex_x"], d["axis_apex_y"]  # floats, exact apex (stretched)
    vx, vy = ax - d["axis_base_mid_x"], ay - d["axis_base_mid_y"]  # floats, base mid -> apex
    n = math.hypot(vx, vy)
    return (ax, ay), (vx / n, vy / n)


def test_tip_label_is_visible_tip():
    for seed in range(6):
        a, b = _cut_pair(seed, **_NO_STRETCH)
        assert a.angle_deg == b.angle_deg
        assert _ideal(a) == _ideal(b)  # the cut does not move the triangle
        for res in (a, b):
            (ix, iy), (ux, uy) = _axis(res)  # exact apex, unit axis (floats)
            dx, dy = res.tip_x - ix, res.tip_y - iy  # float px, label - apex
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
        for res in (a, b):
            (ix, iy), (ux, uy) = _axis(res)  # exact apex, unit stretched axis (floats)
            dx, dy = res.tip_x - ix, res.tip_y - iy  # float px, label - apex
            # on the stretched axis, behind the apex
            assert abs(-dx * uy + dy * ux) < 1e-3, seed
            assert dx * ux + dy * uy <= 1e-3, seed
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

# "section.key" -> value pins (dict): a slender pointer, chord_half = 0.1 x
# dial radius, so the wedge half-angle is ~5-6 deg (the user's thin-tip case)
_SLENDER = {"geometry.hub_ratio_range": [0.2, 0.2], "geometry.chord_ratio_range": [0.5, 0.5]}


# ── tip label = first base-parallel line with > n pointer pixels ──────────

def _tri_mask(apex, tan_a, height, size=96):
    """Bool mask (size, size) of an isosceles triangle pointing up (-y): apex
    (x, y) float px, half-angle tan tan_a (float), height (float px), base at
    y = apex_y + height.  Pixel (x, y) is in iff 0 <= y - ay <= height and
    |x - ax| <= (y - ay) tan_a (pixel centres at integer coords)."""
    ys, xs = np.mgrid[:size, :size]  # (size, size) int pixel grids
    r = ys - apex[1]  # (size, size) float, depth below the apex
    return (r >= 0) & (r <= height) & (np.abs(xs - apex[0]) <= r * tan_a + 1e-12)


def _tri_corners(apex, tan_a, height):
    """(B_l, B_r) base corners (x, y) floats of the _tri_mask triangle."""
    w = height * tan_a  # float px, half base
    return (apex[0] - w, apex[1] + height), (apex[0] + w, apex[1] + height)


@pytest.mark.parametrize("tan_a", [0.21, 0.3, 0.37, 0.45])
def test_tip_count_label_closed_form(tan_a):
    # axis-aligned triangle (height along +y, base parallel to x, so the band
    # around L(t) is the pixel row r with t in [r - 0.5, r + 0.5)).
    # Apex on a pixel centre (32, 10): row r holds 2 floor(r tan a) + 1 pixels
    # (row 0 = the apex pixel), so n = 0 puts the label at t = 0, the first row.
    apex = (32.0, 10.0)  # (x, y) float px
    pix = _tri_mask(apex, tan_a, 60.0)  # (96, 96) bool
    (x, y), t, cnt, fb = g7.tip_count_label(pix, apex, *_tri_corners(apex, tan_a, 60.0), 0)
    assert (x, y, t, cnt, fb) == (32.0, 10.0, 0.0, 1, False)
    # Apex on a pixel-column boundary (32.5, 10): row r holds
    # 2 floor(r tan a + 1/2) pixels (even), so count > n (n even) first holds on
    # row r* = ceil((n + 1) / (2 tan a)) and t* = r* - 1/2 exactly; the closed
    # form t = (n + 1) / (2 tan a) is within 1/2 of it (the row discretisation).
    apex = (32.5, 10.0)  # (x, y) float px
    pix = _tri_mask(apex, tan_a, 60.0)
    corners = _tri_corners(apex, tan_a, 60.0)  # (B_l, B_r)
    first_row = math.ceil(0.5 / tan_a)  # int, first non-empty row (2 px)
    (x, y), t, cnt, fb = g7.tip_count_label(pix, apex, *corners, 0)
    assert x == 32.5 and abs(t - (first_row - 0.5)) < 1e-9 and cnt == 2 and not fb, (t, cnt)
    for n in (2, 4, 8):
        closed = (n + 1) / (2 * tan_a)  # float px, continuous cross-section W = n + 1
        want = math.ceil(closed) - 0.5  # float px, exact discrete expectation
        (x, y), t, cnt, fb = g7.tip_count_label(pix, apex, *corners, n)
        assert abs(t - want) < 1e-9 and abs(t - closed) <= 0.5, (n, t, want, closed)
        assert x == 32.5 and abs(y - (10.0 + t)) < 1e-12 and cnt > n and not fb
        assert cnt == 2 * math.floor(math.ceil(closed) * tan_a + 0.5)


def _brute_count(pix, apex, base_left, base_right, t):
    """count(t) of tip_count_label by direct evaluation (int): pointer pixels whose
    unit square the base-parallel line L(t) passes through, judged from the
    square's 4 corners (signed distances to L(t) along the unit base normal:
    not all on one side; the half-open tie rule -r < d - t k <= r of the
    generator only matters on exact corner touches, which random floats avoid)."""
    mx, my = (base_left[0] + base_right[0]) / 2, (base_left[1] + base_right[1]) / 2
    hl = math.hypot(mx - apex[0], my - apex[1])
    hx, hy = (mx - apex[0]) / hl, (my - apex[1]) / hl  # unit height (floats)
    bx, by = base_right[0] - base_left[0], base_right[1] - base_left[1]
    bl = math.hypot(bx, by)
    nx, ny = -by / bl, bx / bl  # unit base normal (floats)
    if hx * nx + hy * ny < 0:
        nx, ny = -nx, -ny
    k = hx * nx + hy * ny  # float
    ys, xs = np.nonzero(pix)
    d = (xs - apex[0]) * nx + (ys - apex[1]) * ny - t * k  # (N,) float64, centre distance
    # (4, N) float64 corner distances: centre +- 0.5 px in x and y
    corners = np.stack([d + sx * 0.5 * nx + sy * 0.5 * ny for sx in (-1, 1) for sy in (-1, 1)])
    return int(((corners.min(axis=0) < 0) & (corners.max(axis=0) > 0)).sum())


def test_tip_count_no_dropout_on_tilted_sliver():
    # user sample inspect_tip_location/v7_seed593719257: the pointer points
    # right, its base-parallel lines tilt ~9 deg off vertical, and the tip is a
    # continuous 1-px sliver; a fixed 1-px band (|d| <= 0.5) skipped it at some
    # t (count 0 inside the pointer).  Counting the squares the line passes
    # through must see >= 1 pixel at every t along the sliver.
    for tilt_deg in (5.0, 9.0, 20.0, 35.0, 45.0):
        th = math.radians(tilt_deg)  # float rad, base tilt off vertical
        ax, ay = 10.0, 30.2  # float px, apex
        # height points +x; base direction tilted th off vertical
        bl_, br_ = (60.0 - 20 * math.sin(th), 30.2 - 20 * math.cos(th)), \
                   (60.0 + 20 * math.sin(th), 30.2 + 20 * math.cos(th))
        # (64, 80) bool, a 1-px staircase sliver climbing 0.15 px per column (a
        # row step adds |nb_x| + |nb_y| > 1 to the distance: the old band's gap)
        pix = np.zeros((64, 80), dtype=bool)
        i = np.arange(50)  # (50,) int, column index along the sliver
        pix[30 - np.floor(i * 0.15).astype(int), 10 + i] = True
        ts = np.arange(0.0, 40.0, g7.TIP_T_STEP)  # (T,) float px along the height
        counts = [_brute_count(pix, (ax, ay), bl_, br_, t) for t in ts]
        assert min(counts) >= 1, (tilt_deg, counts.index(min(counts)))
        # and the generator agrees: with n = 0 the label is at the sliver's start
        (lx, ly), t, cnt, fb = g7.tip_count_label(pix, (ax, ay), bl_, br_, 0)
        assert not fb and t <= 0.5 and cnt >= 1, (tilt_deg, t, cnt)


def test_tip_count_label_matches_brute_force_any_angle():
    rng = np.random.default_rng(0)  # np Generator for random triangles
    ys, xs = np.mgrid[:128, :128]  # (128, 128) int grids
    for _ in range(25):
        ang = rng.uniform(0, 2 * math.pi)  # float rad, height direction
        ax, ay = rng.uniform(50, 78, 2)  # float px apex
        hl, half = rng.uniform(25, 45), rng.uniform(4, 14)  # float px height, half base
        hx, hy = math.cos(ang), math.sin(ang)
        # slightly skewed base (a stretched triangle): base direction off the normal
        sk = rng.uniform(-0.1, 0.1)  # float rad
        bx, by = -math.sin(ang + sk), math.cos(ang + sk)
        mx, my = ax + hl * hx, ay + hl * hy
        bl_, br_ = (mx - half * bx, my - half * by), (mx + half * bx, my + half * by)
        # pixel mask: inside the triangle (barycentric half-planes) or a hub disc
        def side(p, q):
            return (q[0] - p[0]) * (ys - p[1]) - (q[1] - p[1]) * (xs - p[0])
        s1, s2, s3 = side((ax, ay), bl_), side(bl_, br_), side(br_, (ax, ay))
        tri = ((s1 >= 0) & (s2 >= 0) & (s3 >= 0)) | ((s1 <= 0) & (s2 <= 0) & (s3 <= 0))
        pix = tri | (np.hypot(xs - mx, ys - my) <= half * 1.2)  # (128, 128) bool
        for n in (0, 2, 5):
            (lx, ly), t, cnt, fb = g7.tip_count_label(pix, (ax, ay), bl_, br_, n)
            assert not fb
            assert cnt == _brute_count(pix, (ax, ay), bl_, br_, t) > n, (n, t, cnt)
            if t > 0:
                assert _brute_count(pix, (ax, ay), bl_, br_, t - g7.TIP_T_STEP) <= n
            # on the height A -> M
            assert abs((lx - ax) * hy - (ly - ay) * hx) < 1e-9


def test_tip_count_label_fallback_and_monotonic_unit():
    apex = (32.5, 10.0)  # (x, y) float px
    pix = _tri_mask(apex, 0.3, 20.0)  # (96, 96) bool, base width 12 px
    corners = _tri_corners(apex, 0.3, 20.0)  # (B_l, B_r)
    ts = [g7.tip_count_label(pix, apex, *corners, n)[1] for n in range(40)]  # list[float]
    assert all(b >= a for a, b in zip(ts, ts[1:])), ts
    (x, y), t, cnt, fb = g7.tip_count_label(pix, apex, *corners, 30)
    assert fb and (x, y) == (32.5, 30.0) and t == 20.0  # no line exceeds 30 px: M'
    empty = np.zeros((96, 96), dtype=bool)  # (96, 96) bool, no pointer at all
    assert g7.tip_count_label(empty, apex, *corners, 0)[3] is True


# n values (list[int]) for the render-level monotonicity test, increasing
_N_LADDER = [0, 1, 2, 3, 5, 8, 12, 20, 30]


@pytest.mark.parametrize("pins", [{}, _SLENDER], ids=["default", "slender"])
def test_tip_label_monotonic_in_min_pixels(pins):
    for seed in range(12):
        ts = []  # list[float] px, tip_t per n
        for n in _N_LADDER:
            res = g7.render_image(_cfg(**{**pins, "tip_label.min_pixels": n}), seed)
            d = res.params["derived"]  # dict
            ts.append(d["tip_t"])
            if not d["tip_fallback"]:
                assert d["tip_count"] > n, (seed, n, d["tip_count"])
        assert all(b >= a - 1e-9 for a, b in zip(ts, ts[1:])), (seed, ts)
        assert ts[-1] > ts[0], (seed, ts)


def _dark_on_line(res, back, dark=100, reach=12.0):
    """Dark pixels a person counts on a base-parallel line of the final image (int).

    The line runs through the label moved `back` px (float) along the height
    toward the apex (A'), parallel to the stretched base
    (RenderResult.tip_axis); a pixel counts when its gray < dark (int), the
    line passes through its unit square (its 4 corners are not all on one side
    of the line) and it lies within `reach` px of the line point along the base
    direction (keeps the far face / boundary out).
    """
    (ax, ay), (mx, my), (bx, by) = res.tip_axis  # floats, image px
    hl = math.hypot(mx - ax, my - ay)
    hx, hy = (mx - ax) / hl, (my - ay) / hl  # unit height (floats)
    nx, ny = -by, bx  # unit base normal (floats)
    if hx * nx + hy * ny < 0:
        nx, ny = -nx, -ny
    px, py = res.tip_x - back * hx, res.tip_y - back * hy  # float px, point on the line
    ys, xs = np.nonzero(res.image < dark)  # (N,) int dark pixel coords
    d = (xs - px) * nx + (ys - py) * ny  # (N,) float64 distance from the line
    lat = (xs - px) * bx + (ys - py) * by  # (N,) float64 position along the line
    # (4, N) float64 corner distances: centre +- 0.5 px in x and y
    corners = np.stack([d + sx * 0.5 * nx + sy * 0.5 * ny for sx in (-1, 1) for sy in (-1, 1)])
    crossed = (corners.min(axis=0) < 0) & (corners.max(axis=0) > 0)  # (N,) bool
    return int((crossed & (np.abs(lat) <= reach)).sum())


# "section.key" -> value pins (dict): _CLEAN_TIP (flat face 200, black pointer,
# no noise / blur / effects / shadows) with a light rim / outer fill, so the
# only dark pixels near the tip are pointer pixels.  Default edge band [1, 3]
# and edge_max_frac 0.25: near the label the band is clamped away, so no
# half-blended ring (gray 100 on this face: a pointer pixel by the rule that a
# "< 100" reader never calls dark) sits on the counted line
_COUNT_CLEAN = {**_CLEAN_TIP, "palette.outer_gray_range": [220, 220],
                "palette.rim_gray_range": [220, 220]}


@pytest.mark.parametrize("cut", [False, True], ids=["cut_off", "cut_on"])
def test_tip_label_counts_dark_pixels(cut):
    # a person draws the base-parallel line through the label on the final
    # image and counts the dark pixels on it: more than n; one pixel closer to
    # the apex: at most n
    n = v7_config.default_config()["tip_label"]["min_pixels"]  # int, default n
    cfg = _cfg(**{**_COUNT_CLEAN, "cut_tip.enabled": cut})
    at, closer = [], []  # list[int], dark counts on the label line / 1 px closer
    for seed in range(60):
        res = g7.render_image(cfg, seed)
        assert not res.params["derived"]["tip_fallback"], seed
        at.append(_dark_on_line(res, 0.0))
        closer.append(_dark_on_line(res, 1.0))
    at, closer = np.array(at), np.array(closer)  # (60,) int64 each
    print("%s n=%d: count at label %s; 1 px closer %s; closer <= n for %.0f%%"
          % ("cut on" if cut else "cut off", n, np.bincount(at).tolist(),
             np.bincount(closer).tolist(), 100 * (closer <= n).mean()))
    assert (at > n).all(), at
    assert (closer <= n).mean() >= 0.9, closer


# ── edge clamp: pointer.edge_max_frac ──────────────────────────────────────

def _edge_case(frac, thickness=3):
    """Draw a slender pointer's edge band on a synthetic canvas.

    Triangle apex (20.3, 64.6) pointing -x, base x = 100.4, half base 6
    (W grows 12 / 80 = 0.15 px per px of height), hub disc r 9 at the base
    midpoint; body gray 50, background 200, edge gray 0, no noise.
    Returns (arr (128, 128) uint8 after the band, dist (128, 128) float32 ring
    index, mask (128, 128) bool, (A, B_l, B_r) exact float vertices).
    The triangle is drawn like draw_pointer draws it: raster_pointer_triangle.
    """
    from PIL import Image, ImageDraw
    apex, bl, br = (20.3, 64.6), (100.4, 58.6), (100.4, 70.6)  # (x, y) float px each
    in_tri = g7.raster_pointer_triangle(apex, bl, br, 128)  # (128, 128) bool, triangle raster
    mask = Image.new("L", (128, 128), 0)  # "L" pointer mask
    ImageDraw.Draw(mask).ellipse([100.4 - 9, 64.6 - 9, 100.4 + 9, 64.6 + 9], fill=255)
    m = np.array(mask)  # (128, 128) uint8
    m[in_tri] = 255
    mask = Image.fromarray(m)
    img = Image.fromarray(np.where(m > 0, 50, 200).astype(np.uint8))
    bg = np.full((128, 128), 200, dtype=np.uint8)  # (128, 128) uint8 pre-pointer canvas
    out, painted = g7._draw_pointer_edge(img, mask, 0, bg, thickness, 0, np.random.default_rng(0),
                                         (apex, bl, br, in_tri), frac)
    return np.array(out), g7._erosion_dist(m, thickness), m > 0, (apex, bl, br)


def test_edge_clamp_unit_slender_apex():
    t, frac = 3, 0.25  # int band, float clamp
    arr, dist, mask, (a, bl, br) = _edge_case(frac, t)
    mx, my = (bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2
    hl = math.hypot(mx - a[0], my - a[1])  # float px, |M - A|
    ys, xs = np.nonzero(mask & (arr != 50))  # (N,) int, pixels the band repainted
    h = ((xs - a[0]) * (mx - a[0]) + (ys - a[1]) * (my - a[1])) / hl  # (N,) float64
    w = math.hypot(br[0] - bl[0], br[1] - bl[1]) * h / hl  # (N,) float64, W(h)
    near = h < 3 * 4  # (N,) bool, within the first 12 px of the apex
    k = dist[ys, xs]  # (N,) float32 ring index
    # floor with the + 1e-9 of edge_thickness_map: with the exact vertices W
    # can land on 4.0 up to float error (cap 1, not 0)
    assert (k[near] < np.floor(frac * w[near] + 1e-9)).all(), (k[near], w[near])
    # and along the whole triangle: a repainted ring index is always < floor(frac W)
    tri = (h >= 0) & (h <= hl) & (xs < 92)  # (N,) bool, triangle part, clear of the hub disc
    assert tri.sum() > 20, tri.sum()
    assert (k[tri] < np.floor(frac * w[tri] + 1e-9)).all()
    # within 12 px W < 1.8, so floor(0.25 W) = 0: no band at all there, body only
    ys_n, xs_n = np.nonzero(mask)
    h_all = ((xs_n - a[0]) * (mx - a[0]) + (ys_n - a[1]) * (my - a[1])) / hl
    assert (arr[ys_n[h_all < 12], xs_n[h_all < 12]] == 50).all()
    # frac 1: no clamp, the full 3-ring band (ring 0 = background 200) everywhere
    arr1, dist1, mask1, _ = _edge_case(1.0, t)
    band = mask1 & (dist1 < t)  # (128, 128) bool
    assert (arr1[band & (dist1 == 0)] == 200).all() and (arr1[mask1 & ~band] == 50).all()
    # thickness 0: no band at all, nothing repainted
    arr0, _, mask0, _ = _edge_case(frac, 0)
    assert (arr0[mask0] == 50).all()


def test_edge_clamp_keeps_draw_count():
    # the edge noise is drawn for the unclamped band: the pointer stream ends in
    # the same state whatever edge_max_frac is (nothing else moves)
    for seed in range(4):
        a = g7.render_image(_cfg(**{"pointer.edge_max_frac": 1.0}), seed)
        b = g7.render_image(_cfg(**{"pointer.edge_max_frac": 0.0}), seed)
        assert a.params["pointer"] == b.params["pointer"]
        assert a.params["tri_artifacts"] == b.params["tri_artifacts"]


@pytest.mark.parametrize("frac", [0.25, 1.0])
def test_edge_clamp_apex_shows_body(frac):
    # a person looks at the apex of a black pointer (edge band 2-3 px) on a
    # flat face 200: with the clamp, the pixel on the height where the triangle
    # is 1.5 .. 3.5 px wide shows the body (dark), not the background; without
    # it (frac 1) the background ring eats that thin end
    # no rim band, no tip jitter or pointer shift: the apex is never covered
    cfg = _cfg(**{**_COUNT_CLEAN, **_NO_STRETCH, "cut_tip.enabled": False,
                  "outer.rim_width": 0, "geometry.tip_variation": 0.0,
                  "geometry.ptr_displacement_range": [0.0, 0.0],
                  "pointer.edge_thickness_range": [2, 3], "pointer.edge_max_frac": frac})
    dark_frac = []  # list[float], share of the sampled apex points that are dark, per seed
    for seed in range(30):
        res = g7.render_image(cfg, seed)
        g = res.params["geometry"]  # dict
        (ax, ay), (mx, my), _ = res.tip_axis  # floats, image px
        hl = math.hypot(mx - ax, my - ay)
        per_px = 2 * g["chord_half"] / (g["tip_dist"] + g["chord_dist"])  # float, dW / dh
        hits = []  # list[bool], dark at each sampled width
        for wid in (1.5, 2.5, 3.5):
            h = wid / per_px  # float px along the height
            x, y = ax + h * (mx - ax) / hl, ay + h * (my - ay) / hl
            hits.append(int(res.image[int(round(y)), int(round(x))]) < 100)
        dark_frac.append(float(np.mean(hits)))
    print("frac %.2f: apex dark share mean %.3f min %.3f" % (frac, np.mean(dark_frac), min(dark_frac)))
    if frac < 1:
        assert min(dark_frac) == 1.0, dark_frac
    else:
        assert np.mean(dark_frac) < 0.8, dark_frac  # the check tells the two apart


# ── the label sits on the principal axis (after stretch, displacements, offset) ──

# "section.key" -> value pins (dict): the schema maxima of every transform that
# moves or skews the pointer (stretch, dial / pointer displacement)
_AXIS_EXTREMES = {"geometry.stretch_h_range": [0.8, 1.1], "geometry.stretch_v_range": [0.8, 1.1],
                  "geometry.displacement_range": [-10.0, 10.0],
                  "geometry.ptr_displacement_range": [-6.0, 6.0],
                  "dataset.angle_deg_range": [0, 359]}


def _line_dist(px, py, ax, ay, bx, by):
    """Perpendicular distance (float px) of (px, py) from the line through a and b."""
    dx, dy = bx - ax, by - ay  # float, line direction
    return abs((px - ax) * dy - (py - ay) * dx) / math.hypot(dx, dy)


def test_tip_on_principal_axis_geometry():
    # 200 seeds at the schema-max stretch / displacements: the label lies on the
    # stretched height A' -> M' (unrounded RenderResult.tip_axis)
    cfg = _cfg(**_AXIS_EXTREMES)
    dist, ts = [], []  # list[float] px, label distance from A'M', and tip_t
    for seed in range(200):
        res = g7.render_image(cfg, seed)
        (ax, ay), (mx, my), _bdir = res.tip_axis  # floats, image px
        dist.append(_line_dist(res.tip_x, res.tip_y, ax, ay, mx, my))
        along = ((res.tip_x - ax) * (mx - ax) + (res.tip_y - ay) * (my - ay)) / math.hypot(mx - ax, my - ay)
        assert -1e-9 <= along <= math.hypot(mx - ax, my - ay) + 1e-9, seed  # between A' and M'
        ts.append(along)
    dist = np.array(dist)  # (200,) float64
    print("axis (geometry) dist: max %.2e; tip_t mean %.2f max %.2f"
          % (dist.max(), np.mean(ts), np.max(ts)))
    assert dist.max() < 1e-6, dist.max()


def _stretched_centres(res):
    """Pointer (hub) centre and mounting-hole centre of a render, stretched, image
    px, from its recorded geometry -> ((px, py), (hx, hy)) floats."""
    g, d = res.params["geometry"], res.params["derived"]  # dicts
    c = g["canvas_size"] // 2  # int px, canvas centre

    def img_pt(x, y):
        """Canvas point (x, y) -> stretched image px (stretch about c - 0.5)."""
        return (c - 0.5 + (x - c + 0.5) * g["stretch_h"] + d["canvas_offset_x"],
                c - 0.5 + (y - c + 0.5) * g["stretch_v"] + d["canvas_offset_y"])
    pcx = c + g["ptr_displacement_mag"] * math.cos(g["ptr_displacement_angle_rad"])  # float
    pcy = c + g["ptr_displacement_mag"] * math.sin(g["ptr_displacement_angle_rad"])
    hcx = pcx + g["hole_displacement_mag"] * math.cos(g["hole_displacement_angle_rad"])
    hcy = pcy + g["hole_displacement_mag"] * math.sin(g["hole_displacement_angle_rad"])
    return img_pt(pcx, pcy), img_pt(hcx, hcy)


# geometry case (str) -> ("section.key" -> value pins dict, p95 bound float px
# or None).  PCA of the triangle pixels pins the height of the long, slender
# triangle well, so there the label must be within 0.6 px.  The default
# pointer's triangle left after the hub disc is short and fat (length ~ 1.2 x
# width), so its principal direction swings with a 1-px raster asymmetry and
# the fitted line misses even the axis apex by ~1.3 px p95; there (None) the
# label must merely be no farther from it than the axis apex is, + 0.1 px p95
_PCA_CASES = {
    "slender": ({**_SLENDER, "geometry.stretch_h_range": [0.8, 1.1],
                 "geometry.stretch_v_range": [0.8, 1.1]}, 0.6),
    "default": ({}, None),
}


@pytest.mark.parametrize("case", list(_PCA_CASES))
def test_tip_on_principal_axis_pixels(case):
    # a person fits the triangle's height to the dark TRIANGLE pixels of a clean
    # render (PCA; the hub disc and the mounting hole are left out) and checks
    # the label lies on it
    pins, bound = _PCA_CASES[case]  # dict, float px or None
    cfg = _cfg(**{**_CLEAN_TIP, "cut_tip.enabled": False,
                  "palette.outer_gray_range": [220, 220], "palette.rim_gray_range": [220, 220],
                  **pins})
    ys, xs = np.mgrid[:IMG, :IMG]  # (256, 256) int pixel grids
    dist, d_apex = [], []  # list[float] px, label / axis apex distance from the fitted axis
    for seed in range(100):
        res = g7.render_image(cfg, seed)
        d, g = res.params["derived"], res.params["geometry"]  # dicts
        smax = max(g["stretch_h"], g["stretch_v"])  # float, largest stretch
        (pcx, pcy), (hcx, hcy) = _stretched_centres(res)  # floats, hub / hole centres
        inside = (((xs - IMG / 2) / d["outer_rx"]) ** 2
                  + ((ys - IMG / 2) / d["outer_ry"]) ** 2) < 0.95  # (256, 256) bool, inside B
        sel = ((res.image < 100) & inside
               & (np.hypot(xs - pcx, ys - pcy) > g["hub_radius"] * 1.2 * smax)
               & (np.hypot(xs - hcx, ys - hcy) > g["hole_radius"] * smax + 1.5))  # (256, 256) bool
        pts = np.stack([xs[sel], ys[sel]], axis=1).astype(np.float64)  # (N, 2) float64
        mean = pts.mean(axis=0)  # (2,) float64 centroid
        evals, evecs = np.linalg.eigh(np.cov((pts - mean).T))  # (2,), (2, 2) float64
        ax_dir = evecs[:, np.argmax(evals)]  # (2,) float64 principal direction
        dist.append(_line_dist(res.tip_x, res.tip_y, mean[0], mean[1],
                               mean[0] + ax_dir[0], mean[1] + ax_dir[1]))
        d_apex.append(_line_dist(d["axis_apex_x"], d["axis_apex_y"], mean[0], mean[1],
                                 mean[0] + ax_dir[0], mean[1] + ax_dir[1]))
    dist, d_apex = np.array(dist), np.array(d_apex)  # (100,) float64 each
    print("%s axis (pixels) dist: mean %.3f p50 %.3f p95 %.3f max %.3f; axis apex p95 %.3f"
          % (case, dist.mean(), np.median(dist), np.percentile(dist, 95), dist.max(),
             np.percentile(d_apex, 95)))
    if bound is None:
        assert np.percentile(dist, 95) <= np.percentile(d_apex, 95) + 0.1, (dist, d_apex)
    else:
        assert np.percentile(dist, 95) < bound, dist



# ── min_bg_contrast restrains the pointer gray (drawn after bg + shadow) ─────

def test_pointer_forbidden_interval_unit():
    import random
    # bare face 180, darkest shadowed face 180 * (1 - 0.5) = 90, c = 40
    f_lo, f_hi = g7.pointer_forbidden_interval(180, 40, 0.5)  # floats
    assert (f_lo, f_hi) == (50.0, 220.0)
    assert g7.pointer_forbidden_interval(180, 40) == (140.0, 220.0)  # no shadow: bg +- c
    rng = random.Random(3)  # shared stream (random.Random)
    vals = {g7._pick_pointer_gray(rng, 0, 255, 180, 40, 0.5) for _ in range(3000)}  # set[int]
    assert vals == set(range(0, 51)) | set(range(220, 256))  # endpoints allowed, both sides
    # a fractional f_lo: 175 * (1 - 0.3) - 30 = 92.5 -> the lower interval ends at 92
    vals = {g7._pick_pointer_gray(rng, 0, 255, 175, 30, 0.3) for _ in range(3000)}
    assert max(v for v in vals if v < 175) == 92 and min(v for v in vals if v > 92) == 205
    # c = 0: exactly the old randint(lo, hi) draw, whatever the shadow
    for s_max in (0.0, 0.4, 1.0):
        for seed in range(200):
            want = random.Random(seed).randint(10, 240)  # int, old draw
            assert g7._pick_pointer_gray(random.Random(seed), 10, 240, 160, 0, s_max) == want
    # s_max = 0 reproduces the bare rule draw for draw
    for seed in range(200):
        a = g7._pick_pointer_gray(random.Random(seed), 0, 255, 160, 60, 0.0)  # int
        assert abs(a - 160) >= 60
    # empty set: one draw consumed, endpoint farthest from the forbidden interval
    c1, c2 = random.Random(9), random.Random(9)  # two identical streams
    # bg 150, s 0.6 -> faces [60, 150], c 80 -> forbidden (-20, 230); range [0, 200]:
    # lo is 60 below the shadowed face, hi 50 above the bare face -> lo
    assert g7._pick_pointer_gray(c1, 0, 200, 150, 80, 0.6) == 0
    c2.randint(0, 200)
    assert c1.random() == c2.random(), "infeasible case must consume one draw"
    info = {}  # dict, fallback flag out
    g7._pick_pointer_gray(random.Random(1), 0, 200, 150, 80, 0.6, info)
    assert info == {"fallback": True, "headroom_ok": True}  # headroom 0: I = [lo, hi]
    g7._pick_pointer_gray(random.Random(1), 0, 255, 150, 0, 0.6, info)  # c = 0: never
    assert info == {"fallback": False, "headroom_ok": True}
    # bg 100, s 0.1 -> faces [90, 100], c 200: hi is 140 above, lo 90 below -> hi
    assert g7._pick_pointer_gray(random.Random(0), 0, 240, 100, 200, 0.1) == 240


def test_render_records_forbidden_interval():
    # no gradient / noise: headroom H = 0, so the base interval I is the whole
    # gray range and the contrast rule alone decides (H > 0 would leave
    # I = [51, 204] inside the forbidden (14, 210): the endpoint fallback)
    pins = {"shadows.prob": 1.0, "shadows.strength_range": [0.6, 0.6],
            "shadows.strength_jitter": 0.0, "palette.bg_gray_range": [160, 160],
            "pointer.min_bg_contrast": 50, "pointer.gradient_range": [0.0, 0.0],
            "pointer.noise_amplitude": 0}
    for seed in range(6):
        p = g7.render_image(_cfg(**pins), seed).params  # dict, sampled values
        pt = p["pointer"]  # dict
        assert pt["forbidden_lo"] == pytest.approx(160 * 0.4 - 50, abs=1e-3)
        assert pt["forbidden_hi"] == pytest.approx(210.0)
        assert pt["gray"] <= pt["forbidden_lo"] or pt["gray"] >= pt["forbidden_hi"], pt
        assert pt["gray_fallback"] is False
        assert p["shadows"]["s_p"] == pytest.approx(0.6) and "d_max" not in p["shadows"]
    # no shadow: the interval is bg +- c
    pt = g7.render_image(_cfg(**{**pins, "shadows.prob": 0.0}), 0).params["pointer"]
    assert (pt["forbidden_lo"], pt["forbidden_hi"]) == (110.0, 210.0)


# "section.key" -> value pins (dict): strong shadows (strength 0.8 -> darkest face
# bg * 0.2), any pointer gray, flat face 150-170, nothing that perturbs gray
# levels (noise, blur, jitter, effects), so a person can read the levels off
_GAP_PINS = {"shadows.prob": 1.0, "shadows.strength_range": [0.8, 0.8],
             "shadows.strength_jitter": 0.0, "shadows.blur_radius": 0.0,
             "shadows.rim_blur_range": [0.5, 0.5],
             "pointer.gray_range": [0, 255], "palette.bg_gray_range": [150, 170],
             "pointer.noise_amplitude": 0, "pointer.gradient_range": [0.0, 0.0],
             "pointer.edge_noise_amplitude": 0,
             "background.noise_intensity": 0, "background.variation": 0,
             "background.dial_circle_blur": False, "hier_noise.block_variance": 0,
             "hier_noise.pixel_variance": 0, "digits.show_prob": 0.0,
             "tri_artifacts.enabled": False, "reflections.prob": 0.0, "glares.prob": 0.0,
             "jitter.brightness_range": [1.0, 1.0], "jitter.contrast_range": [1.0, 1.0],
             "jitter.sharpness_range": [1.0, 1.0]}


def _pointer_face_levels(c, seed):
    """Median grays a person reads off one render (float levels):
    (pointer body, deep-shadow face, bare face), or None if a region is tiny.

    cfg = _GAP_PINS + pointer.min_bg_contrast c (int).  Pointer body = the
    pointer footprint (gray 0 vs 255 probe renders) eroded 3 px (no edge band);
    deep shadow = where a strength-1 shadow blackens the face (and the face is
    lit without a shadow), off the pointer; bare face = central face pixels the
    strength-1 shadow leaves unchanged, off the pointer.
    """
    cfg = _cfg(**{**_GAP_PINS, "pointer.min_bg_contrast": c})
    img = g7.render_image(cfg, seed).image.astype(np.int32)  # (256, 256) int32
    full = g7.render_image(_cfg(**{**_GAP_PINS, "pointer.min_bg_contrast": c,
                                   "shadows.strength_range": [1.0, 1.0]}), seed).image
    none = g7.render_image(_cfg(**{**_GAP_PINS, "pointer.min_bg_contrast": c,
                                   "shadows.prob": 0.0}), seed).image
    foot = _pointer_footprint(cfg, seed)  # (256, 256) bool, pointer pixels
    off_ptr = ~_dilate(foot, 4)  # (256, 256) bool, well away from the pointer
    umbra = _erode((full == 0) & (none >= 60), 2) & off_ptr  # (256, 256) bool
    bare = (full == none) & off_ptr & (_radius_grid() < 0.6 * DIAL)  # (256, 256) bool
    body = _erode(foot, 3)  # (256, 256) bool
    if umbra.sum() < 50 or bare.sum() < 50 or body.sum() < 20:
        return None
    return (float(np.median(img[body])), float(np.median(img[umbra])),
            float(np.median(img[bare])))


def test_min_bg_contrast_keeps_pointer_visible_over_shadow():
    c, tol = 80, 3  # int, pointer.min_bg_contrast under test; int levels tolerance
    gaps = {80: [], 0: []}  # c -> list of (shadow gap, bare gap) floats per seed
    for seed in range(40):
        for cc in gaps:
            lv = _pointer_face_levels(cc, seed)
            if lv is not None:
                ptr, shd, bare = lv
                gaps[cc].append((abs(ptr - shd), abs(ptr - bare)))
    on, off = np.array(gaps[80]), np.array(gaps[0])  # (N, 2) float64
    print("c=80 shadow / bare gaps p0 p10 p50:", np.percentile(on, [0, 10, 50], axis=0).T,
          "c=0:", np.percentile(off, [0, 10, 50], axis=0).T, len(on), len(off))
    assert len(on) >= 30 and len(off) >= 30
    assert on.min(axis=0)[0] >= c - tol and on.min(axis=0)[1] >= c - tol, on.min(axis=0)
    assert (off.min(axis=1) < c - tol).any(), "the measurement cannot tell c = 0 from c = 80"


# ── pointer triangle: own hard raster (per-line round(width), centred) ──────

# path (str) of the user's seed-8 sample config (copied from the studio Save zip
# inspect_tip_location/v7_seed8.zip: edge thickness [0, 0], cut on, slender pointer)
_SEED8_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                           "v7_seed8_config.yaml")


def _line_interval(tri, s, rows):
    """Exact [lo, hi] (floats) of scan line s (row y = s when rows, else column
    x = s) inside triangle tri (3 (x, y) float vertices), by half-plane clipping
    (independent of the generator's edge-intersection code); None if empty."""
    (x0, y0), (x1, y1), (x2, y2) = tri
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)  # float, signed 2x area
    lo, hi = -math.inf, math.inf  # floats, running interval of u
    for (px, py), (qx, qy) in (((x0, y0), (x1, y1)), ((x1, y1), (x2, y2)), ((x2, y2), (x0, y0))):
        # inside iff sign(area) * cross(q - p, r - p) >= 0, linear in u: a u + b >= 0
        ex, ey = qx - px, qy - py  # float, edge vector
        if rows:   # r = (u, s): cross = ex (s - py) - ey (u - px)
            a, b = -ey, ex * (s - py) + ey * px
        else:      # r = (s, u): cross = ex (u - py) - ey (s - px)
            a, b = ex, -ex * py - ey * (s - px)
        sg = math.copysign(1.0, area)  # float, orientation sign
        a, b = a * sg, b * sg
        if abs(a) < 1e-12:
            if b < -1e-9:
                return None
            continue
        if a > 0:
            lo = max(lo, -b / a)
        else:
            hi = min(hi, -b / a)
    return (lo, hi) if lo <= hi + 1e-9 else None


def _random_pointer_triangles(n, seed=0):
    """n random isosceles triangles (apex, B_l, B_r), (x, y) float vertices, at
    many angles (incl. exact / near horizontal, vertical, 45 deg), half-angles
    2..60 deg (slender .. wide), heights 10..90 px, inside a 400 px canvas."""
    rng = np.random.default_rng(seed)  # np Generator
    fixed = [0.0, 90.0, 180.0, 270.0, 44.9, 45.0, 45.1, 134.9, 135.1, 1.0, 179.0, 359.0]
    out = []  # list of 3-tuples of (x, y) floats
    for i in range(n):
        ang = math.radians(fixed[i] if i < len(fixed) else rng.uniform(0, 360))  # height dir
        half_a = math.radians(rng.uniform(2, 8) if i % 2 == 0 else rng.uniform(8, 60))  # half-angle
        h = rng.uniform(10, 90)  # float px, height
        ax, ay = rng.uniform(195, 205, 2)  # float px, apex
        mx, my = ax + h * math.cos(ang), ay + h * math.sin(ang)  # base midpoint
        hb = h * math.tan(half_a)  # float px, half base
        px, py = -math.sin(ang), math.cos(ang)  # unit across
        out.append(((ax, ay), (mx + hb * px, my + hb * py), (mx - hb * px, my - hb * py)))
    return out


def test_raster_pointer_triangle_rule():
    # every scan line: one contiguous run of n = round(w) pixels (|n - w| <= 0.5),
    # centred on the true midline within 0.5 px, nothing where w < 0.5; n is
    # non-decreasing from the apex to the nearer base corner (no ball / neck);
    # nothing outside the triangle's bounding box dilated by 1
    import time
    C = 400  # int px, canvas side (every test triangle fits)
    checked = 0  # int, monotone segments checked
    ms = []  # list[float], raster time per triangle (ms)
    for tri in _random_pointer_triangles(400):
        (ax, ay), bl, br = tri
        mx, my = (bl[0] + br[0]) / 2, (bl[1] + br[1]) / 2
        rows = abs(my - ay) >= abs(mx - ax)  # bool, scan rows (else columns)
        t0 = time.perf_counter()
        r = g7.raster_pointer_triangle(*tri, C)  # (C, C) bool
        ms.append(1000 * (time.perf_counter() - t0))
        assert r.shape == (C, C) and r.dtype == bool
        xs_v, ys_v = [p[0] for p in tri], [p[1] for p in tri]  # vertex x / y floats
        yy, xx = np.nonzero(r)
        if xx.size:
            assert xx.min() >= min(xs_v) - 1 and xx.max() <= max(xs_v) + 1, tri
            assert yy.min() >= min(ys_v) - 1 and yy.max() <= max(ys_v) + 1, tri
        sv = [p[1] if rows else p[0] for p in tri]  # vertex scan coords (floats)
        ns = {}  # scan line (int) -> pixel count n (int)
        for s in range(C):
            line = r[s, :] if rows else r[:, s]  # (C,) bool
            idx = np.flatnonzero(line)  # (n,) int64 filled pixels
            iv = _line_interval(tri, s, rows)
            if iv is None:
                assert idx.size == 0, (tri, s)
                continue
            w, c = iv[1] - iv[0], (iv[0] + iv[1]) / 2  # float px width / midpoint
            ns[s] = idx.size
            assert abs(idx.size - w) <= 0.5 + 1e-6, (tri, s, idx.size, w)
            if w < 0.5 - 1e-6:
                assert idx.size == 0, (tri, s, w)
            if idx.size:
                assert idx[-1] - idx[0] + 1 == idx.size, (tri, s)  # contiguous
                assert abs((idx[0] + idx[-1]) / 2 - c) <= 0.5 + 1e-6, (tri, s, idx, c)
        # apex -> nearer base corner along the scan (where the apex is the
        # extreme vertex along the scan, w grows linearly over that stretch)
        a_s, b_s = sv[0], sv[1:]
        if all(b > a_s for b in b_s) or all(b < a_s for b in b_s):
            near = min(b_s, key=lambda b: abs(b - a_s))  # float, nearer base corner's scan coord
            seg = [s for s in sorted(ns) if min(a_s, near) <= s <= max(a_s, near)]  # list[int]
            if a_s > near:
                seg = seg[::-1]  # walk away from the apex
            counts = [ns[s] for s in seg]  # list[int]
            assert all(b >= a for a, b in zip(counts, counts[1:])), (tri, counts)
            checked += 1
    print("raster: %d monotone segments; %.3f ms mean, %.3f ms max per triangle"
          % (checked, np.mean(ms), np.max(ms)))
    assert checked >= 300, checked


def test_raster_pointer_triangle_fixes_seed8_ball():
    # the user's seed-8 tip (pre-stretch canvas, apex ~(144.68, 40.18), 14 deg,
    # slender, cut r 7): Pillow filled rows 48..54 with 3 3 2 3 3 3 4 px (a ball
    # on a neck); now, from the cut boundary down 12 rows, the count never drops
    cfg = v7_config.load_config(_SEED8_YAML)
    cap = {}  # dict, the draw_pointer outputs (spy)
    orig = g7.draw_pointer

    def spy(*a, **k):
        out = orig(*a, **k)
        cap["tri"], cap["pix"] = (out[1], out[2], out[3]), out[5]
        return out
    g7.draw_pointer = spy
    try:
        g7.render_image(cfg, 8)
    finally:
        g7.draw_pointer = orig
    tri, pix = cap["tri"], cap["pix"]  # 3 (x, y) floats; (C, C) bool pre-stretch pointer pixels
    ax, ay = tri[0]
    win = slice(int(ax) - 15, int(ax) + 15)  # columns around the tip
    first = next(r for r in range(int(ay), int(ay) + 40) if pix[r, win].any())  # int, cut boundary row
    rr = range(first, first + 13)  # the cut boundary row + 12
    counts = [int(pix[r, win].sum()) for r in rr]  # list[int]
    widths = [_line_interval(tri, r, True) for r in rr]  # list[(lo, hi)]
    print("seed 8 rows %d..%d counts %s widths %s" % (
        first, first + 12, counts, [round(w[1] - w[0], 2) for w in widths]))
    assert all(b >= a for a, b in zip(counts, counts[1:])), counts
    # past the boundary row (the cut arc may trim it) every row holds round(w)
    for r, n, (lo, hi) in list(zip(rr, counts, widths))[1:]:
        assert abs(n - (hi - lo)) <= 0.5 + 1e-6, (r, n, hi - lo)


# "section.key" -> value pins (dict): flat face 200, black pointer 0, rim /
# outer 220, no noise / blur / effects / shadows / stretch, and NO edge band
# (the band repaints the outline, which would hide the raster under test)
_RASTER_CLEAN = {**_CLEAN_TIP, **_NO_STRETCH, "palette.outer_gray_range": [220, 220],
                 "palette.rim_gray_range": [220, 220],
                 "pointer.edge_thickness_range": [0, 0]}


def _dark_counts_along(res, frac=0.6, dark=100):
    """Dark-pixel count (list[int]) on each scan line along the pointer, from the
    apex toward the base, down to `frac` (float) of the height; stops early at
    the first line the hub disc / mounting hole could reach.  Lines touching the
    rim band / outer fill (an apex under the rim) are skipped until the pointer
    comes out on the dial face.

    Scan lines are image rows when the height A' -> M' (RenderResult.tip_axis)
    is closer to vertical, else columns, the way a person would count them; a
    pixel counts when its gray < dark (int) and it lies within a window of
    1.5 x the triangle's half width + 3 px about the height line."""
    (ax, ay), (mx, my), _ = res.tip_axis  # floats, image px
    g, d = res.params["geometry"], res.params["derived"]  # dicts
    rows = abs(my - ay) >= abs(mx - ax)  # bool, scan rows
    (pcx, pcy), (hcx, hcy) = _stretched_centres(res)  # floats, hub / hole centres
    ys, xs = np.mgrid[:IMG, :IMG]  # (256, 256) int grids
    excl = ((np.hypot(xs - pcx, ys - pcy) <= g["hub_radius"] + 1.5)
            | (np.hypot(xs - hcx, ys - hcy) <= g["hole_radius"] + 1.5))  # (256, 256) bool
    c = g["canvas_size"] // 2  # int px, canvas centre = dial centre (canvas coords)
    dcx, dcy = c + d["canvas_offset_x"], c + d["canvas_offset_y"]  # int px, dial centre (image)
    rim = v7_config.default_config()["outer"]["rim_width"] + 1.5  # float px, rim band + margin
    face = (((xs - dcx) / (d["dial_rx"] - rim)) ** 2
            + ((ys - dcy) / (d["dial_ry"] - rim)) ** 2) <= 1  # (256, 256) bool, bare dial face
    s0, s1 = (ay, my) if rows else (ax, mx)  # float, apex / base midpoint scan coords
    u0, u1 = (ax, mx) if rows else (ay, my)  # float, their along-line coords
    step = 1 if s1 > s0 else -1  # int, scan direction (apex -> base)
    s = math.ceil(s0) if step > 0 else math.floor(s0)  # int, first line past the apex
    s_end = s0 + frac * (s1 - s0)  # float, last scan coordinate
    dark_img = res.image < dark  # (256, 256) bool
    counts = []  # list[int]
    while (s - s_end) * step <= 0:
        f = (s - s0) / (s1 - s0)  # float in [0, frac], fraction of the height
        uc = u0 + f * (u1 - u0)  # float, height line on this line
        reach = 1.5 * g["chord_half"] * f + 3.0  # float px, half window along the line
        lo = max(int(math.ceil(uc - reach)), 0)  # int, window start
        hi = min(int(math.floor(uc + reach)), IMG - 1)  # int, window end (incl.)
        line = dark_img[s, lo:hi + 1] if rows else dark_img[lo:hi + 1, s]  # (k,) bool
        ex = excl[s, lo:hi + 1] if rows else excl[lo:hi + 1, s]  # (k,) bool
        on_face = face[s, lo:hi + 1] if rows else face[lo:hi + 1, s]  # (k,) bool
        if ex.any() or (counts and not on_face.all()):
            break
        if on_face.all():
            counts.append(int(line.sum()))
        s += step
    return counts


@pytest.mark.parametrize("geom", ["default", "slender"])
@pytest.mark.parametrize("cut", [False, True], ids=["cut_off", "cut_on"])
def test_pointer_width_never_shrinks_toward_base(cut, geom):
    # a person counts the dark pixels on every row (or column) of a clean
    # render, walking from the tip toward the hub: the count never drops (no
    # ball at the tip, no neck behind it), for 100 seeds
    cfg = _cfg(**{**_RASTER_CLEAN, **(_SLENDER if geom == "slender" else {}),
                  "cut_tip.enabled": cut})
    bad, lines = [], 0  # list[(seed, counts)], int lines checked
    for seed in range(100):
        counts = _dark_counts_along(g7.render_image(cfg, seed))  # list[int]
        lines += len(counts)
        if any(b < a for a, b in zip(counts, counts[1:])):
            bad.append((seed, counts))
    print("%s %s: %d lines, %d seeds with a drop %s" % (
        geom, "cut" if cut else "no cut", lines, len(bad), bad[:2]))
    assert lines >= 100 * 5, lines
    assert not bad, bad[:5]


# ── pointer gray range warnings (config-level, pointer_gray_warnings) ──

def test_gray_warnings_default_config_is_clean():
    assert g7.pointer_gray_warnings(v7_config.default_config()) == []


def test_headroom_bound_is_worst_case_of_geometry():
    cfg = v7_config.default_config()
    # smallest dial R = int(85 * 0.9) = 76, tip = 74 * 0.98, hub = 76 * 0.45 < tip
    tip = 74 * 0.98  # float px, shortest tip
    want = 25.0 * (tip + 2) / tip + 25  # float, |gradient|max * along_max + noise
    assert g7.pointer_headroom_bound(cfg) == pytest.approx(want)
    assert 50 < want < 51  # H ≈ 51 -> I = [51, 204]


def test_gray_warning_no_headroom():
    w = g7.pointer_gray_warnings(_cfg(**{"pointer.gray_range": [230, 255]}))
    assert len(w) == 1 and w[0].startswith("pointer.gray_range: [230, 255] leaves no room")
    assert "[51, 204]" in w[0] and "H≈51" in w[0]


def test_gray_warning_too_narrow_counts_levels():
    w = g7.pointer_gray_warnings(_cfg(**{"pointer.gray_range": [100, 120]}))
    assert len(w) == 1 and w[0].startswith("pointer.gray_range:")
    assert "21 levels" in w[0]


def test_gray_warning_contrast_fallback_share():
    cfg = _cfg(**{"pointer.min_bg_contrast": 90, "palette.bg_gray_range": [100, 160],
                  "shadows.prob": 0.5})
    # hand enumeration, I = [51, 204], c = 90: the upper part bg + 90 <= 204 needs
    # bg <= 114; the lower part needs bg (1 - s) - 90 >= 51.  No shadow: empty for
    # bg 115..140 (26 of 61); shadow (s_max >= 0.45): empty for bg 115..160 (46)
    want = 0.5 * 26 / 61 + 0.5 * 46 / 61  # float, 36 / 61
    assert g7.pointer_fallback_share(cfg) == pytest.approx(want)
    w = g7.pointer_gray_warnings(cfg)
    assert len(w) == 1 and w[0].startswith("pointer.min_bg_contrast: 90")
    assert f"{want * 100:.1f}%" in w[0] and "point mass" in w[0]


def test_max_strength_cdf_matches_grid():
    # P(clip(s + max(jp, jr)) <= t) by a fine midpoint grid over (s, jp, jr)
    s_lo, s_hi, j = 0.5, 0.62, 0.05  # float, default shadow strengths
    n = 120  # int, grid points per axis
    s = s_lo + (np.arange(n) + 0.5) / n * (s_hi - s_lo)  # (n,) float64
    jg = -j + (np.arange(n) + 0.5) / n * 2 * j  # (n,) float64
    smax = np.clip(s[:, None, None] + np.maximum(jg[None, :, None], jg[None, None, :]), 0, 1)
    for t in (0.47, 0.5, 0.55, 0.6, 0.66):
        assert g7._max_strength_cdf(t, s_lo, s_hi, j) == pytest.approx(
            float((smax <= t).mean()), abs=0.01)
    assert g7._max_strength_cdf(0.4, s_lo, s_hi, j) == 0.0
    assert g7._max_strength_cdf(0.7, s_lo, s_hi, j) == 1.0


def test_gray_warnings_predict_render_rates():
    """2000 seeds through render_image: the empirical headroom_ok == False and
    gray_fallback rates agree with the predictions within sampling error.
    (a) is all-or-nothing given the worst-case H (here every image's H rounds
    to the same band, so 100 %); c = 40 puts s* inside the shadow strength
    distribution, so the CDF part of the enumeration is exercised."""
    n = 2000  # int, seeds per config
    cfg_a = _cfg(**{"pointer.gray_range": [230, 255]})
    no_room = sum(not g7.render_image(cfg_a, s).params["pointer"]["headroom_ok"]
                  for s in range(n))  # int, images without headroom
    print("no headroom: predicted 1.0, empirical %.4f" % (no_room / n))
    assert no_room == n
    cfg_c = _cfg(**{"pointer.min_bg_contrast": 40})
    p = g7.pointer_fallback_share(cfg_c)  # float, predicted share
    fb = sum(g7.render_image(cfg_c, s).params["pointer"]["gray_fallback"]
             for s in range(n))  # int, fallback images
    print("fallback: predicted %.4f, empirical %.4f" % (p, fb / n))
    assert 0.1 < p < 0.9
    assert abs(fb / n - p) < 4 * math.sqrt(p * (1 - p) / n)
