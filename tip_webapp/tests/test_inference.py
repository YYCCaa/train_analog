"""Unit tests for tip_webapp.app.inference (pure logic + the real TFLite model, no web server).

What is covered:

    intake       box parsing / default box / bounds, byte + pixel limits, bad bytes, EXIF upright
    preprocess   crop -> convert("L") -> 256 x 256 Lanczos; a 256-px box is bit-exact; 16-bit scaling
    mapping      model px <-> image px through a box is exact and invertible (centre convention)
    decode       numpy decode == train_v9_tip.decode_logits (torch) on 60+ maps, incl. borders/corners
    candidates   2-peak map, plateau de-duplication, mass threshold, argmax first
    samples      the shipped TFLite model on the 6 labelled dials: 1-5 within 7 px, 6 is a known failure
"""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from PIL import Image

from tip_webapp.app import inference as inf

# SAMPLES: list[dict], entries of tip_webapp/samples/samples.json ({"file", "label": {"x", "y"}, "note", ...})
SAMPLES = json.loads((inf.PACKAGE_DIR / "samples" / "samples.json").read_text(encoding="utf-8"))["samples"]
# BUDGET_PX: float, the error budget (model px == image px for the 256-px samples)
BUDGET_PX = 7.0


@pytest.fixture(scope="module")
def model() -> inf.TipModel:
    """The shipped TFLite model, loaded once for this module."""
    return inf.TipModel(inf.DEFAULT_MODEL_PATH)


def _png(arr: np.ndarray) -> bytes:
    """PNG bytes of a uint8 array (H, W) gray or (H, W, 3) RGB."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


# ── intake ─────────────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("w", "h", "expected"),
    [(256, 256, (0, 0, 256)), (400, 300, (50, 0, 300)), (301, 500, (0, 99, 301)), (640, 481, (79, 0, 481))],
)
def test_default_box_is_largest_centred_square(w, h, expected):
    box = inf.default_box(w, h)
    assert (box.left, box.top, box.size) == expected


def test_parse_box_accepts_ints_with_spaces_and_blank_means_default():
    assert inf.parse_box(" 137 , 61,256 ", 640, 480) == inf.Box(137, 61, 256)
    assert inf.parse_box("0,0,480", 640, 480) == inf.Box(0, 0, 480)
    assert inf.parse_box(None, 640, 480) == inf.Box(80, 0, 480)
    assert inf.parse_box("   ", 640, 480) == inf.Box(80, 0, 480)


@pytest.mark.parametrize("text", ["1,2", "a,b,c", "1,2,3,4", "1.5,2,20", "10;10;20", "0,0,"])
def test_parse_box_malformed(text):
    with pytest.raises(inf.InvalidBoxError, match="Malformed"):
        inf.parse_box(text, 640, 480)


@pytest.mark.parametrize(
    ("text", "needle"),
    [("-1,0,100", "outside"), ("0,-5,100", "outside"), ("600,0,100", "outside"), ("0,400,100", "outside"),
     ("0,0,481", "outside"), ("10,10,15", "at least 16"), ("0,0,0", "at least 16")],
)
def test_parse_box_outside_or_too_small(text, needle):
    with pytest.raises(inf.InvalidBoxError, match=needle):
        inf.parse_box(text, 640, 480)


def test_tiny_image_has_no_valid_default_box():
    with pytest.raises(inf.InvalidBoxError, match="at least 16"):
        inf.parse_box(None, 12, 40)


def test_load_image_rejects_bad_bytes():
    with pytest.raises(inf.InvalidImageError, match="empty"):
        inf.load_image(b"")
    with pytest.raises(inf.InvalidImageError, match="not an image"):
        inf.load_image(b"just some text, definitely not a picture")
    truncated = _png(np.zeros((64, 64), np.uint8))[:60]
    with pytest.raises(inf.InvalidImageError):
        inf.load_image(truncated)


def test_load_image_limits():
    data = _png(np.zeros((40, 50), np.uint8))
    with pytest.raises(inf.ImageTooLargeError, match="MB"):
        inf.load_image(data, max_bytes=10)
    with pytest.raises(inf.ImageTooLargeError, match="megapixels"):
        inf.load_image(data, max_pixels=1999)
    assert inf.load_image(data, max_pixels=2000).size == (50, 40)


def test_load_image_applies_exif_orientation():
    # stored: 30 wide x 20 tall, EXIF orientation 6 (rotate 90 degrees clockwise to view) -> upright 20 x 30
    img = Image.fromarray(np.tile(np.arange(30, dtype=np.uint8) * 8, (20, 1)))
    exif = Image.Exif()
    exif[0x0112] = 6
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=exif.tobytes(), quality=95)
    assert inf.load_image(buf.getvalue()).size == (20, 30)


# ── preprocess ─────────────────────────────────────────────────────────────────────────────────


def test_preprocess_256_box_is_bit_exact_gray_crop():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8)
    img = Image.fromarray(rgb)
    out = inf.preprocess(img, inf.Box(100, 20, 256))
    assert out.shape == (256, 256) and out.dtype == np.uint8
    expected = np.asarray(img.crop((100, 20, 356, 276)).convert("L"))
    assert np.array_equal(out, expected)


def test_preprocess_resizes_with_lanczos():
    rng = np.random.default_rng(2)
    gray = rng.integers(0, 256, size=(600, 600), dtype=np.uint8)
    img = Image.fromarray(gray)
    out = inf.preprocess(img, inf.Box(50, 40, 512))
    expected = np.asarray(img.crop((50, 40, 562, 552)).resize((256, 256), Image.Resampling.LANCZOS))
    assert np.array_equal(out, expected)
    small = inf.preprocess(img, inf.Box(0, 0, 100))  # upsampling path
    assert small.shape == (256, 256)


def test_preprocess_scales_16bit_instead_of_clipping():
    wide = (np.arange(64, dtype=np.uint16)[None, :].repeat(64, 0) * 1000).astype(np.uint16)  # 0 .. 63000
    img = Image.frombytes("I;16", (64, 64), wide.astype("<u2").tobytes())
    out = inf.preprocess(img, inf.Box(0, 0, 64))
    assert out.max() > 200 and out.min() == 0 and len(np.unique(out)) > 30  # not a white sheet


# ── coordinate mapping ─────────────────────────────────────────────────────────────────────────


def test_mapping_is_exact_and_invertible():
    box = inf.Box(137, 61, 256)
    assert inf.model_to_image(10.25, 200.0, box) == (147.25, 261.0)  # 256-px box: pure offset
    big = inf.Box(100, 50, 512)
    # pixel EDGES map to edges: model -0.5 -> image left - 0.5, model 255.5 -> left + size - 0.5
    assert inf.model_to_image(-0.5, -0.5, big) == (99.5, 49.5)
    assert inf.model_to_image(255.5, 255.5, big) == (611.5, 561.5)
    # model pixel 0 (centre 0) covers image pixels 100 and 101 -> centre 100.5
    assert inf.model_to_image(0.0, 0.0, big) == (100.5, 50.5)
    rng = np.random.default_rng(3)
    for x, y in rng.uniform(-0.5, 255.5, size=(20, 2)):
        for b in (box, big, inf.Box(7, 3, 33)):
            back = inf.image_to_model(*inf.model_to_image(x, y, b), b)
            assert abs(back[0] - x) < 1e-9 and abs(back[1] - y) < 1e-9


# ── decode ─────────────────────────────────────────────────────────────────────────────────────


def _random_maps(n_random: int, seed: int = 0) -> np.ndarray:
    """(N, 128, 128) float32 logit maps: noise + one sub-pixel Gaussian peak each; borders and corners included."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128].astype(np.float64)
    # centres: list[(cy, cx)] float heatmap px: random ones, then every corner and border mid-point
    centres = [tuple(rng.uniform(0, 127, size=2)) for _ in range(n_random)]
    centres += [(0, 0), (0, 127), (127, 0), (127, 127), (0, 63.4), (127, 64.6), (31.2, 0), (90.7, 127),
                (0.4, 0.3), (126.6, 126.8), (1, 126), (126, 1)]
    maps = []
    for cy, cx in centres:
        sigma = rng.uniform(0.6, 2.5)
        amp = rng.uniform(6.0, 25.0)
        logits = rng.normal(0.0, 1.5, size=(128, 128)) + amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))
        maps.append(logits)
    return np.stack(maps).astype(np.float32)


def test_numpy_decode_matches_train_v9_tip():
    torch = pytest.importorskip("torch")
    tv = pytest.importorskip("train_v9_tip")
    maps = _random_maps(50)
    assert len(maps) >= 60
    ref = tv.decode_logits(torch.from_numpy(maps[:, None])).numpy()  # (N, 2) image px
    ours = np.array([inf.decode_logits(m)[:2] for m in maps])
    diff = np.abs(ours - ref).max()
    print(f"\nmax |numpy - torch| decode difference over {len(maps)} maps: {diff:.2e} px")
    assert diff < 1e-4


def test_decode_off_map_cells_are_ignored():
    logits = np.full((128, 128), -20.0, np.float32)
    logits[0, 0] = 10.0
    logits[0, 1] = 10.0  # equal weight right of the corner -> u = 0.5, v = 0 (cells at -1 do not exist)
    x, y, iy, ix = inf.decode_logits(logits)
    assert (iy, ix) == (0, 0)
    assert abs(x - (2 * 0.5 + 0.5)) < 1e-6 and abs(y - 0.5) < 1e-6


def test_window_mass_is_clipped_at_corner():
    probs = np.full((128, 128), 1.0 / (128 * 128))
    assert math.isclose(inf.window_mass(probs, 0, 0), 9 / (128 * 128))
    assert math.isclose(inf.window_mass(probs, 64, 64), 25 / (128 * 128))


# ── candidates ─────────────────────────────────────────────────────────────────────────────────


def _blob(logits: np.ndarray, cy: float, cx: float, amp: float, sigma: float = 1.0) -> None:
    """Add a Gaussian bump of height amp at (cy, cx) heatmap px to logits (in place)."""
    yy, xx = np.mgrid[0:128, 0:128]
    logits += amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2))


def test_candidates_two_peaks():
    logits = np.full((128, 128), -10.0)
    _blob(logits, 30.3, 40.0, 20.0)
    _blob(logits, 90.0, 100.6, 19.0)
    _blob(logits, 10.0, 120.0, 7.0)  # far too weak: mass < 0.001 -> dropped
    logits = logits.astype(np.float32)
    probs = inf.softmax_map(logits)
    cands = inf.find_candidates(logits, probs)
    assert len(cands) == 2
    first, second = cands
    x, y, iy, ix = inf.decode_logits(logits)
    assert (first["x_model"], first["y_model"]) == (x, y) and (first["cell_y"], first["cell_x"]) == (iy, ix)
    assert (second["cell_y"], second["cell_x"]) == (90, 101)
    assert first["mass"] > second["mass"] >= inf.MIN_CANDIDATE_MASS
    # mass = probability summed over the 5x5 cells around each peak cell
    assert math.isclose(second["mass"], float(probs[88:93, 99:104].sum()), rel_tol=1e-12)
    assert math.isclose(first["mass"], float(probs[28:33, 38:43].sum()), rel_tol=1e-12)
    assert 0.1 < second["mass"] / first["mass"] < 0.5  # amp 19 vs 20, off-centre: clearly the runner-up
    assert abs(second["x_model"] - (2 * 100.6 + 0.5)) < 0.5 and abs(second["y_model"] - (2 * 90 + 0.5)) < 0.5
    assert math.isclose(first["mass"] + second["mass"], 1.0, abs_tol=0.02)


def test_candidates_plateau_reported_once_and_capped_at_five():
    logits = np.full((128, 128), -10.0, np.float32)
    logits[50, 50] = logits[50, 51] = 5.0  # a tie: both cells are local maxima of their 9x9 windows
    for k in range(6):
        logits[5 + 20 * k, 110] = 4.0 - 0.1 * k  # six single-cell peaks, decreasing
    probs = inf.softmax_map(logits)
    cands = inf.find_candidates(logits, probs)
    assert len(cands) == inf.MAX_CANDIDATES
    assert sum(1 for c in cands if c["cell_x"] in (50, 51)) == 1
    assert (cands[0]["cell_y"], cands[0]["cell_x"]) == (50, 50)  # argmax (first in row-major order) first
    masses = [c["mass"] for c in cands[1:]]
    assert masses == sorted(masses, reverse=True)


# ── the real model on the labelled samples ─────────────────────────────────────────────────────


def test_sample_errors(model):
    errors = {}
    for s in SAMPLES:
        pred = inf.predict(model, (inf.PACKAGE_DIR / "samples" / s["file"]).read_bytes())
        tx, ty = pred.tip
        errors[s["file"]] = math.hypot(tx - s["label"]["x"], ty - s["label"]["y"])
        assert pred.box == inf.Box(0, 0, 256)
        assert pred.tip == pred.tip_model  # 256-px image, 256-px box: image px == model px
    print("\nsample errors (px): " + ", ".join(f"{k} {v:.3f}" for k, v in errors.items()))
    files = [s["file"] for s in SAMPLES]
    for f in files[:5]:
        assert errors[f] < BUDGET_PX, f"{f}: {errors[f]:.2f} px"
    assert errors[files[5]] > BUDGET_PX, "sample 6 is the known look-alike failure"


def test_model_reports_io(model):
    assert model.input_shape == [1, 256, 256, 1] and model.output_shape == [1, 128, 128, 1]
    assert model.input_dtype == model.output_dtype == "float32"
    assert 1 <= model.num_threads <= 4
    assert model.file_bytes == inf.DEFAULT_MODEL_PATH.stat().st_size
