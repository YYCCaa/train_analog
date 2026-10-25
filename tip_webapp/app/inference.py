"""Pure image -> pointer-tip inference with the shipped TFLite TipNet (no web code).

Pipeline for one upload (shapes and dtypes on every arrow):

    bytes (<= 20 MB)
      | open + guards (<= 40 MP), first frame                      load_image()
      v
    PIL.Image, upright (ImageOps.exif_transpose)       W x H, any mode
      | box "left,top,size" (or the largest centred square)       parse_box() / default_box()
      v
    crop  size x size  --convert("L")-->  gray uint8 (size, size)
      | resize 256 x 256, Lanczos                                  preprocess()
      v
    model_gray uint8 (256, 256)  --/255-->  model_input float32 (1, 256, 256, 1)  NHWC
      | TFLite TipNet (one interpreter, behind a lock)             TipModel.run()
      v
    logits float32 (128, 128)  --softmax over all cells-->  probs float64 (128, 128), sum 1
      | argmax cell (iy, ix) -> softmax of the 5x5 logits around it -> centroid (u, v)
      v                                                            decode_logits()
    tip_model (x, y) = (2u + 0.5, 2v + 0.5)   in 256-px model-input coordinates
      | exact centre-convention mapping back through the crop      model_to_image()
      v
    tip (x, y) in uploaded-image px     + candidates (look-alike peaks), heatmap, timings

Coordinate conventions (pixel i is centred at coordinate i, its edges are i - 0.5 and i + 0.5):

    heatmap cell u             model-input px                     uploaded-image px
    (128 per side)             (256 per side)                     (the crop is `size` per side)
    ┌──┬──┐                    ┌─┬─┬─┬─┐                          ┌───────────── size ─────────────┐
    │u │  │   x = 2u + 0.5     │ │ │ │ │   X_crop = (x + 0.5)     │ X = left + X_crop              │
    └──┴──┘ ─────────────────▶ └─┴─┴─┴─┘   * size / 256 - 0.5 ──▶ │                                │
    one cell = 2 x 2 model px  x in [-0.5, 255.5]                 └────────────────────────────────┘
                               x = (X_crop + 0.5) * 256 / size - 0.5   (the inverse mapping)

    Because every mapping is "scale the pixel EDGES", a cell / pixel / crop that tiles its parent
    exactly maps back exactly: the 128 x 128 heatmap tiles the 256 x 256 model input, which tiles
    the size x size crop, which sits at (left, top) in the uploaded image.

The public entry point is :func:`predict`; it raises subclasses of :class:`PredictError`
that the web layer maps to HTTP status codes (400 / 413 / 422).
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import threading
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# MODEL_SIZE: int, side of the square gray image the model takes (model-input px).
MODEL_SIZE = 256
# HEATMAP_SIZE: int, side of the square logit map the model returns (heatmap px; 1 cell = 2x2 model px).
HEATMAP_SIZE = 128
# DECODE_WIN: int, side of the softmax-centroid window around the argmax cell (heatmap px).
DECODE_WIN = 5
# PEAK_NEIGHBOURHOOD: int, a candidate is the largest probability in its PEAK_NEIGHBOURHOOD^2 window.
PEAK_NEIGHBOURHOOD = 9
# MAX_CANDIDATES: int, at most this many look-alike peaks are reported.
MAX_CANDIDATES = 5
# MIN_CANDIDATE_MASS: float, peaks whose 5x5 probability mass is below this are dropped.
MIN_CANDIDATE_MASS = 1e-3
# MIN_BOX: int, smallest accepted crop side in uploaded-image px.
MIN_BOX = 16
# MAX_BYTES: int, largest accepted upload in bytes (20 MB).
MAX_BYTES = 20 * 1024 * 1024
# MAX_PIXELS: int, largest accepted width * height (40 megapixels); guards decompression bombs.
MAX_PIXELS = 40_000_000

# PACKAGE_DIR: Path, the tip_webapp/ folder (model/, samples/, static/ live inside it).
PACKAGE_DIR = Path(__file__).resolve().parent.parent
# DEFAULT_MODEL_PATH: Path, the shipped float32 TFLite model.
DEFAULT_MODEL_PATH = PACKAGE_DIR / "model" / "tipnet_v9.tflite"
# DEFAULT_CARD_PATH: Path, the JSON model card that describes DEFAULT_MODEL_PATH.
DEFAULT_CARD_PATH = PACKAGE_DIR / "model" / "model_card.json"

# _BOX_RE: compiled regex, "left,top,size" as three (possibly negative) integers, spaces allowed.
_BOX_RE = re.compile(r"^\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*$")


# ── errors ───────────────────────────────────────────────────────────────────────────────────────


class PredictError(Exception):
    """Base class for every expected failure; str(exc) is a user-facing message."""


class InvalidImageError(PredictError):
    """The bytes are missing, empty, not a decodable image, or corrupt (HTTP 400)."""


class ImageTooLargeError(PredictError):
    """The upload exceeds the byte or pixel limit (HTTP 413)."""


class InvalidBoxError(PredictError):
    """The box is malformed, outside the image, or smaller than MIN_BOX (HTTP 422)."""


# ── image intake ─────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Box:
    """A square crop in uploaded-image px (after EXIF rotation).

    left: int, x of the crop's left-most pixel column
    top:  int, y of the crop's top-most pixel row
    size: int, side length in px; the crop covers columns left .. left + size - 1
    """

    left: int
    top: int
    size: int

    def to_json(self) -> dict[str, int]:
        """-> {"left": int, "top": int, "size": int}."""
        return {"left": self.left, "top": self.top, "size": self.size}


def load_image(data: bytes, *, max_bytes: int = MAX_BYTES, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Decode upload bytes into an upright PIL image (first frame only).

    data: bytes, the uploaded file; max_bytes / max_pixels: int limits (overridable for tests).
    Returns a loaded PIL.Image of any mode, size W x H after ImageOps.exif_transpose.
    Raises InvalidImageError (empty / not an image / corrupt) or ImageTooLargeError.
    """
    if len(data) > max_bytes:
        raise ImageTooLargeError(f"File is larger than the {max_bytes / (1024 * 1024):g} MB limit.")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")
    try:
        with warnings.catch_warnings():
            # Promote Pillow's "possible decompression bomb" warning to an error we can catch.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            width, height = img.size  # int, int: header size, read before any pixel is decoded
            if width * height > max_pixels:
                raise ImageTooLargeError(
                    f"Image is {width}x{height} ({width * height / 1e6:.1f} megapixels); "
                    f"the limit is {max_pixels / 1e6:g} megapixels."
                )
            img.seek(0)  # animated GIF / WebP / APNG: only the first frame is used
            img.load()
    except ImageTooLargeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError(f"Image has too many pixels (limit {max_pixels / 1e6:g} megapixels).") from exc
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The uploaded file is not an image (not a recognised image format).") from exc
    except (OSError, SyntaxError, ValueError, EOFError, IndexError) as exc:
        raise InvalidImageError("The image file is corrupt or truncated and could not be decoded.") from exc
    try:
        return ImageOps.exif_transpose(img)  # phone photos come out upright, like the browser shows them
    except (ValueError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The image's EXIF orientation data is corrupt.") from exc


def default_box(width: int, height: int) -> Box:
    """The largest centred square of a width x height image (ints) -> Box; odd slack rounds left/up."""
    size = min(width, height)
    return Box(left=(width - size) // 2, top=(height - size) // 2, size=size)


def parse_box(text: str | None, width: int, height: int) -> Box:
    """Validate the optional "left,top,size" form field against a width x height image.

    text: str | None, e.g. "137,61,256"; None or blank means "use default_box".
    Returns a Box fully inside the image with size >= MIN_BOX, else raises InvalidBoxError.
    """
    if text is None or not text.strip():
        box = default_box(width, height)
        if box.size < MIN_BOX:
            raise InvalidBoxError(
                f"The image is only {width}x{height} px; the square must be at least {MIN_BOX} px."
            )
        return box
    match = _BOX_RE.match(text)
    if match is None:
        raise InvalidBoxError(f"Malformed box '{text[:40]}': expected 'left,top,size' as three integers.")
    left, top, size = (int(g) for g in match.groups())
    if size < MIN_BOX:
        raise InvalidBoxError(f"The box is {size} px; it must be at least {MIN_BOX} px on a side.")
    if left < 0 or top < 0 or left + size > width or top + size > height:
        raise InvalidBoxError(
            f"The box {left},{top} size {size} lies outside the {width}x{height} image."
        )
    return Box(left=left, top=top, size=size)


def _to_gray(img: Image.Image) -> Image.Image:
    """Any PIL mode -> mode "L" via convert("L"); 16-bit / float images are first scaled to 0..255.

    (Pillow's own I;16 -> L conversion clips everything above 255 to white, which would turn a
    16-bit PNG into a blank picture, so those modes are scaled instead. 8-bit inputs - every
    upload from the web page - go straight through convert("L").)
    """
    if img.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
        # wide: (H, W) float64 raw sample values
        wide = np.asarray(img, dtype=np.float64)
        if img.mode.startswith("I;16") or wide.max(initial=0) > 255:
            wide = wide * (255.0 / 65535.0)
        return Image.fromarray(np.clip(np.floor(wide + 0.5), 0, 255).astype(np.uint8))
    return img.convert("L")


def preprocess(img: Image.Image, box: Box) -> np.ndarray:
    """Crop the box, gray it, resize to the model's input size.

    img: upright PIL.Image (any mode); box: Box inside it.
    Returns model_gray: (256, 256) uint8, exactly the picture the model sees (before / 255).
    A 256-px box is not resampled at all (Pillow returns a copy), so it is bit-exact.
    """
    crop = img.crop((box.left, box.top, box.left + box.size, box.top + box.size))
    gray = _to_gray(crop)
    if gray.size != (MODEL_SIZE, MODEL_SIZE):
        gray = gray.resize((MODEL_SIZE, MODEL_SIZE), Image.Resampling.LANCZOS)
    return np.asarray(gray, dtype=np.uint8)


# ── coordinate mapping (centre convention, see the module docstring) ────────────────────────────


def model_to_image(x_model: float, y_model: float, box: Box) -> tuple[float, float]:
    """Model-input px (floats in [-0.5, 255.5]) -> uploaded-image px (floats) through `box`.

    X_crop = (x_model + 0.5) * size / 256 - 0.5 ;  X = left + X_crop  (same for y).
    """
    k = box.size / MODEL_SIZE  # float: image px per model px
    return box.left + (x_model + 0.5) * k - 0.5, box.top + (y_model + 0.5) * k - 0.5


def image_to_model(x: float, y: float, box: Box) -> tuple[float, float]:
    """Uploaded-image px -> model-input px; the exact inverse of model_to_image."""
    k = MODEL_SIZE / box.size  # float: model px per image px
    return (x - box.left + 0.5) * k - 0.5, (y - box.top + 0.5) * k - 0.5


# ── decode ───────────────────────────────────────────────────────────────────────────────────────


def softmax_map(logits: np.ndarray) -> np.ndarray:
    """Spatial softmax over every cell. logits: (h, w) float -> probs: (h, w) float64 summing to 1."""
    z = logits.astype(np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def _window_centroid(logits: np.ndarray, iy: int, ix: int, win: int = DECODE_WIN) -> tuple[float, float]:
    """Softmax-weighted centroid of the win x win logits around cell (iy, ix), cells off the map ignored.

    logits: (h, w) float; iy, ix: int cell. Returns (u, v) float heatmap px (u = column, v = row).
    Mirrors train_v9_tip.decode_logits: padding with -inf makes off-map cells weigh exactly 0.
    """
    h, w = logits.shape
    r = win // 2
    y0, y1 = max(0, iy - r), min(h, iy + r + 1)  # int: rows of the window that exist
    x0, x1 = max(0, ix - r), min(w, ix + r + 1)  # int: cols of the window that exist
    # patch: (rows, cols) float64 logits of the in-map part of the window
    patch = logits[y0:y1, x0:x1].astype(np.float64)
    wts = np.exp(patch - patch.max())
    wts /= wts.sum()
    # ys / xs: (rows,) / (cols,) float64 absolute cell indices
    ys = np.arange(y0, y1, dtype=np.float64)
    xs = np.arange(x0, x1, dtype=np.float64)
    v = float((wts.sum(axis=1) * ys).sum())
    u = float((wts.sum(axis=0) * xs).sum())
    return u, v


def heatmap_to_model(u: float, v: float) -> tuple[float, float]:
    """Heatmap px (u, v) -> model-input px (x, y) = (2u + 0.5, 2v + 0.5)."""
    return 2.0 * u + 0.5, 2.0 * v + 0.5


def decode_logits(logits: np.ndarray, win: int = DECODE_WIN) -> tuple[float, float, int, int]:
    """Numpy port of train_v9_tip.decode_logits for one map.

    logits: (128, 128) float logits. Returns (x_model, y_model, iy, ix):
    x/y float model-input px of the tip, iy/ix int argmax cell (first one in row-major order on ties).
    """
    h, w = logits.shape
    flat = int(np.argmax(logits))  # int: row-major index of the largest logit
    iy, ix = divmod(flat, w)
    u, v = _window_centroid(logits, iy, ix, win)
    x, y = heatmap_to_model(u, v)
    return x, y, iy, ix


def window_mass(probs: np.ndarray, iy: int, ix: int, win: int = DECODE_WIN) -> float:
    """Sum of probabilities in the win x win window around cell (iy, ix), clipped to the map -> float in [0, 1]."""
    r = win // 2
    return float(probs[max(0, iy - r): iy + r + 1, max(0, ix - r): ix + r + 1].sum())


def local_maxima(probs: np.ndarray, neighbourhood: int = PEAK_NEIGHBOURHOOD) -> list[tuple[int, int]]:
    """Cells that hold the largest value of their neighbourhood x neighbourhood window (window clipped to the map).

    probs: (h, w) float. Returns [(iy, ix), ...] in row-major order. Plateaus of equal values
    yield every tied cell; find_candidates suppresses those duplicates.
    """
    r = neighbourhood // 2
    padded = np.pad(probs, r, mode="constant", constant_values=-np.inf)
    # windows: (h, w, n, n) strided view; window_max: (h, w) neighbourhood maxima
    windows = np.lib.stride_tricks.sliding_window_view(padded, (neighbourhood, neighbourhood))
    window_max = windows.max(axis=(2, 3))
    iy, ix = np.nonzero(probs >= window_max)
    return list(zip(iy.tolist(), ix.tolist()))


def find_candidates(
    logits: np.ndarray,
    probs: np.ndarray,
    *,
    max_count: int = MAX_CANDIDATES,
    min_mass: float = MIN_CANDIDATE_MASS,
) -> list[dict[str, float | int]]:
    """Look-alike peaks of the probability map, the decoded tip first.

    logits: (128, 128) float; probs: (128, 128) float64 = softmax_map(logits).
    Returns up to max_count dicts {"x_model", "y_model": float model px, "mass": float,
    "peak": float max prob, "cell_x", "cell_y": int heatmap cell}, each a local maximum
    (largest value in its 9x9 neighbourhood) with 5x5 mass >= min_mass, sorted by mass, largest
    first. The argmax peak is the decoded tip and always comes first (it is also, in practice,
    the heaviest); its coordinates are exactly decode_logits'. Every other peak is decoded the
    same way (softmax-weighted centroid of the 5x5 logits around it).
    """
    _, _, top_y, top_x = decode_logits(logits)
    # peaks: list of (mass, iy, ix) for every local maximum with enough mass
    peaks = [(window_mass(probs, iy, ix), iy, ix) for iy, ix in local_maxima(probs)]
    peaks = [p for p in peaks if p[0] >= min_mass or (p[1], p[2]) == (top_y, top_x)]
    # The argmax cell first, then the rest by mass (largest first); ties keep row-major order.
    peaks.sort(key=lambda p: ((p[1], p[2]) != (top_y, top_x), -p[0]))
    r = PEAK_NEIGHBOURHOOD // 2
    kept: list[tuple[float, int, int]] = []  # accepted (mass, iy, ix)
    for mass, iy, ix in peaks:
        # Two distinct local maxima can only sit within r cells of each other when they tie
        # (each is in the other's window); report such a plateau once.
        if any(max(abs(iy - ky), abs(ix - kx)) <= r for _, ky, kx in kept):
            continue
        kept.append((mass, iy, ix))
        if len(kept) == max_count:
            break
    out = []
    for mass, iy, ix in kept:
        u, v = _window_centroid(logits, iy, ix)
        x, y = heatmap_to_model(u, v)
        out.append({"x_model": x, "y_model": y, "mass": mass, "peak": float(probs[iy, ix]),
                    "cell_x": ix, "cell_y": iy})
    return out


# ── model ────────────────────────────────────────────────────────────────────────────────────────


def _interpreter_class():
    """The LiteRT Interpreter class (ai_edge_litert), falling back to tensorflow.lite; -> (cls, runtime name str)."""
    try:
        from ai_edge_litert.interpreter import Interpreter  # type: ignore[import-not-found]

        try:
            from importlib.metadata import version

            return Interpreter, f"ai-edge-litert {version('ai-edge-litert')}"
        except Exception:  # noqa: BLE001 - version lookup is cosmetic
            return Interpreter, "ai-edge-litert"
    except ImportError:
        import tensorflow as tf  # type: ignore[import-not-found]

        return tf.lite.Interpreter, f"tensorflow.lite {tf.__version__}"


class TipModel:
    """One TFLite interpreter for the whole process, safe to call from several threads.

    path:          Path, the .tflite file
    file_bytes:    int, its size on disk
    num_threads:   int, CPU threads the interpreter may use = min(4, os.cpu_count())
    runtime:       str, e.g. "ai-edge-litert 2.1.2"
    input_shape:   list[int], e.g. [1, 256, 256, 1] (NHWC), read from the interpreter
    output_shape:  list[int], e.g. [1, 128, 128, 1] (NHWC)
    input_dtype / output_dtype: str, numpy dtype names, e.g. "float32"
    """

    def __init__(self, path: Path | str, num_threads: int | None = None) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"TFLite model not found: {self.path}")
        self.file_bytes = self.path.stat().st_size
        self.num_threads = num_threads or min(4, os.cpu_count() or 1)
        cls, self.runtime = _interpreter_class()
        # _interp: the LiteRT Interpreter; tensors are allocated once, then reused per call
        self._interp = cls(model_path=str(self.path), num_threads=self.num_threads)
        self._interp.allocate_tensors()
        # _lock: threading.Lock, serialises set_tensor / invoke / get_tensor (the interpreter is not re-entrant)
        self._lock = threading.Lock()
        inp = self._interp.get_input_details()[0]
        out = self._interp.get_output_details()[0]
        self._in_index, self._out_index = inp["index"], out["index"]  # int tensor indices
        self.input_shape = [int(d) for d in inp["shape"]]
        self.output_shape = [int(d) for d in out["shape"]]
        self.input_dtype = np.dtype(inp["dtype"]).name
        self.output_dtype = np.dtype(out["dtype"]).name
        if self.input_shape != [1, MODEL_SIZE, MODEL_SIZE, 1] or self.output_shape != [1, HEATMAP_SIZE, HEATMAP_SIZE, 1]:
            raise ValueError(
                f"Unexpected model I/O: input {self.input_shape}, output {self.output_shape}; "
                f"expected [1, {MODEL_SIZE}, {MODEL_SIZE}, 1] -> [1, {HEATMAP_SIZE}, {HEATMAP_SIZE}, 1]."
            )

    def run(self, model_gray: np.ndarray) -> np.ndarray:
        """model_gray: (256, 256) uint8 -> logits: (128, 128) float32 (a private copy)."""
        x = (model_gray.astype(np.float32) / 255.0)[None, :, :, None]  # (1, 256, 256, 1) float32 in [0, 1]
        with self._lock:
            self._interp.set_tensor(self._in_index, x)
            self._interp.invoke()
            out = self._interp.get_tensor(self._out_index)  # (1, 128, 128, 1) float32
            return np.array(out[0, :, :, 0], dtype=np.float32, copy=True)

    def describe(self) -> dict[str, object]:
        """Interpreter facts for /api/model -> dict of JSON-ready values."""
        return {
            "runtime": self.runtime,
            "num_threads": self.num_threads,
            "input_shape": self.input_shape,
            "input_dtype": self.input_dtype,
            "output_shape": self.output_shape,
            "output_dtype": self.output_dtype,
        }


def load_card(model_path: Path) -> dict[str, object]:
    """The model card for model_path: <its folder>/model_card.json if present, else the shipped card -> dict."""
    for candidate in (model_path.parent / "model_card.json", DEFAULT_CARD_PATH):
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    return {"name": model_path.stem}


# ── full prediction ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Prediction:
    """Everything one /api/predict call returns, before JSON encoding.

    image_size: (int, int), uploaded image (W, H) after EXIF rotation
    box:        Box, the square actually used
    model_gray: (256, 256) uint8, the picture the model saw
    logits:     (128, 128) float32, raw model output
    probs:      (128, 128) float64, softmax over all cells (sums to 1)
    tip_model:  (float, float), decoded tip in model-input px
    peak_cell:  (int, int), argmax cell (ix, iy) in heatmap px
    confidence: float, probability mass in the 5x5 decode window
    peak_prob:  float, max probability
    candidates: list[dict], from find_candidates (model px), image px added by to_json
    timing_ms:  dict[str, float], {"preprocess", "inference", "total"}
    """

    image_size: tuple[int, int]
    box: Box
    model_gray: np.ndarray
    logits: np.ndarray
    probs: np.ndarray
    tip_model: tuple[float, float]
    peak_cell: tuple[int, int]
    confidence: float
    peak_prob: float
    candidates: list[dict[str, float | int]]
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def tip(self) -> tuple[float, float]:
        """Decoded tip in uploaded-image px -> (x, y) floats."""
        return model_to_image(*self.tip_model, self.box)

    def to_json(self) -> dict[str, object]:
        """The /api/predict response body (see README "Response")."""
        buf = io.BytesIO()
        Image.fromarray(self.model_gray).save(buf, format="PNG")  # (256, 256) uint8 -> mode "L"
        tip_x, tip_y = self.tip
        candidates = []
        for c in self.candidates:
            cx, cy = model_to_image(float(c["x_model"]), float(c["y_model"]), self.box)
            candidates.append({"x": cx, "y": cy, "x_model": c["x_model"], "y_model": c["y_model"],
                               "mass": c["mass"], "peak": c["peak"],
                               "cell": {"x": c["cell_x"], "y": c["cell_y"]}})
        return {
            "image": {"width": self.image_size[0], "height": self.image_size[1]},
            "box": self.box.to_json(),
            "tip": {"x": tip_x, "y": tip_y},
            "tip_model": {"x": self.tip_model[0], "y": self.tip_model[1]},
            "peak_cell": {"x": self.peak_cell[0], "y": self.peak_cell[1]},
            "confidence": self.confidence,
            "peak_prob": self.peak_prob,
            "candidates": candidates,
            "heatmap": {
                "width": HEATMAP_SIZE,
                "height": HEATMAP_SIZE,
                "dtype": "float32",
                "encoding": "base64-le",
                "data": base64.b64encode(self.probs.astype("<f4").tobytes()).decode("ascii"),
            },
            "model_input_png": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii"),
            "timing_ms": self.timing_ms,
        }


def predict(
    model: TipModel,
    data: bytes,
    box_text: str | None = None,
    *,
    max_bytes: int = MAX_BYTES,
    max_pixels: int = MAX_PIXELS,
) -> Prediction:
    """Run the whole pipeline on one upload.

    model: TipModel; data: bytes of the image file; box_text: str | None "left,top,size".
    Raises InvalidImageError (400), ImageTooLargeError (413), InvalidBoxError (422).
    """
    t0 = time.perf_counter()
    img = load_image(data, max_bytes=max_bytes, max_pixels=max_pixels)
    box = parse_box(box_text, img.width, img.height)
    model_gray = preprocess(img, box)
    t1 = time.perf_counter()
    logits = model.run(model_gray)
    t2 = time.perf_counter()
    probs = softmax_map(logits)
    x_model, y_model, iy, ix = decode_logits(logits)
    pred = Prediction(
        image_size=(img.width, img.height),
        box=box,
        model_gray=model_gray,
        logits=logits,
        probs=probs,
        tip_model=(x_model, y_model),
        peak_cell=(ix, iy),
        confidence=window_mass(probs, iy, ix),
        peak_prob=float(probs[iy, ix]),
        candidates=find_candidates(logits, probs),
    )
    pred.timing_ms = {
        "preprocess": round((t1 - t0) * 1000.0, 2),
        "inference": round((t2 - t1) * 1000.0, 2),
        "total": round((time.perf_counter() - t0) * 1000.0, 2),
    }
    return pred
