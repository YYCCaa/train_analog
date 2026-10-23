"""Pure image -> grayscale conversion logic (no web code).

Pipeline for one upload:

    bytes --open/guard--> PIL.Image --first frame, EXIF upright--> RGB(A)
          --method--> gray (H, W) uint8 [+ alpha (H, W) uint8]
          --encode (PNG / JPEG / WebP)--> bytes

The single public entry point is :func:`convert_to_grayscale`; it raises
subclasses of :class:`ConversionError` that the web layer maps to HTTP codes.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

# MAX_BYTES: int, largest accepted upload in bytes (20 MB).
MAX_BYTES = 20 * 1024 * 1024
# MAX_PIXELS: int, largest accepted width*height (50 megapixels); guards decompression bombs.
MAX_PIXELS = 50_000_000
# JPEG_QUALITY: int, quality used when encoding JPEG output.
JPEG_QUALITY = 95

# Make Pillow's own bomb check agree with ours (it warns above 1x and raises above 2x this value).
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ConversionError(Exception):
    """Base class for every expected conversion failure; str(exc) is a user-facing message."""


class InvalidImageError(ConversionError):
    """The bytes are not a decodable image, are corrupt, or use an unsupported pixel format (HTTP 400)."""


class ImageTooLargeError(ConversionError):
    """The upload exceeds the byte or pixel limit (HTTP 413)."""


class UnknownMethodError(ConversionError):
    """The requested grayscale method id does not exist (HTTP 422)."""


class UnknownFormatError(ConversionError):
    """The requested output format is not one of FORMATS (HTTP 422)."""


@dataclass(frozen=True)
class ConversionResult:
    """Encoded grayscale image plus metadata.

    data:       bytes, the encoded output file
    media_type: str, e.g. "image/png"
    extension:  str, file extension without dot, e.g. "png"
    width:      int, output width in pixels (after EXIF rotation)
    height:     int, output height in pixels (after EXIF rotation)
    method:     str, the grayscale method id that was applied
    """

    data: bytes
    media_type: str
    extension: str
    width: int
    height: int
    method: str


@dataclass(frozen=True)
class Method:
    """One grayscale method.

    id:          str, stable identifier used by the API (e.g. "luma601")
    label:       str, short human-readable name for dropdowns
    description: str, one-sentence explanation shown in the UI
    fn:          callable, (H, W, 3) uint8 RGB array -> (H, W) uint8 gray array
    """

    id: str
    label: str
    description: str
    fn: Callable[[np.ndarray], np.ndarray]


def _round_u8(values: np.ndarray) -> np.ndarray:
    """Round half-up and clip a float array to uint8. values: any-shape float array -> same-shape uint8."""
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


def _luma601(rgb: np.ndarray) -> np.ndarray:
    """ITU-R 601-2 luma, delegated to Pillow so it is bit-identical to Image.convert("L"). rgb: (H, W, 3) uint8."""
    return np.asarray(Image.fromarray(rgb).convert("L"))


def _luma709(rgb: np.ndarray) -> np.ndarray:
    """ITU-R BT.709 luma 0.2126R + 0.7152G + 0.0722B. rgb: (H, W, 3) uint8 -> (H, W) uint8."""
    # weights: (3,) float64 per-channel coefficients
    weights = np.array([0.2126, 0.7152, 0.0722])
    return _round_u8(rgb.astype(np.float64) @ weights)


def _average(rgb: np.ndarray) -> np.ndarray:
    """Plain channel mean (R + G + B) / 3. rgb: (H, W, 3) uint8 -> (H, W) uint8."""
    return _round_u8(rgb.astype(np.float64).sum(axis=2) / 3.0)


def _lightness(rgb: np.ndarray) -> np.ndarray:
    """HSL lightness (max + min) / 2. rgb: (H, W, 3) uint8 -> (H, W) uint8."""
    # rgb_i: (H, W, 3) int32 copy so max+min cannot overflow uint8
    rgb_i = rgb.astype(np.int32)
    return _round_u8((rgb_i.max(axis=2) + rgb_i.min(axis=2)) / 2.0)


# METHODS: dict[str, Method], every supported method keyed by id; insertion order = UI order.
METHODS: dict[str, Method] = {
    m.id: m
    for m in (
        Method("luma601", "Luma (BT.601)",
               "Standard perceptual weighting 0.299 R + 0.587 G + 0.114 B (same as most image editors).",
               _luma601),
        Method("luma709", "Luma (BT.709)",
               "HDTV / sRGB weighting 0.2126 R + 0.7152 G + 0.0722 B; greens look slightly brighter.",
               _luma709),
        Method("average", "Average",
               "Simple mean of the three channels (R + G + B) / 3.",
               _average),
        Method("lightness", "Lightness",
               "Midpoint of the brightest and darkest channel (max + min) / 2; flattens saturated colours.",
               _lightness),
    )
}

# DEFAULT_METHOD: str, method id used when none is given.
DEFAULT_METHOD = "luma601"

# FORMATS: dict[str, tuple[str, str, str]], output format id -> (Pillow format, media type, extension).
FORMATS: dict[str, tuple[str, str, str]] = {
    "png": ("PNG", "image/png", "png"),
    "jpeg": ("JPEG", "image/jpeg", "jpg"),
    "webp": ("WEBP", "image/webp", "webp"),
}
# FORMAT_CHOICES: tuple[str, ...], every value accepted for out_format ("same" + FORMATS keys).
FORMAT_CHOICES = ("same", *FORMATS)

# _PIL_TO_FORMAT_ID: dict[str, str], Pillow's Image.format name -> our format id, for out_format="same".
_PIL_TO_FORMAT_ID = {pil: fid for fid, (pil, _, _) in FORMATS.items()}


def _open_first_frame(data: bytes, max_pixels: int) -> tuple[Image.Image, str | None]:
    """Decode the first frame of an image with size guards.

    data: bytes of the upload; max_pixels: int pixel budget.
    Returns (image, source format name such as "PNG"/"GIF", or None).
    """
    try:
        with warnings.catch_warnings():
            # Promote Pillow's "possible decompression bomb" warning to an error we can catch.
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(data))
            # src_format: str | None, container format detected from the header
            src_format = img.format
            width, height = img.size
            if width * height > max_pixels:
                raise ImageTooLargeError(
                    f"Image is {width}x{height} ({width * height / 1e6:.1f} megapixels); "
                    f"the limit is {max_pixels / 1e6:g} megapixels."
                )
            img.seek(0)  # animated GIF/WebP/APNG: only the first frame is converted
            img.load()
    except ImageTooLargeError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ImageTooLargeError(f"Image has too many pixels (limit {max_pixels / 1e6:g} megapixels).") from exc
    except UnidentifiedImageError as exc:
        raise InvalidImageError("The uploaded file is not a recognised image format.") from exc
    except (OSError, SyntaxError, ValueError, EOFError, IndexError) as exc:
        raise InvalidImageError("The image file is corrupt or truncated and could not be decoded.") from exc
    return img, src_format


def _to_rgb_and_alpha(img: Image.Image) -> tuple[np.ndarray, np.ndarray | None]:
    """Normalise any Pillow mode to 8-bit RGB plus optional alpha.

    Returns (rgb: (H, W, 3) uint8, alpha: (H, W) uint8 or None when fully opaque / absent).
    """
    try:
        if img.mode in ("I", "I;16", "I;16L", "I;16B", "I;16N", "F"):
            # High-bit-depth single channel: scale 16-bit range down to 8 bits, then replicate to RGB.
            # wide: (H, W) float64 raw sample values
            wide = np.asarray(img, dtype=np.float64)
            if img.mode.startswith("I;16") or wide.max(initial=0) > 255:
                wide = wide * (255.0 / 65535.0)
            img = Image.fromarray(_round_u8(wide))

        if img.has_transparency_data:
            # rgba: (H, W, 4) uint8
            rgba = np.asarray(img.convert("RGBA"))
            alpha = rgba[..., 3]
            return np.ascontiguousarray(rgba[..., :3]), (None if alpha.min() == 255 else alpha.copy())
        return np.asarray(img.convert("RGB")), None
    except (ValueError, OSError) as exc:
        raise InvalidImageError(f"Unsupported image pixel format '{img.mode}'.") from exc


def _resolve_format(out_format: str, src_format: str | None) -> str:
    """Map the requested out_format ("same"/"png"/"jpeg"/"webp") to a FORMATS key; "same" falls back to png."""
    if out_format == "same":
        return _PIL_TO_FORMAT_ID.get(src_format or "", "png")
    return out_format


def convert_to_grayscale(
    data: bytes,
    method: str = DEFAULT_METHOD,
    out_format: str = "same",
    *,
    max_bytes: int = MAX_BYTES,
    max_pixels: int = MAX_PIXELS,
) -> ConversionResult:
    """Convert encoded image bytes to an encoded 8-bit grayscale image.

    data:       bytes, the uploaded file
    method:     str, a key of METHODS
    out_format: str, one of FORMAT_CHOICES
    max_bytes / max_pixels: int limits (overridable for tests)
    Raises InvalidImageError, ImageTooLargeError, UnknownMethodError, UnknownFormatError.
    """
    if method not in METHODS:
        raise UnknownMethodError(f"Unknown method '{method}'. Choose one of: {', '.join(METHODS)}.")
    if out_format not in FORMAT_CHOICES:
        raise UnknownFormatError(f"Unknown format '{out_format}'. Choose one of: {', '.join(FORMAT_CHOICES)}.")
    if len(data) > max_bytes:
        raise ImageTooLargeError(f"File is larger than the {max_bytes / (1024 * 1024):g} MB limit.")
    if not data:
        raise InvalidImageError("The uploaded file is empty.")

    img, src_format = _open_first_frame(data, max_pixels)
    try:
        img = ImageOps.exif_transpose(img)  # honour camera orientation so the output is upright
    except (ValueError, OSError, SyntaxError) as exc:
        raise InvalidImageError("The image's EXIF orientation data is corrupt.") from exc

    rgb, alpha = _to_rgb_and_alpha(img)
    # gray: (H, W) uint8 grayscale plane
    gray = METHODS[method].fn(rgb)

    format_id = _resolve_format(out_format, src_format)
    pil_format, media_type, extension = FORMATS[format_id]

    if alpha is not None and format_id == "jpeg":
        # JPEG has no alpha channel: composite onto a white background.
        a = alpha.astype(np.float64) / 255.0
        gray = _round_u8(gray * a + 255.0 * (1.0 - a))
        alpha = None

    out_img = Image.fromarray(gray)
    if alpha is not None:
        out_img = Image.merge("LA", (out_img, Image.fromarray(alpha)))

    # save_kwargs: dict[str, object], encoder options for the chosen format
    save_kwargs: dict[str, object] = {}
    if format_id == "jpeg":
        save_kwargs = {"quality": JPEG_QUALITY}
    elif format_id == "webp":
        save_kwargs = {"lossless": True}  # lossless keeps the gray values exact

    buf = io.BytesIO()
    out_img.save(buf, format=pil_format, **save_kwargs)
    return ConversionResult(
        data=buf.getvalue(),
        media_type=media_type,
        extension=extension,
        width=out_img.width,
        height=out_img.height,
        method=method,
    )
