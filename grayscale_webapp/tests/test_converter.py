"""Unit tests for grayscale_webapp.app.converter (pure conversion logic)."""

from __future__ import annotations

import io
import zlib

import numpy as np
import pytest
from PIL import Image

from grayscale_webapp.app import converter
from grayscale_webapp.app.converter import convert_to_grayscale

# RGB_2X2: (2, 2, 3) uint8, four known colours: red, green / blue, a mixed colour
RGB_2X2 = np.array(
    [[[255, 0, 0], [0, 255, 0]],
     [[0, 0, 255], [200, 100, 50]]],
    dtype=np.uint8,
)


def encode(img: Image.Image, fmt: str = "PNG", **kwargs) -> bytes:
    """Encode a Pillow image (img) to bytes in format fmt."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, **kwargs)
    return buf.getvalue()


def decode(data: bytes) -> Image.Image:
    """Decode bytes into a fully loaded Pillow image."""
    img = Image.open(io.BytesIO(data))
    img.load()
    return img


def round_half_up(values: np.ndarray) -> np.ndarray:
    """Reference rounding used by the spec: floor(x + 0.5), clipped to uint8."""
    return np.clip(np.floor(values + 0.5), 0, 255).astype(np.uint8)


@pytest.fixture
def png_2x2() -> bytes:
    """PNG bytes of RGB_2X2."""
    return encode(Image.fromarray(RGB_2X2))


def test_luma601_matches_pil_convert_L(png_2x2):
    result = convert_to_grayscale(png_2x2, "luma601", "png")
    expected = np.asarray(Image.fromarray(RGB_2X2).convert("L"))
    out = decode(result.data)
    assert out.mode == "L"
    assert np.array_equal(np.asarray(out), expected)
    assert (result.width, result.height, result.method) == (2, 2, "luma601")
    assert result.media_type == "image/png" and result.extension == "png"


def test_luma601_matches_pil_on_random_image():
    # rgb: (32, 48, 3) uint8 random image covering many values
    rgb = np.random.default_rng(0).integers(0, 256, size=(32, 48, 3), dtype=np.uint8)
    out = decode(convert_to_grayscale(encode(Image.fromarray(rgb)), "luma601", "png").data)
    assert np.array_equal(np.asarray(out), np.asarray(Image.fromarray(rgb).convert("L")))


@pytest.mark.parametrize(
    "method, formula",
    [
        ("luma709", lambda f: f @ np.array([0.2126, 0.7152, 0.0722])),
        ("average", lambda f: f.sum(axis=2) / 3.0),
        ("lightness", lambda f: (f.max(axis=2) + f.min(axis=2)) / 2.0),
    ],
)
def test_numpy_methods(png_2x2, method, formula):
    out = decode(convert_to_grayscale(png_2x2, method, "png").data)
    expected = round_half_up(formula(RGB_2X2.astype(np.float64)))
    assert np.array_equal(np.asarray(out), expected)


def test_known_values_by_hand(png_2x2):
    # average of (200, 100, 50) = 116.67 -> 117 ; lightness = (200 + 50) / 2 = 125 ; pure red average = 85
    avg = np.asarray(decode(convert_to_grayscale(png_2x2, "average", "png").data))
    light = np.asarray(decode(convert_to_grayscale(png_2x2, "lightness", "png").data))
    assert avg[1, 1] == 117 and avg[0, 0] == 85
    assert light[1, 1] == 125 and light[0, 0] == 128  # (255 + 0) / 2 = 127.5 rounds half up


def test_alpha_preserved_as_LA_for_png_and_webp():
    # rgba: (2, 2, 4) uint8, varying alpha
    rgba = np.dstack([RGB_2X2, np.array([[255, 128], [0, 64]], dtype=np.uint8)])
    data = encode(Image.fromarray(rgba))
    out = decode(convert_to_grayscale(data, "luma601", "same").data)
    assert out.mode == "LA"
    assert np.array_equal(np.asarray(out)[..., 1], rgba[..., 3])
    assert np.array_equal(np.asarray(out)[..., 0], np.asarray(Image.fromarray(RGB_2X2).convert("L")))

    webp = decode(convert_to_grayscale(data, "luma601", "webp").data)
    assert webp.format == "WEBP"
    # Pillow stores LA as RGBA in WebP; alpha must survive and the colour channels must be equal (gray)
    arr = np.asarray(webp.convert("RGBA"))
    assert np.array_equal(arr[..., 3], rgba[..., 3])
    visible = arr[..., 3] > 0
    assert np.array_equal(arr[..., 0][visible], arr[..., 1][visible])


def test_jpeg_flattens_alpha_onto_white():
    # Fully transparent black pixels must become white; opaque black stays black.
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[:, :4, 3] = 255  # left half opaque black, right half transparent
    result = convert_to_grayscale(encode(Image.fromarray(rgba)), "luma601", "jpeg")
    out = decode(result.data)
    assert out.format == "JPEG" and out.mode == "L"
    assert result.media_type == "image/jpeg" and result.extension == "jpg"
    arr = np.asarray(out).astype(int)
    assert arr[:, :3].max() < 10
    assert arr[:, 5:].min() > 245


def test_exif_orientation_honoured():
    # A 4x2 (W x H) image tagged Orientation=6 (rotate 90 CW) must come out 2x4.
    img = Image.new("RGB", (4, 2), (255, 0, 0))
    img.putpixel((0, 0), (0, 0, 255))
    exif = Image.Exif()
    exif[0x0112] = 6
    data = encode(img, "JPEG", exif=exif.tobytes(), quality=100)
    result = convert_to_grayscale(data, "luma601", "png")
    assert (result.width, result.height) == (2, 4)
    assert decode(result.data).size == (2, 4)


def test_same_format_keeps_png_jpeg_webp():
    img = Image.fromarray(RGB_2X2)
    for fmt, ext in (("PNG", "png"), ("JPEG", "jpg"), ("WEBP", "webp")):
        result = convert_to_grayscale(encode(img, fmt), "luma601", "same")
        assert result.extension == ext
        assert decode(result.data).format == fmt


@pytest.mark.parametrize("fmt", ["GIF", "BMP", "TIFF"])
def test_other_formats_fall_back_to_png(fmt):
    result = convert_to_grayscale(encode(Image.fromarray(RGB_2X2), fmt), "luma601", "same")
    assert result.extension == "png" and result.media_type == "image/png"
    assert decode(result.data).format == "PNG"


def test_animated_gif_uses_first_frame():
    frames = [Image.new("RGB", (4, 4), c) for c in ((255, 255, 255), (0, 0, 0))]
    buf = io.BytesIO()
    frames[0].save(buf, "GIF", save_all=True, append_images=frames[1:], duration=100, loop=0)
    out = np.asarray(decode(convert_to_grayscale(buf.getvalue(), "luma601", "same").data))
    assert out.min() == 255


def test_palette_cmyk_and_16bit_inputs():
    pal = Image.fromarray(RGB_2X2).convert("P")
    assert decode(convert_to_grayscale(encode(pal), "average", "png").data).mode == "L"

    cmyk = Image.fromarray(RGB_2X2).convert("CMYK")
    assert decode(convert_to_grayscale(encode(cmyk, "JPEG"), "luma601", "same").data).mode == "L"

    # sixteen: (2, 2) uint16 grayscale spanning the full 16-bit range
    sixteen = np.array([[0, 65535], [32896, 257]], dtype=np.uint16)
    out = np.asarray(decode(convert_to_grayscale(encode(Image.fromarray(sixteen)), "luma601", "png").data))
    assert out.tolist() == [[0, 255], [128, 1]]


def test_corrupt_and_non_image_bytes_raise():
    with pytest.raises(converter.InvalidImageError):
        convert_to_grayscale(b"hello, this is plain text", "luma601", "png")
    truncated = encode(Image.fromarray(np.full((64, 64, 3), 7, np.uint8)), "PNG")[:60]
    with pytest.raises(converter.InvalidImageError):
        convert_to_grayscale(truncated, "luma601", "png")
    with pytest.raises(converter.InvalidImageError):
        convert_to_grayscale(b"", "luma601", "png")


def test_limits_and_bad_arguments(png_2x2):
    with pytest.raises(converter.ImageTooLargeError):
        convert_to_grayscale(png_2x2, max_bytes=10)
    with pytest.raises(converter.ImageTooLargeError):
        convert_to_grayscale(png_2x2, max_pixels=3)
    with pytest.raises(converter.UnknownMethodError):
        convert_to_grayscale(png_2x2, "sepia")
    with pytest.raises(converter.UnknownFormatError):
        convert_to_grayscale(png_2x2, "luma601", "gif")


def test_decompression_bomb_header_rejected():
    # A tiny PNG whose IHDR claims 20000 x 20000 = 400 MP must be refused before any pixel is decoded.
    data = bytearray(encode(Image.new("L", (1, 1))))
    # IHDR chunk layout: 8-byte signature, 4-byte length, b"IHDR" at 12..16, width 16..20, height 20..24, CRC 29..33
    data[16:24] = (20000).to_bytes(4, "big") * 2
    data[29:33] = zlib.crc32(bytes(data[12:29])).to_bytes(4, "big")
    with pytest.raises(converter.ImageTooLargeError):
        convert_to_grayscale(data, "luma601", "png")
