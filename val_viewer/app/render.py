"""PNG rendering of dials and heatmaps.

    dial  (256,256) uint8 ---------------------------------> PNG mode L
    logits (128,128) --log:    t = 1 + (l - max) / (6 ln10) (6 decades below the max, clipped)
                     --linear: t = exp(l - max)             (probability / max probability)
                     --floor(255 t)--> COLORMAP (256,3) --> PNG RGB 128x128
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

# COLORMAP: (256, 3) uint8, one blue hue, dark navy (index 0) to light blue (index 255), luminance rising
COLORMAP = np.round(np.linspace([4, 6, 36], [200, 226, 255], 256)).astype(np.uint8)
# DECADES: float, log-scale range below the map maximum (powers of ten of probability)
DECADES = 6.0


def _png(arr) -> bytes:
    """uint8 ndarray (h, w) -> mode L PNG bytes, (h, w, 3) -> RGB PNG bytes."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def dial_png(img: np.ndarray) -> bytes:
    """(256, 256) uint8 -> PNG bytes, mode L."""
    return _png(np.ascontiguousarray(img, dtype=np.uint8))


def heatmap_png(logits: np.ndarray, scale: str) -> bytes:
    """(128, 128) float logits, scale 'log' | 'linear' -> PNG bytes, 128 x 128 RGB.

    The softmax maximum cell is always colour index 255; no other cell reaches it (floor)."""
    d = np.asarray(logits, dtype=np.float64)
    d = d - d.max()                                  # (128,128) <= 0, log of p / p_max
    if scale == "log":
        t = np.clip(1.0 + d / (DECADES * np.log(10.0)), 0.0, 1.0)  # (128,128) in [0, 1]
    elif scale == "linear":
        t = np.exp(d)                                # (128,128) in (0, 1]
    else:
        raise ValueError(f"unknown scale {scale!r}")
    idx = np.minimum(np.floor(t * 255.0), 255).astype(np.int64)
    idx[t >= 1.0] = 255                              # (128,128) colour indices
    return _png(COLORMAP[idx])
