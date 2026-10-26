# TipNet v9 (TFLite): integration guide

## 1. What it is

`tipnet_v9.tflite` finds the **tip of the pointer** of an analog dial in a 256 x 256 gray picture.
It does not read the dial value: it returns one sub-pixel (x, y) point plus a heat map that tells
you how far to trust it. It is a small fully convolutional net (one logit per 2 x 2 pixel cell);
the tip is decoded from that map with ~10 lines of numpy.

| Item | Value |
|---|---|
| File | `tip_webapp/model/tipnet_v9.tflite`, 416,408 bytes, float32, 100,001 parameters |
| Input | `input` (tensor `serving_default_input:0`): float32 NHWC `(1, 256, 256, 1)`, gray / 255 in [0, 1] |
| Output | `logits` (tensor `PartitionedCall:0`): float32 NHWC `(1, 128, 128, 1)`, raw logits |
| Signature | `serving_default`: inputs `['input']`, outputs `['logits']` |
| Accuracy | synthetic val, 22,500 images: median 0.415 px, p95 1.547 px, **p99 3.073 px**, max 140.1 px |
| Speed | desktop, Python, batch 1, 4 threads: ~27 ms per image with LiteRT (on-device speed is unmeasured) |
| Same facts as JSON | `tip_webapp/model/model_card.json` |

## 2. What to copy and install

Copy `tipnet_v9.tflite` into your project. For Python:

```bash
pip install ai-edge-litert numpy pillow     # tested with ai-edge-litert 2.1.2
```

`tflite-runtime` or `tensorflow` (`tf.lite.Interpreter`) work too, and so does any TFLite/LiteRT
runtime in another language (Android, iOS, C++). There is one input and one output: address them by
position, not by name.

## 3. Input

Float32 `(1, 256, 256, 1)`, NHWC, value = gray / 255 in [0, 1]. No mean/std, no ImageNet stats.
To build it from a photo:

1. **Apply the EXIF orientation** (`ImageOps.exif_transpose`) so your coordinates are in the upright picture.
2. **Crop a square** around the dial. The dial face should be about 60-73 % of the crop side, near
   the centre (training: centre within 10 px of the crop centre, side = 256).
3. **Convert to gray** with ITU-R 601 luma (`L = 0.299 R + 0.587 G + 0.114 B`, Pillow `convert("L")`).
4. **Resize to 256 x 256** with Lanczos or area filtering (a 256-px crop needs no resize).
5. **Divide by 255**, add the batch and channel axes.

```
 square crop, side S                  dial face diameter = 0.60..0.73 * S
 +------------------------+           bezel may touch or leave the frame
 |       .--------.       |           mirrored / 90-degree-rotated dials are fine
 |     /            \     |           (trained with the 8 flips/rotations)
 |    |       o---> * tip |
 |     \            /     |
 |       '--------'       |
 +------------------------+
```

## 4. Output

`(1, 128, 128, 1)` float32 **logits**: not probabilities, not sigmoid outputs. The tip probability
of a cell is the **softmax over all 16,384 cells jointly** (the map sums to 1); training used a
spatial-softmax cross-entropy against a 1-cell Gaussian. Per-cell thresholds or a sigmoid are wrong.
Cell (row v, col u) covers model-input pixels x in {2u, 2u+1}, y in {2v, 2v+1}, so its centre is at
x = 2u + 0.5 (pixel i is centred at coordinate i, edges at i -/+ 0.5).

## 5. Decode and map back

1. `(iy, ix)` = argmax cell of the 128 x 128 logits (ties: first in row-major order).
2. Softmax the logits in the 5 x 5 window centred there (cells off the map are ignored, not padded).
3. Centroid in cell units: `u = sum(w * col)`, `v = sum(w * row)`.
4. Model px: `x = 2u + 0.5`, `y = 2v + 0.5`.
5. Your image px, for a crop `left, top, size`: `X = left + (x + 0.5) * size / 256 - 0.5` (same for Y with `top`).

```
logits (128,128) --argmax--> cell (iy,ix) --5x5 softmax centroid--> (u,v) cell units
   --x=2u+0.5, y=2v+0.5--> (x,y) model px in [-0.5,255.5] --X=left+(x+0.5)*size/256-0.5--> (X,Y) your px
```

Inverse: `x = (X - left + 0.5) * 256 / size - 0.5`. Worked example (shipped sample 1, run through the
block below): `tip_model = (203.988, 116.177)`, label `(203.71, 115.26)`, error 0.958 px. Pasted at
(137, 61) into a 640 x 480 canvas and cropped with box `(137, 61, 256)` it gives
`(340.988, 177.177)`, i.e. the same tip plus the offset. With box `(100, 40, 300)` the same model
point maps to `(339.13, 176.23)`.

## 6. Drop-in Python (`tipnet.py`)

```python
"""tipnet.py - drop-in wrapper for tipnet_v9.tflite. Needs: numpy, Pillow, ai-edge-litert."""
import threading

import numpy as np
from PIL import Image, ImageOps

try:  # any of the three runtimes exposes the same Interpreter API
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite import Interpreter


class TipNet:
    def __init__(self, path="tipnet_v9.tflite", num_threads=4):
        self._it = Interpreter(model_path=str(path), num_threads=num_threads)
        self._it.allocate_tensors()
        self._in = self._it.get_input_details()[0]["index"]    # int, tensor id of (1,256,256,1) f32
        self._out = self._it.get_output_details()[0]["index"]  # int, tensor id of (1,128,128,1) f32
        self._lock = threading.Lock()  # an Interpreter is not re-entrant: one call at a time

    def predict(self, img, box=None):
        """img: PIL.Image or path. box: (left, top, size) square in img px, or None = centred square.
        Returns {"tip": (X, Y) image px, "tip_model": (x, y) 256-px model px, "confidence": float,
                 "runner_up": {"tip": (X, Y), "ratio": float} | None, "ambiguous": bool}."""
        img = ImageOps.exif_transpose(img if isinstance(img, Image.Image) else Image.open(img))
        if box is None:
            s = min(img.size)
            box = ((img.width - s) // 2, (img.height - s) // 2, s)
        left, top, size = box
        gray = img.crop((left, top, left + size, top + size)).convert("L")  # PIL "L" = 8-bit, 601 luma
        if size != 256:
            gray = gray.resize((256, 256), Image.LANCZOS)
        x = (np.asarray(gray, np.float32) / 255.0)[None, :, :, None]  # float32 (1,256,256,1), [0,1]
        with self._lock:
            self._it.set_tensor(self._in, x)
            self._it.invoke()
            logits = self._it.get_tensor(self._out)[0, :, :, 0].astype(np.float64)  # (128,128) raw logits
        e = np.exp(logits - logits.max())
        probs = e / e.sum()  # (128,128) float64, softmax over ALL cells, sums to 1

        def centroid(iy, ix):  # 5x5 softmax centroid around cell (iy, ix); off-map cells ignored
            y0, y1, x0, x1 = max(iy - 2, 0), min(iy + 3, 128), max(ix - 2, 0), min(ix + 3, 128)
            w = np.exp(logits[y0:y1, x0:x1] - logits[y0:y1, x0:x1].max())
            w /= w.sum()
            v, u = (w.sum(1) * np.arange(y0, y1)).sum(), (w.sum(0) * np.arange(x0, x1)).sum()
            return 2 * u + 0.5, 2 * v + 0.5  # cell -> model px: x = 2u + 0.5

        def to_image(p):  # model px -> image px: X = left + (x + 0.5) * size / 256 - 0.5
            return tuple(o + (c + 0.5) * size / 256 - 0.5 for o, c in zip((left, top), p))

        mass = lambda iy, ix: probs[max(iy - 2, 0):iy + 3, max(ix - 2, 0):ix + 3].sum()  # 5x5 mass
        iy, ix = np.unravel_index(np.argmax(logits), logits.shape)  # first max in row-major order
        conf = float(mass(iy, ix))
        # Runner-up = heaviest other local maximum (largest in its 9x9 area, >4 cells from the winner).
        nb = np.lib.stride_tricks.sliding_window_view(np.pad(probs, 4, constant_values=-1), (9, 9)).max((2, 3))
        others = [(mass(a, b), a, b) for a, b in np.argwhere(probs >= nb) if max(abs(a - iy), abs(b - ix)) > 4]
        ru = max(others, default=None)
        ru = ru if ru and ru[0] >= 1e-3 else None  # ignore peaks with < 0.1 % mass
        ratio = float(ru[0] / conf) if ru else 0.0
        tip_model = centroid(iy, ix)
        return {"tip": to_image(tip_model), "tip_model": tip_model, "confidence": conf,
                "runner_up": {"tip": to_image(centroid(ru[1], ru[2])), "ratio": ratio} if ru else None,
                "ambiguous": ratio > 0.2 or conf < 0.5}
```

Usage: `r = TipNet("tipnet_v9.tflite").predict("photo.jpg", box=(137, 61, 256))`, then read
`r["tip"]` (your px), `r["confidence"]`, `r["ambiguous"]`. Checked against the six shipped samples
(`tip_webapp/samples/`): errors 0.958 / 0.072 / 0.201 / 0.092 / 0.212 / 140.1 px, identical to
`tip_webapp/app/inference.py` (the tested reference). It assumes 8-bit images; 16-bit or float
images must be scaled to 0-255 first (`convert("L")` would clip them), see `_to_gray` there.

## 7. Trust signals

- **`confidence`** = softmax mass inside the 5 x 5 decode window. On val: 1 % of images are below
  0.881, 5 % below 0.935, median 0.970.
- **`runner_up["ratio"]`** = mass of the second-best look-alike peak / mass of the winner (0 if there
  is none). A peak is the largest cell in its 9 x 9 neighbourhood with 5 x 5 mass >= 0.001.
- **`ambiguous`** = `ratio > 0.2 or confidence < 0.5`. It flags 7 of 22,500 images (0.03 %) and
  catches 4 of the 13 misses (> 7 px), including every miss larger than 20.7 px (the largest
  unflagged error is 20.68 px). p99 of unflagged images 3.048 px; of flagged ones 135.4 px.

| Rule | Images flagged | Of the 13 misses caught |
|---|---|---|
| confidence < 0.5 | 0.01 % | 1 |
| confidence < 0.7 | 0.05 % | 3 |
| confidence < 0.8 | 0.23 % | 5 |
| confidence < 0.9 | 1.63 % | 10 |

Suggested use: accept silently if not `ambiguous` and `confidence >= 0.9`; otherwise show
`runner_up["tip"]` or ask for a retake. Sample 6 (confidence 0.788, ratio 0.237) is `ambiguous`; its
runner-up (70.73, 180.89) is 0.23 px from the true tip (70.96, 180.94). Some misses are confident
single-peak errors of 7-21 px, mostly along the pointer; no signal here catches those.

## 8. Pitfalls

| Mistake | Symptom | Fix |
|---|---|---|
| NCHW `(1,1,256,256)` input | shape error, or garbage | NHWC `(1,256,256,1)` |
| 0-255 values | tip 25-193 px off on all 6 shipped samples | divide by 255 |
| ImageNet mean/std | tip 20-166 px off on 5 of the 6 samples | none: plain gray / 255 |
| sigmoid / per-cell threshold on logits | many "tips" | softmax over the whole map, then decode |
| `x = 2u` (missing +0.5) | constant 0.5 px bias per axis (0.7 px diagonal) vs 0.415 px median | `x = 2u + 0.5` |
| non-square crop, or squashing the photo | dial distorted, big errors | always crop a square |
| crop far larger than the dial | pointer too thin / small, out of training range | dial face 60-73 % of the side |
| EXIF orientation ignored | tip is in the rotated frame, not the one you display | `exif_transpose` first |
| 16-bit / float images through `convert("L")` | clipped to white | scale to 0-255 first |
| one interpreter shared across threads | corrupt output or crash | a lock (as above) or one interpreter per thread |

## 9. Limits and regeneration

- **Synthetic validation only.** Trained on `sv9_150k` (150,000 synthetic gray dials; 127,500 train /
  22,500 val): any pointer angle and gray (contrast >= 35 levels), shadows, glares, 0-6 decoy
  triangles, cut tips, noise, jitter. **No real-photo accuracy is known**: measure on your own
  labelled photos before trusting the numbers above.
- **Failure modes:** a look-alike wins (the far end of a needle-shaped pointer, a decoy, a shadow;
  worst val image 140.1 px), and confident 7-21 px slips along the pointer. 13 of 22,500 val images
  are over 7 px. There is no "no dial here" output; the model always returns one tip.
- **float16:** `model_float16.tflite` (232,480 bytes) exists only in the training runs folder, not
  here. Its logits drift up to 0.11 from float32 and it was ~2x slower on a desktop CPU with
  LiteRT; mobile GPU delegates were not measured.
- **Provenance:** `train_v9_tip.py` (run `v9_tipnet_a`, best-p99 weights, epoch 50) -> `model.onnx`
  (opset 17) -> `export_v9_tflite.py` (onnx2tf 2.3.8) -> this file.
  `python export_v9_tflite.py --check all` re-verifies it against ONNX (max logit difference 1.34e-4,
  max decoded-tip difference 0.0000 px) and writes `<out stem>_check.json`.
- **Try it:** `python -m tip_webapp` serves the same pipeline with a heat-map view
  (`tip_webapp/app/inference.py` is the reference implementation).
