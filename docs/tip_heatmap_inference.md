# Tip-Heatmap Inference Pipeline

Engineering documentation for `tip_heatmap_net.pth` and `infer.py`.

---

## 1. System overview

The system predicts the angular position of an analog dial meter's pointer from a photograph. It works in two stages:

1. A U-Net (`tip_heatmap_net.pth`) takes a 192x192 RGB image and produces a 192x192 single-channel heatmap whose peak indicates the pointer tip location.
2. The peak pixel coordinate is converted to a clockwise angle from the 12-o'clock (up) direction.

```
raw image  ──►  preprocess (crop/pad to 192x192)  ──►  U-Net  ──►  heatmap  ──►  argmax  ──►  angle
```

---

## 2. Model architecture

`TipHeatmapNet` is a lightweight U-Net defined in `train.py`.

```
Encoder                          Decoder
───────                          ───────
3  → 32   (Conv-BN-ReLU x2)     256+256 → 256  (skip-cat + Conv-BN-ReLU x2)
32 → 64      ↓ MaxPool 2x       128+128 → 128
64 → 128     ↓ MaxPool 2x        64+64  →  64
128→ 256     ↓ MaxPool 2x        32+32  →  32
             ↓ MaxPool 2x
         Bottleneck: 256 → 512    Head: 32 → 1 (1x1 conv + sigmoid)
```

- **Input**: `[B, 3, 192, 192]` float32, pixel values normalised to [0, 1].
- **Output**: `[B, 1, 192, 192]` float32, values in [0, 1]. The output is a probability-like map; the brightest pixel is the predicted tip.
- **Parameters**: ~7.8M.
- **Fully convolutional**: accepts any spatial size divisible by 16 (192 / 16 = 12). However, the model was trained on 192x192 and accuracy degrades at other sizes.

### Why U-Net for this task

Keypoint heatmap regression requires both global context (to understand which blob is the pointer) and precise localisation (sub-10px accuracy on the tip). The U-Net encoder captures global context through pooling; the skip connections restore spatial detail lost during downsampling. This combination outperforms encoder-only architectures (e.g. ResNet + deconv without skips) on precise single-point localisation tasks.

### Training target

The ground-truth heatmap is a 2-D Gaussian centred on the true tip coordinate:

```
heatmap[y, x] = exp( -((x - tip_x)^2 + (y - tip_y)^2) / (2 * sigma^2) )
```

`sigma = 5 px` at 192x192 resolution. This produces a blob approximately 30 px in diameter (6-sigma), which provides a smooth gradient signal for backpropagation while maintaining localisation precision. The heatmap is generated on-the-fly during training from `(tip_x, tip_y)` in `labels.csv`.

Loss function: MSE between predicted and target heatmaps.

---

## 3. Coordinate system

All three scripts (`generate_synthetic.py`, `train.py`, `infer.py`) share a single coordinate convention:

| Property | Value |
|---|---|
| Image size | 192 x 192 pixels |
| Origin | top-left corner, (0, 0) |
| x axis | increases rightward (column index) |
| y axis | increases downward (row index) |
| Centre pixel | `(96, 96)` — computed as `IMG_SIZE // 2` |
| Angle 0 deg | positive y direction in geometric space = upward = toward row 0 |
| Angle direction | clockwise: 0=up, 90=right, 180=down, 270=left |

### Angle recovery formula

Given the predicted tip at `(tip_x, tip_y)` in the 192x192 canvas:

```
dx = tip_x - 96
dy = -(tip_y - 96)       # flip y so that "up" is positive
angle = atan2(dx, dy)     # radians, CW from up
angle_deg = angle * 180 / pi  % 360
```

This is the exact inverse of the generator's tip placement:

```python
# generate_synthetic.py
a = math.radians(angle_deg - 90)
tip_x = cx + r * cos(a)
tip_y = cy + r * sin(a)
```

Verified by roundtrip test across all quadrants (0, 45, 90, 135, 180, 270, 359 deg): recovered angles match input angles to < 0.1 deg.

---

## 4. Preprocessing in `infer.py`

### Problem

Real input images may not be exactly 192x192. The model expects 192x192 input, and the angle calculation assumes the dial centre sits at pixel (96, 96). Preprocessing must resize the canvas without shifting the dial centre.

### Design constraint

**The centre of the raw input image must map to pixel (96, 96) in the preprocessed canvas.** This ensures the angle calculation — which is measured from (96, 96) — remains valid regardless of input size.

### Algorithm

Each axis (width, height) is handled independently with the same logic:

#### Case 1: axis > 192 — centre crop

```
crop_left  = (raw_size - 192) // 2
crop_right = raw_size - 192 - crop_left
```

Equal pixels are removed from both sides. The raw centre at `raw_size // 2` maps to `raw_size // 2 - crop_left`. For even `raw_size`, this equals exactly 96. For odd `raw_size`, the error is at most 0.5 px.

**Why cropping works**: the dial meter is assumed to be roughly centred in the input photograph. Peripheral content (background, housing) is not relevant to the prediction.

#### Case 2: axis < 192 — symmetric padding

```
pad_left  = (192 - raw_size) // 2
pad_right = 192 - raw_size - pad_left
```

Equal padding is added to both sides. The fill colour is the mean RGB of the two edge rows/columns of the raw image. This avoids introducing hard edges or an unnatural uniform colour that the model has never seen during training.

**Why edge-mean padding works**: the synthetic training data has near-white, subtly textured backgrounds. The edge pixels of a real dial photo are typically background. Extending that colour into the padding region produces a smooth continuation that does not create false features for the model to latch onto.

#### Case 3: axis == 192 — no-op

No transformation. Offset is 0.

### Offset tracking

The preprocessing returns an offset `(ox, oy)` so the predicted tip can be mapped back to raw-image coordinates:

```
original_x = preprocessed_x + ox
original_y = preprocessed_y + oy
```

| Scenario | ox / oy |
|---|---|
| Cropped | positive (pixels were removed from the left/top) |
| Padded | negative (pixels were added to the left/top) |
| Exact | 0 |

The angle is computed **before** applying the offset, in preprocessed space where the centre is always (96, 96).

### Centre alignment precision

For axes with an odd pixel difference, `pad_left` and `pad_right` (or `crop_left` and `crop_right`) differ by 1 px. This introduces a maximum centre misalignment of 0.5 px, which at a typical tip distance of ~50 px from centre corresponds to < 0.6 deg of angular error. This is below the model's prediction noise floor and can be disregarded.

---

## 5. Inference pipeline summary

```
1. Load image, convert to RGB
2. Preprocess:
   a. Centre-crop any axis > 192
   b. Edge-mean-pad any axis < 192
   c. Record (ox, oy) offset
3. Normalise to float32 [0, 1], reshape to [1, 3, 192, 192]
4. Forward pass through TipHeatmapNet → [1, 1, 192, 192] heatmap
5. argmax on heatmap → (pred_x, pred_y) in preprocessed space
6. angle = atan2(pred_x - 96, -(pred_y - 96)) * 180/pi  % 360
7. original coords = (pred_x + ox, pred_y + oy)
```

---

## 6. Usage

```bash
python infer.py photo.png --model tip_heatmap_net.pth --save-heatmap hm.png
```

Output:
```
Input: photo.png  (200x180)
Centre convention: (96, 96) in 192x192 canvas
Predicted tip (original coords): (150, 68)
Angle: 47.3 deg  (CW from up, 0=up 90=right 180=down 270=left)
Heatmap saved to hm.png
```

---

## 7. Limitations

- **Centre assumption**: the dial must be approximately centred in the input image. If the dial is off-centre, the crop/pad logic will not align it correctly and the angle will be wrong.
- **Single pointer**: the model predicts exactly one heatmap peak. Images with multiple pointers or no pointer will produce unreliable results.
- **Resolution**: the model was trained at 192x192. Inputs significantly larger or smaller than this (even after crop/pad) may have different effective feature scales than the training data.
- **Argmax quantisation**: the tip coordinate is integer-valued (pixel index). At a tip distance of 50 px from centre, 1 px of error corresponds to ~1.1 deg of angular error. Sub-pixel refinement (e.g. soft-argmax) is not implemented.
