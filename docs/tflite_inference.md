# TFLite Tip-Heatmap Inference Pipeline

Engineering documentation for `tip_heatmap_net.tflite` and `infer_tflite.py`.

---

## 1. System overview

The TFLite model is a converted version of the PyTorch `tip_heatmap_net.pth`. It predicts the angular position of an analog dial meter's pointer from a photograph, without requiring PyTorch or CUDA at runtime.

```
raw image  ──►  preprocess (crop/pad to 192x192)  ──►  TFLite model  ──►  heatmap  ──►  argmax  ──►  angle
```

Runtime dependencies: `numpy`, `Pillow`, and one of:
- `ai_edge_litert` (recommended, lightweight)
- `tensorflow` (fallback)

---

## 2. Model details

The TFLite model is a float32 conversion of the PyTorch U-Net, produced via the `PyTorch → ONNX → onnx2tf → TFLite` pipeline in `convert_tflite.py`.

| Property | Value |
|---|---|
| Input tensor | `[1, 192, 192, 3]` float32, NHWC, pixels in [0, 1] |
| Output tensor | `[1, 192, 192, 1]` float32, NHWC, values in [0, 1] |
| Quantisation | none (float32) |
| Training sigma | 3 px (Gaussian heatmap target) |

### Key difference from PyTorch model

The TFLite model uses **NHWC** channel ordering (height, width, channels) for both input and output, whereas the PyTorch model uses **NCHW**. `infer_tflite.py` handles this transparently — the preprocessing produces `[H, W, 3]` arrays directly instead of transposing to `[3, H, W]`.

### Training target

The ground-truth heatmap is a 2-D Gaussian centred on the true tip coordinate:

```
heatmap[y, x] = exp( -((x - tip_x)^2 + (y - tip_y)^2) / (2 * sigma^2) )
```

`sigma = 3 px` at 192x192 resolution. This produces a blob approximately 18 px in diameter (6-sigma). The tighter sigma (compared to the default of 5 px) yields sharper heatmap peaks and improved localisation precision at the cost of a slightly harder optimisation surface during training.

---

## 3. Coordinate system

Identical to the PyTorch pipeline. All scripts share a single coordinate convention:

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

```
dx = tip_x - 96
dy = -(tip_y - 96)
angle = atan2(dx, dy) * 180 / pi  % 360
```

---

## 4. Preprocessing

Identical to `infer.py`. Each axis is handled independently:

| Input axis size | Action | Offset |
|---|---|---|
| > 192 | Centre-crop to 192 | positive |
| < 192 | Edge-mean-pad to 192 | negative |
| = 192 | No-op | 0 |

The preprocessing returns `(ox, oy)` to map predictions back to original image coordinates:

```
original_x = preprocessed_x + ox
original_y = preprocessed_y + oy
```

The angle is always computed in preprocessed space where the centre is (96, 96).

---

## 5. Inference pipeline summary

```
1. Load image, convert to RGB
2. Preprocess:
   a. Centre-crop any axis > 192
   b. Edge-mean-pad any axis < 192
   c. Record (ox, oy) offset
3. Normalise to float32 [0, 1], shape [1, 192, 192, 3]  (NHWC)
4. Run TFLite interpreter → [1, 192, 192, 1] heatmap
5. Squeeze to [192, 192], argmax → (pred_x, pred_y)
6. angle = atan2(pred_x - 96, -(pred_y - 96)) * 180/pi  % 360
7. original coords = (pred_x + ox, pred_y + oy)
```

---

## 6. Usage

```bash
python infer_tflite.py photo.png --model tip_heatmap_net.tflite --save-heatmap hm.png
```

Output:
```
Input: photo.png  (200x180)
Centre convention: (96, 96) in 192x192 canvas
Predicted tip (original coords): (150, 68)
Angle: 47.3 deg  (CW from up, 0=up 90=right 180=down 270=left)
Heatmap saved to hm.png
```

### Programmatic usage

```python
from PIL import Image
from infer_tflite import _load_interpreter, predict

interp = _load_interpreter("tip_heatmap_net.tflite")
img = Image.open("photo.png").convert("RGB")
angle, tip_x, tip_y, heatmap = predict(interp, img)
```

---

## 7. Numerical equivalence with PyTorch

The TFLite model produces predictions within ~0.001 max heatmap difference of the PyTorch model. `validate_pth_vs_tflite.py` confirms this across thousands of synthetic samples — tip coordinates match exactly in >98% of cases, with the remaining cases differing by at most 1 px.

---

## 8. Limitations

- **Centre assumption**: the dial must be approximately centred in the input image. If the dial is off-centre, the crop/pad logic will not align it correctly and the angle will be wrong.
- **Single pointer**: the model predicts exactly one heatmap peak. Images with multiple pointers or no pointer will produce unreliable results.
- **Resolution**: the model was trained at 192x192. Inputs significantly larger or smaller than this (even after crop/pad) may have different effective feature scales than the training data.
- **Argmax quantisation**: the tip coordinate is integer-valued (pixel index). At a tip distance of 50 px from centre, 1 px of error corresponds to ~1.1 deg of angular error. Sub-pixel refinement (e.g. soft-argmax) is not implemented.
- **Float32 only**: the model is not quantised. If targeting constrained devices, INT8 quantisation can be added to the `convert_tflite.py` pipeline with a representative dataset.
